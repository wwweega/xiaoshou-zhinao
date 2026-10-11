"""
Step 6 · LangGraph 状态图编排（项目心脏）
============================================================
面试考点（这是简历上「LangGraph 状态图 / 条件边 / 循环反思 / 断点续跑」的代码落点，
也是最容易被深挖的地方，务必吃透）：

Q1: 为什么用 LangGraph，不用自己写 while 循环？
A:  自己写循环能跑，但会缺三样企业级必需的东西：
    ①检查点（Checkpoint）：每个 super-step 后自动持久化，进程挂了能续跑
    ②可观测：每个节点是独立 span，天然可追踪
    ③可视化：图结构能导出成 Mermaid 图，评审/交接成本低
    另外 LangGraph 的 StateGraph 是「显式状态机」，比隐式 while 循环更容易推理——
    你能一眼说出"什么情况下会走到哪个节点"，这对排查线上问题至关重要。

Q2: 什么是 super-step？和普通 step 有什么区别？
A:  一个 super-step = 图中所有「当前可执行节点」并发执行一轮 + 状态合并。
    这是 LangGraph 的并行模型（Pregel 风格，BSP 计算模型）：
    同一 super-step 内的节点并行跑，跑完后一起做状态合并，再进入下一轮。
    所以我们能用它做并行分支，而不用手写线程池。

Q3: 条件边和普通边的区别？
A:  普通边是"必然走向下一个节点"，条件边是"根据状态决定走向哪"。
    critic_router 就是条件边：通过→去 finalize，不通过→回灌重跑。
    这就是循环反思（Reflection）的实现方式。

Q4: 断点续跑怎么做到 100%？
A:  thread_id 是恢复的唯一钥匙。只要有 thread_id，任何时刻都能从
    最后一个 checkpoint 恢复。注意：恢复点必须落在「状态已提交」的边界上，
    所以长耗时操作（如工具调用）不能跨越 super-step 边界。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.roles import (
    analyst_agent,
    critic_agent,
    planner_agent,
    researcher_agent,
    reviewer_agent,
    writer_agent,
)
from app.core.config import settings
from app.core.context import get_tracer, reset_tracer, set_task_context, set_tracer, span_of
from app.core.observability import logger, metrics, TraceRecorder
from app.core.schemas import (
    AgentRole,
    CritiqueResult,
    Deliverable,
    Plan,
    SubTask,
    TaskStatus,
    TaskSummary,
)
from app.memory.persistence import build_checkpointer, task_store
from app.memory.store import ConversationMemory, build_task_context, long_term_memory


# ============================================================
# 图状态定义
# ============================================================
def _merge_results(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    """
    自定义 reducer。

    面试考点：为什么需要 reducer？
      → LangGraph 在并行 super-step 里，多个节点可能同时更新同一个字段。
        默认行为是"后写覆盖"，会丢数据。给字段加 Annotated[dict, reducer]
        就告诉框架"合并而不是覆盖"。
        这是并行状态机的核心机制——不懂 reducer 就会写出偶发丢数据的 bug。
    """
    return {**(a or {}), **(b or {})}


class GraphState(TypedDict, total=False):
    """图的共享状态。所有节点读写同一个状态对象（黑板模式）。"""

    task_id: str
    query: str

    plan: Plan | None
    subtask_results: Annotated[dict[str, str], _merge_results]
    current_batch: list[str]          # 当前 super-step 要执行的子任务 id

    deliverable: str
    critique: CritiqueResult | None
    reflection_rounds: int
    revision_round: int
    approved: bool
    archived_path: str                # 交付物归档路径（落盘后可追溯）

    memory_summary: str
    recall_context: str               # 长期记忆召回结果
    review_history: list[dict[str, Any]]

    events: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    errors: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
    llm_calls: int
    tokens_used: int
    started_at: float


# ============================================================
# 调度辅助：DAG → 批次（实现"任务级并行"）
# ============================================================
def next_ready_subtasks(plan: Plan) -> list[SubTask]:
    """
    找出当前可以执行的所有子任务（依赖已全部完成）。

    面试考点：为什么要"批次"而不是"逐个"？
      → 逐个执行会浪费并行机会。比如 t1、t2 互不依赖，
        串行 2×3s=6s，并发 3s。批次化是 Task 级并行的实现基础。

    注意与 LangGraph 的分工：
      LangGraph 管「节点级」编排（agent 之间怎么走），
      这个函数管「任务级」编排（一个 agent 内部多个 subtask 怎么排）。
      两层编排粒度不同，不能混为一谈。
    """
    done = {t.id for t in plan.subtasks if t.status == TaskStatus.COMPLETED}
    failed = {t.id for t in plan.subtasks if t.status == TaskStatus.FAILED}
    ready: list[SubTask] = []
    for t in plan.subtasks:
        if t.status != TaskStatus.PENDING:
            continue
        if t.id in failed:
            continue
        if all(d in done for d in t.depends_on):
            ready.append(t)
    return ready


# ============================================================
# 节点 1：准备（召回长期记忆 + 装配上下文）
# ============================================================
async def prepare_node(state: GraphState, tracer: TraceRecorder | None = None) -> dict[str, Any]:
    tracer = tracer or get_tracer()
    """
    面试考点：为什么准备阶段要单独成节点，而不塞进 planner？
      → 单一职责 + 可观测。这样 trace 里能清楚看到"记忆召回花了多少时间、
        召回了什么"，排查"为什么 Agent 引用了一个不相关的历史结论"时非常有用。
    """
    events = ["🚀 任务启动，正在召回历史记忆…"]
    query = state["query"]

    # L3 长期记忆召回
    recalled = await long_term_memory.recall(query, top_k=settings.memory_top_k)
    recall_context = ""
    if recalled:
        lines = ["【历史相关结论（来自以往任务，仅供参考，需自行核实）】"]
        for r in recalled:
            lines.append(f"- (相似度 {r['similarity']:.2f}) {r['content'][:200]}")
        recall_context = "\n".join(lines)
        events.append(f"🧠 召回 {len(recalled)} 条历史结论（最高相似度 {recalled[0]['similarity']:.2f}）")
    else:
        events.append("🧠 无历史相关结论（首次处理该类任务）")

    return {
        "recall_context": recall_context,
        "started_at": time.time(),
        "events": events,
        "reflection_rounds": 0,
        "subtask_results": {},
        "errors": [],
        "review_history": [],
        "approved": False,
        "llm_calls": 0,
        "tokens_used": 0,
    }


# ============================================================
# 节点 2：规划
# ============================================================
async def planner_node(state: GraphState, tracer: TraceRecorder | None = None) -> dict[str, Any]:
    tracer = tracer or get_tracer()
    query = state["query"]
    recall = state.get("recall_context", "")
    full_query = f"{query}\n\n{recall}" if recall else query

    plan, events = await planner_agent.plan(full_query, tracer=tracer)

    # 记录任务到 TaskStore（面向业务的查询视图）
    task_store.emit(state["task_id"], {"type": "plan", "plan": plan.model_dump(mode="json")})

    return {
        "plan": plan,
        "events": events,
        "current_batch": [t.id for t in next_ready_subtasks(plan)],
    }


# ============================================================
# 节点 3：执行一批子任务（并行）
# ============================================================
async def execute_batch_node(state: GraphState, tracer: TraceRecorder | None = None) -> dict[str, Any]:
    """
    面试考点（高频）：这里的并发是怎么做的？要注意什么？
      ①用 asyncio.gather 并发执行同批次子任务
      ②用 return_exceptions=True，单个子任务失败不能拖垮整批
      ③并发度要有上限：外部 API 有限流，无限并发会把自己打死
        （生产用 asyncio.Semaphore 控制，这里按批次天然限制）

    另一个高频追问：为什么外部 API 调用要多 Agent 并发，但 LLM 调用要小心？
      → LLM API 通常有 RPM/TPM 限制，并发过高会被 429。
        所以真实生产会在 LLM 客户端层加信号量做全局限流。
    """
    tracer = tracer or get_tracer()
    plan: Plan = state["plan"]  # type: ignore[assignment]
    batch_ids = state.get("current_batch", [])
    results: dict[str, str] = {}
    events: list[str] = []
    errors: list[str] = []
    llm_calls = 0

    by_id = {t.id: t for t in plan.subtasks}
    batch = [by_id[i] for i in batch_ids if i in by_id]
    if not batch:
        return {"events": ["（无待执行子任务）"]}

    events.append(f"⚙️ 并行执行 {len(batch)} 个子任务：{[t.id for t in batch]}")

    async def run_subtask(st: SubTask) -> tuple[SubTask, str, list[str], int]:
        st.status = TaskStatus.RUNNING
        st.started_at = datetime.now()

        # 上下文工程：只注入本子任务声明的依赖项产出
        dep_results = {
            d: state["subtask_results"][d]
            for d in st.depends_on
            if d in state.get("subtask_results", {})
        }
        context = build_task_context(st, dep_results)

        handler = {
            AgentRole.RESEARCHER: researcher_agent,
            AgentRole.ANALYST: analyst_agent,
            AgentRole.WRITER: writer_agent,
        }.get(st.agent)

        if handler is None:
            st.status = TaskStatus.FAILED
            st.finished_at = datetime.now()
            return st, f"（不支持的执行角色：{st.agent}）", [f"⚠️ {st.id} 角色不支持"], 0

        # 编排 agent 会用它下面的子 agent 完成
        if st.agent == AgentRole.PLANNER:
            out, ev = "（规划类子任务已由 Planner 阶段完成）", []
        else:
            out, ev = await handler.execute(st, context, tracer=tracer)  # type: ignore[attr-defined]

        st.result = out
        st.status = TaskStatus.COMPLETED
        st.finished_at = datetime.now()
        return st, out, ev, 1

    # ---------- 并发执行 ----------
    gathered = await asyncio.gather(
        *(run_subtask(st) for st in batch), return_exceptions=True
    )

    for item in gathered:
        if isinstance(item, BaseException):
            errors.append(f"批次执行异常: {item}")
            events.append(f"❌ 子任务异常：{item}")
            continue
        st, out, ev, calls = item
        results[st.id] = out
        events.extend(ev)
        llm_calls += calls
        if st.status == TaskStatus.FAILED:
            errors.append(f"{st.id} 执行失败")
        events.append(f"   ✔ [{st.id}] {st.name} 完成（{st.duration_ms:.0f}ms）")

    # 计算下一批
    remaining = next_ready_subtasks(plan)
    if remaining:
        events.append(f"↪️ 下一批可执行：{[t.id for t in remaining]}")
    else:
        events.append("🏁 所有子任务执行完毕")

    return {
        "subtask_results": results,
        "current_batch": [t.id for t in remaining],
        "events": events,
        "errors": errors,
        "llm_calls": state.get("llm_calls", 0) + llm_calls,
    }


# ============================================================
# 节点 4：汇总成稿
# ============================================================
async def writer_node(state: GraphState, tracer: TraceRecorder | None = None) -> dict[str, Any]:
    tracer = tracer or get_tracer()
    plan: Plan = state["plan"]  # type: ignore[assignment]
    results = state.get("subtask_results", {})

    # 汇总所有依赖产出作为最终写作依据
    context_parts = []
    for st in plan.subtasks:
        if st.id in results:
            context_parts.append(f"### {st.name}\n{results[st.id][:2000]}")
    context = "\n\n".join(context_parts)

    # 若最后一个子任务已经是 Writer 产出，则直接作为初稿，不再重复生成（省 token）
    last = plan.subtasks[-1] if plan.subtasks else None
    if last and last.agent == AgentRole.WRITER and last.id in results:
        draft = results[last.id]
        events = ["📝 复用已有 Writer 产出作为初稿（避免重复生成，节省 token）"]
    else:
        fake_task = SubTask(
            id="final", name="生成最终运营交付文档", agent=AgentRole.WRITER,
            description="整合全部上游结论，输出可直接交付运营使用的完整文档",
        )
        with tracer.span("agent:writer:final", kind="agent"):
            draft, events = await writer_agent.execute(fake_task, context, tracer=tracer)

    return {"deliverable": draft, "events": events}


# ============================================================
# 节点 5：评审
# ============================================================
async def critic_node(state: GraphState, tracer: TraceRecorder | None = None) -> dict[str, Any]:
    tracer = tracer or get_tracer()
    plan: Plan = state["plan"]  # type: ignore[assignment]
    results = state.get("subtask_results", {})
    context = "\n\n".join(f"### {st.name}\n{results.get(st.id, '')[:1500]}" for st in plan.subtasks)

    rounds = state.get("reflection_rounds", 0)
    critique, events = await critic_agent.review(state.get("deliverable", ""), context, tracer=tracer)

    history = list(state.get("review_history", []))
    history.append({
        "round": rounds + 1,
        "score": critique.score,
        "passed": critique.passed,
        "dimensions": critique.dimensions,
        "issues": critique.issues,
        "ts": datetime.now().isoformat(timespec="seconds"),
    })

    events.insert(0, f"🔁 反思轮次 {rounds + 1}/{settings.max_reflection_rounds}")

    return {
        "critique": critique,
        "approved": critique.passed,
        "review_history": history,
        "events": events,
    }


# ============================================================
# 节点 6：回灌重跑（把 Critic 的意见喂回 Writer）
# ============================================================
async def revise_node(state: GraphState, tracer: TraceRecorder | None = None) -> dict[str, Any]:
    tracer = tracer or get_tracer()
    """
    面试考点（必问）：循环反思怎么防止死循环 + 怎么防止越改越差？
      三道保险：
        ①硬上限 max_reflection_rounds（配置化），到了就放行不再改
        ②保留历史最佳版本：如果改完分数更低，就回滚到上一版
           （这是很多人想不到的点——LLM 修改可能把好的地方改坏）
        ③致命项单独处理：合规问题必须改，非致命问题允许"通过但记录"
    """
    plan: Plan = state["plan"]  # type: ignore[assignment]
    critique: CritiqueResult = state["critique"]  # type: ignore[assignment]
    results = state.get("subtask_results", {})
    rounds = state.get("reflection_rounds", 0)

    events = [f"🔧 根据评审意见修订（第 {rounds + 1} 轮）"]

    # 保存当前版本，用于"越改越差"时回滚
    best = state.get("deliverable", "")
    best_score = critique.score

    context = "\n\n".join(f"### {st.name}\n{results.get(st.id, '')[:1500]}" for st in plan.subtasks)

    revise_task = SubTask(
        id="revise", name="按评审意见修订交付文档", agent=AgentRole.WRITER,
        description=(
            "根据质量审核意见修订文档，必须解决以下问题：\n"
            + "\n".join(f"- {i}" for i in critique.issues[:5])
            + "\n\n修改建议：\n"
            + "\n".join(f"- {s}" for s in critique.suggestions[:5])
        ),
    )

    with tracer.span(f"agent:revise:round{rounds + 1}", kind="agent"):
        revised, ev = await writer_agent.execute(revise_task, context, tracer=tracer)

    # 如果 Critic 明确列了问题，把原稿和意见一起给它，效果更好
    events.extend(ev)
    events.append("🧷 已保留历史版本，若新版评分下降将自动回滚")

    return {
        "deliverable": revised,
        "reflection_rounds": rounds + 1,
        "current_batch": [],       # 修订阶段不再跑子任务
        "events": events,
        "_best_draft": best,       # 供 finalize 回滚判断
        "_best_score": best_score,
    }


# ============================================================
# 节点 7：定稿与沉淀
# ============================================================
async def finalize_node(state: GraphState, tracer: TraceRecorder | None = None) -> dict[str, Any]:
    tracer = tracer or get_tracer()
    plan: Plan = state["plan"]  # type: ignore[assignment]
    critique = state.get("critique")
    results = state.get("subtask_results", {})

    # 回滚判断：如果改完分数更低，用回滚版本（防"越改越差"）
    best_draft = state.get("_best_draft")     # type: ignore[arg-type]
    best_score = state.get("_best_score", 0.0)  # type: ignore[arg-type]
    final_draft = state.get("deliverable", "")
    if best_draft and critique and critique.score < best_score:
        logger.info("修订后评分下降（%.2f → %.2f），回滚到上一版", best_score, critique.score)
        final_draft = best_draft
        note = f"⚠️ 修订后评分下降（{best_score:.2f}→{critique.score:.2f}），已自动回滚到上一版本"
    else:
        note = ""

    report, events = await reviewer_agent.assemble(state["query"], results, plan, critique)
    if note:
        events.insert(0, note)

    # 最终交付以 Reviewer 的完整报告为准（含人工审核清单）
    final_content = report if len(report) > len(final_draft) * 0.3 else final_draft

    # ---------- 沉淀长期记忆（供未来任务复用）----------
    try:
        conclusion = (
            f"任务「{plan.goal or state['query']}」的结论："
            + "；".join(
                f"{st.name}: {results.get(st.id, '')[:150]}"
                for st in plan.subtasks if st.id in results
            )
        )
        await long_term_memory.remember(
            conclusion[:1500], kind="task_conclusion",
            key=f"goal:{plan.goal[:50]}" if plan.goal else None,
            payload={"task_id": state["task_id"], "score": critique.score if critique else None},
        )
        events.append("💾 已沉淀任务结论到长期记忆（下次同类任务可复用）")
    except Exception as exc:  # noqa: BLE001
        logger.warning("长期记忆写入失败: %s", exc)

    # ---------- 交付物归档（踩坑修复）----------
    # 面试考点：为什么长任务的产物必须立刻落盘，而不能只留在内存/trace 里？
    #   实测教训：一次 11.6 分钟、5.8 万 token 跑出的 1.3 万字报告，
    #   因为只存在内存 state 里，服务一停就永久丢失——钱白花了。
    #   trace 记录的是「指标」（token/耗时），不是「产物」；两者要分开归档。
    #   这对应「产物可追溯」这一工程要求：
    #   ① 内容落 Markdown（可读、可 diff、可直接交付业务方）
    #   ② 元信息写在文件头注释（评分/轮次/目标，供复盘）
    #   生产应换成对象存储（S3/OSS），但接口收敛在这一处（依赖倒置）。
    archived_path = ""
    try:
        out_dir = settings.data_dir / "deliverables"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c for c in state["task_id"] if c.isalnum() or c in "-_")
        md_path = out_dir / f"{safe_id}.md"
        head = (
            f"<!-- task_id: {state['task_id']} -->\n"
            f"<!-- goal: {plan.goal or state['query']} -->\n"
            f"<!-- score: {critique.score if critique else 'N/A'} -->\n"
            f"<!-- rounds: {state.get('revision_round', 0)} -->\n\n"
        )
        md_path.write_text(head + final_content, encoding="utf-8")
        archived_path = str(md_path)
        events.append(f"📦 交付物已归档：data/deliverables/{md_path.name}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("交付物归档失败: %s", exc)

    return {
        "deliverable": final_content,
        "events": events,
        "current_batch": [],
        "archived_path": archived_path,
    }


# ============================================================
# 条件边路由
# ============================================================
def route_after_execute(state: GraphState) -> Literal["execute", "writer"]:
    """
    执行循环条件边：还有可执行子任务就继续循环，否则去写作。

    面试考点：这个条件边实现了什么？
      → 实现了「任务级 DAG 的动态调度」。LangGraph 的循环边 + 状态里的
        current_batch 共同构成一个"批次推进器"，
        每轮 super-step 消化一批（并行的），直到 DAG 全部完成。
    """
    if state.get("current_batch"):
        return "execute"
    return "writer"


def route_after_critic(state: GraphState) -> Literal["revise", "finalize"]:
    """
    反思循环条件边：不通过且还有轮次预算 → 回灌重跑；否则定稿。

    面试考点：为什么要判 rounds 上限？
      → 没有上限的反思循环会烧钱到天荒地老。企业级系统里，
        一切循环都必须有预算上限（token 预算、轮次预算、时间预算），
        这是"成本可控"的根本。
    """
    critique = state.get("critique")
    rounds = state.get("reflection_rounds", 0)

    if critique and not critique.passed and rounds < settings.max_reflection_rounds:
        return "revise"
    return "finalize"


# ============================================================
# 构图
# ============================================================
def build_graph(checkpointer: Any = None):
    """
    组装状态图。

    面试考点：为什么要提供 build_graph 函数而不是模块级单例图？
      → 测试时可以传入不同的 checkpointer（单测用内存，集成测试用 PG），
        也让"图结构"成为可参数化的对象，便于多租户下定制不同流程。
    """
    graph = StateGraph(GraphState)

    graph.add_node("prepare", _wrap("prepare", prepare_node))
    graph.add_node("planner", _wrap("planner", planner_node))
    graph.add_node("execute", _wrap("execute", execute_batch_node))
    graph.add_node("writer", _wrap("writer", writer_node))
    graph.add_node("critic", _wrap("critic", critic_node))
    graph.add_node("revise", _wrap("revise", revise_node))
    graph.add_node("finalize", _wrap("finalize", finalize_node))

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "planner")
    graph.add_edge("planner", "execute")

    # 循环边 1：批次推进
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"execute": "execute", "writer": "writer"},
    )
    graph.add_edge("writer", "critic")

    # 循环边 2：反思回灌
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"revise": "revise", "finalize": "finalize"},
    )
    graph.add_edge("revise", "critic")   # 改完再评一次，形成闭环
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def _wrap(name: str, fn):
    """
    包装节点：用 contextvars 注入 tracer。

    面试考点：这里为什么要用 contextvars 而不是把 tracer 放进 state？
      → 因为 LangGraph 的 checkpointer 会在每个 super-step 后序列化整个 state，
        而 TraceRecorder 持有 span 列表等不可序列化对象，会直接报错：
            TypeError: Type is not msgpack serializable: TraceRecorder
        根本原因是职责混淆：state 是业务数据（必须可序列化、可跨进程恢复），
        trace 是运行时观测数据（进程内、临时）。两者不该混在一起。
        contextvars 天然按 asyncio task 隔离，并发安全，且对业务代码零侵入。
    """

    async def _node(state: GraphState) -> dict[str, Any]:
        tracer = get_tracer()
        if tracer is None:
            return await fn(state, None)
        with tracer.span(f"node:{name}", kind="node"):
            return await fn(state, tracer)

    _node.__name__ = f"node_{name}"
    return _node


# ============================================================
# 对外执行入口
# ============================================================
@dataclass
class RunResult:
    summary: TaskSummary
    trace_text: str


class ZhinaoRunner:
    """任务执行器：编排 + 追踪 + 指标 + 持久化。"""

    def __init__(self) -> None:
        self.checkpointer = build_checkpointer()
        self.graph = build_graph(self.checkpointer)
        self._llm_client_ref = None

    async def run(
        self,
        query: str,
        task_id: str | None = None,
        thread_id: str | None = None,
        stream: bool = False,
    ) -> RunResult:
        task_id = task_id or f"task_{uuid.uuid4().hex[:10]}"
        thread_id = thread_id or f"thread_{task_id}"

        tracer = TraceRecorder(task_id=task_id, query=query)
        task_store.save(task_id, {
            "task_id": task_id,
            "query": query,
            "status": TaskStatus.RUNNING.value,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        task_store.emit(task_id, {"type": "start", "task_id": task_id, "query": query})

        init: GraphState = {
            "task_id": task_id,
            "query": query,
            "plan": None,
            "subtask_results": {},
            "current_batch": [],
            "deliverable": "",
            "critique": None,
            "reflection_rounds": 0,
            "approved": False,
            "memory_summary": "",
            "recall_context": "",
            "review_history": [],
            "events": [],
            "errors": [],
            "llm_calls": 0,
            "tokens_used": 0,
            "started_at": time.time(),
        }

        config = {"configurable": {"thread_id": thread_id}}
        t0 = time.perf_counter()

        # 关键：tracer 通过 contextvars 绑定，不进入图状态（不可序列化）
        token = set_tracer(tracer)
        set_task_context(task_id, thread_id)
        ok = True
        final_state: dict[str, Any] = {}

        try:
            # LangGraph 的 invoke：内部按 super-step 推进，每个 step 后写 checkpoint
            final_state = await self.graph.ainvoke(init, config=config)  # type: ignore[arg-type]

            # 把执行过程中的事件推给 SSE
            for ev in final_state.get("events", []):
                task_store.emit(task_id, {"type": "log", "message": ev})
        except Exception as exc:  # noqa: BLE001
            ok = False
            logger.exception("任务执行失败: %s", exc)
            task_store.emit(task_id, {"type": "error", "message": str(exc)})
            final_state = {**init, "errors": [f"{type(exc).__name__}: {exc}"]}
        finally:
            # 必须重置，否则痕迹会残留到下一个请求（长驻进程里的经典泄漏）
            reset_tracer(token)

        latency = (time.perf_counter() - t0) * 1000

        plan: Plan | None = final_state.get("plan")  # type: ignore[assignment]
        critique: CritiqueResult | None = final_state.get("critique")  # type: ignore[assignment]

        summary = TaskSummary(
            task_id=task_id,
            query=query,
            status=TaskStatus.COMPLETED if ok else TaskStatus.FAILED,
            plan=plan,
            deliverable=Deliverable(
                title=f"运营智脑交付报告 · {plan.goal if plan else query}",
                content=final_state.get("deliverable", ""),
                meta={"review_history": final_state.get("review_history", [])},
            ) if final_state.get("deliverable") else None,
            critique=critique,
            reflection_rounds=final_state.get("reflection_rounds", 0),
            total_tokens=tracer.total_tokens,
            total_cost_usd=tracer.total_cost_usd,
            latency_ms=round(latency, 2),
            trace_id=tracer.trace_id,
            errors=final_state.get("errors", []),
            review_history=final_state.get("review_history", []),
        )

        metrics.record_task(tracer, ok=ok)
        tracer.save()

        task_store.save(task_id, {
            "task_id": task_id,
            "query": query,
            "status": summary.status.value,
            "score": critique.score if critique else None,
            "tokens": summary.total_tokens,
            "cost_usd": summary.total_cost_usd,
            "latency_ms": summary.latency_ms,
            "trace_id": tracer.trace_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "deliverable_content": (summary.deliverable.content if summary.deliverable else ""),
            "archived_path": final_state.get("archived_path", ""),
        })
        task_store.emit(task_id, {"type": "done", "task_id": task_id,
                                  "score": critique.score if critique else None})

        return RunResult(summary=summary, trace_text=tracer.summary())

    async def resume(self, thread_id: str, task_id: str | None = None) -> RunResult:
        """
        断点续跑：用同一个 thread_id 重新 invoke，LangGraph 会从最后 checkpoint 继续。

        面试考点：这里最容易答错的点是什么？
          → "用 thread_id 就能续跑"不完全对。必须满足：
            ①编译时传了 checkpointer ②状态可序列化 ③图结构没变
            （改了图结构后旧的 checkpoint 可能不兼容，生产中要版本化图）
        """
        logger.info("尝试从 thread_id=%s 恢复", thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        state = await self.graph.aget_state(config)
        if not state or not state.values:
            raise ValueError(f"未找到 thread_id={thread_id} 的检查点")
        logger.info("已恢复到状态，继续执行")
        return await self.run(
            query=state.values.get("query", ""),
            task_id=task_id or state.values.get("task_id"),
            thread_id=thread_id,
        )


runner = ZhinaoRunner()
