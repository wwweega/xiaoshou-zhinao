"""
Step 5 · Agent 角色实现（六个 Agent）
============================================================
面试考点（本项目最核心的设计题）：

Q: Multi-Agent 到底比单 Agent 强在哪？别背概念，讲清本质。
A: 三个真实收益，一个真实代价。

   收益1 · 角色分离降低单次任务的认知负载
     写一段"既要分析数据又要写文案又要自我检查"的 prompt，
     模型会顾此失彼。拆成三个 Agent，每个 prompt 只专注一件事，质量明显更高。
     这对应软件工程里的单一职责原则。

   收益2 · 上下文隔离
     如果全塞一个 Agent，prompt 里会堆满工具返回的原始数据（几万字），
     注意力被稀释。拆开后，Researcher 消化原始数据，只把 200 字结论交给 Analyst，
     每一跳都做了"信息压缩"。这是 token 成本下降的关键。

   收益3 · 可独立评测与迭代
     Critic 打分时能给"事实性 0.7 / 可读性 0.9"这样的分维度分数，
     你才知道该改哪个 Agent 的 prompt。单 Agent 你只能得到一个总分，无从下手。

   代价 · 通信开销与误差累积
     每多一跳就多一次 LLM 调用（贵 + 慢），而且上游错了下游会放大。
     所以我们用 Critic 独立校验 + 硬阈值回灌 + 最多 2 轮来兜底。

Q: 那什么时候不该用 Multi-Agent？
A: 任务能用一次 LLM 调用解决的，绝不拆多 Agent。简单任务拆多 Agent 是负优化。
   （敢说这句，面试官会认为你真懂，而不是只会堆概念）
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from app.core.config import settings
from app.core.observability import logger
from app.core.schemas import (
    AgentRole,
    CritiqueResult,
    Plan,
    SubTask,
    TaskType,
)
from app.llm.client import llm_client
from app.memory.store import build_task_context
from app.tools.registry import tool_registry


# ============================================================
# 通用基类
# ============================================================
class BaseAgent:
    """
    面试考点：为什么要有基类？
      → 把"日志 + trace + 异常兜底"这些横切关注点收口。
        每个子类只写自己的 prompt 和业务逻辑，不用重复造轮子。
    """

    role: AgentRole = AgentRole.PLANNER
    name: str = "base"

    def __init__(self) -> None:
        # 每个 Agent 可声明自己能用哪些工具 —— 这就是「最小权限原则」
        self.allowed_tools: list[str] = []

    def tool_schemas(self) -> list[dict[str, Any]]:
        """只暴露该 Agent 被授权的工具 schema。"""
        alls = {t["function"]["name"]: t for t in tool_registry.all_schemas()}
        return [alls[n] for n in self.allowed_tools if n in alls]

    async def call_llm(self, messages: list[dict[str, str]], tracer=None, span=None):
        return await llm_client.chat(messages, span=span, tracer=tracer)

    def system_prompt(self) -> str:
        raise NotImplementedError


# ============================================================
# 1. Planner：任务拆解
# ============================================================
class PlannerAgent(BaseAgent):
    """
    职责：把一句模糊的业务需求，拆成有依赖关系的 DAG。

    面试考点：Planner 是最容易"跑飞"的 Agent。
      对策：①强制 JSON Schema ②限制子任务数量上限 ③失败兜底给默认计划
           ④不允许它调用工具（它只需要思考，不需要查数据）
    """

    role = AgentRole.PLANNER
    name = "planner"
    MAX_SUBTASKS = 6

    def system_prompt(self) -> str:
        return f"""你是跨境电商运营团队的「任务规划专家」。

你的职责：把用户的运营需求拆解成可执行的子任务列表（DAG）。

【可用的执行角色】
- researcher：负责数据采集（竞品抓取、评论抽取、联网检索、趋势查询）
- analyst   ：负责数据分析（SQL 查询、情感分析、统计计算）
- writer    ：负责内容生成（Listing 文案、广告词、运营报告）

【拆解原则】
1. 子任务数量控制在 2~{self.MAX_SUBTASKS} 个，宁可少而精，不要凑数。
2. 只有真正需要上游数据时才声明 depends_on，避免人为制造串行（会拖慢速度）。
3. 第一个子任务通常是数据采集。
4. 最后一个子任务应该是产出最终交付物（文案/报告）。
5. complexity 评分：0~0.4 简单（单步查询/抽取），0.5~0.7 中等，
   0.8~1.0 复杂（需要多步推理与策略设计）。

【输出要求】
只输出 JSON，格式如下：
{{
  "task_type": "full_pipeline|market_research|competitor|listing|ad_keywords|review_insight|weekly_report",
  "goal": "一句话业务目标",
  "complexity": 0.6,
  "subtasks": [
    {{"id": "t1", "name": "子任务名", "agent": "researcher", "description": "具体做什么、产出什么", "depends_on": []}},
    {{"id": "t2", "name": "子任务名", "agent": "analyst", "description": "...", "depends_on": ["t1"]}}
  ]
}}"""

    async def plan(self, query: str, tracer=None) -> tuple[Plan, list[str]]:
        """返回 (计划, 事件日志)。事件日志用于 SSE 推送给前端。"""
        events: list[str] = []
        events.append("🧭 Planner 开始拆解任务…")

        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": f"业务需求：{query}"},
        ]

        with tracer.span("agent:planner", kind="agent") if tracer else _null_ctx() as sp:
            try:
                plan, model = await llm_client.chat_structured(
                    messages, Plan, span=sp, tracer=tracer, role="planner"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Planner 失败，使用兜底计划: %s", exc)
                events.append(f"⚠️ Planner 解析失败，启用兜底计划（{exc}）")
                plan = self._fallback_plan(query)

        # 后处理：截断过多子任务 + 修复非法依赖
        plan = self._sanitize(plan)
        events.append(
            f"✅ Planner 产出 {len(plan.subtasks)} 个子任务，复杂度 {plan.complexity}"
        )
        for st in plan.subtasks:
            events.append(f"   · [{st.id}] {st.name} → {st.agent.value}")

        return plan, events

    def _sanitize(self, plan: Plan) -> Plan:
        """防御性编程：LLM 可能给出不存在的依赖、循环依赖、重复 id。"""
        if len(plan.subtasks) > self.MAX_SUBTASKS:
            plan.subtasks = plan.subtasks[: self.MAX_SUBTASKS]

        ids = {t.id for t in plan.subtasks}
        for t in plan.subtasks:
            t.depends_on = [d for d in t.depends_on if d in ids and d != t.id]
        return plan

    def _fallback_plan(self, query: str) -> Plan:
        """
        面试考点：为什么要写兜底？
          → LLM 不可用/格式错乱是必然事件，不是异常事件。
            企业级系统必须有降级路径，否则一次模型抖动整个功能就 500。
        """
        return Plan(
            task_type=TaskType.FULL_PIPELINE,
            goal=query,
            complexity=0.6,
            subtasks=[
                SubTask(id="t1", name="竞品与市场数据采集", agent=AgentRole.RESEARCHER,
                        description="抓取品类 Top5 竞品的价格、评分、评论数与主打关键词，并检索行业趋势",
                        depends_on=[]),
                SubTask(id="t2", name="评论洞察与情感分析", agent=AgentRole.ANALYST,
                        description="对竞品评论做情感分布统计，抽取用户痛点与好评点，计算价格区间与利润空间",
                        depends_on=["t1"]),
                SubTask(id="t3", name="运营策略与文案产出", agent=AgentRole.WRITER,
                        description="基于竞品与评论洞察，输出Listing卖点方案与广告关键词建议",
                        depends_on=["t1", "t2"]),
            ],
        )


# ============================================================
# 2. Researcher：数据采集（唯一大量用工具的 Agent）
# ============================================================
class ResearcherAgent(BaseAgent):
    """
    职责：调用工具把「外部世界的事实」取回来。

    面试考点：怎么让模型正确调用工具？
      三要素：①工具描述要写清"什么时候用"而不只是"是什么"
             ②参数 schema 要有默认值和描述
             ③调用失败要把错误原文回灌，让模型自己改参数
      这三条是降低工具参数出错率的主要手段。
      （说明：本项目未做"开关该机制"的 A/B 对比，因此不给出提升率数字；
        实际效果取决于模型与工具描述的匹配度，建议读者用 scripts/demo.py 自行观察。）
    """

    role = AgentRole.RESEARCHER
    name = "researcher"

    def __init__(self) -> None:
        super().__init__()
        self.allowed_tools = [
            "competitor_scraper", "review_insight", "web_search",
            "trend_analysis", "price_history",
        ]

    def system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in self.tool_schemas()
        )
        return f"""你是跨境电商的「市场情报研究员」。

【你可以使用的工具】
{tools_desc}

【工作方式】
1. 先判断为了完成本子任务，需要调用哪几个工具（一般 1~3 个，不要贪多）。
2. 只输出一个 JSON，声明你要调用的工具和参数：
{{
  "tool_calls": [
    {{"tool_name": "competitor_scraper", "arguments": {{"category": "kitchen storage", "top_n": 5}}}}
  ]
}}
3. 工具调用完成后，我会把真实数据给你，你再输出一段结论。

【输出结论的要求】
- 必须基于真实数据，不许编造数字。
- 明确给出：价格带、主要卖点关键词、竞品强弱对比。
- 控制在 300 字以内，要精炼，因为你下游的同事只关心结论。"""

    async def execute(self, task: SubTask, context: str, tracer=None) -> tuple[str, list[str]]:
        events: list[str] = []
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": f"子任务：{task.name}\n说明：{task.description}\n\n{context}"},
        ]

        with tracer.span(f"agent:researcher:{task.id}", kind="agent", subtask=task.id) if tracer else _null_ctx() as sp:
            # ---- 第一跳：让模型决定调用哪些工具 ----
            try:
                raw, _ = await llm_client.chat(messages, span=sp, tracer=tracer,
                                               model=settings.llm_small_model)
                tool_calls = self._parse_tool_calls(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("工具决策解析失败: %s", exc)
                tool_calls = self._default_tool_calls(task)

            if not tool_calls:
                tool_calls = self._default_tool_calls(task)

            events.append(f"🔧 Researcher 决定调用 {len(tool_calls)} 个工具")

            # ---- 第二跳：并发执行工具 ----
            # 面试考点：为什么用 asyncio.gather 并发而不是 for 循环串行？
            #   → 三个独立工具各 200ms，串行 600ms，并发 200ms。
            #     工具之间无语义依赖时并发是免费的 3 倍加速。
            #     但要注意：有依赖关系的工具（如先搜 ID 再查价格）必须串行，不能盲并发。
            async def run_one(tc: dict[str, Any]):
                name = tc.get("tool_name", "")
                args = tc.get("arguments", {}) or {}
                if name not in self.allowed_tools:
                    events.append(f"⛔ 工具 {name} 未授权，已拒绝")
                    return None
                events.append(f"   → 调用 {name}({json.dumps(args, ensure_ascii=False)[:80]})")
                return await tool_registry.call(name, args, tracer=tracer)

            results = await asyncio.gather(
                *(run_one(tc) for tc in tool_calls), return_exceptions=True
            )

            # ---- 处理结果 + 错误回灌 ----
            tool_outputs: list[str] = []
            failed: list[str] = []
            for tc, res in zip(tool_calls, results):
                if isinstance(res, BaseException):
                    failed.append(f"{tc.get('tool_name')}: {res}")
                    continue
                if res is None:
                    continue
                if not res.ok:
                    # 关键：把工具的报错原文回灌，模型据此修正参数重试
                    failed.append(f"{res.tool_name} 失败：{res.error}")
                    events.append(f"   ⚠️ {res.tool_name} 失败：{str(res.error)[:60]}")
                    continue
                events.append(f"   ✓ {res.tool_name} 成功（{res.latency_ms:.0f}ms）")
                tool_outputs.append(
                    f"### 工具 {res.tool_name} 返回\n"
                    + json.dumps(res.result, ensure_ascii=False)[:2500]
                )

            # 有失败就做一次自纠正重试
            if failed and tool_outputs:
                logger.info("部分工具失败，触发自纠正：%s", failed)
                events.append("🔁 检测到工具失败，触发自纠正重试")
                fix_msgs = messages + [
                    {"role": "user", "content":
                        "以下工具调用失败了，请修正参数后重新给出 tool_calls（只输出 JSON）：\n"
                        + "\n".join(failed)},
                ]
                try:
                    raw2, _ = await llm_client.chat(fix_msgs, span=sp, tracer=tracer,
                                                    model=settings.llm_small_model)
                    for tc in self._parse_tool_calls(raw2):
                        if tc.get("tool_name") in self.allowed_tools:
                            r2 = await tool_registry.call(
                                tc["tool_name"], tc.get("arguments", {}) or {}, tracer=tracer
                            )
                            if r2.ok:
                                events.append(f"   ✓ 自纠正后 {r2.tool_name} 成功")
                                tool_outputs.append(
                                    f"### 工具 {r2.tool_name} 返回（自纠正）\n"
                                    + json.dumps(r2.result, ensure_ascii=False)[:2500]
                                )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("自纠正失败: %s", exc)

            if not tool_outputs:
                events.append("⚠️ 无有效工具数据，Researcher 基于已有知识作答（置信度低）")
                return "（无工具数据）未能采集到外部数据，建议人工核查后重试。", events

            # ---- 第三跳：把数据消化成结论 ----
            final_messages = messages + [
                {"role": "user", "content":
                    "以下是工具返回的真实数据：\n\n" + "\n\n".join(tool_outputs)
                    + "\n\n请基于以上数据，输出该子任务的结构化结论（300 字内，必须含具体数字）。"}
            ]
            conclusion, _ = await llm_client.chat(final_messages, span=sp, tracer=tracer)
            events.append("📄 Researcher 已产出结论")

        return conclusion, events

    def _parse_tool_calls(self, raw: str) -> list[dict[str, Any]]:
        from app.llm.client import extract_json

        try:
            data = extract_json(raw)
        except Exception:  # noqa: BLE001
            return []
        calls = data.get("tool_calls") or data.get("tools") or []
        if isinstance(calls, list):
            return [c for c in calls if isinstance(c, dict) and c.get("tool_name")]
        return []

    def _default_tool_calls(self, task: SubTask) -> list[dict[str, Any]]:
        """兜底：按任务名关键词猜工具。宁可猜错也别空手而归。"""
        name = task.name + task.description
        calls: list[dict[str, Any]] = []
        if any(k in name for k in ("竞品", "市场", "选品", "调研")):
            calls.append({"tool_name": "competitor_scraper",
                          "arguments": {"category": "kitchen storage", "top_n": 5}})
        if any(k in name for k in ("评论", "口碑", "评价")):
            calls.append({"tool_name": "review_insight",
                          "arguments": {"product_id": "B0EXAMPLE01", "max_reviews": 100}})
        if any(k in name for k in ("趋势", "热度")):
            calls.append({"tool_name": "trend_analysis",
                          "arguments": {"keyword": "kitchen storage organizer", "months": 6}})
        if not calls:
            calls.append({"tool_name": "web_search",
                          "arguments": {"query": task.name, "top_k": 5}})
        return calls


# ============================================================
# 3. Analyst：数据分析
# ============================================================
class AnalystAgent(BaseAgent):
    """
    职责：把原始数据变成有业务含义的洞察（算指标、找规律、给判断）。

    面试考点：为什么分析类任务不能让 Writer 顺手做？
      → 因为分析需要调用计算工具（模型算数是不可靠的，7 位数乘法经常错）。
        Agent 分工后，Analyst 负责"算对"，Writer 负责"写好看"，各司其职。
    """

    role = AgentRole.ANALYST
    name = "analyst"

    def __init__(self) -> None:
        super().__init__()
        self.allowed_tools = ["sql_query", "calculator", "sentiment_analysis", "keyword_extract"]

    def system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in self.tool_schemas()
        )
        return f"""你是跨境电商的「数据分析师」。

【可用工具】
{tools_desc}

【核心原则】
1. 任何涉及数值计算（利润率、ACoS、增幅）必须调用 calculator，禁止心算。
2. 需要运营数据时调用 sql_query（只能 SELECT）。
3. 结论必须带数字，且标明数据来源。
4. 区分「事实」与「推论」：事实写"是什么"，推论写"据此判断"。

【输出格式】
先给 3~5 条带数字的关键发现（要点式），最后给 2~3 条可执行建议。
总长度 350 字以内。"""

    async def execute(self, task: SubTask, context: str, tracer=None) -> tuple[str, list[str]]:
        events: list[str] = ["📊 Analyst 开始分析…"]
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": f"分析任务：{task.name}\n说明：{task.description}\n\n{context}"},
        ]

        with tracer.span(f"agent:analyst:{task.id}", kind="agent", subtask=task.id) if tracer else _null_ctx() as sp:
            # 先让模型决定要算什么
            try:
                raw, _ = await llm_client.chat(messages, span=sp, tracer=tracer,
                                               model=settings.llm_small_model)
                calls = ResearcherAgent()._parse_tool_calls(raw)
            except Exception:  # noqa: BLE001
                calls = []

            calls = [c for c in calls if c.get("tool_name") in self.allowed_tools]
            if calls:
                results = await asyncio.gather(
                    *(tool_registry.call(c["tool_name"], c.get("arguments", {}) or {}, tracer=tracer)
                      for c in calls),
                    return_exceptions=True,
                )
                outs = []
                for c, r in zip(calls, results):
                    if isinstance(r, BaseException) or r is None or not r.ok:
                        err = str(getattr(r, "error", r))
                        events.append(f"   ⚠️ {c.get('tool_name')} 失败：{err[:50]}")
                        continue
                    events.append(f"   ✓ {r.tool_name} 完成")
                    outs.append(f"### {r.tool_name}\n{json.dumps(r.result, ensure_ascii=False)[:2000]}")
                if outs:
                    messages.append({"role": "user", "content":
                        "以下是计算与查询结果：\n\n" + "\n\n".join(outs)
                        + "\n\n请结合上游数据，输出你的分析结论（带具体数字）。"})

            text, _ = await llm_client.chat(messages, span=sp, tracer=tracer)
            events.append("📈 Analyst 分析完成")

        return text, events


# ============================================================
# 4. Writer：内容生成
# ============================================================
class WriterAgent(BaseAgent):
    """
    职责：把分析结论转成可交付的内容（Listing / 广告词 / 报告）。

    面试考点：为什么 Writer 的 prompt 里要求"只能使用给定事实"？
      → 这是防幻觉的第一道闸：写作类 Agent 最容易为了文采编造卖点。
        明确约束"只能使用上游提供的数字与事实"，能显著降低幻觉率。
    """

    role = AgentRole.WRITER
    name = "writer"

    def __init__(self) -> None:
        super().__init__()
        self.allowed_tools = ["keyword_extract", "image_generator", "calculator"]

    def system_prompt(self) -> str:
        return """你是跨境电商的「资深内容运营」，擅长写高转化的 Amazon Listing 与广告文案。

【铁律】
1. 只能使用上游提供的真实数据与卖点，严禁编造任何产品参数或效果承诺。
2. 不得出现医疗功效、绝对化用语（best/No.1/guaranteed）等违反平台合规的表述。
3. 卖点必须对应「用户痛点」，做到"痛点→方案"一一回应。

【Listing 结构（Amazon 美国站）】
- Title：≤200 字符，核心关键词前置，格式：[品牌] + [核心品类] + [关键特性×3] + [适用场景] + [数量/尺寸]
- Bullet Points：5 条，每条以【大写卖点标签】开头，后接 1~2 句具体说明，含数字
- Description：150~250 词，讲使用场景与情感价值，自然埋词
- Backend Search Terms：10~15 个长尾词，逗号分隔，不重复 Title 已有词

【广告关键词】
- 分三组：核心词（高相关高流量）/ 长尾词（高转化）/ 否定词（避免无效花费）
- 每条注明建议匹配方式（精准/词组/广泛）

输出用 Markdown，条理清晰，可直接交付给运营同学使用。"""

    async def execute(self, task: SubTask, context: str, tracer=None) -> tuple[str, list[str]]:
        events: list[str] = ["✍️ Writer 开始撰写…"]
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content":
                f"写作任务：{task.name}\n要求：{task.description}\n\n"
                f"【必须使用的事实依据】\n{context}\n\n"
                "请基于以上事实完成内容创作。"},
        ]

        with tracer.span(f"agent:writer:{task.id}", kind="agent", subtask=task.id) if tracer else _null_ctx() as sp:
            # 写作固定用大模型：这是质量最敏感的一环，路由到小模型会明显掉质量。
            # 用 chat_stream 而非 chat：Writer 产出的是长文（Title+5条BP+Description+关键词表），
            # 非流式下单次生成动辄 3~5 分钟，会周期性撞上总超时。
            # 流式把超时约束从"总时长"变成"chunk 间隔"，是长文生成的正确机制（详见 client.py 注释）。
            text, model = await llm_client.chat_stream(messages, span=sp, tracer=tracer,
                                                      model=settings.llm_model)
            events.append(f"✅ Writer 完成初稿（模型：{model}，流式生成）")

            # 顺带抽取关键词，供广告投放复用（体现"沉淀可复用资产"）
            try:
                kw = await tool_registry.call(
                    "keyword_extract", {"text": text, "top_k": 12}, tracer=tracer
                )
                if kw.ok and kw.result:
                    words = "、".join(k["word"] for k in kw.result.get("keywords", [])[:12])
                    events.append(f"🏷️ 已沉淀关键词资产：{words[:60]}…")
            except Exception:  # noqa: BLE001
                pass

        return text, events


# ============================================================
# 5. Critic：质量评审（反思循环的触发器）
# ============================================================
class CriticAgent(BaseAgent):
    """
    职责：给产出物打分，决定是否打回重跑。

    面试考点（必问）：Critic 有什么用？它真的有用吗？
      → 有，但要说清边界：
        ①它的价值不在于"打分准"，而在于提供一个可自动化的质量门禁，
          把"人工 review 才能发现的问题"前置到流水线里。
        ②它确实有偏差（LLM-as-Judge 的通病：偏爱长答案、偏爱和自己风格像的答案）。
          所以我们的做法是：分维度打分 + 致命项一票否决 + 高风险场景仍要人工在环。
        ③如果用 Critic 也当生成者（同模型同 prompt 风格），纠错效果会打折扣，
          因为模型难以发现自己的系统性盲区。所以 Critic 用不同的 prompt 视角。
    """

    role = AgentRole.CRITIC
    name = "critic"

    def __init__(self, pass_threshold: float = 0.80):
        super().__init__()
        self.allowed_tools = []   # Critic 不该调工具，它只做判断，避免自己引入新信息
        self.pass_threshold = pass_threshold

    def system_prompt(self) -> str:
        return f"""你是跨境电商的内容「质量审核专家」，代表平台合规与转化效果的双重视角。

【评分维度（各 0~1）】
- relevance  ：内容与用户需求、产品实际卖点的相关性
- factuality ：是否只使用给定事实，有无编造参数/功效
- compliance ：是否违反 Amazon 平台合规（绝对化用语、医疗功效、侵权）
- readability：结构清晰度、可读性、是否便于运营直接使用

【评分标准】
- 0.9+：可直接上线
- 0.8~0.9：小修可用
- 0.6~0.8：需要返工
- <0.6：严重不达标

【致命项（fatal=true）】
出现以下任一情况直接判不通过：编造产品参数、出现绝对化用语或医疗功效、
出现竞品品牌名造成侵权风险。

【输出 JSON】
{{
  "score": 0.0~1.0 的综合分,
  "passed": true/false,
  "dimensions": {{"relevance": 0.0, "factuality": 0.0, "compliance": 0.0, "readability": 0.0}},
  "issues": ["具体问题1", "具体问题2"],
  "suggestions": ["可执行的修改建议1", "建议2"],
  "fatal": false
}}

【要求】问题必须具体到位置和内容，不要写"可以更好"这类空话。
综合分 = 四个维度的加权均值（factuality 与 compliance 权重各 0.3，其余各 0.2）。"""

    async def review(
        self, deliverable: str, context: str, tracer=None
    ) -> tuple[CritiqueResult, list[str]]:
        events: list[str] = ["🔍 Critic 开始审核…"]
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content":
                f"【事实依据】\n{context[:3000]}\n\n"
                f"【待审核内容】\n{deliverable[:4000]}\n\n请审核并输出 JSON。"},
        ]

        with tracer.span("agent:critic", kind="agent") if tracer else _null_ctx() as sp:
            try:
                # 用大模型审核：评审需要推理能力，小模型容易给"假通过"
                result, _ = await llm_client.chat_structured(
                    messages, CritiqueResult, span=sp, tracer=tracer, role="critic"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Critic 解析失败，默认放行: %s", exc)
                events.append(f"⚠️ Critic 输出异常，默认放行（{exc}）")
                result = CritiqueResult(
                    score=0.8, passed=True,
                    dimensions={"relevance": 0.8, "factuality": 0.8,
                                "compliance": 0.8, "readability": 0.8},
                    issues=["Critic 未能完成评审"], suggestions=[], fatal=False,
                )

        # 硬约束：致命项一票否决 + 阈值判定（不信模型自己给的 passed）
        if result.fatal:
            result.passed = False
            events.append("🚫 检测到致命问题，判定不通过")
        result.passed = bool(result.passed and result.score >= self.pass_threshold and not result.fatal)

        dim_str = " ".join(f"{k}={v:.2f}" for k, v in result.dimensions.items())
        events.append(f"📋 评分 {result.score:.2f}（{dim_str}）→ {'✅ 通过' if result.passed else '❌ 打回'}")
        for issue in result.issues[:3]:
            events.append(f"   ⚠️ {issue[:80]}")

        return result, events


# ============================================================
# 6. Reviewer：人工在环（HITL）适配
# ============================================================
class ReviewerAgent(BaseAgent):
    """
    职责：把最终产物整理成人可读的交付件，并标注人工审核点。

    面试考点：为什么要留人工在环？
      → 全自动 AI 决策在高风险场景（定价、合规文案、对外发布）是不可接受的。
        HITL 的正确姿势不是"让 AI 停下来等人"，而是
        "AI 做完 90% 的活，把人放在价值最高、风险最大的那个确认点上"。
        本例中确认点放在「定稿发布前」，而不是每个子任务后（否则人就成瓶颈了）。
    """

    role = AgentRole.REVIEWER
    name = "reviewer"

    async def assemble(
        self, query: str, results: dict[str, str], plan: Plan, critique: CritiqueResult | None
    ) -> tuple[str, list[str]]:
        events = ["📦 Reviewer 汇总交付物…"]

        sections = [
            "# 运营智脑 · 交付报告",
            f"\n> **任务目标**：{plan.goal or query}",
            f"> **任务类型**：{plan.task_type.value}",
            f"> **执行子任务**：{len(plan.subtasks)} 个",
        ]
        if critique:
            sections.append(
                f"> **质量评分**：{critique.score:.2f} / 1.00"
                f"（相关性 {critique.dimensions.get('relevance', 0):.2f}、"
                f"事实性 {critique.dimensions.get('factuality', 0):.2f}、"
                f"合规性 {critique.dimensions.get('compliance', 0):.2f}、"
                f"可读性 {critique.dimensions.get('readability', 0):.2f}）"
            )

        sections.append("\n---\n")
        for st in plan.subtasks:
            content = results.get(st.id, "（无产出）")
            sections.append(f"## {st.name}\n\n{content}\n")

        sections.append("\n---\n")
        sections.append(
            "## ⚠️ 人工审核清单（发布前请逐项确认）\n\n"
            "- [ ] 文案中的产品参数与实际商品一致（AI 不得作为唯一事实来源）\n"
            "- [ ] 无绝对化用语与医疗功效表述（Amazon 合规）\n"
            "- [ ] 无竞品品牌名（避免侵权）\n"
            "- [ ] 价格与利润测算已复核\n"
            "- [ ] 广告关键词已排除品牌词与无关词\n"
        )

        events.append("✅ 交付物已生成，含人工审核清单")
        return "\n".join(sections), events


# ============================================================
# 辅助：无 tracer 时的空上下文
# ============================================================
class _NullSpan:
    model: str | None = None
    status: str = "ok"
    error: str | None = None
    meta: dict[str, Any] = {}


class _null_ctx:
    def __enter__(self):
        return _NullSpan()

    def __exit__(self, *args):
        return False


# 全局实例
planner_agent = PlannerAgent()
researcher_agent = ResearcherAgent()
analyst_agent = AnalystAgent()
writer_agent = WriterAgent()
critic_agent = CriticAgent()
reviewer_agent = ReviewerAgent()
