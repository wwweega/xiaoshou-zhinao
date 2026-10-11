"""
单元测试
============================================================
面试考点：为什么 AI 项目也要写单测？
  → 因为 AI 项目里有大量「非 AI 的确定性逻辑」：DAG 拓扑排序、模型路由打分、
    JSON 提取、缓存、权限控制。这些必须 100% 正确，不能"跑一次看着没问题"就上线。
    把确定性部分测透，才能让不确定性（LLM）的调试成本可控。

运行：python -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.core.schemas import AgentRole, Plan, SubTask, TaskStatus, TaskType
from app.llm.client import ComplexityRouter, extract_json
from app.memory.store import ConversationMemory, LongTermMemory, build_task_context, estimate_tokens
from app.tools.registry import build_default_registry, tool_registry


# ============================================================
# 1. JSON 提取（LLM 输出的容错能力）
# ============================================================
class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        """模型最爱加的 markdown 代码块。"""
        raw = '好的，这是结果：\n```json\n{"a": 1, "b": "中文"}\n```\n希望有帮助！'
        assert extract_json(raw) == {"a": 1, "b": "中文"}

    def test_prefix_chatter(self):
        """模型在前面寒暄一通。"""
        raw = '根据你的要求，我分析如下：{"score": 0.9} 以上。'
        assert extract_json(raw)["score"] == 0.9

    def test_trailing_comma(self):
        """尾逗号是模型最常犯的格式错误。"""
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_invalid_raises(self):
        from app.llm.client import LLMFormatError
        with pytest.raises(LLMFormatError):
            extract_json("完全没有 JSON 的一段话")


# ============================================================
# 2. 模型路由（成本控制的核心逻辑）
# ============================================================
class TestComplexityRouter:
    def test_planner_is_complex(self):
        """规划类任务应该走大模型。"""
        model, use_small = ComplexityRouter.pick("planner", "做一个完整的市场分析方案")
        assert use_small is False

    def test_researcher_is_simple(self):
        """简单抽取任务应该走小模型。"""
        model, use_small = ComplexityRouter.pick("researcher", "抓取数据")
        assert use_small is True

    def test_score_bounds(self):
        for role in ("planner", "writer", "researcher", "analyst"):
            s = ComplexityRouter.score(role, "策略 优化 分析 对比 建议 推理 规划 预测 洞察 方案 取舍")
            assert 0.0 <= s <= 1.0

    def test_reasoning_keywords_raise_score(self):
        low = ComplexityRouter.score("researcher", "抓取数据")
        high = ComplexityRouter.score("researcher", "抓取数据并分析策略优化方案对比建议")
        assert high > low


# ============================================================
# 3. DAG 拓扑排序（任务调度的正确性）
# ============================================================
class TestPlanTopology:
    def _plan(self) -> Plan:
        return Plan(
            task_type=TaskType.FULL_PIPELINE, goal="测试",
            subtasks=[
                SubTask(id="t3", name="C", agent=AgentRole.WRITER, depends_on=["t2"]),
                SubTask(id="t1", name="A", agent=AgentRole.RESEARCHER, depends_on=[]),
                SubTask(id="t2", name="B", agent=AgentRole.ANALYST, depends_on=["t1"]),
            ],
        )

    def test_topological_order(self):
        order = [t.id for t in self._plan().topological_order()]
        assert order.index("t1") < order.index("t2") < order.index("t3")

    def test_parallel_detection(self):
        """t1 无依赖，应该被识别为可并发执行。"""
        plan = Plan(task_type=TaskType.FULL_PIPELINE, goal="x", subtasks=[
            SubTask(id="t1", name="A", agent=AgentRole.RESEARCHER),
            SubTask(id="t2", name="B", agent=AgentRole.RESEARCHER),
            SubTask(id="t3", name="C", agent=AgentRole.WRITER, depends_on=["t1", "t2"]),
        ])
        assert set(plan.missing_deps()) == {"t1", "t2"}

    def test_cycle_detection(self):
        """循环依赖必须被发现，否则任务会死锁。"""
        plan = Plan(task_type=TaskType.FULL_PIPELINE, goal="x", subtasks=[
            SubTask(id="t1", name="A", agent=AgentRole.RESEARCHER, depends_on=["t2"]),
            SubTask(id="t2", name="B", agent=AgentRole.ANALYST, depends_on=["t1"]),
        ])
        with pytest.raises(ValueError, match="环"):
            plan.topological_order()

    def test_empty_subtasks_rejected(self):
        with pytest.raises(Exception):
            Plan(task_type=TaskType.FULL_PIPELINE, goal="x", subtasks=[])


# ============================================================
# 4. 工具层（校验、缓存、权限、安全）
# ============================================================
class TestToolRegistry:
    def test_all_tools_registered(self):
        assert len(tool_registry.names()) >= 12

    def test_schema_format(self):
        """schema 必须是合法的 Function Calling 格式，否则模型无法调用。"""
        for s in tool_registry.all_schemas():
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert s["function"]["parameters"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_sql_injection_blocked(self):
        """安全红线：模型生成的 SQL 绝不能有写操作。"""
        r = await tool_registry.call("sql_query", {"sql": "DROP TABLE orders"})
        assert r.ok is False
        assert "只读" in (r.error or "")

    @pytest.mark.asyncio
    async def test_calculator_blocks_unsafe(self):
        """安全：计算器不能被执行任意代码。"""
        r = await tool_registry.call("calculator", {"expression": "__import__('os').system('ls')"})
        assert r.ok is False

    @pytest.mark.asyncio
    async def test_calculator_works(self):
        r = await tool_registry.call("calculator", {"expression": "(22.9-8.4)/22.9*100"})
        assert r.ok is True
        assert abs(r.result["result"] - 63.318) < 0.01

    @pytest.mark.asyncio
    async def test_idempotent_cache(self):
        """幂等缓存：同一参数重复调用应命中缓存（降本关键）。"""
        reg = build_default_registry()
        r1 = await reg.call("competitor_scraper", {"category": "unit-test-cat", "top_n": 3})
        assert r1.ok
        r2 = await reg.call("competitor_scraper", {"category": "unit-test-cat", "top_n": 3})
        assert r2.ok
        assert r2.result == r1.result

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_helpful_error(self):
        """工具不存在时，错误信息要能指导模型自我修正。"""
        r = await tool_registry.call("not_exist_tool", {})
        assert r.ok is False
        assert "不存在" in (r.error or "")
        assert "可用工具" in (r.error or "")

    @pytest.mark.asyncio
    async def test_bad_args_no_retry(self):
        """参数错误不该重试（重试也是错的），应快速失败并回灌提示。"""
        r = await tool_registry.call("competitor_scraper", {"wrong_arg": 1})
        assert r.ok is False
        assert r.attempt == 1


# ============================================================
# 5. 记忆系统
# ============================================================
class TestMemory:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("中文中文中文") > 0
        assert estimate_tokens("hello world") > 0

    def test_window_keeps_recent(self):
        m = ConversationMemory(window_size=3)
        for i in range(10):
            m.add("user", f"消息{i}")
        ctx = m.build_context()
        assert len(ctx) == 3
        assert "消息9" in ctx[-1]["content"]

    def test_context_order(self):
        """摘要必须排在最前，保证模型先知道历史全貌。"""
        m = ConversationMemory(window_size=2)
        m.summary = "历史摘要内容"
        m.add("user", "最近1")
        m.add("assistant", "最近2")
        ctx = m.build_context()
        assert "历史摘要" in ctx[0]["content"]

    @pytest.mark.asyncio
    async def test_long_term_recall(self):
        """长期记忆：写入后应能按语义召回。"""
        ltm = LongTermMemory()
        await ltm.remember("厨房收纳品类的平均毛利率是 63%，主推价格带 19.9-24.9 美金")
        await ltm.remember("家居清洁类目竞争激烈，不建议新卖家进入")
        hits = await ltm.recall("厨房收纳的利润情况", top_k=1)
        assert len(hits) == 1
        assert "厨房收纳" in hits[0]["content"]

    @pytest.mark.asyncio
    async def test_fact_exact_lookup(self):
        ltm = LongTermMemory()
        await ltm.remember("结论内容", key="goal:选品分析")
        assert ltm.get_fact("goal:选品分析") is not None

    def test_context_engineering_trims(self):
        """上下文工程：长结果必须截断，否则会吃掉整个 token 预算。"""
        ctx = build_task_context(
            SubTask(id="t1", name="x", agent=AgentRole.WRITER),
            {"t0": "A" * 5000},
        )
        assert "已截断" in ctx
        assert len(ctx) < 2000

    def test_no_dep_context(self):
        ctx = build_task_context(SubTask(id="t1", name="x", agent=AgentRole.WRITER), {})
        assert "无上游依赖" in ctx


# ============================================================
# 6. 可观测性
# ============================================================
class TestObservability:
    def test_span_tree_and_cost(self):
        from app.core.observability import TraceRecorder, MetricsCollector

        t = TraceRecorder(task_id="t_test", query="q")
        with t.span("outer", kind="node") as sp1:
            with t.span("inner", kind="llm") as sp2:
                t.record_llm(sp2, "deepseek-chat", 1000, 500)
        assert len(t.spans) == 2
        assert t.spans[1].parent_id == t.spans[0].span_id
        assert t.spans[1].total_tokens == 1500
        assert t.spans[1].cost_usd > 0
        assert t.total_cost_usd > 0

    def test_error_span(self):
        from app.core.observability import TraceRecorder

        t = TraceRecorder(task_id="t_err")
        with pytest.raises(ValueError):
            with t.span("bad", kind="tool"):
                raise ValueError("boom")
        assert t.spans[0].status == "error"
        assert "boom" in (t.spans[0].error or "")

    def test_p95_calculation(self):
        """P95 的插值算法要对：100 个值里最小的 95 个都该小于等于 P95。"""
        from app.core.observability import MetricsCollector

        m = MetricsCollector()
        m.latencies = [float(i) for i in range(1, 101)]
        p95 = m._pct(m.latencies, 0.95)
        assert 94 <= p95 <= 96

    def test_wall_latency_is_true_elapsed_not_max_span(self):
        """
        回归测试：墙钟耗时必须是「时间区间并集」，不是「最长单 span」。

        背景：原实现 `max(根 span 的 latency)` 在本项目实测 trace 上
        把 1396.7s 的端到端误报成 377.1s（偏差 3.7 倍）。
        这里构造两个「串行且互不包含」的根 span，真实跨度为两段之和。
        """
        import time as _t

        from app.core.observability import TraceRecorder

        t = TraceRecorder(task_id="t_wall")
        with t.span("node_a", kind="node"):
            _t.sleep(0.05)
        _t.sleep(0.02)  # 段间隔：证明墙钟 > 单段，而不是等于最长段
        with t.span("node_b", kind="node"):
            _t.sleep(0.05)

        wall = t.total_latency_ms
        longest = max(s.latency_ms for s in t.spans)
        # 墙钟应覆盖 a + gap + b，必然严格大于任何单个 span
        assert wall > longest, "墙钟必须大于最长单节点（否则就是取 max 的旧 bug）"
        # 期望约 120ms（50 + 20 + 50），给足容差
        assert 110 <= wall <= 400, f"墙钟量级异常: {wall}ms"

    def test_wall_latency_matches_span_timeline(self):
        """墙钟 == max(end_ts) - min(start_ts)，可直接由 span 时间线复算。"""
        import time as _t

        from app.core.observability import TraceRecorder

        t = TraceRecorder(task_id="t_wall2")
        with t.span("n1", kind="node"):
            _t.sleep(0.03)
        with t.span("n2", kind="node"):
            _t.sleep(0.03)

        starts = [s.start_ts for s in t.spans]
        ends = [s.end_ts for s in t.spans]
        expected_ms = (max(ends) - min(starts)) * 1000
        assert abs(t.total_latency_ms - round(expected_ms, 2)) < 1.0

    def test_sum_latency_ge_wall_latency(self):
        """累加口径 ≥ 墙钟口径（串行时基本相等；并行/嵌套时更大）。"""
        import time as _t

        from app.core.observability import TraceRecorder

        t = TraceRecorder(task_id="t_sum")
        with t.span("outer", kind="node"):
            with t.span("inner", kind="agent"):  # 嵌套：outer 包含 inner
                _t.sleep(0.03)
        # 嵌套导致累加把同一段时间算了两遍 → sum > wall
        assert t.sum_latency_ms > t.total_latency_ms


# ============================================================
# 7. 配置安全
# ============================================================
class TestConfig:
    """
    面试考点：为什么这几个测试要显式传 llm_provider？
      → 因为环境变量会污染测试。CI 里我们用 LLM_PROVIDER=mock 跑，
        如果不显式覆盖，测「有 Key 时不该走 mock」就会失败。
        这本身就是个好案例：测试必须与运行环境解耦，否则 CI 里过、本地挂，
        或者反过来，团队就会开始不信任测试。
    """

    def test_mock_autodetect(self):
        """没配 Key 时应自动降级 mock，保证项目开箱可跑。"""
        from app.core.config import Settings

        s = Settings(llm_provider="openai_compatible", llm_api_key="")
        assert s.use_mock is True

    def test_valid_key_not_mock(self):
        from app.core.config import Settings

        s = Settings(llm_provider="openai_compatible", llm_api_key="sk-abcdefghijklmn")
        assert s.use_mock is False

    def test_explicit_mock_provider_wins(self):
        """显式指定 mock 时，即使有 Key 也走 mock（CI 强制开关）。"""
        from app.core.config import Settings

        s = Settings(llm_provider="mock", llm_api_key="sk-abcdefghijklmn")
        assert s.use_mock is True

    def test_small_model_fallback(self):
        """小模型未单独配置时自动复用大模型配置。"""
        from app.core.config import Settings

        s = Settings(llm_api_key="sk-abc1234567", llm_small_base_url="", llm_small_api_key="")
        assert s.llm_small_api_key == "sk-abc1234567"


class TestMockResponseDispatch:
    """
    回归测试：mock 分支判定必须「从具体到宽泛」。

    背景（真实踩坑）：
      最初 _mock_response 把 "子任务" 判定放在最前面。结果 Critic 的 prompt
      因为携带了上游 Planner 的子任务上下文，也命中了该分支，
      拿到的是 Plan JSON 而非 CritiqueResult，导致连续 3 次 Pydantic 校验失败。
      修复方式：把特征最独特的 CritiqueResult 判定提到最前。
    """

    def test_critic_prompt_returns_critique_not_plan(self):
        """Critic 的 prompt 里即使含「子任务」字样，也必须返回 CritiqueResult 结构。"""
        from app.llm.client import _mock_response
        from app.core.schemas import CritiqueResult

        # 模拟真实的 Critic prompt：既含 pass_threshold，又含上游子任务上下文
        messages = [
            {"role": "system", "content": "你是质量评审员，pass_threshold=0.80，输出 score/passed"},
            {"role": "user", "content": "待评审内容：\n子任务 t1 结果……\n子任务 t2 结果……"},
        ]
        raw = _mock_response(messages)
        data = extract_json(raw)
        # 必须能通过 CritiqueResult 校验，而不是 Plan
        result = CritiqueResult.model_validate(data)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)

    def test_planner_prompt_returns_plan(self):
        """Planner 的 prompt 仍应正确返回 Plan 结构。"""
        from app.llm.client import _mock_response

        messages = [
            {"role": "system", "content": "你是任务规划师，请拆解为子任务，输出 subtasks 数组"},
            {"role": "user", "content": "帮我调研厨房收纳品类"},
        ]
        raw = _mock_response(messages)
        data = extract_json(raw)
        plan = Plan.model_validate(data)
        assert len(plan.subtasks) >= 1


class TestPriceLookup:
    """回归测试：成本统计不能静默变成 0（模型名带日期后缀也要能匹配）。"""

    def test_exact_match(self):
        from app.core.observability import lookup_price

        px_in, px_out = lookup_price("qwen3.8-flash")
        assert px_in > 0 and px_out > 0

    def test_prefix_fallback_for_snapshot_versions(self):
        """带日期后缀的快照版本（如 -0915）应回退到同系列单价，而不是返回 0。"""
        from app.core.observability import lookup_price

        assert lookup_price("qwen3.8-flash-0915") == lookup_price("qwen3.8-flash")
        assert lookup_price("qwen3.8-max-0902") == lookup_price("qwen3.8-max")


class TestTaskStorePersistence:
    """回归测试：任务产物必须能跨进程重启存活，否则长任务成本白花。"""

    def test_survives_restart(self, tmp_path):
        from app.memory.persistence import TaskStore

        s1 = TaskStore(persist_dir=tmp_path)
        s1.save("task_x", {"task_id": "task_x", "status": "completed",
                           "deliverable_content": "# 报告\n" + "内容" * 500})
        # 模拟进程重启：全新实例从磁盘恢复
        s2 = TaskStore(persist_dir=tmp_path)
        got = s2.get("task_x")
        assert got is not None
        assert got["status"] == "completed"
        assert len(got["deliverable_content"]) > 1000

    def test_path_traversal_sanitized(self, tmp_path):
        """task_id 来自外部输入，必须防止路径穿越。"""
        from app.memory.persistence import TaskStore

        s = TaskStore(persist_dir=tmp_path)
        s.save("../../etc/evil", {"task_id": "../../etc/evil", "status": "ok"})
        # 不应在 tmp_path 之外产生文件
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        assert ".." not in files[0].name


class TestDeliverableArchival:
    """回归测试：交付物必须落盘成 Markdown，可追溯、可下载。

    踩坑背景：一次 11.6 分钟 / 5.8 万 token 的报告因为只存在内存里，
    服务一停就永久丢失。归档是"产物可追溯"的底线要求。
    """

    def test_finalize_archives_markdown(self, tmp_path, monkeypatch):
        import asyncio
        import importlib

        from app.core import config as cfg

        monkeypatch.setattr(
            type(cfg.settings), "data_dir",
            property(lambda self: tmp_path), raising=False,
        )
        from app.graph import workflow as wf
        importlib.reload(wf)

        state = {
            "task_id": "task_archive1",
            "query": "归档测试任务",
            "plan": _make_plan(),
            "subtask_results": {"t1": "结论一", "t2": "结论二"},
            "deliverable": "# 测试交付物\n" + "正文内容" * 200,
            "revision_round": 1,
        }
        out = asyncio.run(wf.finalize_node(state))
        assert out["archived_path"], "归档路径不能为空"
        from pathlib import Path

        f = Path(out["archived_path"])
        assert f.exists(), "归档文件必须真实落盘"
        text = f.read_text(encoding="utf-8")
        # 元信息头必须带上任务可追溯信息
        assert "task_archive1" in text
        assert "测试目标" in text
        assert "rounds: 1" in text
        # 正文必须是最终交付内容，且已写入文件（去掉注释头后与 deliverable 一致）
        body = text.split("-->", 1)[-1]
        assert out["deliverable"].strip() in body, "归档文件必须包含完整交付正文"

    def test_archive_path_is_sanitized(self, tmp_path, monkeypatch):
        """task_id 可能来自外部，归档文件名必须净化，防路径穿越。"""
        import asyncio
        import importlib
        from pathlib import Path

        from app.core import config as cfg

        monkeypatch.setattr(
            type(cfg.settings), "data_dir",
            property(lambda self: tmp_path), raising=False,
        )
        from app.graph import workflow as wf
        importlib.reload(wf)
        state = {
            "task_id": "../../evil",
            "query": "x",
            "plan": _make_plan(),
            "subtask_results": {},
            "deliverable": "abc" * 100,
            "revision_round": 0,
        }
        out = asyncio.run(wf.finalize_node(state))
        p = Path(out["archived_path"])
        assert p.exists()
        assert ".." not in p.name
        assert p.parent.name == "deliverables"


def _make_plan():
    """构造一个最小可用的 Plan 供测试使用。"""
    from app.core.schemas import AgentRole, Plan, SubTask, TaskType

    return Plan(
        goal="测试目标",
        task_type=TaskType.FULL_PIPELINE,
        subtasks=[
            SubTask(id="t1", name="采集", agent=AgentRole.RESEARCHER, description="采集"),
            SubTask(id="t2", name="分析", agent=AgentRole.ANALYST, description="分析"),
        ],
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
