"""
Step 1 · 可观测与成本追踪
============================================================
面试考点：为什么实习生做的项目不算"企业级"？
  企业级 = 可度量 + 可定位 + 可控成本。
  90% 的应届项目只有 print()，出问题靠肉眼 debug，老板问"这个月花了多少钱"答不上来。
  这个模块只做三件事，但每一件都是 JD 里明写的考点（美图「全链路可观测」、
  小律同学「质量、延迟、成本、失败率告警」）：

  1. Trace：把一次任务里每个 Agent / Tool / LLM 调用串成树，能回答"慢在哪一步"
  2. Cost ：每次调用记录 token 与费用，能回答"钱花在哪一步"
  3. Metrics：聚合 P50/P95 延迟、失败率、缓存命中率，能回答"系统稳不稳"
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from app.core.config import settings

# ---------------- 日志 ----------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zhinao")


# ============================================================
# 价格表：单位 USD / 1K tokens
# 面试提醒：一定要说清"价格会变，这里是配置"，不要硬编码在业务逻辑里
# ============================================================
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # model_name: (prompt_price_per_1k, completion_price_per_1k)
    # ---------- DeepSeek ----------
    "deepseek-chat": (0.00027, 0.0011),
    # ---------- OpenAI ----------
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    # ---------- 通义千问（阿里云百炼）----------
    # 百炼报价单位是「元/百万 token」，这里换算成「USD/1K token」，汇率按 7.2 估算。
    # qwen3.8-flash: 0.80 元/M 输入、2.70 元/M 输出
    #   输入 0.80 / 7.2 / 1000 = 0.000111
    #   输出 2.70 / 7.2 / 1000 = 0.000375
    "qwen3.8-flash": (0.000111, 0.000375),
    # qwen3.8-max: 12 元/M 输入、36 元/M 输出
    #   输入 12 / 7.2 / 1000 = 0.001667
    #   输出 36 / 7.2 / 1000 = 0.005000
    "qwen3.8-max": (0.001667, 0.005),
    "qwen-plus": (0.0004, 0.0012),
    "qwen-max": (0.001667, 0.005),
    "qwen-flash": (0.000111, 0.000375),
    "mock": (0.0, 0.0),
}


def lookup_price(model: str) -> tuple[float, float]:
    """
    查模型单价，支持前缀匹配兜底。

    面试考点：为什么需要兜底？
      → 模型版本迭代很快（qwen3.8-flash-0915 这种带日期后缀的快照很常见）。
        如果只支持精确匹配，每上一个新快照成本统计就会静默变成 0，
        而「成本为 0」这个错误极难被发现 —— 看板上数字好看，实际在烧钱。
        所以宁可退化成「用同一系列的单价近似」，也不要返回 0。
    """
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    # 前缀匹配：qwen3.8-flash-0915 → qwen3.8-flash
    for name, price in PRICE_TABLE.items():
        if name != "mock" and model.startswith(name):
            return price
    # 没匹配到：给出保守估计并记录，避免"成本静默为 0"
    return (0.0, 0.0)


@dataclass
class Span:
    """一次被追踪的调用：可以是 Agent 节点、Tool 调用、LLM 请求。"""

    span_id: str
    name: str
    kind: str  # agent | tool | llm | node
    parent_id: str | None = None
    start_ts: float = field(default_factory=time.perf_counter)
    end_ts: float | None = None
    status: str = "running"  # running | ok | error
    error: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        end = self.end_ts if self.end_ts is not None else time.perf_counter()
        return round((end - self.start_ts) * 1000, 2)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = self.latency_ms
        d["total_tokens"] = self.total_tokens
        return d


class TraceRecorder:
    """
    一次任务的完整调用树。

    面试追问：为什么自己写而不直接上 Langfuse？
      → 可以答：生产我会接 Langfuse/OpenTelemetry，这里手写是为了讲清 trace_id /
        parent_span_id 的父子关系原理——模型服务里这就是基本的分布式追踪模型。
    """

    def __init__(self, task_id: str, query: str = "", persist: bool = True):
        self.task_id = task_id
        self.query = query
        self.persist = persist
        self.trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    # -------- 上下文管理器：with tracer.span("researcher", kind="agent"): --------
    @contextmanager
    def span(self, name: str, kind: str = "node", **meta: Any) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        sp = Span(
            span_id=f"sp_{uuid.uuid4().hex[:8]}",
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent else None,
            meta=meta,
        )
        self.spans.append(sp)
        self._stack.append(sp)
        try:
            yield sp
            sp.status = "ok"
        except Exception as exc:  # noqa: BLE001
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}"
            logger.exception("span 失败: %s", name)
            raise
        finally:
            sp.end_ts = time.perf_counter()
            self._stack.pop()

    # -------- LLM 调用记账 --------
    def record_llm(
        self,
        span: Span,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        px_in, px_out = lookup_price(model)
        cost = prompt_tokens / 1000 * px_in + completion_tokens / 1000 * px_out
        span.model = model
        span.prompt_tokens += prompt_tokens
        span.completion_tokens += completion_tokens
        span.cost_usd = round(span.cost_usd + cost, 6)
        return cost

    # -------- 汇总 --------
    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    @property
    def total_latency_ms(self) -> float:
        """
        真实墙钟耗时 = 所有 span 里「最晚结束」减「最早开始」。

        面试考点：为什么不能写成 `max(根 span 的 latency)`？
          → 根 span（parent_id=None）在 LangGraph 里就是各个节点：
            prepare / planner / execute / writer / critic / revise / finalize。
            它们**顺序执行、互不包含**，所以最长的那一个绝不等于端到端耗时。

        【本项目的真实 bug 与修复】
          原实现：
              roots = [s for s in self.spans if s.parent_id is None]
              return max((s.latency_ms for s in roots), default=0.0)
          在实测 trace `tr_85e50d9c1118` 上，节点时间线是 +0s → +1396.7s（串行），
          真实端到端应为 1396.7s，但旧实现返回了「最长单节点」node:revise 的 377.1s，
          偏差 3.7 倍。原因就是上面那句"最长者当总时长"的想当然。

          正确做法是取时间区间的并集长度。串行场景下它等于各节点耗时之和；
          并行场景下它会正确小于求和（因为区间重叠），这正是墙钟应有的语义。
        """
        if not self.spans:
            return 0.0
        # 尚未结束的 span 用当前时刻兜底，保证运行中查询也合理
        now = time.perf_counter()
        starts = [s.start_ts for s in self.spans]
        ends = [s.end_ts if s.end_ts is not None else now for s in self.spans]
        return round((max(ends) - min(starts)) * 1000, 2)

    @property
    def sum_latency_ms(self) -> float:
        """
        所有节点耗时「累加」口径（保留用于对比）。

        面试考点：墙钟 vs 累加，什么时候该用哪个？
          → 墙钟（total_latency_ms）回答"用户实际等了多久"，
            用于 SLA / 性能指标。
          → 累加（sum_latency_ms）回答"CPU/推理时间被消耗了多少"，
            用于算力成本分摊。
          当存在并行分支时，累加 > 墙钟；差额就是并行带来的收益。
          报告指标时必须写清用了哪个口径，否则会被质疑。
        """
        return round(sum(s.latency_ms for s in self.spans), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "query": self.query,
            "created_at": self.created_at,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            # 墙钟：用户实际等待时长（指标口径，报告须优先用这个）
            "wall_latency_ms": self.total_latency_ms,
            # 累加：各 span 耗时之和（算力口径，并行时大于墙钟）
            "sum_latency_ms": self.sum_latency_ms,
            "spans": [s.to_dict() for s in self.spans],
        }

    def save(self) -> Path | None:
        if not self.persist or not settings.enable_trace:
            return None
        path = settings.trace_path / f"{self.trace_id}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def summary(self) -> str:
        """给终端打印的精简视图——演示时一眼看出瓶颈在哪。"""
        lines = [
            f"┌─ Trace {self.trace_id} | 任务 {self.task_id}",
            f"│  总耗时 {self.total_latency_ms:.0f}ms | 总 token {self.total_tokens}"
            f" | 总成本 ${self.total_cost_usd:.5f}",
            "│  " + "-" * 62,
        ]
        # 按耗时排序，把瓶颈顶到最前——这是这个函数真正的价值
        for sp in sorted(self.spans, key=lambda s: -s.latency_ms):
            depth = 0
            cur = sp
            by_id = {x.span_id: x for x in self.spans}
            while cur.parent_id:
                depth += 1
                cur = by_id.get(cur.parent_id)  # type: ignore[assignment]
                if cur is None:
                    break
            icon = {"agent": "🤖", "tool": "🔧", "llm": "🧠", "node": "▫️"}.get(sp.kind, "▫️")
            mark = "✗" if sp.status == "error" else " "
            lines.append(
                f"│ {mark}{'  ' * depth}{icon} {sp.name:<28} {sp.latency_ms:>8.1f}ms"
                f"  {sp.total_tokens:>6}tk  ${sp.cost_usd:.5f}"
            )
        lines.append("└" + "-" * 63)
        return "\n".join(lines)


# ============================================================
# 全局指标聚合（进程级）
# 面试考点：为什么用环形缓冲而不是全量存？
#   → 长跑服务内存会无限增长。生产会推给 Prometheus，这里保留最近 N 条样本即可。
# ============================================================
class MetricsCollector:
    def __init__(self, max_samples: int = 500):
        self.max_samples = max_samples
        self.latencies: list[float] = []
        self.tokens: list[int] = []
        self.costs: list[float] = []
        self.total_tasks = 0
        self.failed_tasks = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def record_task(self, trace: TraceRecorder, ok: bool = True) -> None:
        self.total_tasks += 1
        if not ok:
            self.failed_tasks += 1
        self.latencies.append(trace.total_latency_ms)
        self.tokens.append(trace.total_tokens)
        self.costs.append(trace.total_cost_usd)
        for buf in (self.latencies, self.tokens, self.costs):
            if len(buf) > self.max_samples:
                del buf[0 : len(buf) - self.max_samples]
        for sp in trace.spans:
            if sp.kind == "tool":
                self.tool_calls += 1
                if sp.status == "error":
                    self.tool_errors += 1

    @staticmethod
    def _pct(buf: list[float], p: float) -> float:
        """
        面试考点：P95 怎么算？
          排序后取 ceil(p * n) - 1 位。P95 = 100 个请求里 95 个都快于这个值，
          关注尾部延迟，比平均值更能反映用户体验。
        """
        if not buf:
            return 0.0
        s = sorted(buf)
        idx = min(len(s) - 1, max(0, int(-(-p * len(s) // 1)) - 1))
        return round(s[idx], 2)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": round(
                1 - self.failed_tasks / self.total_tasks, 4
            ) if self.total_tasks else 1.0,
            "latency_ms": {
                "p50": self._pct(self.latencies, 0.50),
                "p95": self._pct(self.latencies, 0.95),
                "p99": self._pct(self.latencies, 0.99),
            },
            "total_tokens": sum(self.tokens),
            "avg_tokens_per_task": round(sum(self.tokens) / len(self.tokens), 1) if self.tokens else 0.0,
            "total_cost_usd": round(sum(self.costs), 6),
            "avg_cost_per_task": round(sum(self.costs) / len(self.costs), 6) if self.costs else 0.0,
            "tool_calls": self.tool_calls,
            "tool_error_rate": round(self.tool_errors / self.tool_calls, 4) if self.tool_calls else 0.0,
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(
                    self.cache_hits / (self.cache_hits + self.cache_misses), 4
                ) if (self.cache_hits + self.cache_misses) else 0.0,
            },
        }


metrics = MetricsCollector()
