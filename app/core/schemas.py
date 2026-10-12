"""
Step 1 · 统一数据模型（全项目共享的「契约」）
============================================================
设计要点：为什么先定 Schema 再写代码？
  - LLM 的输出是不确定的。用 Pydantic 把「自由文本」约束成「结构体」，
    是 LLM 工程化最核心的一招（Structured Output）。
  - 结构体一旦定死，Agent 之间就能用类型安全的方式传数据，
    JSON 解析失败率从 ~15% 降到接近 0（配合重试）。
  - 这些模型同样是 API 的响应契约，前后端不会扯皮。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ============================================================
# 枚举：用 Enum 而不是裸字符串，防止"researching"/"research" 这类拼写漂移
# ============================================================
class AgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    REVIEWER = "reviewer"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"   # 人工在环（HITL）中断点
    REFLECTING = "reflecting"          # Critic 打回重跑
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """任务类型决定用哪个 Agent 流水线、以及路由到哪个模型。"""

    MARKET_RESEARCH = "market_research"   # 市场调研
    COMPETITOR = "competitor"             # 竞品分析
    LISTING = "listing"                   # Listing 文案生成
    AD_KEYWORDS = "ad_keywords"           # 广告关键词优化
    REVIEW_INSIGHT = "review_insight"     # 评论洞察
    WEEKLY_REPORT = "weekly_report"       # 运营周报
    FULL_PIPELINE = "full_pipeline"       # 全链路


# ============================================================
# 子任务：Planner 的产物，也是 DAG 的节点
# ============================================================
class SubTask(BaseModel):
    id: str = Field(description="子任务唯一标识，如 t1")
    name: str = Field(description="子任务名称，如 竞品价格抓取")
    agent: AgentRole = Field(description="负责执行的 Agent 角色")
    description: str = Field(default="", description="该子任务的详细说明")
    depends_on: list[str] = Field(default_factory=list, description="依赖的子任务 id，构成 DAG")
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    tokens_used: int = 0
    retry_count: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return 0.0


# ============================================================
# Planner 输出的计划（LLM 结构化输出目标）
# ============================================================
class Plan(BaseModel):
    """
    设计要点：为什么要给 LLM 明确的 JSON Schema？
      → 模型有「格式幻觉」，不给 schema 就会自由发挥。
        给出 schema + 少量示例（few-shot），输出符合率能到 95%+。
    """

    task_type: TaskType = TaskType.FULL_PIPELINE
    goal: str = Field(default="", description="本次任务的业务目标，一句话")
    subtasks: list[SubTask] = Field(default_factory=list)
    complexity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="任务复杂度评分，用于模型路由：<0.5 走小模型，>=0.5 走大模型",
    )

    @field_validator("subtasks")
    @classmethod
    def _at_least_one(cls, v: list[SubTask]) -> list[SubTask]:
        if not v:
            raise ValueError("计划至少包含一个子任务")
        return v

    def topological_order(self) -> list[SubTask]:
        """
        设计要点：为什么这里要自己实现拓扑排序？
          → LangGraph 负责「节点级」编排，而子任务 DAG 是「任务级」编排，
            两层编排是不同粒度。Kahn 算法 O(V+E)，同时可检测环。
        """
        by_id = {t.id: t for t in self.subtasks}
        indeg = {t.id: 0 for t in self.subtasks}
        for t in self.subtasks:
            for dep in t.depends_on:
                if dep in indeg:
                    indeg[t.id] += 1

        queue = [tid for tid, d in indeg.items() if d == 0]
        order: list[SubTask] = []
        while queue:
            tid = queue.pop(0)
            order.append(by_id[tid])
            for t in self.subtasks:
                if tid in t.depends_on:
                    indeg[t.id] -= 1
                    if indeg[t.id] == 0:
                        queue.append(t.id)

        if len(order) != len(self.subtasks):
            raise ValueError("子任务依赖存在环，无法拓扑排序")
        return order

    def missing_deps(self) -> list[str]:
        """返回已完成子任务所依赖但尚未完成的 id —— 用于并行调度判断。"""
        done = {t.id for t in self.subtasks if t.status == TaskStatus.COMPLETED}
        ready: list[str] = []
        for t in self.subtasks:
            if t.status != TaskStatus.PENDING:
                continue
            if all(d in done for d in t.depends_on):
                ready.append(t.id)
        return ready


# ============================================================
# Critic 评审结果
# ============================================================
class CritiqueResult(BaseModel):
    """
    设计要点：多 Agent 怎么防止「错误互相放大」？
      → Critic 独立打分 + 硬阈值回灌重跑。分数维度拆开打，便于定位到底哪里不行。
    """

    score: float = Field(ge=0.0, le=1.0, description="综合评分")
    passed: bool = Field(description="是否达到通过阈值")
    dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="分维度得分，如 {'relevance':0.9,'factuality':0.7,'compliance':0.95,'readability':0.8}",
    )
    issues: list[str] = Field(default_factory=list, description="发现的问题清单")
    suggestions: list[str] = Field(default_factory=list, description="修改建议")
    fatal: bool = Field(default=False, description="是否存在致命问题（如违反平台合规）")


# ============================================================
# 工具调用记录（MCP 风格）
# ============================================================
class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    attempt: int = 1


# ============================================================
# 最终产物
# ============================================================
class Deliverable(BaseModel):
    title: str
    content: str
    format: Literal["markdown", "json", "text"] = "markdown"
    meta: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 任务总结（对外返回体）
# ============================================================
class TaskSummary(BaseModel):
    task_id: str
    query: str
    status: TaskStatus
    plan: Plan | None = None
    deliverable: Deliverable | None = None
    critique: CritiqueResult | None = None
    reflection_rounds: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0
    trace_id: str | None = None
    errors: list[str] = Field(default_factory=list)
    review_history: list[dict[str, Any]] = Field(default_factory=list)
