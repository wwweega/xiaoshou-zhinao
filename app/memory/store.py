"""
Step 4 · 记忆系统（长短期记忆 + 上下文工程）
============================================================
设计要点（美图 AI Agent 岗 JD 原话：「构建具备长短时记忆管理、复杂工作流状态机编排
以及外部知识库检索能力的服务架构」——这是 Agent 岗区分度最高的一环）：

为什么 Agent 需要记忆？
  1. 短期记忆：多轮对话里"他""那个产品"指代什么。没有它 Agent 就是失忆症。
  2. 长期记忆：上一个任务得出的结论，这个任务能不能复用？否则每次都从零开始，
     既烧钱又慢，而且结论不一致。

为什么不能只靠"把全部历史塞进 prompt"？
  → Token 成本是 O(n²) 增长的：第 20 轮的 prompt 里有前 19 轮的全部内容。
    而且长上下文会稀释注意力（lost in the middle），模型反而抓不住重点。
  所以要用「窗口 + 摘要压缩 + 相关性筛选」三件套。

本模块实现三层记忆：
  L1 短期缓冲  —— 最近 N 条消息，原样保留（保证指代消解）
  L2 摘要记忆  —— 超出窗口的历史压缩成结构化摘要（省 token，保信息）
  L3 长期记忆  —— 历史任务的结论向量化存储，按语义相关性召回（跨任务复用）
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.observability import logger, metrics
from app.core.schemas import SubTask
from app.llm.client import llm_client


# ============================================================
# L1 + L2：会话级记忆（短期窗口 + 摘要压缩）
# ============================================================
@dataclass
class Message:
    role: str          # system | user | assistant | tool
    content: str
    ts: float = field(default_factory=time.time)
    tokens: int = 0

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    设计要点：窗口大小怎么定？
      → 不是越大越好。要看「指代消解需要多少上文」。
        运营类任务实测 4~8 条足够；代码类任务需要更长。这里默认 6，可配置。
    """

    def __init__(self, window_size: int | None = None, summary_trigger: int | None = None):
        self.window_size = window_size or settings.memory_window_size
        self.summary_trigger = summary_trigger or settings.memory_summary_trigger_tokens
        self.messages: list[Message] = []
        self.summary: str = ""            # L2 压缩摘要
        self._summarized_count: int = 0   # 已被摘要掉的原始消息数

    # -------- 写入 --------
    def add(self, role: str, content: str) -> None:
        msg = Message(role=role, content=content, tokens=estimate_tokens(content))
        self.messages.append(msg)

    # -------- 上下文装配（上下文工程的核心）--------
    def build_context(self) -> list[dict[str, str]]:
        """
        设计要点（高频追问）：上下文工程到底在做什么？
          → 三件事：①保住指代消解必需的原样对话 ②把远期历史压成摘要
            ③不让无关内容占据注意力预算。

        装配顺序（顺序本身影响效果）：
          系统摘要（远期）→ 最近 N 条原文（近期）→ 当前指令
        """
        ctx: list[dict[str, str]] = []

        # L2：远期摘要放最前，让模型知道"之前发生过什么"
        if self.summary:
            ctx.append({
                "role": "system",
                "content": f"【历史任务摘要】\n{self.summary}",
            })

        # L1：最近 window_size 条原样保留
        recent = self.messages[-self.window_size :]
        for m in recent:
            ctx.append(m.to_dict())

        return ctx

    def should_summarize(self) -> bool:
        total = sum(m.tokens for m in self.messages)
        return total > self.summary_trigger and len(self.messages) > self.window_size * 2

    async def compress(self, tracer=None) -> str:
        """
        把「窗口之外」的旧消息压缩成结构化摘要。

        设计要点：为什么摘要要「结构化」而不是让模型随便总结？
          → 自由总结会丢掉关键实体（品牌名、数字、结论），
            结构化（目标/已确认事实/关键数字/未决问题）能保证关键信息不丢。
        """
        older = self.messages[: -self.window_size]
        if not older:
            return self.summary

        raw = "\n".join(f"{m.role}: {m.content[:400]}" for m in older)
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是上下文压缩器。把下面的对话历史压缩成结构化摘要，"
                    "必须保留：①任务目标 ②已确认的事实与结论 ③所有关键数字与专有名词 "
                    "④尚未解决的问题。不要评论，不要遗漏数字。用中文，控制在 300 字内。"
                ),
            },
            {"role": "user", "content": raw},
        ]

        with_chunk = None
        if tracer:
            with_chunk = tracer.span("memory:compress", kind="node", compressed=len(older))
        if with_chunk:
            with_chunk.__enter__()
        try:
            text, _ = await llm_client.chat(prompt, model=settings.llm_small_model)
            self.summary = text.strip()
            self._summarized_count += len(older)
            # 把已摘要的消息从缓冲里移除，只保留窗口内的
            self.messages = self.messages[-self.window_size :]
            logger.info("记忆压缩完成：%d 条 → %d 字摘要", len(older), len(self.summary))
            return self.summary
        finally:
            if with_chunk:
                with_chunk.__exit__(None, None, None)

    def token_cost(self) -> int:
        """当前上下文的 token 开销，用于成本核算。"""
        return sum(m.tokens for m in self.messages) + estimate_tokens(self.summary)

    def reset(self) -> None:
        self.messages.clear()
        self.summary = ""
        self._summarized_count = 0


# ============================================================
# L3：长期记忆（向量检索）
# ============================================================
class LongTermMemory:
    """
    设计要点：没有向量数据库怎么演示长期记忆？
      → 用「哈希向量 + 余弦相似度」做一个降级实现。
        这正好能把 RAG 的原理讲清楚：embedding → 相似度 → Top-K 召回。
        生产直接换成 Milvus / pgvector，接口不变（这就是依赖倒置的价值）。

    为什么长期记忆要双写（向量库 + 事实表）？
      → 向量检索擅长"模糊语义"，但不擅长"精确查某个 SKU 的上次结论"。
        所以生产上是「向量库做召回 + 结构化表做精确查询」，
        这跟 RAG 里的「向量检索 + BM25」混合检索是同一个思路。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vectors: list[tuple[str, list[float], dict[str, Any]]] = []  # (id, vec, payload)
        self.facts: dict[str, dict[str, Any]] = {}   # 结构化事实表
        self._lock = asyncio.Lock()

    # -------- 极简 embedding（无外部依赖，可复现）--------
    def _embed(self, text: str) -> list[float]:
        """
        设计要点：为什么这里不用真 embedding 模型？
          → 演示环境不该有外部依赖。而且这个哈希词袋向量能把
            「相似度计算」的原理展示清楚。生产会换成 BGE-M3 这类中文效果好的模型。
        """
        vec = [0.0] * self.dim
        tokens = [t for t in text.lower().replace("\n", " ").split() if t]
        # 中文按字切分，保证中文也能算相似度
        chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        for tok in tokens + chars:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        # L2 归一化：让余弦相似度可以直接做点积，省一次除法
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """归一化后余弦相似度 = 点积。O(d)，d=256 时极快。"""
        return sum(x * y for x, y in zip(a, b))

    # -------- 写入 --------
    async def remember(
        self,
        content: str,
        kind: str = "conclusion",
        key: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        async with self._lock:
            mem_id = f"mem_{hashlib.md5(content.encode()).hexdigest()[:10]}"
            vec = self._embed(content)
            self.vectors.append((mem_id, vec, {"kind": kind, "content": content, **(payload or {})}))
            # 结构化事实双写
            if key:
                self.facts[key] = {"content": content, **(payload or {})}
            return mem_id

    # -------- 召回 --------
    async def recall(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """按语义相似度召回 Top-K 记忆。"""
        k = top_k or settings.memory_top_k
        if not self.vectors:
            return []
        qv = self._embed(query)
        scored = [
            (self._cosine(qv, vec), payload)
            for _, vec, payload in self.vectors
        ]
        scored.sort(key=lambda x: -x[0])
        # 设计要点：为什么要设相似度阈值？
        #   → 不设的话永远能召回 Top-K，哪怕全是垃圾，会污染上下文。
        return [
            {**payload, "similarity": round(s, 4)}
            for s, payload in scored[:k]
            if s > 0.05
        ]

    def get_fact(self, key: str) -> dict[str, Any] | None:
        """精确查询：走结构化表，不走向量。"""
        return self.facts.get(key)

    def size(self) -> int:
        return len(self.vectors)


# ============================================================
# 工具函数
# ============================================================
def estimate_tokens(text: str) -> int:
    """
    设计要点：为什么不用 tiktoken 精确算？
      → 这里只需要"量级正确"来做预算判断。中文约 1 字 ≈ 1.5 token，
        英文约 1 词 ≈ 1.3 token。量级对了就够触发压缩逻辑，省一次依赖。
        精确计费用 API 返回的 usage 字段（我们在 llm client 里就是这么做的）。
    """
    if not text:
        return 0
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    others = len(text) - chinese
    return int(chinese * 1.5 + others / 4)


def build_task_context(task: SubTask, depends_results: dict[str, str]) -> str:
    """
    上下文工程：只把「当前子任务真正需要的」上游结果注入 prompt。

    设计要点：为什么不把所有上游结果全塞进去？
      → 三个理由：①token 成本 ②注意力稀释 ③不相关上下文会诱导模型跑题。
        我们只注入 depends_on 声明依赖的那几个子任务结果，
        这就是「依赖驱动的最小上下文」（相较"全量注入"，输入 token 明显更低；
        具体降幅取决于依赖图的稀疏程度，本项目未做 A/B 对照，故不给百分比）。
    """
    if not depends_results:
        return "（无上游依赖，你从零开始完成本子任务）"

    parts = ["【上游子任务的产出，供你参考】"]
    for tid, result in depends_results.items():
        # 单条结果做截断，防止一个长结果吃掉全部预算
        trimmed = result[:1200] + ("...（已截断）" if len(result) > 1200 else "")
        parts.append(f"--- 来自 {tid} ---\n{trimmed}")
    return "\n".join(parts)


# 全局单例（按 task_id 隔离的会话记忆在 Graph 状态里，这里只放长期记忆）
long_term_memory = LongTermMemory()
