"""
Step 2 · LLM 客户端层
============================================================
面试考点（这一层是「模型路由」「成本控制」的技术底座）：

1. 统一抽象 LLMClient：业务代码不关心背后是 DeepSeek / GPT / vLLM
   —— 这是依赖倒置，换模型只改配置不改业务代码。

2. 模型路由（Model Routing）：
   不是所有子任务都需要大模型。抽取、分类、意图识别用 7B 小模型就够，
   策略规划和长文写作才需要大模型。按复杂度打分路由，可显著降低平均单价。
   （说明：降本幅度取决于"大小模型单价差"与"被路由到小模型的任务占比"，
     本项目未做 A/B 对照测量，故不给出具体百分比；可用 scripts/demo.py 实测 token 与成本。）

3. 指数退避重试（Exponential Backoff + Jitter）：
   API 限流（429）是常态。固定间隔重试会造成「惊群」，
   退避 + 随机抖动能把重试请求打散。这是京东/美图 JD 里"API 限流优化"的具体做法。

4. 结构化输出：
   调用方传 pydantic 模型，客户端负责「提示模型输出 JSON → 解析 → 校验 → 失败重试」，
   把不确定性收敛在客户端内部，业务代码拿到的一定是合法对象。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from typing import Any, TypeVar, Type

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)

from app.core.config import settings
from app.core.observability import logger, TraceRecorder, Span

TModel = TypeVar("TModel", bound=BaseModel)


# ============================================================
# 异常体系：分层捕获，让上层能区分「该重试」和「不该重试」
# ============================================================
class LLMError(Exception):
    """LLM 调用基础异常。"""


class LLMRateLimitError(LLMError):
    """限流，可重试。"""


class LLMTimeoutError(LLMError):
    """超时，可重试。"""


class LLMFormatError(LLMError):
    """输出格式非法，可通过重新提示修复。"""


# ============================================================
# 复杂度评估 → 模型路由
# ============================================================
class ComplexityRouter:
    """
    面试考点：复杂度怎么量化？不要拍脑袋，要有可解释的规则（后续可用数据训练分类器）。

    规则设计（可解释、可灰度）：
      - 任务角色权重：写作/规划/评审是「认知密集」，抽取类是「机械劳动」
      - 描述长度：越长通常信息越多
      - 关键词命中：出现"策略/优化/分析/对比/建议"等词，说明需要推理
    """

    # 角色 → 基础复杂度（这是路由用的启发式权重，不是"任务完成率"等指标）
    ROLE_WEIGHT = {
        "planner": 0.75,
        "writer": 0.80,
        "critic": 0.70,
        "analyst": 0.55,
        "researcher": 0.40,
        "reviewer": 0.50,
    }

    # 推理型关键词：命中即加权
    REASONING_HINTS = (
        "策略", "优化", "分析", "对比", "建议", "推理", "规划",
        "为什么", "预测", "洞察", "方案", "取舍",
    )

    @classmethod
    def score(cls, role: str, description: str = "", extra: float = 0.0) -> float:
        base = cls.ROLE_WEIGHT.get(role, 0.5)

        # 长度因子：200 字以上认为信息量充足，最多加 0.1
        length_factor = min(len(description) / 2000.0, 0.1)

        # 关键词因子
        hits = sum(1 for kw in cls.REASONING_HINTS if kw in description)
        kw_factor = min(hits * 0.05, 0.2)

        score = base + length_factor + kw_factor + extra
        return round(min(max(score, 0.0), 1.0), 3)

    @classmethod
    def pick(cls, role: str, description: str = "") -> tuple[str, bool]:
        """
        返回 (模型名, 是否用小模型)。
        面试追问：阈值怎么定？
          → 先拍一个 0.5 作为冷启动，再根据「小模型在该任务上的失败率」调。
            失败率 >15% 就调高阈值（更多任务交给大模型），这是数据驱动的调参。
        """
        c = cls.score(role, description)
        use_small = c < settings.model_routing_threshold
        model = settings.llm_small_model if use_small else settings.llm_model
        return model, use_small


# ============================================================
# 主客户端
# ============================================================
class LLMClient:
    """
    线程/协程安全的 LLM 客户端。生产用连接池复用 AsyncOpenAI 实例。
    """

    def __init__(self) -> None:
        self._clients: dict[str, AsyncOpenAI] = {}
        self._lock = asyncio.Lock()
        self.call_count = 0          # 统计：便于演示"路由省了多少次大模型调用"
        self.small_model_calls = 0
        self.big_model_calls = 0
        self.retry_count = 0

    async def _get_client(self, base_url: str, api_key: str) -> AsyncOpenAI:
        """
        面试考点：为什么缓存 client？
          → AsyncOpenAI 内部维护 httpx 连接池，每次 new 一个会丢连接复用，
            高并发下 TIME_WAIT 暴涨。按 base_url 缓存即可复用连接池。
        """
        key = f"{base_url}|{api_key[:8]}"
        async with self._lock:
            if key not in self._clients:
                self._clients[key] = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    timeout=settings.llm_timeout,
                    max_retries=0,  # 重试我们自己做，避免双重重试放大流量
                )
            return self._clients[key]

    # ---------------------------------------------------------
    # 核心：带重试与记账的 chat 调用
    # ---------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        span: Span | None = None,
        tracer: TraceRecorder | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        force_big: bool = False,
    ) -> tuple[str, str]:
        """
        返回 (文本内容, 实际使用的模型名)。
        """
        model = model or settings.llm_model

        # ---------- Mock 通道：无 Key 也能跑通全链路（演示兜底）----------
        if settings.use_mock:
            text = _mock_response(messages)
            if tracer and span:
                tracer.record_llm(span, "mock", len(str(messages)) // 4, len(text) // 4)
            return text, "mock"

        base_url = settings.llm_base_url
        api_key = settings.llm_api_key
        if model == settings.llm_small_model:
            base_url = settings.llm_small_base_url or base_url
            api_key = settings.llm_small_api_key or api_key

        client = await self._get_client(base_url, api_key)

        @retry(
            stop=stop_after_attempt(settings.llm_max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
            before_sleep=before_sleep_log(logger, 20),
            reraise=True,
        )
        async def _call() -> Any:
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=(
                        temperature if temperature is not None else settings.llm_temperature
                    ),
                    max_tokens=max_tokens or settings.llm_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                msg = str(exc)
                if "429" in msg or "rate" in msg.lower():
                    raise LLMRateLimitError(msg) from exc
                if "timeout" in msg.lower() or "Timeout" in name:
                    raise LLMTimeoutError(msg) from exc
                raise LLMError(f"{name}: {msg}") from exc

        try:
            resp = await _call()
        except (LLMRateLimitError, LLMTimeoutError) as exc:
            self.retry_count += 1
            raise LLMError(f"重试耗尽: {exc}") from exc

        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        # 记账
        if tracer and span:
            tracer.record_llm(span, model, pt, ct)
        self.call_count += 1
        if model == settings.llm_small_model:
            self.small_model_calls += 1
        else:
            self.big_model_calls += 1

        logger.debug("LLM[%s] in=%d out=%d", model, pt, ct)
        return content, model

    # ---------------------------------------------------------
    # 流式生成：长文写作专用（解决"总耗时超时"问题）
    # ---------------------------------------------------------
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        span: Span | None = None,
        tracer: TraceRecorder | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        """
        流式调用，返回 (完整文本, 模型名)。

        面试考点：为什么长文写作必须用流式，而不是把 timeout 调大？
        ────────────────────────────────────────────────────────
        非流式请求下，httpx 的 timeout 是「首字节到完整响应」的总时长约束。
        qwen3.8-flash 带推理生成一份 3000 字 Listing 方案，端到端可能要 3~5 分钟，
        无论把 timeout 设成 180s 还是 600s，都是在跟模型的生成速度赛跑 ——
        而且超时后会触发重试，同一个请求被重复计费，成本直接翻倍。

        流式的本质区别：服务端一有 token 就立刻推回来。
        此时 read timeout 约束的是「两个 chunk 之间的最大间隔」，
        而不是「整个响应的总时长」。
        只要模型持续吐字（间隔远小于 timeout），生成 10 分钟也不会被判定超时。

        这就是「用正确的机制，而不是调大参数」——调参治标，换机制治本。
        """
        model = model or settings.llm_model

        # Mock 通道：保持与非流式一致的行为
        if settings.use_mock:
            text = _mock_response(messages)
            if tracer and span:
                tracer.record_llm(span, "mock", len(str(messages)) // 4, len(text) // 4)
            return text, "mock"

        base_url = settings.llm_base_url
        api_key = settings.llm_api_key
        if model == settings.llm_small_model:
            base_url = settings.llm_small_base_url or base_url
            api_key = settings.llm_small_api_key or api_key

        client = await self._get_client(base_url, api_key)

        @retry(
            stop=stop_after_attempt(settings.llm_max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
            before_sleep=before_sleep_log(logger, 20),
            reraise=True,
        )
        async def _call() -> tuple[str, int, int]:
            chunks: list[str] = []
            pt = ct = 0
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=(
                        temperature if temperature is not None else settings.llm_temperature
                    ),
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    # 最后一个 chunk 携带 usage，choices 可能为空
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        pt = getattr(usage, "prompt_tokens", 0) or pt
                        ct = getattr(usage, "completion_tokens", 0) or ct
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        chunks.append(piece)
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                msg = str(exc)
                if "429" in msg or "rate" in msg.lower():
                    raise LLMRateLimitError(msg) from exc
                if "timeout" in msg.lower() or "Timeout" in name:
                    raise LLMTimeoutError(msg) from exc
                raise LLMError(f"{name}: {msg}") from exc

            text = "".join(chunks)
            # 兜底：流式未返回 usage 时，按字符数粗估，保证成本统计不断档
            return text, pt or (len(str(messages)) // 4), ct or (len(text) // 4)

        try:
            content, pt, ct = await _call()
        except (LLMRateLimitError, LLMTimeoutError) as exc:
            self.retry_count += 1
            raise LLMError(f"重试耗尽: {exc}") from exc

        if tracer and span:
            tracer.record_llm(span, model, pt, ct)

        self.call_count += 1
        if model == settings.llm_small_model:
            self.small_model_calls += 1
        else:
            self.big_model_calls += 1

        logger.debug("LLM-stream[%s] in=%d out=%d chars=%d", model, pt, ct, len(content))
        return content, model

    # ---------------------------------------------------------
    # 结构化输出：LLM 输出 → Pydantic 对象
    # ---------------------------------------------------------
    async def chat_structured(
        self,
        messages: list[dict[str, str]],
        schema: Type[TModel],
        *,
        span: Span | None = None,
        tracer: TraceRecorder | None = None,
        role: str = "default",
        max_fix_attempts: int = 2,
    ) -> tuple[TModel, str]:
        """
        面试考点：为什么不让模型直接吐 JSON 就完事？
          → 模型有 5%~15% 概率输出非法 JSON（多打引号、用 ```json 包裹、
            中文字段名、注释、尾逗号）。所以必须：提取 → 解析 → 校验 → 失败回灌修复。
            这套「解析-校验-重试」是 LLM 工程化的核心，不是锦上添花。
        """
        # 按复杂度路由模型
        model, use_small = ComplexityRouter.pick(role, messages[-1].get("content", ""))
        if use_small:
            logger.info("路由 → 小模型 %s（role=%s）", model, role)

        # 把 schema 注入 system prompt，减少格式漂移
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        sys_hint = (
            "\n\n【输出格式要求】\n"
            "你必须只输出一个合法的 JSON 对象，不要任何解释文字、不要 markdown 代码块标记。\n"
            "严格遵循以下 JSON Schema：\n" + schema_json
        )
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + sys_hint}
        else:
            msgs.insert(0, {"role": "system", "content": sys_hint.strip()})

        last_err: Exception | None = None
        for attempt in range(max_fix_attempts + 1):
            raw, used_model = await self.chat(
                msgs, span=span, tracer=tracer, model=model, max_tokens=settings.llm_max_tokens
            )
            try:
                data = extract_json(raw)
                return schema.model_validate(data), used_model
            except (LLMFormatError, ValidationError, json.JSONDecodeError) as exc:
                last_err = exc
                logger.warning(
                    "结构化输出失败（第 %d 次）: %s | 原始输出前 200 字: %s",
                    attempt + 1, exc, raw[:200],
                )
                # 错误回灌：把失败原因告诉模型，让它自己修
                msgs = msgs + [
                    {"role": "assistant", "content": raw[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"你的输出不符合要求，解析失败：{exc}。\n"
                            "请重新输出**纯 JSON**（不要 markdown 代码块、不要解释），"
                            "字段名必须与 Schema 完全一致。"
                        ),
                    },
                ]

        raise LLMFormatError(f"结构化输出修复失败: {last_err}")

    async def close(self) -> None:
        for c in self._clients.values():
            await c.close()
        self._clients.clear()


# ============================================================
# 工具函数：从各种「脏输出」里抠出 JSON
# ============================================================
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """
    面试考点：这是 LLM 工程里最土但最有效的函数。真实项目里 90% 的"模型不稳定"
    都是格式问题，不是智能问题。

    处理顺序：代码块 → 首个 { 到末个 } → 直接解析
    """
    if not text or not text.strip():
        raise LLMFormatError("空输出")

    candidates: list[str] = []

    # 1) ```json ... ``` 包裹
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))

    # 2) 第一个 { 到最后一个 }（应对模型在前面寒暄）
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        candidates.append(text[s : e + 1])

    # 3) 原样
    candidates.append(text)

    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            # 4) 兜底：去掉尾逗号再试一次（模型很爱犯）
            fixed = re.sub(r",\s*([}\]])", r"\1", cand)
            try:
                obj = json.loads(fixed)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    raise LLMFormatError(f"无法从输出中提取 JSON: {text[:200]}")


def prompt_hash(text: str) -> str:
    """提示词哈希：用于语义缓存 key 与提示词版本追踪。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ============================================================
# Mock 实现：无 API Key 时的兜底，保证项目可跑通
# ============================================================
def _mock_response(messages: list[dict[str, str]]) -> str:
    """
    面试考点：为什么要写 mock？
      → 三个真实理由：①CI 里不能真的花钱调模型 ②单测要确定性输出
        ③演示环境断网时不能翻车。生产项目里 mock provider 是标配。

    【踩坑记录】判定顺序必须「从具体到宽泛」：
      最初把 "子任务" 判定放在最前面，结果 Critic 的 prompt 里因为携带了上游
      Planner 的子任务上下文，也命中了这个分支，导致 Critic 拿到的是 Plan JSON
      而不是 CritiqueResult —— 连续 3 次 Pydantic 校验失败。
      这暴露了一个通用设计原则：**多分支识别时，先判特征最强的分支**，
      否则宽泛的判据会「吃掉」本该属于具体判据的请求。
    """
    content = "\n".join(m.get("content", "") for m in messages)

    # ① CritiqueResult 的特征最独特（score + passed + dimensions 同时出现）
    if "pass_threshold" in content or ("critique" in content.lower() and "score" in content):
        return json.dumps(
            {"score": 0.86, "passed": True,
             "dimensions": {"relevance": 0.9, "factuality": 0.82,
                            "compliance": 0.95, "readability": 0.8},
             "issues": [], "suggestions": ["可补充 2 个长尾关键词"], "fatal": False},
            ensure_ascii=False,
        )
    # ② Function Calling / 工具选择
    if "MCP" in content or "tool" in content.lower() or "工具" in content:
        return json.dumps(
            {"tool_name": "competitor_scraper",
             "arguments": {"category": "kitchen storage", "top_n": 5}},
            ensure_ascii=False,
        )
    # ③ Plan（最后判：特征最宽泛，容易被其他 prompt 误命中）
    if "subtasks" in content or "子任务" in content:
        return json.dumps(
            {
                "task_type": "full_pipeline",
                "goal": "完成跨境电商选品与运营分析（mock）",
                "complexity": 0.6,
                "subtasks": [
                    {"id": "t1", "name": "竞品数据采集", "agent": "researcher",
                     "description": "抓取 Top5 竞品的价格、评分、评论数", "depends_on": []},
                    {"id": "t2", "name": "评论情感分析", "agent": "analyst",
                     "description": "对竞品评论做情感与需求点抽取", "depends_on": ["t1"]},
                    {"id": "t3", "name": "Listing 文案生成", "agent": "writer",
                     "description": "基于竞品与评论洞察生成 Listing", "depends_on": ["t1", "t2"]},
                ],
            },
            ensure_ascii=False,
        )
    return "（mock 输出）已基于给定上下文完成该步骤分析，结论：主打「可调节、免打孔、防水」三个卖点。"


# 全局单例
llm_client = LLMClient()
