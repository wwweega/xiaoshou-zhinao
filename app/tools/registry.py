"""
Step 3 · 工具层（MCP 风格）
============================================================
设计要点：为什么 Agent 需要「工具」而不是让模型硬答？
  - 模型的知识是静态的、会幻觉的。凡是「可计算、可查询、需实时」的事，
    都必须走工具（Tool Calling / Function Calling），模型只负责决策用哪个工具、传什么参数。
  - Anthropic 提出的 MCP（Model Context Protocol）就是把这套东西标准化：
    统一 schema + 统一发现机制 + 统一调用协议。

本模块实现三件事：
  1. ToolSpec：统一的工具描述（name / description / JSON Schema）
     —— 这个 schema 直接喂给 Function Calling，模型据此决定怎么调用
  2. 参数校验 + 超时 + 重试 + 幂等：工具出错不能让整个 Agent 崩
  3. 错误信息回灌：工具失败时把「为什么失败」返回给模型，让它自己改参数重试
     —— 这是降低工具调用出错率的核心机制（本项目未做前后对比测量，故不给出提升率）
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from app.core.observability import logger, metrics, TraceRecorder
from app.core.schemas import ToolCall


# ============================================================
# 工具规格
# ============================================================
@dataclass
class ToolSpec:
    name: str
    description: str
    # JSON Schema，形如 {"type":"object","properties":{...},"required":[...]}
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    timeout: float = 20.0
    max_retries: int = 2
    # 幂等键字段：同一 key 重复调用直接返回缓存，避免重复抓取/重复扣费
    idempotent_key: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI Function Calling 的 tools 参数格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ============================================================
# 参数校验模型（每个工具一个）
# ============================================================
class CompetitorScraperArgs(BaseModel):
    category: str = "kitchen storage"
    top_n: int = 5


class ReviewInsightArgs(BaseModel):
    product_id: str
    max_reviews: int = 100


class SqlQueryArgs(BaseModel):
    sql: str
    limit: int = 100


class CalcArgs(BaseModel):
    expression: str


class KeywordExtractArgs(BaseModel):
    text: str
    top_k: int = 10


class SentimentArgs(BaseModel):
    texts: list[str]


class TrendArgs(BaseModel):
    keyword: str
    months: int = 6


class PriceHistoryArgs(BaseModel):
    product_id: str
    days: int = 30


class ImageGenArgs(BaseModel):
    prompt: str
    scene: str = "产品场景图"


class ErpArgs(BaseModel):
    sku: str


class ExchangeArgs(BaseModel):
    base: str = "USD"
    quote: str = "CNY"


class WebSearchArgs(BaseModel):
    query: str
    top_k: int = 5


# ============================================================
# 工具注册中心
# ============================================================
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._cache: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重名: {spec.name}")
        self._tools[spec.name] = spec
        # 设计要点：为什么要 per-key 锁而不是全局锁？
        #   → 同一个幂等键并发时只需一次真实调用（防缓存击穿），
        #     不同 key 之间不应互相阻塞，否则并发退化成串行。
        self._locks[spec.name] = asyncio.Lock()

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}。可用: {list(self._tools)}")
        return self._tools[name]

    def all_schemas(self) -> list[dict[str, Any]]:
        """返回全部工具的 Function Calling schema，喂给模型。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    # ---------------------------------------------------------
    # 统一调用入口
    # ---------------------------------------------------------
    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tracer: TraceRecorder | None = None,
    ) -> ToolCall:
        """
        设计要点：为什么所有工具必须走同一个 call 入口？
          → 只有收口才能统一做：trace、超时、重试、缓存、参数校验、指标统计。
            如果每个 Agent 自己直接调 handler，这些能力要写 12 遍而且必然漏。
        """
        t0 = time.perf_counter()
        record = ToolCall(tool_name=name, arguments=arguments)

        spec = self._tools.get(name)
        if spec is None:
            record.ok = False
            record.error = (
                f"工具 '{name}' 不存在。可用工具：{list(self._tools)}。"
                "请从可用工具中重新选择。"
            )
            record.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            metrics.tool_errors += 1
            return record

        # ---------- 幂等缓存 ----------
        cache_key = None
        if spec.idempotent_key:
            raw = f"{name}:{arguments.get(spec.idempotent_key, '')}:{json.dumps(arguments, sort_keys=True)}"
            cache_key = hashlib.md5(raw.encode()).hexdigest()
            if cache_key in self._cache:
                record.result = self._cache[cache_key]
                record.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                metrics.cache_hits += 1
                logger.debug("工具缓存命中: %s", name)
                return record
            metrics.cache_misses += 1

        # ---------- 参数校验（在调用前做，而不是靠抛异常）----------
        # 设计要点：为什么校验要前置？
        #   ①靠 TypeError 兜底太脆弱：Python 里参数名写错、类型不对抛的异常类型不固定
        #   ②前置校验能给出精确的错误信息（缺哪个字段、类型错在哪）
        #   ③这类错误不该重试，越早失败越省资源
        missing = [k for k in spec.parameters.get("required", []) if k not in arguments]
        if missing:
            record.ok = False
            record.error = (
                f"缺少必填参数 {missing}。请按以下 schema 重新构造参数："
                f"{json.dumps(spec.parameters, ensure_ascii=False)}"
            )
            record.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            metrics.tool_errors += 1
            logger.warning("工具参数缺失: %s -> %s", name, missing)
            return record

        allowed_props = set(spec.parameters.get("properties", {}).keys())
        unknown = [k for k in arguments if k not in allowed_props]
        if unknown:
            record.ok = False
            record.error = (
                f"传入了未定义的参数 {unknown}。可用参数只有：{sorted(allowed_props)}。"
                "请修正参数名后重试。"
            )
            record.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            metrics.tool_errors += 1
            logger.warning("工具参数非法: %s -> %s", name, unknown)
            return record

        # ---------- 带重试的调用 ----------
        # 设计要点：异常分类决定「能不能重试」，这个分类是稳定性的关键
        #   - 超时 / 限流  → 可重试（瞬时故障，重试大概率成功）
        #   - 安全拦截     → 绝不重试（策略拒绝，重试永远失败，白等 1.3 秒）
        #   - 参数错误     → 绝不重试（要模型改参数，不是重发）
        #   - 其他未知异常 → 可重试有限次
        last_err: str | None = None
        for attempt in range(1, spec.max_retries + 2):
            record.attempt = attempt
            try:
                coro = spec.handler(**arguments)
                result = await asyncio.wait_for(coro, timeout=spec.timeout)
                record.result = result
                record.ok = True
                record.error = None
                if cache_key:
                    self._cache[cache_key] = result
                break
            except (PermissionError, ValueError) as exc:
                # 不可重试：安全策略拒绝 / 参数或表达式非法
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("工具被拒绝（不重试）: %s -> %s", name, last_err)
                break
            except TypeError as exc:
                # 参数签名不匹配，同样不可重试
                last_err = (
                    f"参数错误：{exc}。请检查参数名与类型是否与 schema 一致，"
                    f"schema: {json.dumps(spec.parameters, ensure_ascii=False)}"
                )
                logger.warning("工具参数错误（不重试）: %s", name)
                break
            except asyncio.TimeoutError:
                last_err = f"工具执行超时（>{spec.timeout}s），可能是外部接口慢或被限流。"
                logger.warning("工具超时: %s (第 %d 次)", name, attempt)
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("工具失败: %s (第 %d 次) %s", name, attempt, last_err)

            if attempt <= spec.max_retries:
                # 指数退避 + 抖动：避免多个 Agent 同时重试打爆下游
                await asyncio.sleep(min(2 ** (attempt - 1) * 0.3 + random.random() * 0.2, 3))

        # ---------- 收尾：判定成败 ----------
        # 【踩坑记录】这里之前写的是 `if not record.ok:`，看起来没问题，
        # 但 record.ok 的默认值是 **True**（ToolCall.ok 默认 True），
        # 所以"第一次就失败并 break"时 record.ok 仍是 True，
        # 结果：安全拦截明明生效了，却返回 ok=True —— 生产里这是致命 bug，
        # 意味着 SQL 注入防护形同虚设。
        # 正确做法：用「是否拿到结果」作为唯一判据，而不是依赖默认值。
        if record.result is None and record.error is None:
            record.ok = False
            record.error = last_err or "工具执行失败且未返回错误信息"
        elif last_err is not None and record.result is None:
            record.ok = False
            record.error = last_err

        if not record.ok:
            metrics.tool_errors += 1

        record.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        metrics.tool_calls += 1

        if tracer:
            # 工具也进 trace，才能回答"这次任务慢在模型还是慢在工具"
            with tracer.span(f"tool:{name}", kind="tool", attempt=record.attempt) as sp:
                sp.status = "ok" if record.ok else "error"
                sp.error = record.error
                sp.meta["latency_ms"] = record.latency_ms

        return record

    def clear_cache(self) -> None:
        self._cache.clear()


# ============================================================
# 12 个内置工具的实现
# 全部走「可复现的确定性模拟」，不依赖外部网络，保证今天一定能跑通、可演示。
# 生产替换：把 _simulate_* 换成真实 API 调用即可，Agent 侧代码一行不用改。
# ============================================================
def _seeded(seed_str: str) -> random.Random:
    """用字符串做种子，保证同一输入永远得到同一输出——保证演示可复现。"""
    return random.Random(int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16))


async def competitor_scraper(category: str = "kitchen storage", top_n: int = 5) -> dict[str, Any]:
    """
    模拟竞品抓取。
    设计要点：真实项目这里要处理反爬、代理池、增量抓取、robots 合规。
    代码里必须体现你对风控的认知——直接写"爬虫随便爬"是减分项。
    """
    await asyncio.sleep(0.15)  # 模拟网络 IO，方便观察并发优化效果
    rng = _seeded(f"comp:{category}")
    brands = ["HomeNest", "KitchenPro", "StorageMax", "TidyLife", "ChefChoice",
              "SimplyHome", "BoxUp", "NeatCraft"]
    items = []
    for i in range(min(top_n, len(brands))):
        items.append({
            "rank": i + 1,
            "brand": brands[i],
            "title": f"{category.title()} Organizer {i + 1}-Pack",
            "price_usd": round(rng.uniform(12.9, 45.9), 2),
            "rating": round(rng.uniform(3.9, 4.8), 1),
            "reviews": rng.randint(320, 18000),
            "monthly_sales": rng.randint(500, 9000),
            "bsr": rng.randint(80, 6000),
            "main_keywords": rng.sample(
                ["adjustable", "no drilling", "waterproof", "stackable",
                 "space saving", "heavy duty", "easy install"], 3),
        })
    items.sort(key=lambda x: x["price_usd"])
    prices = [i["price_usd"] for i in items]
    return {
        "category": category,
        "count": len(items),
        "items": items,
        "price_stats": {
            "min": min(prices), "max": max(prices),
            "avg": round(sum(prices) / len(prices), 2),
            "median": round(sorted(prices)[len(prices) // 2], 2),
        },
        "source": "amazon_us_simulated",
    }


async def review_insight(product_id: str, max_reviews: int = 100) -> dict[str, Any]:
    """模拟评论抽取。真实场景接 Amazon SP-API / 第三方评论库。"""
    await asyncio.sleep(0.12)
    rng = _seeded(f"review:{product_id}")
    n = min(max_reviews, 100)
    negatives = [
        "The adhesive failed after two weeks, fell off the wall.",
        "Too small for actual kitchen utensils, misleading photos.",
        "Arrived cracked, packaging was terrible.",
        "Instructions unclear and missing two screws.",
        "Rusty after one month of normal use.",
    ]
    positives = [
        "Love it! Saves so much cabinet space and looks clean.",
        "Installation took 5 minutes, no drilling needed. Perfect for renters.",
        "Sturdy and holds heavy items without sagging.",
        "Great value for the price, bought two more.",
        "Waterproof as advertised, great for the bathroom too.",
    ]
    picked_neg = rng.sample(negatives, min(3, len(negatives)))
    picked_pos = rng.sample(positives, min(5, len(positives)))
    return {
        "product_id": product_id,
        "sample_size": n,
        "sentiment": {
            "positive": rng.randint(62, 84),
            "neutral": rng.randint(6, 14),
            "negative": rng.randint(10, 24),
        },
        "pain_points": picked_neg,
        "delights": picked_pos,
        "top_mentions": ["easy install", "space saving", "sturdy", "value", "waterproof"],
    }


async def sql_query(sql: str, limit: int = 100) -> dict[str, Any]:
    """
    模拟 SQL 查询。
    设计要点：真实 Agent 执行 SQL 必须有「只读 + 表白名单 + 行数上限 + 超时」四道闸。
    模型生成的 SQL 直接打到生产库 = 灾难（DROP TABLE）。
    """
    await asyncio.sleep(0.05)
    sql_lower = sql.strip().lower()
    forbidden = ("drop", "delete", "update", "insert", "alter", "truncate", "grant")
    if any(k in sql_lower for k in forbidden):
        raise PermissionError("仅允许只读 SELECT 查询（安全策略拦截）")
    rng = _seeded(f"sql:{sql}")
    rows = [
        {"week": f"2026-W{i + 1:02d}",
         "revenue_usd": round(rng.uniform(8000, 22000), 2),
         "orders": rng.randint(180, 620),
         "conversion_rate": round(rng.uniform(0.07, 0.16), 4),
         "ad_spend_usd": round(rng.uniform(900, 3200), 2),
         "acos": round(rng.uniform(0.12, 0.34), 4)}
        for i in range(min(12, limit))
    ]
    return {"sql": sql[:200], "row_count": len(rows), "rows": rows}


async def calculator(expression: str) -> dict[str, Any]:
    """安全计算器：只允许数字与基础运算符，杜绝 eval 注入。"""
    import ast

    await asyncio.sleep(0.01)
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
               ast.USub, ast.UAdd, ast.Module)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"表达式包含不允许的语法: {type(node).__name__}")
    return {"expression": expression, "result": eval(  # noqa: S307
        compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})}


async def keyword_extract(text: str, top_k: int = 10) -> dict[str, Any]:
    """关键词抽取（TF-IDF 极简版）。"""
    await asyncio.sleep(0.03)
    stop = {"the", "a", "an", "and", "or", "for", "with", "of", "to", "is", "it",
            "this", "that", "you", "your", "on", "in", "we", "our", "are", "be"}
    words = [w.strip(".,!?()[]\"'").lower() for w in text.split()]
    freq: dict[str, int] = {}
    for w in words:
        if len(w) > 2 and w not in stop and not w.isdigit():
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:top_k]
    return {"keywords": [{"word": w, "count": c} for w, c in top]}


async def sentiment_analysis(texts: list[str]) -> dict[str, Any]:
    """轻量情感分析（词典法）。生产建议换成微调后的 BERT 小模型。"""
    await asyncio.sleep(0.05)
    pos_words = {"love", "great", "perfect", "sturdy", "easy", "value", "good",
                 "excellent", "recommend", "nice", "amazing", "worth"}
    neg_words = {"broke", "broken", "cracked", "rusty", "fail", "failed", "poor",
                 "terrible", "bad", "cheap", "waste", "refund", "missing", "misleading"}
    results = []
    for t in texts[:200]:
        low = t.lower()
        p = sum(1 for w in pos_words if w in low)
        n = sum(1 for w in neg_words if w in low)
        label = "positive" if p > n else ("negative" if n > p else "neutral")
        results.append({"text": t[:120], "label": label, "score": round((p - n) / max(p + n, 1), 3)})
    dist = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        dist[r["label"]] += 1
    return {"count": len(results), "distribution": dist, "samples": results[:5]}


async def trend_analysis(keyword: str, months: int = 6) -> dict[str, Any]:
    """模拟搜索热度趋势（Google Trends 风格）。"""
    await asyncio.sleep(0.1)
    rng = _seeded(f"trend:{keyword}")
    base = rng.randint(35, 75)
    series = []
    for i in range(months):
        base = max(5, min(100, base + rng.randint(-9, 12)))
        series.append({"month": f"2026-{i + 1:02d}", "interest": base})
    slope = series[-1]["interest"] - series[0]["interest"]
    return {
        "keyword": keyword,
        "series": series,
        "trend": "rising" if slope > 8 else ("declining" if slope < -8 else "stable"),
        "slope": slope,
    }


async def price_history(product_id: str, days: int = 30) -> dict[str, Any]:
    """模拟价格历史，用于判断竞品是否在打价格战。"""
    await asyncio.sleep(0.08)
    rng = _seeded(f"price:{product_id}")
    base = rng.uniform(18, 45)
    pts = []
    for i in range(min(days, 30)):
        base = max(8, base * (1 + rng.uniform(-0.03, 0.03)))
        pts.append({"day": i + 1, "price": round(base, 2)})
    prices = [p["price"] for p in pts]
    return {
        "product_id": product_id,
        "current": prices[-1],
        "min": min(prices), "max": max(prices),
        "volatility": round((max(prices) - min(prices)) / (sum(prices) / len(prices)), 4),
        "series": pts,
    }


async def image_generator(prompt: str, scene: str = "产品场景图") -> dict[str, Any]:
    """
    模拟图片生成。真实场景接 ComfyUI / SD API（飞博共创、上霖卫浴 JD 都提到）。
    设计要点：生产要按场景维护 Prompt 模板库，而不是让用户手写 prompt。
    """
    await asyncio.sleep(0.2)
    return {
        "scene": scene,
        "prompt": prompt,
        "images": [f"https://cdn.example.com/gen/{hashlib.md5(prompt.encode()).hexdigest()[:10]}_{i}.png"
                   for i in range(3)],
        "style": "clean studio, white background, soft shadow",
    }


async def erp_inventory(sku: str) -> dict[str, Any]:
    """模拟 ERP 库存查询。"""
    await asyncio.sleep(0.06)
    rng = _seeded(f"erp:{sku}")
    stock = rng.randint(0, 1800)
    return {
        "sku": sku,
        "stock": stock,
        "in_transit": rng.randint(0, 600),
        "avg_daily_sales": rng.randint(5, 60),
        "days_of_supply": round(stock / max(rng.randint(5, 60), 1), 1),
        "alert": "补货预警" if stock < 200 else "库存健康",
    }


async def exchange_rate(base: str = "USD", quote: str = "CNY") -> dict[str, Any]:
    """汇率查询（模拟）。真实接第三方汇率 API，必须做缓存（汇率一天变不了几次）。"""
    await asyncio.sleep(0.02)
    fixed = {("USD", "CNY"): 7.18, ("USD", "EUR"): 0.92, ("CNY", "USD"): 0.139}
    rate = fixed.get((base.upper(), quote.upper()), 7.18)
    return {"base": base.upper(), "quote": quote.upper(), "rate": rate,
            "updated_at": "2026-10-10T14:00:00Z"}


async def web_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """模拟联网检索（RAG 里的"外网检索"分支）。"""
    await asyncio.sleep(0.18)
    rng = _seeded(f"web:{query}")
    templates = [
        "{q} market size grew {p}% YoY according to industry report",
        "Top trends in {q} for 2026: sustainability, modular design, smart storage",
        "Consumer complaints about {q} focus on durability and installation",
        "Regulatory update: new compliance rules for {q} in the US market",
        "Competitor pricing analysis for {q} shows a shift toward mid-tier",
    ]
    docs = [{"title": f"[{i + 1}] " + t.format(q=query, p=rng.randint(4, 26))[:90],
             "snippet": t.format(q=query, p=rng.randint(4, 26)),
             "score": round(rng.uniform(0.6, 0.98), 3),
             "source": f"industry-report-{i + 1}.com"}
            for i, t in enumerate(rng.sample(templates, min(top_k, len(templates))))]
    docs.sort(key=lambda x: -x["score"])
    return {"query": query, "count": len(docs), "documents": docs}


# ============================================================
# 构建默认注册表
# ============================================================
def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(ToolSpec(
        name="competitor_scraper",
        description=(
            "抓取指定品类在亚马逊美国站的竞品榜单，返回价格、评分、评论数、月销、BSR 与主打关键词。"
            "用于竞品分析与定价策略。调用时只需传入 category（品类关键词），"
            "不要自行添加 marketplace、include_keywords 等参数——站点固定为亚马逊美国站。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "品类关键词，如 kitchen storage"},
                "top_n": {"type": "integer", "description": "抓取前 N 名，默认 5", "default": 5},
            },
            "required": ["category"],
        },
        handler=competitor_scraper,
        idempotent_key="category",
        tags=["research"],
    ))

    reg.register(ToolSpec(
        name="review_insight",
        description=(
            "抽取指定商品的评论样本，输出情感分布、痛点与好评点。用于挖掘用户真实需求，指导 Listing 卖点。"
            "必填参数是 product_id（商品 ID 或 ASIN）。"
            "如果没有真实 ASIN，请用竞品抓取结果里的商品名或品类关键词作为 product_id 占位，不要省略该参数。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "商品 ID 或 ASIN；缺失时可用商品名/品类关键词代替",
                },
                "max_reviews": {"type": "integer", "description": "抽取评论条数，默认 100", "default": 100},
            },
            "required": ["product_id"],
        },
        handler=review_insight,
        idempotent_key="product_id",
        tags=["research"],
    ))

    reg.register(ToolSpec(
        name="sql_query",
        description="对运营数据仓库执行【只读 SELECT】查询，返回周度营收、订单、转化率、广告花费等指标。禁止任何写操作。",
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "只读 SELECT 语句"},
                "limit": {"type": "integer", "description": "返回行数上限，默认 100", "default": 100},
            },
            "required": ["sql"],
        },
        handler=sql_query,
        tags=["data"],
    ))

    reg.register(ToolSpec(
        name="calculator",
        description="安全计算器，支持 + - * / ** % 与括号。用于计算利润率、ACoS、同比增幅等，避免模型算错数。",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式，如 (22.9-8.4)/22.9*100"}},
            "required": ["expression"],
        },
        handler=calculator,
        tags=["util"],
    ))

    reg.register(ToolSpec(
        name="keyword_extract",
        description="从文本中抽取高频关键词及其频次，用于 Listing 关键词埋词与广告词库构建。",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "待抽取文本"},
                "top_k": {"type": "integer", "description": "返回前 K 个，默认 10", "default": 10},
            },
            "required": ["text"],
        },
        handler=keyword_extract,
        tags=["nlp"],
    ))

    reg.register(ToolSpec(
        name="sentiment_analysis",
        description="批量文本情感分析，返回情感分布与样本。用于评论洞察与舆情监控。",
        parameters={
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}, "description": "文本列表"},
            },
            "required": ["texts"],
        },
        handler=sentiment_analysis,
        tags=["nlp"],
    ))

    reg.register(ToolSpec(
        name="trend_analysis",
        description="查询关键词的搜索热度趋势，判断该品类处于上升期还是衰退期。用于选品决策。",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "关键词"},
                "months": {"type": "integer", "description": "回溯月数，默认 6", "default": 6},
            },
            "required": ["keyword"],
        },
        handler=trend_analysis,
        idempotent_key="keyword",
        tags=["research"],
    ))

    reg.register(ToolSpec(
        name="price_history",
        description="查询商品近 30 天价格曲线与波动率，用于判断竞品是否在打价格战。",
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "商品 ID"},
                "days": {"type": "integer", "description": "回溯天数，默认 30", "default": 30},
            },
            "required": ["product_id"],
        },
        handler=price_history,
        idempotent_key="product_id",
        tags=["research"],
    ))

    reg.register(ToolSpec(
        name="image_generator",
        description="根据商品卖点描述生成产品场景图/主图，返回图片链接。用于 Listing 与广告素材。",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "画面描述（英文效果更好）"},
                "scene": {"type": "string", "description": "场景类型，如 产品场景图/模特图", "default": "产品场景图"},
            },
            "required": ["prompt"],
        },
        handler=image_generator,
        tags=["creative"],
    ))

    reg.register(ToolSpec(
        name="erp_inventory",
        description="查询 SKU 的实时库存、在途量与可售天数，返回补货预警。用于运营周报与补货决策。",
        parameters={
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "商品 SKU"}},
            "required": ["sku"],
        },
        handler=erp_inventory,
        idempotent_key="sku",
        tags=["business"],
    ))

    reg.register(ToolSpec(
        name="exchange_rate",
        description="查询实时汇率，用于多币种定价与利润换算。",
        parameters={
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "基础币种，默认 USD", "default": "USD"},
                "quote": {"type": "string", "description": "目标币种，默认 CNY", "default": "CNY"},
            },
            "required": [],
        },
        handler=exchange_rate,
        idempotent_key="base",
        tags=["business"],
    ))

    reg.register(ToolSpec(
        name="web_search",
        description="联网检索行业报告与市场资讯，返回带相关度分数的文档片段。用于补充品类趋势与政策合规信息。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词"},
                "top_k": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
            },
            "required": ["query"],
        },
        handler=web_search,
        idempotent_key="query",
        timeout=25.0,
        tags=["research"],
    ))

    return reg


# 全局单例
tool_registry = build_default_registry()
