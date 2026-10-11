"""
Step 7 · FastAPI 接口层（SSE 流式 + 生命周期管理）
============================================================
面试考点：

Q: 为什么 Agent 服务必须用流式（SSE）而不是普通 HTTP 响应？
A:  Agent 任务动辄几十秒到几分钟。普通请求会：
    ①前端一直转圈，用户以为卡死 ②Nginx/网关默认 60s 超时直接 504 ③用户无法中断
    SSE（Server-Sent Events）让中间过程实时可见，体验从"黑盒等待"变成"看着它干活"。
    选 SSE 而不是 WebSocket 的原因：单向推送够用、基于 HTTP 无需协议升级、
    能天然穿透大多数企业网关（WebSocket 常被防火墙拦）。

Q: 长任务为什么要用后台任务队列（Celery）而不是直接 await？
A:  ①HTTP 连接不该占用几分钟 ②任务要能重试/重放 ③要能横向扩展 worker。
    本项目的做法：提供「同步等待」和「提交后台任务 + 轮询/流式订阅」两条路径，
    小任务同步返回便于演示，长任务走异步队列。

Q: FastAPI 的 lifespan 用来干什么？
A:  管理应用级资源：启动时初始化连接池、加载模型；关闭时优雅释放。
    不用 lifespan 而每次请求新建连接 = 连接泄漏 + 性能灾难。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.observability import logger, metrics
from app.core.schemas import TaskStatus
from app.graph.workflow import runner
from app.llm.client import llm_client
from app.memory.persistence import task_store
from app.memory.store import long_term_memory
from app.tools.registry import tool_registry

router = APIRouter()


# ============================================================
# 请求 / 响应体
# ============================================================
class RunRequest(BaseModel):
    query: str = Field(..., description="自然语言运营需求", min_length=2, max_length=2000)
    wait: bool = Field(True, description="True=同步等待结果；False=立即返回 task_id 走后台")
    stream: bool = Field(False, description="True=返回 SSE 流")


class RunResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskDetail(BaseModel):
    task_id: str
    status: str
    query: str
    score: float | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    trace_id: str | None = None
    deliverable_content: str | None = None


# ============================================================
# 1. 核心接口：执行任务（三种模式）
# ============================================================
@router.post("/run", summary="执行一个运营任务")
async def run_task(req: RunRequest, background: BackgroundTasks):
    """
    三种调用方式：
      - 默认（wait=true, stream=false）：同步等待完整结果，适合演示与调试
      - wait=false：立即返回 task_id，任务在后台跑，之后轮询 /task/{id}
      - stream=true：SSE 流式，实时推送每一步进展，适合前端接入

    面试考点：为什么参数要设计成三种模式而不是三个接口？
      → 底层执行逻辑完全相同，差异只在"结果投递方式"。
        用参数区分能避免逻辑重复，也不会让 API 面膨胀。
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    if req.stream:
        return StreamingResponse(
            _stream_task(query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 关键：禁用 Nginx 缓冲，否则 SSE 会被攒着一起发
            },
        )

    if not req.wait:
        # 后台执行：先占位再启动
        task_id = f"task_{int(time.time() * 1000)}"
        task_store.save(task_id, {
            "task_id": task_id, "query": query, "status": TaskStatus.PENDING.value,
        })

        async def _bg() -> None:
            try:
                await runner.run(query, task_id=task_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("后台任务失败: %s", exc)

        background.add_task(_bg)
        return RunResponse(task_id=task_id, status=TaskStatus.PENDING.value,
                           message="任务已提交后台执行，请轮询 /api/task/{task_id}")

    # 同步等待
    result = await runner.run(query)
    logger.info("同步任务完成\n%s", result.trace_text)
    return {
        "summary": result.summary.model_dump(mode="json"),
        "trace": result.trace_text,
    }


# ============================================================
# SSE 流式实现
# ============================================================
async def _stream_task(query: str) -> AsyncGenerator[str, None]:
    """
    SSE 生成器。

    面试考点：SSE 的坑有哪些？（答出这些会让面试官觉得你真做过）
      ①Nginx 缓冲：必须加 X-Accel-Buffering: no，否则事件被攒着不发
      ②代理超时：网关的 read timeout 要大于任务时长
      ③前端断连检测：客户端关页面后服务端要能感知并取消任务，否则白烧 token
      ④心跳：中间长时间无事件会被中间设备掐断，要定期发注释行保活
      ⑤事件格式必须严格：data: xxx\n\n（两个换行），写错前端收不到
    """
    task_id = f"task_{int(time.time() * 1000)}"

    # 先把任务提交到后台，这样主协程可以一边推事件一边等
    task = asyncio.create_task(runner.run(query, task_id=task_id))

    yield _sse({"type": "start", "task_id": task_id, "query": query})

    seen = 0
    last_heartbeat = time.time()

    try:
        while not task.done():
            # 增量推送新事件
            events = task_store.events(task_id)
            while seen < len(events):
                yield _sse(events[seen])
                seen += 1

            # 心跳保活（每 10s 一条注释）
            if time.time() - last_heartbeat > 10:
                yield ": heartbeat\n\n"
                last_heartbeat = time.time()

            await asyncio.sleep(0.3)

        # 收尾：把剩余事件全部推完
        events = task_store.events(task_id)
        while seen < len(events):
            yield _sse(events[seen])
            seen += 1

        result = task.result()
        yield _sse({
            "type": "result",
            "task_id": task_id,
            "score": result.summary.critique.score if result.summary.critique else None,
            "tokens": result.summary.total_tokens,
            "cost_usd": result.summary.total_cost_usd,
            "latency_ms": result.summary.latency_ms,
            "reflection_rounds": result.summary.reflection_rounds,
            "deliverable": result.summary.deliverable.content if result.summary.deliverable else "",
            "trace": result.trace_text,
        })
    except asyncio.CancelledError:
        # 客户端断开：取消任务，避免继续烧 token
        task.cancel()
        logger.info("客户端断开，已取消任务 %s", task_id)
        raise


def _sse(payload: dict[str, Any]) -> str:
    """SSE 帧格式：data: <json>\n\n"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ============================================================
# 2. 任务查询
# ============================================================
@router.get("/task/{task_id}", summary="查询任务详情")
async def get_task(task_id: str):
    data = task_store.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return data


@router.get("/tasks", summary="任务列表")
async def list_tasks(limit: int = 20):
    return {"items": task_store.list_all(limit=limit)}


@router.get("/task/{task_id}/events", summary="任务事件流（已结束任务）")
async def task_events(task_id: str):
    if not task_store.get(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"task_id": task_id, "events": task_store.events(task_id)}


@router.get("/task/{task_id}/deliverable", summary="下载归档的交付物（Markdown）")
async def download_deliverable(task_id: str):
    """
    面试考点：为什么交付物要单独落盘 + 提供下载接口？
      - 长任务产物只存内存 → 服务重启即永久丢失（本项目踩过的坑）
      - 归档成 Markdown → 可读、可 diff、可直接交给业务方
      - 单独 endpoint 而非塞进 task 详情 → 避免大文本污染列表接口响应
    """
    from pathlib import Path

    safe = "".join(c for c in task_id if c.isalnum() or c in "-_")
    path = settings.data_dir / "deliverables" / f"{safe}.md"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"未找到 task {task_id} 的归档交付物（可能任务尚未完成或未归档）",
        )
    return {"task_id": task_id, "path": str(path), "content": path.read_text(encoding="utf-8")}


# ============================================================
# 3. 可观测接口
# ============================================================
@router.get("/metrics", summary="运行指标（P95 延迟 / 成本 / 缓存命中）")
async def get_metrics():
    """
    面试考点：这四个指标分别回答什么问题？
      - latency p95/p99：用户体验（平均值会骗人，尾部延迟才是真实体感）
      - tool_error_rate：Agent 稳定性（工具挂了 Agent 就是瞎子）
      - cache_hit_rate：降本效果（缓存是性价比最高的优化）
      - total_cost_usd / avg_cost_per_task：老板最关心的问题
    """
    snap = metrics.snapshot()
    snap["llm"] = {
        "total_calls": llm_client.call_count,
        "big_model_calls": llm_client.big_model_calls,
        "small_model_calls": llm_client.small_model_calls,
        "routing_saving_ratio": round(
            llm_client.small_model_calls / llm_client.call_count, 4
        ) if llm_client.call_count else 0.0,
        "retry_count": llm_client.retry_count,
    }
    snap["memory"] = {"long_term_entries": long_term_memory.size()}
    snap["tools"] = {"registered": len(tool_registry.names()), "names": tool_registry.names()}
    snap["config"] = {
        "provider": settings.llm_provider,
        "mock_mode": settings.use_mock,
        "big_model": settings.llm_model,
        "small_model": settings.llm_small_model,
        "routing_threshold": settings.model_routing_threshold,
        "max_reflection_rounds": settings.max_reflection_rounds,
    }
    return snap


@router.get("/traces", summary="Trace 列表")
async def list_traces(limit: int = 20):
    files = sorted(settings.trace_path.glob("tr_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:limit]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({
                "trace_id": d.get("trace_id"),
                "task_id": d.get("task_id"),
                "query": d.get("query"),
                "created_at": d.get("created_at"),
                "total_tokens": d.get("total_tokens"),
                "total_cost_usd": d.get("total_cost_usd"),
                "wall_latency_ms": d.get("wall_latency_ms"),
                "span_count": len(d.get("spans", [])),
            })
        except Exception:  # noqa: BLE001
            continue
    return {"items": out}


@router.get("/trace/{trace_id}", summary="Trace 详情（带耗时排序）")
async def get_trace(trace_id: str):
    f = settings.trace_path / f"{trace_id}.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="trace 不存在")
    data = json.loads(f.read_text(encoding="utf-8"))
    # 按耗时降序，瓶颈一眼可见
    data["spans"].sort(key=lambda s: -s.get("latency_ms", 0))
    return data


# ============================================================
# 4. 工具与记忆管理
# ============================================================
@router.get("/tools", summary="工具清单（MCP 风格 schema）")
async def list_tools():
    return {"count": len(tool_registry.names()), "tools": tool_registry.all_schemas()}


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


@router.post("/tool/call", summary="直接调用工具（调试用）")
async def call_tool(req: ToolCallRequest):
    record = await tool_registry.call(req.tool_name, req.arguments)
    if not record.ok:
        return {"ok": False, "error": record.error, "latency_ms": record.latency_ms}
    return {"ok": True, "result": record.result, "latency_ms": record.latency_ms,
            "cached": record.attempt == 1 and record.latency_ms < 5}


@router.get("/memory/search", summary="检索长期记忆")
async def search_memory(q: str, top_k: int = 3):
    items = await long_term_memory.recall(q, top_k=top_k)
    return {"query": q, "count": len(items), "items": items}


# ============================================================
# 5. 健康检查
# ============================================================
@router.get("/health", summary="健康检查")
async def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "mock_mode": settings.use_mock,
        "tools": len(tool_registry.names()),
        "checkpointer": type(runner.checkpointer).__name__ if runner.checkpointer else None,
    }
