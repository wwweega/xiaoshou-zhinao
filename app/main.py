"""
Step 7 补充 · FastAPI 应用入口
============================================================
面试考点：lifespan 里该做什么、不该做什么？
  该做：初始化连接池、预热模型、校验配置、注册优雅关闭钩子
  不该做：跑耗时的训练/大批量预热（会拖长启动时间，K8s 探针会判定失败）
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.core.config import BASE_DIR, settings
from app.core.observability import logger
from app.llm.client import llm_client
from app.tools.registry import tool_registry

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期。

    面试考点（真实故障场景）：为什么不在这里做重初始化？
      → K8s 的 readinessProbe 通常只等 30s。如果启动时跑 5 分钟预热，
        Pod 会被判定启动失败反复重启，形成 CrashLoopBackOff。
        正确做法：重初始化放到后台任务，服务先以"降级可用"状态起来。
    """
    logger.info("=" * 68)
    logger.info("  运营智脑 · 跨境电商 Multi-Agent 运营自动化平台")
    logger.info("=" * 68)
    logger.info("LLM provider : %s", settings.llm_provider)
    logger.info("Mock 模式    : %s %s", settings.use_mock,
                "（未检测到有效 API Key，将使用离线模拟输出）" if settings.use_mock else "")
    logger.info("大模型        : %s", settings.llm_model)
    logger.info("小模型        : %s", settings.llm_small_model)
    logger.info("工具数量      : %d", len(tool_registry.names()))
    logger.info("反思轮次上限  : %d", settings.max_reflection_rounds)
    logger.info("-" * 68)

    yield

    await llm_client.close()
    logger.info("服务已关闭，连接已释放")


app = FastAPI(
    title="运营智脑 · 跨境电商 Multi-Agent 运营自动化平台",
    description=(
        "基于 LangGraph 的多智能体协作系统：Planner → Researcher → Analyst → "
        "Writer → Critic → Reviewer，覆盖市场调研、竞品分析、Listing 生成、"
        "广告关键词与运营周报全流程。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：本地开发放开，生产要收紧为白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    """返回内置的演示前端。"""
    html_file = BASE_DIR / "frontend" / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>运营智脑</h1><p>前端文件未找到，请访问 /docs 使用 Swagger UI</p>")


@app.get("/stats", include_in_schema=False)
async def stats():
    return {"uptime_seconds": round(time.time() - START_TIME, 1)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
