"""
Step 4 补充 · 持久化层（Checkpointer）
============================================================
面试考点：Agent 的"断点续跑"是怎么实现的？
  → 核心是 Checkpointer：把图的完整状态在每个 super-step 后存下来。
    LangGraph 内置了这个抽象，换存储只改一行（MemorySaver ↔ PostgresSaver）。

为什么企业级必须落库而不是放内存？
  1. 服务重启不丢任务（内存态重启即归零）
  2. 长任务可以中断数小时等人工审核，进程不可能一直挂着
  3. 多个 worker 实例要能接续同一个任务（横向扩展的前提）

本模块做了「自动降级」：
  - 配了 PostgreSQL 且依赖可装可用 → 用官方 PostgresSaver（生产）
  - 否则 → 用官方 MemorySaver（本地/演示，重启即丢）
  注意：当前**没有**自建 SQLite Checkpointer。业务侧的任务视图落在
  TaskStore（data/tasks/*.json），那是 CQRS 的读侧，不是图的 checkpoint。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.observability import logger


def build_checkpointer() -> Any:
    """
    返回一个 LangGraph 兼容的 checkpointer。

    面试考点：为什么用 try-import 而不是硬依赖？
      → 本地开发和 CI 不该被迫装 PostgreSQL。可选依赖降级是工程常识，
        PostgresSaver 属于可选依赖，装不上就退到内存实现。
    """
    # ---------- 方案 A：生产 Postgres ----------
    if settings.database_url.startswith("postgresql"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore

            saver = PostgresSaver.from_conn_string(settings.database_url)
            saver.setup()  # 自动建表
            logger.info("Checkpointer: PostgreSQL（生产模式）")
            return saver
        except Exception as exc:  # noqa: BLE001
            logger.warning("PostgresSaver 不可用（%s），降级到内存模式", exc)

    # ---------- 方案 B：本地/演示 内存存储 ----------
    # 说明：这里用 MemorySaver 保证开箱即跑；要演示"重启不丢"，
    #      需安装 langgraph-checkpoint-postgres 并把 DATABASE_URL 切到 postgresql。
    try:
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Checkpointer: InMemory（演示模式，重启会丢）")
        return MemorySaver()
    except Exception:  # noqa: BLE001
        logger.warning("LangGraph checkpointer 不可用，断点续跑功能关闭")
        return None


# ============================================================
# 轻量任务状态仓库（用于 API 侧查询任务进度）
# ============================================================
class TaskStore:
    """
    面试考点：为什么任务状态要和 Graph 的 checkpoint 分开存？
      → 职责不同。checkpoint 是「执行引擎的内部状态」（为了能恢复），
        TaskStore 是「面向业务的查询视图」（前端要展示任务列表、状态、产物）。
        混在一起会导致：①内部结构变更频繁，污染 API 契约
        ②查询要反序列化整张图，性能差。这是典型的 CQRS 思想——读写分离。

    【踩坑记录 · 实测发现】
      最初的实现是纯内存 dict，进程重启后所有任务产物全部丢失，
      包括花 11 分钟、5.8 万 token 跑出来的 13000 字交付报告 —— 成本白花了。
      这说明：**内存态只适合做缓存，不能做唯一存储**。
      修复：落盘为 JSON 文件（data/tasks/*.json），保持接口不变。
      生产可直接换 PostgreSQL / S3，因为读写都收敛在这一个类里（依赖倒置）。
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        # 落盘目录：默认 data/tasks/
        self._dir = persist_dir or (settings.data_dir / "tasks")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    # -------- 持久化辅助 --------
    def _path(self, task_id: str) -> Path:
        # 只保留安全字符，避免路径穿越（task_id 来自外部输入）
        safe = "".join(c for c in task_id if c.isalnum() or c in "-_")
        return self._dir / f"{safe}.json"

    def _load_from_disk(self) -> None:
        """启动时把历史任务读回内存，保证服务重启后产物不丢。"""
        count = 0
        for f in self._dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                tid = data.get("task_id")
                if tid:
                    self._tasks[tid] = data
                    count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取历史任务失败 %s: %s", f.name, exc)
        if count:
            logger.info("TaskStore 从磁盘恢复 %d 条历史任务", count)

    def save(self, task_id: str, data: dict[str, Any]) -> None:
        payload = json.loads(json.dumps(data, ensure_ascii=False, default=str))
        payload["task_id"] = task_id
        self._tasks[task_id] = payload
        # 落盘（大产物也一并存，这正是价值所在）
        try:
            self._path(task_id).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            # 落盘失败不能影响主流程（降级为纯内存）
            logger.warning("任务落盘失败 %s: %s", task_id, exc)

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self._tasks.get(task_id)

    def list_all(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list(self._tasks.values())
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items[:limit]

    # -------- 事件流：给 SSE 前端做实时进度推送 --------
    def emit(self, task_id: str, event: dict[str, Any]) -> None:
        self._events.setdefault(task_id, []).append(event)

    def events(self, task_id: str) -> list[dict[str, Any]]:
        return self._events.get(task_id, [])

    def clear(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._events.pop(task_id, None)


task_store = TaskStore()
