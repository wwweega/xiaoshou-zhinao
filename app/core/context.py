"""
Step 6 补充 · 追踪上下文的传递方案
============================================================
【这是一个真实的架构坑，值得单独讲】

问题：
  最初我把 TraceRecorder 放进了 GraphState 里传给各个节点。
  跑起来直接报错：
      TypeError: Type is not msgpack serializable: TraceRecorder

原因：
  LangGraph 的 Checkpointer 在**每个 super-step 结束后**会把整个状态
  序列化后存盘（为了断点续跑）。而 TraceRecorder 里持有
  span 列表、asyncio 相关的对象，是不可序列化的。

  更深层的原因是设计理念冲突：
    - 图状态 = 业务数据，必须可序列化（要能存、能跨进程恢复）
    - Trace   = 运行时观测数据，天然是进程内的、临时的
  
  把"观测数据"塞进"业务状态"，是典型的职责混淆。

【三种解法，以及我为什么选第三个】

  方案 A：把 TraceRecorder 从状态里删掉，全局单例
    ✗ 缺陷：多任务并发时会互相污染（A 任务的 span 混进 B 任务）

  方案 B：自己实现序列化接口，让 TraceRecorder 可被 msgpack 打包
    ✗ 缺陷：图状态会变得巨大（几十个 span × 每个任务），
      每次 super-step 都要序列化一遍，性能急剧下降。
      而且恢复时反序列化出旧的 trace 也没意义——追踪的是"本次执行"。

  方案 C：用 contextvars 传递（Python 3.7+）
    ✓ 优点 ①不进图状态，不参与序列化 ②天然按协程隔离，
      每个 async task 有自己的上下文副本，并发安全
      ③业务代码零侵入（节点函数通过 get_tracer() 取，不用从 state 里挖）

  这就是最终采用 contextvars 的原因——它是 Python 官方为
  "异步上下文隔离"提供的标准方案。

【这个设计的意义】
这是「你踩过什么坑」类问题的绝佳素材，因为它同时展示了：
  ①理解框架的底层机制（checkpointer 会序列化状态）
  ②能定位根因（不是语法问题，是职责混淆）
  ③评估了多个方案并给出取舍理由（而不是撞到一个能跑的就完事）
  ④知道 Python 的 contextvars 标准方案
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Optional

from app.core.observability import TraceRecorder

# 每个 async 上下文独立的 tracer 槽位
_tracer_var: ContextVar[Optional[TraceRecorder]] = ContextVar("zhinao_tracer", default=None)
_task_id_var: ContextVar[str] = ContextVar("zhinao_task_id", default="")
_thread_id_var: ContextVar[str] = ContextVar("zhinao_thread_id", default="")


def set_tracer(tracer: TraceRecorder) -> Token:
    """
    绑定当前上下文的 tracer。

    注意：contextvars 的隔离单位是「asyncio task」。
    asyncio.gather 创建的子任务会**复制**父上下文，
    所以在子任务里 set 的值不会污染父任务——这正是我们想要的。
    """
    return _tracer_var.set(tracer)


def get_tracer() -> Optional[TraceRecorder]:
    """节点函数里随时取当前 tracer，不用从 state 里传。"""
    return _tracer_var.get()


def reset_tracer(token: Token) -> None:
    """恢复上一个上下文，避免跨请求泄漏。"""
    _tracer_var.reset(token)


def set_task_context(task_id: str, thread_id: str) -> None:
    _task_id_var.set(task_id)
    _thread_id_var.set(thread_id)


def get_task_id() -> str:
    return _task_id_var.get()


def get_thread_id() -> str:
    return _thread_id_var.get()


class span_of:
    """
    便捷上下文管理器：有 tracer 就用它开 span，没有就退化为空操作。

    设计要点：为什么不用 if 判断？
      → 因为每个节点里写 `if tracer: with tracer.span(...)` 会让代码
        到处是缩进层级。用统一的替身对象（空对象模式）能让业务代码保持平铺。
    """

    def __init__(self, name: str, kind: str = "node", **meta: Any):
        self.name = name
        self.kind = kind
        self.meta = meta
        self._ctx: Any = None
        self._span: Any = None

    def __enter__(self):
        tracer = get_tracer()
        if tracer is None:
            self._span = _NullSpan()
            return self._span

        class _Ctx:
            def __init__(self, cm):
                self.cm = cm

            def __enter__(self):
                return self.cm.__enter__()

            def __exit__(self, *a):
                return self.cm.__exit__(*a)

            def __call__(self):
                return self

        # 直接把 tracer.span 当上下文管理器用
        cm = tracer.span(self.name, kind=self.kind, **self.meta)
        self._ctx = cm
        self._span = cm.__enter__()
        return self._span

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


class _NullSpan:
    """空对象：没有 tracer 时的替身，避免到处判空。"""

    model: str | None = None
    status: str = "ok"
    error: str | None = None
    meta: dict[str, Any] = {}
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return 0
