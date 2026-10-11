"""
Step 1 · 统一配置层
============================================================
面试考点：为什么要有 config 层？
  - 12-Factor App 原则：配置与代码分离，同一份镜像可用不同环境变量部署到 dev/staging/prod
  - 密钥绝不进 Git：.env 在 .gitignore 里，只提交 .env.example
  - 类型安全：pydantic-settings 会在启动时就校验配置，而不是等到跑任务时报 KeyError
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/core/config.py → 上溯两级）
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """全局配置。所有字段可用环境变量覆盖，也可写在 .env 里。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 允许 .env 里有本项目不认识的变量，不报错
    )

    # ---------------- LLM ----------------
    llm_provider: Literal["openai_compatible", "mock"] = "openai_compatible"

    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # 小模型：用于模型路由，做抽取/分类/意图识别这类"简单但高频"的活
    llm_small_model: str = "deepseek-chat"
    llm_small_base_url: str = ""
    llm_small_api_key: str = ""

    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096

    # 为什么默认给到 180s 而不是 60s？
    # 新一代推理型模型（qwen3.8 系列默认开启深度思考）在生成长文本时，
    # 单次调用 60~130s 是常态。把超时压到 60s 会导致大量「非故障性超时」——
    # 请求其实在正常生成，只是还没返回，客户端就先掐断了。
    # 后果比慢更糟：掐断后触发重试，同一个请求被重复计费三次，成本翻三倍。
    # 所以超时阈值必须大于模型的 P99 生成耗时，宁可等，不要白等重试。
    llm_timeout: int = 180
    llm_max_retries: int = 3

    # ---------------- 服务 ----------------
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ---------------- 存储 ----------------
    # 本项目只用 SQLite：单机、零运维、便于复现。
    # 说明：仓库早期版本里曾预留 redis_url / enable_redis 两个字段，但代码中从未被引用，
    # 属于「配了不用」的死配置。为避免读者误以为项目依赖 Redis，这里已删除。
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'zhinao.db'}"

    # ---------------- 记忆 / 上下文工程 ----------------
    memory_window_size: int = 6
    memory_summary_trigger_tokens: int = 3000
    memory_top_k: int = 3

    # ---------------- 模型路由与成本 ----------------
    model_routing_threshold: float = 0.5
    enable_cost_tracking: bool = True

    # ---------------- Agent 行为 ----------------
    max_reflection_rounds: int = 2
    task_token_budget: int = 80000

    # ---------------- 可观测 ----------------
    enable_trace: bool = True
    trace_dir: str = str(BASE_DIR / "data" / "traces")

    # ---------------- 校验 ----------------
    @field_validator("llm_small_base_url")
    @classmethod
    def _fallback_base_url(cls, v: str, info) -> str:
        """小模型不单独配置时，复用大模型的 base_url，避免漏配。"""
        return v or info.data.get("llm_base_url", "")

    @field_validator("llm_small_api_key")
    @classmethod
    def _fallback_api_key(cls, v: str, info) -> str:
        return v or info.data.get("llm_api_key", "")

    # ---------------- 便捷属性 ----------------
    @property
    def use_mock(self) -> bool:
        """
        是否走离线 mock 通道。

        两条判定（顺序不能反）：
          1. 显式指定 LLM_PROVIDER=mock → 一定 mock（CI / 演示强制开关）
          2. 否则看有没有有效 Key：没有就自动降级 mock，
             保证项目「克隆下来就能跑」，不会因为漏配 Key 直接崩

        面试考点：为什么要有自动降级？
          → 三个真实理由：①CI 里绝不能真的花钱调模型
            ②单测需要确定性输出 ③演示/答辩现场断网或余额不足时不能翻车。
            生产项目里 mock provider 是标配，不是可选项。
        """
        if self.llm_provider == "mock":
            return True
        return not self.llm_api_key.startswith("sk-")

    @property
    def data_dir(self) -> Path:
        d = BASE_DIR / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def trace_path(self) -> Path:
        p = Path(self.trace_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    """
    单例：整个进程只解析一次 .env。
    面试追问：为什么用 lru_cache 而不是模块级变量？
      → 模块级变量在 import 时就执行，测试里不好替换；lru_cache 可显式 clear()
        让单测注入不同配置，同时仍是进程内单例。
    """
    return Settings()


settings = get_settings()
