"""全局配置：统一从 .env 读取，全项目唯一配置入口。

设计要点（面试可讲）：
- pydantic-settings 强类型配置，错误在启动时暴露而不是运行中
- 所有密钥只进 .env（gitignore），代码库零泄露
- 提供派生属性（如 llm_cfg）把「提供商切换」收敛到一处
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（src/autoreport/config.py -> 上三级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ProviderConfig(BaseSettings):
    """单个 LLM 提供商的连接配置。"""

    api_key: str = ""
    base_url: str = ""
    main_model: str = ""
    small_model: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM ----------
    llm_provider: str = "qwen"  # qwen / deepseek / openai_compatible
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_main_model: str = "qwen-plus"
    qwen_small_model: str = "qwen-turbo"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_main_model: str = "deepseek-chat"
    deepseek_small_model: str = "deepseek-chat"
    openai_compatible_api_key: str = ""
    openai_compatible_base_url: str = ""
    openai_compatible_main_model: str = ""
    openai_compatible_small_model: str = ""

    # ---------- 嵌入 ----------
    embedding_provider: str = "auto"  # auto / dashscope / local

    # ---------- LangSmith ----------
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "autoreport-dev"

    # ---------- 存储 ----------
    db_path: str = "data/db/autoreport.sqlite"
    chroma_dir: str = "data/db/chroma"
    checkpoint_path: str = "data/db/checkpoints.sqlite"

    # ---------- 采集 ----------
    crawl_delay_seconds: float = 3.0
    crawl_max_retries: int = 3
    crawl_timeout_seconds: int = 30
    user_agent: str = "AutoReport-Research/0.1 (personal study project)"

    # ---------- 检索 ----------
    retrieval_top_k: int = 6
    retrieval_final_k: int = 8

    # ---------- 多智能体 ----------
    hitl_enabled: bool = True

    # ---------- 联网搜索 ----------
    tavily_api_key: str = ""

    # ---------- 其他 ----------
    hf_endpoint: str = "https://hf-mirror.com"
    log_level: str = "INFO"

    # ---------------- 派生配置 ----------------

    @property
    def llm_cfg(self) -> ProviderConfig:
        """当前 LLM 提供商配置（切换提供商只改 .env 的 LLM_PROVIDER）。"""
        p = self.llm_provider.lower()
        if p == "qwen":
            return ProviderConfig(
                api_key=self.qwen_api_key,
                base_url=self.qwen_base_url,
                main_model=self.qwen_main_model,
                small_model=self.qwen_small_model,
            )
        if p == "deepseek":
            return ProviderConfig(
                api_key=self.deepseek_api_key,
                base_url=self.deepseek_base_url,
                main_model=self.deepseek_main_model,
                small_model=self.deepseek_small_model,
            )
        if p == "openai_compatible":
            return ProviderConfig(
                api_key=self.openai_compatible_api_key,
                base_url=self.openai_compatible_base_url,
                main_model=self.openai_compatible_main_model,
                small_model=self.openai_compatible_small_model,
            )
        raise ValueError(f"未知 LLM_PROVIDER: {self.llm_provider}")

    @property
    def llm_available(self) -> bool:
        """LLM 是否可用（决定系统能否进入「智能模式」）。"""
        cfg = self.llm_cfg
        return bool(cfg.api_key and cfg.base_url and cfg.main_model)

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing and bool(self.langsmith_api_key)

    # ---------------- 路径 ----------------

    def abs_path(self, rel: str) -> Path:
        """相对路径 -> 项目根下的绝对路径，并确保父目录存在。"""
        p = Path(rel)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_file(self) -> Path:
        return self.abs_path(self.db_path)

    @property
    def chroma_dir_abs(self) -> Path:
        p = Path(self.chroma_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def inbox_dir(self) -> Path:
        p = PROJECT_ROOT / "data" / "inbox"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_raw_dir(self) -> Path:
        p = PROJECT_ROOT / "data" / "reports_raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def announcements_dir(self) -> Path:
        p = PROJECT_ROOT / "data" / "announcements"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    """单例配置（lru_cache 避免重复解析 .env）。"""
    return Settings()


def apply_langsmith_env() -> None:
    """把 .env 中的 LangSmith 配置注入标准环境变量。

    LangChain 1.x 通过 LANGSMITH_TRACING / LANGSMITH_API_KEY 环境变量开启追踪，
    这里在进程启动时统一注入，业务代码无需感知。
    无 Key 时静默跳过 —— 整个系统在「无 LangSmith」下照常工作（降级策略）。
    """
    import os

    s = get_settings()
    if s.langsmith_enabled:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
        os.environ.setdefault("LANGSMITH_PROJECT", s.langsmith_project)
