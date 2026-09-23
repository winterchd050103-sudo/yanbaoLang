"""LLM 工厂：统一创建 Chat 模型与 Embedding，屏蔽底层提供商差异。

设计要点：
- Qwen / DeepSeek 都提供 OpenAI 兼容接口，因此统一用 ChatOpenAI + base_url 接入，
  切换提供商只改 .env 的 LLM_PROVIDER，业务代码零改动
- 模型分级控制成本：main（规划/写作）与 small（抽取/改写/judge）两档
- 所有工厂函数「延迟报错」：没配 Key 时创建不失败，调用时才报清晰错误，
  便于离线开发解析/检索层
"""

from __future__ import annotations

import logging

from functools import lru_cache

from autoreport.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_chat_model(role: str = "main", settings: Settings | None = None):
    """创建 Chat 模型。

    Args:
        role: "main"（规划/写作等重任务）或 "small"（抽取/改写/评测等轻任务）。
              模型分级是 token 成本控制的第一手段。
    """
    s = settings or get_settings()
    from langchain_openai import ChatOpenAI

    cfg = s.llm_cfg
    if not s.llm_available:
        raise RuntimeError(
            "LLM 未配置：请在 .env 中填写 API Key（参考 .env.example）。"
            f"当前 LLM_PROVIDER={s.llm_provider}"
        )
    model = cfg.main_model if role == "main" else (cfg.small_model or cfg.main_model)
    return ChatOpenAI(
        model=model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0.1,  # 研报场景重事实、低随机性
        timeout=120,
        max_retries=2,
    )


def structured_invoke(llm, schema_cls, messages):
    """DeepSeek 兼容的结构化输出。

    背景：with_structured_output 默认走 json_schema response_format，
    DeepSeek 仅支持 json_object（会报 "This response_format type is
    unavailable now"）。这里统一改为 json_mode + 在 prompt 显式注入
    schema + Pydantic 校验 —— Qwen/DeepSeek 通吃，且可自动剥离代码围栏。
    """
    import json as _json
    import re as _re

    schema_str = _json.dumps(schema_cls.model_json_schema(), ensure_ascii=False)
    instruction = (
        "输出要求：只输出一个 JSON 对象，禁止 markdown 代码块与任何额外文字。\n"
        f"JSON 必须符合以下 schema：\n{schema_str}"
    )
    msgs = list(messages) if isinstance(messages, (list, tuple)) else [("user", str(messages))]
    if msgs and msgs[0][0] == "system":
        msgs[0] = ("system", f"{msgs[0][1]}\n\n{instruction}")
    else:
        msgs.insert(0, ("system", instruction))

    out = llm.bind(response_format={"type": "json_object"}).invoke(msgs)
    text = out.content if isinstance(out.content, str) else str(out.content)
    text = _re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())  # 剥代码围栏
    return schema_cls.model_validate_json(text)


@lru_cache
def get_embeddings(settings: Settings | None = None):
    """创建 Embedding 模型。

    策略（EMBEDDING_PROVIDER）：
    - auto：有 Qwen Key -> 百炼 text-embedding-v3（在线，效果好）
            否则 -> 本地 BAAI/bge-small-zh-v1.5（离线，需安装 local-embedding 扩展）
    - dashscope / local：显式指定
    """
    s = settings or get_settings()
    provider = s.embedding_provider.lower()

    if provider == "auto":
        provider = "dashscope" if s.qwen_api_key else "local"

    if provider == "dashscope":
        if not s.qwen_api_key:
            raise RuntimeError("dashscope 嵌入需要 QWEN_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-v3",
            api_key=s.qwen_api_key,
            base_url=s.qwen_base_url,
            dimensions=1024,
        )

    if provider == "local":
        # 延迟导入：只有真选本地模式才需要 sentence-transformers
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "本地嵌入需要安装扩展依赖：uv sync --extra local-embedding"
            ) from e
        import os

        os.environ.setdefault("HF_ENDPOINT", s.hf_endpoint)
        os.environ.setdefault("HF_HOME", str(s.hf_cache_dir))  # 缓存放项目 data/（D 盘）
        logger.info("使用本地嵌入模型 BAAI/bge-small-zh-v1.5（HF_ENDPOINT=%s）", s.hf_endpoint)
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    raise ValueError(f"未知 EMBEDDING_PROVIDER: {s.embedding_provider}")
