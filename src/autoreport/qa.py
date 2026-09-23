"""单轮研报问答（RAG）：检索 -> 带引用作答。

与多智能体链路的关系：
- ask 是「轻链路」：一次检索 + 一次生成，延迟低，适合高频事实型问题
- report 是「重链路」：主管拆解 + 多任务循环 + HITL，适合深度报告
- 两者共用同一套 hybrid_retriever 与 CitationManager —— 引用格式全系统一致
"""

from __future__ import annotations

import logging

from autoreport.config import get_settings
from autoreport.retrieval import hybrid_retriever
from autoreport.retrieval.citations import CitationManager

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """你是证券研究助理，根据检索到的研报片段回答用户问题。

规则：
1. 只依据提供的资料回答；资料不足以回答时明确说明
2. 每个事实性结论后面标注引用编号 [n]（沿用资料中的编号）
3. 回答简洁分点，中文

## 检索资料（每段末尾 [n] 为引用编号）
{context}

## 用户问题
{question}"""


def answer_question(question: str, top_k: int | None = None) -> dict:
    """单轮问答。返回 {answer, citations_md, chunks, mode}。

    mode: llm = LLM 生成；retrieval_only = 无 LLM 时的原文摘要降级
    """
    res = hybrid_retriever.retrieve(question, top_k=top_k)
    context = res.get("context", "")
    cm: CitationManager = res["citations"]
    chunks = res["chunks"]

    out = {
        "question": question,
        "chunks": chunks,
        "citations_md": cm.render(),
        "mode": "retrieval_only",
        "answer": "",
    }

    if not chunks:
        out["answer"] = "本地研报库中未找到相关内容。请先导入/采集研报，或换个关键词。"
        return out

    settings = get_settings()
    if not settings.llm_available:
        # 降级：拼接最相关的原文片段（带引用编号），保证无 Key 也可用
        snippets = []
        for i, c in enumerate(chunks, 1):
            cite = cm.cite(
                {
                    "report_id": c.report_id,
                    "page_no": c.page_no,
                    "org": c.org,
                    "title": c.title,
                    "publish_date": c.publish_date,
                    "source_url": c.source_url,
                    "local_path": c.local_path,
                }
            )
            snippets.append(f"【{c.org}·{c.publish_date}】{c.text[:300]}… [{cite.idx}]")
        out["answer"] = (
            "（LLM 未配置，返回最相关原文片段）\n\n" + "\n\n".join(snippets)
        )
        out["citations_md"] = cm.render()
        return out

    from autoreport.llm import get_chat_model

    llm = get_chat_model("main")
    answer = llm.invoke(ANSWER_PROMPT.format(context=context, question=question)).content
    out["answer"] = answer if isinstance(answer, str) else str(answer)
    out["mode"] = "llm"
    return out
