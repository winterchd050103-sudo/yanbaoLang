"""检索工具：把混合检索器包装成 Agent 可调用的 tool。

引用一致性设计：
- 工具工厂接收一个共享的 CitationManager，同一图运行中所有检索调用的
  引用编号在全局单调递增，写作 Agent 拿到的 [n] 与最终引用列表严格一致
- 工具返回「带引用角标的上下文文本」而非结构化数据 —— LLM 直接引用即可
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from autoreport.retrieval.citations import CitationManager


def make_search_reports_tool(cm: CitationManager):
    """创建绑定共享 CitationManager 的检索工具（每个图运行一份）。"""

    class SearchArgs(BaseModel):
        """检索入参 schema（Pydantic 让 LLM 的参数生成更稳定）。"""

        query: str = Field(description="检索查询词，包含股票名称/代码与关注点")
        top_k: int = Field(default=5, ge=1, le=10, description="返回条数")

    @tool(args_schema=SearchArgs)
    def search_reports(query: str, top_k: int = 5) -> str:
        """在本地研报知识库中检索（向量语义 + 关键词双通道融合）。

        返回带 [n] 引用角标的研报原文片段，写作时必须沿用这些编号。
        """
        from autoreport.retrieval import hybrid_retriever

        res = hybrid_retriever.retrieve(query, top_k=top_k)
        chunks = res["chunks"]
        if not chunks:
            return "未检索到相关研报内容。可尝试换关键词（如股票代码、财务指标名）再检索。"

        parts = []
        for c in chunks:
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
            parts.append(f"【{c.org or '未知机构'}·{c.publish_date}】{c.text} [{cite.idx}]")
        return "\n\n---\n\n".join(parts)

    return search_reports
