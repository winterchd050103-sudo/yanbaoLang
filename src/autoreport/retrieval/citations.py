"""引用溯源：把检索块包装成可点击验证的引用列表。

防幻觉核心设计（见方案 6.2）：
- 每条引用 = 机构 + 标题 + 发布日期 + 页码 + 可回跳链接（原始 URL 或本地 PDF 路径）
- 引用编号在上下文组装时确定，写作 Agent 只允许使用已有编号 —— 从机制上
  杜绝"编造来源"：编一个不存在的 [5]，在引用列表里一眼就能看出
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Citation:
    """一条引用：对应 研报 + 页码 粒度。"""

    idx: int  # 编号 [1] [2] ...
    report_id: str
    org: str
    title: str
    publish_date: str
    page_no: int
    source_url: str = ""
    local_path: str = ""

    def line(self) -> str:
        """Markdown 引用条目。"""
        src = self.source_url or self.local_path
        link = f"[原文]({src})" if src else ""
        return (
            f"[{self.idx}] {self.org or '未知机构'}《{self.title or '未知标题'}》"
            f"{self.publish_date} 第{self.page_no}页 {link}".strip()
        )


class CitationManager:
    """管理 检索块 -> 引用编号 的分配（同一 研报+页码 复用同一编号）。

    start_from: 起始编号。多智能体场景下多个子任务各自持有 manager，
    通过递增 start_from 保证全局引用编号不冲突（无 Key/序列化友好的简单方案）。
    """

    def __init__(self, start_from: int = 1) -> None:
        self._by_key: dict[tuple[str, int], Citation] = {}
        self._next = start_from

    def cite(self, chunk: dict) -> Citation:
        key = (chunk["report_id"], chunk["page_no"])
        if key not in self._by_key:
            self._by_key[key] = Citation(
                idx=self._next,
                report_id=chunk["report_id"],
                org=chunk.get("org", ""),
                title=chunk.get("title", ""),
                publish_date=chunk.get("publish_date", ""),
                page_no=chunk["page_no"],
                source_url=chunk.get("source_url", ""),
                local_path=chunk.get("local_path", ""),
            )
            self._next += 1
        return self._by_key[key]

    def list_all(self) -> list[Citation]:
        return sorted(self._by_key.values(), key=lambda c: c.idx)

    def render(self) -> str:
        """渲染引用列表（Markdown）。"""
        return "\n".join(c.line() for c in self.list_all())


def build_context(chunks: list[dict]) -> tuple[str, CitationManager]:
    """把检索块组装成带引用角标的上下文，供写作/问答使用。

    返回 (context_text, citation_manager)。
    context 中每块末尾带 [n]，模型被要求沿用这些编号作答。
    """
    cm = CitationManager()
    parts = []
    for ch in chunks:
        cite = cm.cite(ch)
        parts.append(f"【{ch.get('org', '')}·{ch.get('publish_date', '')}】{ch['text']} [{cite.idx}]")
    return "\n\n---\n\n".join(parts), cm
