"""结构化分块：按页切分 + 段落感知的子块拆分，每块保留页码锚点。

设计要点（面试可讲）：
- 先按页边界切（研报页是天然语义单元，且页码必须保留用于引用）
- 单页过长再按段落（。；！？\n）切子块，带 overlap 防止语义被截断
- 分块大小对 RAG 质量影响极大：太大稀释向量语义，太小丢失上下文
"""

from __future__ import annotations

import re

# 单块目标长度 / 上限 / 相邻块重叠（字符）
CHUNK_TARGET = 500
CHUNK_MAX = 900
CHUNK_OVERLAP = 80

_PARA_SPLIT_RE = re.compile(r"(?<=[。！？；;])\s*")


def split_page_text(text: str, chunk_max: int = CHUNK_MAX,
                    overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把单页文本切成 <=chunk_max 的块：先按段落聚合，超长再硬切。"""
    text = re.sub(r"[ \t\u3000]+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_max:
        return [text]

    # 按句子边界聚合到接近 chunk_max
    sentences = [s for s in _PARA_SPLIT_RE.split(text) if s.strip()]
    blocks: list[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > chunk_max:
            blocks.append(buf)
            # overlap：携带上一块尾部，保证跨块语义连续
            buf = buf[-overlap:] if overlap < len(buf) else buf
            buf += sent
        else:
            buf += sent
    if buf.strip():
        blocks.append(buf)

    # 兜底：仍超长的块按字符硬切（表格流文本无句号的场景）
    final: list[str] = []
    for b in blocks:
        while len(b) > chunk_max * 1.3:
            final.append(b[:chunk_max])
            b = b[chunk_max - overlap:]
        final.append(b)
    return [b.strip() for b in final if b.strip()]


def chunk_report(pages: list[str], chunk_target: int = CHUNK_TARGET) -> list[dict]:
    """整篇研报分块。

    返回元素：{page_no(1-based), chunk_index(全文序号), text}
    """
    out: list[dict] = []
    idx = 0
    # 每页先独立成块；短页（<chunk_target*0.4）并入下一页，避免标题页碎片
    merged_pages: list[tuple[int, str]] = []
    carry = ""
    carry_page = 1
    for i, page_text in enumerate(pages, start=1):
        text = (carry + "\n" + page_text).strip() if carry else (page_text or "").strip()
        if len(text) < chunk_target * 0.4 and i < len(pages):
            if not carry:
                carry_page = i
            carry = text
            continue
        merged_pages.append((carry_page if carry else i, text))
        carry = ""
    if carry:
        merged_pages.append((carry_page, carry))

    for page_no, text in merged_pages:
        for piece in split_page_text(text):
            out.append({"page_no": page_no, "chunk_index": idx, "text": piece})
            idx += 1
    return out
