"""PDF 文本抽取（PyMuPDF）。

设计要点（面试可讲）：
- 按页抽取并保留页码 —— 页码是「引用溯源到研报原文」的关键
- 用字体大小启发式抽取标题（研报 PDF 首页标题通常是最大字号）
- 扫描件检测：平均每页字符数过低判定为疑似扫描件，标记 ocr_needed，
  预留 OCR 接口但不阻塞主流程（降级策略）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 平均每页字符数低于该阈值 -> 疑似扫描件
SCANNED_THRESHOLD = 30


@dataclass
class ExtractedPdf:
    pages: list[str] = field(default_factory=list)  # 下标 0 = 第 1 页
    title_candidates: list[str] = field(default_factory=list)
    page_count: int = 0
    ocr_needed: bool = False


def extract_pdf(pdf_path: str | Path) -> ExtractedPdf:
    """抽取 PDF 全部文本页与标题候选。解析失败抛出异常，由上层决定状态。"""
    import pymupdf

    pdf_path = Path(pdf_path)
    out = ExtractedPdf()
    with pymupdf.open(pdf_path) as doc:
        out.page_count = doc.page_count
        for page in doc:
            out.pages.append(page.get_text("text") or "")

    out.title_candidates = _extract_title_candidates(pdf_path)
    total_chars = sum(len(p) for p in out.pages)
    avg = total_chars / max(out.page_count, 1)
    out.ocr_needed = avg < SCANNED_THRESHOLD
    if out.ocr_needed:
        logger.warning("%s 平均每页仅 %.0f 字符，疑似扫描件（OCR 预留接口，当前降级处理）",
                       pdf_path.name, avg)
    return out


def _extract_title_candidates(pdf_path: str | Path, max_n: int = 5) -> list[str]:
    """从首页按字体大小抽取标题候选：研报标题通常是首页最大的文字块。"""
    import pymupdf

    spans: list[tuple[float, str]] = []
    try:
        with pymupdf.open(pdf_path) as doc:
            page = doc[0]
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line.get("spans", [])).strip()
                    if len(text) < 6:  # 过滤页眉页脚碎片
                        continue
                    max_size = max((s["size"] for s in line.get("spans", [])), default=0)
                    spans.append((max_size, text))
    except Exception as e:  # pragma: no cover
        logger.warning("标题抽取失败 %s: %s", pdf_path, e)
        return []

    spans.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, text in spans[:30]:
        clean = text.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
        if len(out) >= max_n:
            break
    return out
