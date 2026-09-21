"""入库流水线：PDF 文件 -> 解析 -> 元数据 -> 分块 -> 结构化库 + 向量库。

幂等设计（面试可讲）：
- 以 pdf_hash (sha256) 为唯一键，重复导入直接跳过
- 每步状态落库（pending/parsed/indexed/failed），中断后可恢复、可统计
- 向量化失败不回滚结构化入库（status=parsed，关键词检索仍可用）
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from autoreport.config import get_settings
from autoreport.data_ingestion.parsers import chunker, meta_parser, pdf_extract
from autoreport.data_ingestion.storage import db, vector_store

logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(
    pdf_path: str | Path,
    source_site: str = "inbox",
    source_url: str = "",
    use_llm_meta: bool = True,
    meta_override: dict | None = None,
) -> dict:
    """导入单个研报 PDF。返回 {report_id, status, skipped, reason}。

    meta_override: 采集器若已从列表 API 拿到结构化元数据（比正文猜测准），
                   在此覆盖 —— 「API 优先于解析」原则。
    """
    pdf_path = Path(pdf_path)
    settings = get_settings()
    pdf_hash = file_sha256(pdf_path)

    # ---- 幂等检查 ----
    exist = db.get_report_by_hash(pdf_hash)
    if exist is not None:
        return {"report_id": exist.id, "status": exist.status, "skipped": True,
                "reason": "pdf_hash 已存在"}

    # ---- 解析 ----
    try:
        extracted = pdf_extract.extract_pdf(pdf_path)
    except Exception as e:
        logger.error("PDF 解析失败 %s: %s", pdf_path.name, e)
        return {"report_id": "", "status": "failed", "skipped": False,
                "reason": f"PDF 解析失败: {e}"}

    head_text = "\n".join(extracted.pages[:3])
    first_page = extracted.pages[0] if extracted.pages else ""

    # ---- 元数据：规则 -> (可选) LLM 兜底 ----
    meta = meta_parser.parse_meta(
        first_page_text=first_page,
        full_text_head=head_text,
        filename=pdf_path.name,
        title_candidates=extracted.title_candidates,
    )
    if use_llm_meta and settings.llm_available:
        meta = meta_parser.refine_meta_with_llm(meta, first_page)
    # API 元数据覆盖正文抽取（更准：org/rating/日期来自结构化接口）
    from datetime import date as _date

    for key, val in (meta_override or {}).items():
        if not val:
            continue
        if key == "publish_date":
            try:
                meta.publish_date = _date.fromisoformat(str(val)[:10])
            except (ValueError, TypeError):
                pass
        elif key == "target_price":
            try:
                meta.target_price = float(val)
            except (ValueError, TypeError):
                pass
        elif key in ("title", "org", "stock_code", "stock_name", "rating", "report_type"):
            setattr(meta, key, str(val).strip())

    # ---- 结构化入库 ----
    # 原始 PDF 统一移动/复制到 data/reports_raw/，inbox 保持干净
    raw_dir = settings.reports_raw_dir
    dest = raw_dir / pdf_path.name
    if pdf_path.resolve() != dest.resolve():
        if dest.exists() and file_sha256(dest) != pdf_hash:
            dest = raw_dir / f"{pdf_hash[:8]}_{pdf_path.name}"
        if pdf_path.resolve().parent == settings.inbox_dir.resolve():
            shutil.move(str(pdf_path), dest)  # inbox 导入：移动
        else:
            shutil.copy2(str(pdf_path), dest)  # 其他来源：复制
    meta_dict = meta.to_dict()
    if meta.publish_date:
        meta_dict["publish_date"] = meta.publish_date  # date 对象入库（to_dict 里的 ISO 串用于展示）
    report = db.upsert_report(
        **meta_dict,
        pdf_hash=pdf_hash,
        source_site=source_site,
        source_url=source_url,
        local_path=str(dest),
        page_count=extracted.page_count,
        ocr_needed=extracted.ocr_needed,
        status="parsed",
    )

    # ---- 分块入库 + FTS ----
    chunks = chunker.chunk_report(extracted.pages)
    db.add_chunks(
        report.id,
        [
            {
                "text": c["text"],
                "page_no": c["page_no"],
                "chunk_index": c["chunk_index"],
                "meta": {"title": meta.title, "org": meta.org},
            }
            for c in chunks
        ],
    )

    # ---- 向量化（可选：无嵌入配置时降级为关键词检索） ----
    indexed = 0
    try:
        indexed = index_report(report.id)
        db.upsert_report(id=report.id, pdf_hash=pdf_hash, title=meta.title,
                         status="indexed")
    except Exception as e:
        logger.warning("向量化失败（保留关键词检索能力）: %s", e)

    logger.info("入库完成 %s：%s | 分块 %d | 向量 %d",
                pdf_path.name, meta.title[:40] or "(无标题)", len(chunks), indexed)
    return {"report_id": report.id, "status": "indexed" if indexed else "parsed",
            "skipped": False, "chunks": len(chunks), "indexed": indexed}


def index_report(report_id: str) -> int:
    """对已入库研报做向量化（ingest 自动调用；也可单独用于补索引）。"""
    from autoreport.llm import get_embeddings

    embeddings_model = get_embeddings()
    chunks = db.get_chunks(report_ids=[report_id])
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    vectors = embeddings_model.embed_documents(texts)
    items = [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "report_id": c["report_id"],
            "page_no": c["page_no"],
            "chunk_index": c["chunk_index"],
            "stock_code": c["stock_code"],
            "org": c["org"],
            "title": c["title"],
            "publish_date": c["publish_date"],
        }
        for c in chunks
    ]
    n = vector_store.upsert_chunks(items, vectors)
    return n


def ingest_directory(dir_path: str | Path, use_llm_meta: bool = True) -> dict:
    """批量导入目录下全部 PDF（inbox 导入主入口）。"""
    dir_path = Path(dir_path)
    pdfs = sorted(dir_path.glob("*.pdf"))
    result = {"total": len(pdfs), "ok": 0, "skipped": 0, "failed": 0, "details": []}
    for p in pdfs:
        r = ingest_pdf(p, source_site="inbox", use_llm_meta=use_llm_meta)
        if r["skipped"]:
            result["skipped"] += 1
        elif r["status"] in ("indexed", "parsed"):
            result["ok"] += 1
        else:
            result["failed"] += 1
        result["details"].append({"file": p.name, **r})
    return result
