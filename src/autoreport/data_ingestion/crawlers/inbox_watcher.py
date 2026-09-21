"""inbox 目录轮询导入：受限源的合规兜底工作流。

人工从同花顺/慧博等站点下载 PDF 放入 data/inbox/，运行 `autoreport ingest`
即自动解析入库，成功后移入 inbox/processed/，失败留在原处便于排查。
"""

from __future__ import annotations

import logging

from autoreport.config import get_settings
from autoreport.data_ingestion.pipeline import ingest_pdf

logger = logging.getLogger(__name__)


def watch_once(use_llm_meta: bool = True) -> dict:
    """扫描 inbox 目录一次，导入全部 PDF（定时轮询可由系统 cron/任务计划驱动）。"""
    inbox = get_settings().inbox_dir

    pdfs = sorted(inbox.glob("*.pdf"))
    result = {"total": len(pdfs), "ok": 0, "skipped": 0, "failed": 0}
    for pdf in pdfs:
        try:
            r = ingest_pdf(pdf, source_site="inbox", use_llm_meta=use_llm_meta)
        except Exception as e:  # 极端异常：单文件失败不阻塞整批
            logger.exception("导入异常 %s: %s", pdf.name, e)
            result["failed"] += 1
            continue
        if r["skipped"]:
            result["skipped"] += 1
            pdf.unlink(missing_ok=True)  # 库中已有同 hash 的重复文件，直接清理
        elif r["status"] in ("indexed", "parsed"):
            # 成功：ingest 已把 PDF 移入 data/reports_raw/ 归档
            result["ok"] += 1
        else:
            result["failed"] += 1  # 失败文件留在 inbox 便于排查
    return result
