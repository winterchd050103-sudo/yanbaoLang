"""采集调度器：任务编排 + 断点续采 + 状态登记。

流程（方案 5.2 采集流水线）：
    创建 crawl_job -> 列表抓取 -> 逐条下载（去重/限速/重试）
                 -> PDF 解析入库（复用 pipeline.ingest_pdf） -> 更新 job 状态

断点续采：job.items_done 记录进度，中断后按 info_code 去重天然幂等
（pdf_hash 已在 DB 的文件直接跳过），重跑即续采。
"""

from __future__ import annotations

import logging
import time
from datetime import date

from autoreport.config import get_settings
from autoreport.data_ingestion.crawlers.eastmoney import EastmoneyCrawler
from autoreport.data_ingestion.pipeline import ingest_pdf
from autoreport.data_ingestion.storage import db, vector_store

logger = logging.getLogger(__name__)


def crawl_eastmoney_reports(
    stock_code: str = "",
    begin: str | date = "2025-01-01",
    end: str | date | None = None,
    max_pages: int = 2,
    use_llm_meta: bool = False,  # 列表接口已给结构化元数据，默认无需 LLM
) -> dict:
    """东财研报采集 -> 入库 全流程。"""
    crawler = EastmoneyCrawler()
    job = db.upsert_crawl_job(
        source="eastmoney",
        target=f"code={stock_code or 'ALL'} begin={begin} pages={max_pages}",
        status="running",
    )
    result = {"job_id": job.id, "found": 0, "downloaded": 0, "ingested": 0,
              "skipped": 0, "failed": 0}

    try:
        items = crawler.fetch_list(stock_code=stock_code, begin=begin, end=end,
                                   max_pages=max_pages)
        result["found"] = len(items)
        db.upsert_crawl_job(id=job.id, items_found=len(items))

        settings = get_settings()
        for i, item in enumerate(items, 1):
            try:
                out = crawler.download_item(item, settings.reports_raw_dir)
                result["downloaded"] += 1
                meta_override = {
                    k: out.get(k)
                    for k in ("title", "org", "stock_code", "stock_name", "rating",
                              "report_type", "publish_date")
                }
                r = ingest_pdf(out["local_path"], source_site="eastmoney",
                               source_url=item["info_url"], use_llm_meta=use_llm_meta,
                               meta_override=meta_override)
                if r["skipped"]:
                    result["skipped"] += 1
                elif r["status"] in ("indexed", "parsed"):
                    result["ingested"] += 1
                else:
                    result["failed"] += 1
                db.upsert_crawl_job(id=job.id, items_done=i)
            except Exception as e:
                logger.error("条目处理失败 %s: %s", item.get("info_code"), e)
                result["failed"] += 1
            time.sleep(0)  # 限速已在 crawler 内部执行

        db.upsert_crawl_job(id=job.id, status="done", error="")
        logger.info("东财采集完成: %s", result)
    except Exception as e:
        db.upsert_crawl_job(id=job.id, status="failed", error=str(e))
        logger.exception("东财采集任务失败")
        raise
    return result


def reindex_all() -> dict:
    """全量重建向量索引（切换嵌入模型后使用）。"""
    from autoreport.data_ingestion.pipeline import index_report

    reports = db.list_reports(limit=10000)
    ok, fail = 0, 0
    for r in reports:
        try:
            index_report(r.id)
            ok += 1
        except Exception as e:
            logger.warning("索引失败 %s: %s", r.id, e)
            fail += 1
    return {"indexed": ok, "failed": fail, "vector_count": vector_store.count()}
