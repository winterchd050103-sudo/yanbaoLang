"""巨潮资讯公告采集器（官方公开接口，校验源）。

用途：
- 下载年报/半年报等官方公告，为「研报预测 vs 年报实际」偏差校验提供权威数据
- cninfo 接口流程：topSearch 拿 orgId -> hisAnnouncement/query 拿公告列表
  -> static.cninfo.com.cn/{adjunctUrl} 下载 PDF

合规：巨潮是证监会指定信息披露网站，接口公开免费；仍执行限速与 UA 声明。
"""

from __future__ import annotations

import logging
import re
from datetime import date

import requests

from autoreport.config import get_settings
from autoreport.data_ingestion.crawlers.base import BaseCrawler

logger = logging.getLogger(__name__)

TOP_SEARCH = "http://www.cninfo.com.cn/new/information/topSearch/query"
ANN_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
ANN_PDF = "http://static.cninfo.com.cn/{}"

# 公告分类（巨潮 category 代码）
CATEGORY = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
}


class CninfoCrawler(BaseCrawler):
    source_name = "cninfo"

    def _get_org_id(self, stock_code: str) -> str:
        """topSearch 查询股票的 orgId（巨潮内部标识）。"""
        data = {
            "keyWord": stock_code,
            "maxNum": "10",
        }
        resp = self.session.post(TOP_SEARCH, data=data, timeout=self.timeout)
        resp.raise_for_status()
        rows = resp.json()
        for row in rows:
            code = (row.get("code") or "").split(".")[0]
            if code == stock_code:
                return row.get("orgId", "")
        raise ValueError(f"未找到 {stock_code} 的 orgId")

    def fetch_list(
        self,
        stock_code: str,
        ann_type: str = "年报",
        year: int | None = None,
        max_items: int = 10,
    ) -> list[dict]:
        """查询指定股票的公告列表（默认最新年报）。"""
        year = year or date.today().year - 1  # 默认上一年年报
        se_date = f"{year}-01-01~{year}-12-31"
        org_id = self._get_org_id(stock_code)
        # 沪市用 sse，深市/北交所用 szse
        column = "sse" if stock_code.startswith(("6", "9", "68")) else "szse"
        form = {
            "pageNum": "1",
            "pageSize": str(max_items),
            "column": column,
            "tabName": "fulltext",
            "plate": "",
            "stock": f"{stock_code},{org_id}",
            "searchkey": "",
            "secid": "",
            "category": CATEGORY.get(ann_type, CATEGORY["年报"]),
            "trade": "",
            "seDate": se_date,
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        resp = self.session.post(ANN_QUERY, data=form, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        items = []
        for ann in data.get("announcements") or []:
            title = re.sub(r"<[^>]+>", "", ann.get("announcementTitle", ""))
            adjunct = ann.get("adjunctUrl", "")
            if not adjunct:
                continue
            items.append(
                {
                    "url": ANN_PDF.format(adjunct),
                    "title": title,
                    "ann_type": ann_type,
                    "stock_code": stock_code,
                    "stock_name": ann.get("secName", ""),
                    "publish_date": (
                        ann.get("announcementTime", 0) / 1000
                        and date.fromtimestamp(ann["announcementTime"] / 1000).isoformat()
                    ),
                    "source_site": self.source_name,
                }
            )
        logger.info("巨潮公告查询完成: %s %s %d 共 %d 条", stock_code, ann_type, year, len(items))
        return items

    def download_item(self, item: dict, dest_dir) -> dict:
        filename = item["url"].rstrip("/").split("/")[-1]
        local = self.download_pdf(item["url"], dest_dir, filename=filename)
        out = dict(item)
        out["local_path"] = local
        return out


def crawl_announcements(stock_code: str, ann_type: str = "年报",
                        year: int | None = None, max_items: int = 3) -> list[dict]:
    """便捷入口：查询并下载公告到 data/announcements/，同时登记入库。"""
    import hashlib
    from pathlib import Path

    from autoreport.data_ingestion.storage import db

    settings = get_settings()
    crawler = CninfoCrawler()
    items = crawler.fetch_list(stock_code, ann_type, year, max_items=max_items)
    downloaded = []
    for item in items:
        try:
            out = crawler.download_item(item, settings.announcements_dir)
        except Exception as e:
            logger.warning("公告下载失败 %s: %s", item.get("title"), e)
            continue
        pdf_hash = hashlib.sha256(open(out["local_path"], "rb").read()).hexdigest()
        db.upsert_announcement(
            stock_code=out["stock_code"],
            stock_name=out["stock_name"],
            title=out["title"],
            ann_type=out["ann_type"],
            publish_date=out.get("publish_date"),
            source_site=out["source_site"],
            source_url=out["url"],
            local_path=out["local_path"],
            pdf_hash=pdf_hash,
            status="downloaded",
        )
        downloaded.append(out)
    return downloaded
