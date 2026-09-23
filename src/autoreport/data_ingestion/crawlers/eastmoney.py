"""东方财富研报中心采集器（直链模式，方案 3.1/5.2 推荐首选落地源）。

原理：
- 列表接口 reportapi.eastmoney.com/report/list 返回 JSON，其中 infoCode 是关键
- PDF 直链格式稳定：https://pdf.dfcfw.com/pdf/H3_{infoCode}_1.pdf
  —— 用户 docs/ 目录的 5 份文件正是该格式（H3_AP{日期}{序列}_1.pdf）
- 元数据直接来自列表接口（orgSName/ratingNm/researchType），无需正文猜测，
  比正文正则抽取准确率高得多 —— 「API 优先于解析」是采集器的通用原则

合规：仅抓公开列表接口与公开 PDF 直链；限速 3s+；UA 声明个人学习用途。
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from autoreport.data_ingestion.crawlers.base import BaseCrawler
from autoreport.data_ingestion.parsers.meta_parser import parse_date_from_filename

logger = logging.getLogger(__name__)

LIST_API = "https://reportapi.eastmoney.com/report/list"
PDF_URL_TPL = "https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf"

# qType: 0 个股研报 / 1 行业研报 / 2 策略报告 / 3 宏观研究
QTYPE_STOCK = 0


class EastmoneyCrawler(BaseCrawler):
    source_name = "eastmoney"

    def fetch_list(
        self,
        stock_code: str = "",
        q_type: int = QTYPE_STOCK,
        begin: str | date = "2025-01-01",
        end: str | date | None = None,
        page_size: int = 20,
        max_pages: int = 3,
    ) -> list[dict]:
        """抓取研报列表。返回元素含 meta（列表接口直接给出的结构化元数据）。"""
        end = end or date.today()
        begin_s = begin.isoformat() if isinstance(begin, date) else str(begin)
        end_s = end.isoformat() if isinstance(end, date) else str(end)
        # 东财接口要求完整参数（缺省用 * 占位）且需 Referer，否则 400
        base_params = {
            "industryCode": "*",
            "industry": "*",
            "rating": "*",
            "ratingChange": "*",
            "fields": "",
            "orgCode": "",
            "rcode": "",
            "qType": q_type,
            "beginTime": begin_s,
            "endTime": end_s,
            "pageSize": page_size,
        }
        headers = {"Referer": "https://data.eastmoney.com/report/"}

        items: list[dict] = []
        for page in range(1, max_pages + 1):
            params = {**base_params, "pageNo": page}
            if stock_code:
                params["code"] = stock_code
            try:
                data = self.get_json(LIST_API, params=params, headers=headers)
            except Exception as e:
                logger.error("列表接口失败 page=%d: %s", page, e)
                break
            rows = data.get("data") or []
            if not rows:
                break
            for row in rows:
                info_code = row.get("infoCode", "")
                if not info_code:
                    continue
                publish = row.get("publishDate", "") or ""
                items.append(
                    {
                        "info_code": info_code,
                        "url": PDF_URL_TPL.format(info_code=info_code),
                        "title": (row.get("title") or "").strip(),
                        "org": (row.get("orgSName") or "").strip(),
                        "stock_code": (row.get("stockCode") or "").strip(),
                        "stock_name": (row.get("stockName") or "").strip(),
                        "rating": (
                            row.get("emRatingName") or row.get("sRatingName")
                            or row.get("ratingNm") or ""
                        ).strip(),
                        "report_type": (row.get("researchType") or "").strip(),
                        "publish_date": publish[:10] if publish else "",
                        "researcher": (row.get("researcher") or "").strip(),
                        "info_url": f"https://data.eastmoney.com/report/info/{info_code}.html",
                        "source_site": self.source_name,
                    }
                )
            if not data.get("hasMore"):
                break
        logger.info("东财列表抓取完成: code=%s 共 %d 条", stock_code or "(全部)", len(items))
        return items

    def download_item(self, item: dict, dest_dir) -> dict:
        """下载单份研报 PDF 到 dest_dir，返回带 local_path 的条目。"""
        filename = f"H3_{item['info_code']}_1.pdf"  # 与官方直链命名一致，便于溯源
        local = self.download_pdf(item["url"], dest_dir, filename=filename)
        out = dict(item)
        out["local_path"] = local
        if not out.get("publish_date"):
            out["publish_date"] = (
                parse_date_from_filename(filename).isoformat() if parse_date_from_filename(filename) else ""
            )
        return out
