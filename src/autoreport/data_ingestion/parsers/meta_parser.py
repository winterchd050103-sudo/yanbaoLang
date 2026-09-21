"""研报元数据抽取：正则规则为主 + LLM 辅助兜底（两级策略，控制成本）。

设计要点（面试可讲）：
- 第一级：纯正则/启发式，零 token 成本、毫秒级、结果可解释
- 第二级：置信度不足时才调用 LLM（small 模型）补齐，成本可控
- 东财直链文件名 H3_AP{日期}{序列}_1.pdf 内嵌发布日期，优先从文件名解析
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 规则常量
# ---------------------------------------------------------------------------

RATING_KEYWORDS = [
    "买入", "增持", "推荐", "强烈推荐", "审慎推荐", "优于大市", "跑赢行业",
    "跑赢大市", "强于大市", "中性", "持有", "与大市同步", "减持", "卖出",
    "回避", "弱于大市",
]
_RATING_RE = re.compile(
    r"(?:投资评级|评级|投资建议|评级：|评级:)\s*[:：]?\s*(" + "|".join(RATING_KEYWORDS) + ")"
)
_RATING_LOOSE_RE = re.compile("(" + "|".join(RATING_KEYWORDS) + r")\s*(?:\(维持\)|（维持）)?")

_TARGET_PRICE_RE = re.compile(
    r"(?:目标价|目标价格|目标价：|目标价:)\s*[:：]?\s*(?:人民币)?\s*([0-9]+(?:\.[0-9]+)?)\s*元"
)

_STOCK_CODE_RE = re.compile(r"(?:^|\(|（|\s|:|：)([0-9]{6})(?:\.SZ|\.SH|\.BJ|\)|）|\s|$|，|,|。)")
_DATE_TEXT_RE = re.compile(r"(20\d{2})\s*[-年/\.]\s*(\d{1,2})\s*[-月/\.]\s*(\d{1,2})")

# 研报类型判别词（按优先级）
REPORT_TYPE_PATTERNS = [
    ("深度", ["深度报告", "深度研究", "首次覆盖", "投资价值分析"]),
    ("投价", ["投资价值", "询价", "新股定价"]),
    ("点评", ["点评报告", "事件点评", "业绩点评", "动态跟踪", "跟踪报告", "点评"]),
    ("宏观", ["宏观", "策略专题", "流动性"]),
    ("行业", ["行业研究", "行业深度", "行业动态", "行业专题", "行业"]),
]

# 主要券商名单（机构字段启发式匹配；东财采集时可从 API 直接取）
KNOWN_ORGS = [
    "华泰证券", "华泰研究", "中信证券", "中信建投", "国泰君安", "国泰海通",
    "海通证券", "国金证券", "东方财富证券", "东方证券", "申万宏源", "广发证券",
    "招商证券", "兴业证券", "安信证券", "国信证券", "光大证券", "方正证券",
    "中金公司", "中金", "天风证券", "开源证券", "民生证券", "浙商证券",
    "东吴证券", "长江证券", "平安证券", "华西证券", "华安证券", "国联证券",
    "太平洋证券", "西南证券", "东北证券", "银河证券", "中银证券", "瑞银",
    "摩根士丹利", "高盛", "花旗", "野村", "摩根大通",
]

# 常见股票备用词典（库中无该股时兜底）
BUILTIN_STOCKS = {
    "600519": "贵州茅台", "300750": "宁德时代", "002594": "比亚迪",
    "601318": "中国平安", "600036": "招商银行", "000333": "美的集团",
    "601012": "隆基绿能", "603259": "药明康德", "603730": "岱美股份",
    "000858": "五粮液", "600900": "长江电力", "601899": "紫金矿业",
    "000651": "格力电器", "002415": "海康威视", "600030": "中信证券",
    "601166": "兴业银行", "002714": "牧原股份", "600887": "伊利股份",
    "603288": "海天味业", "688981": "中芯国际", "688111": "金山办公",
}


@dataclass
class ReportMeta:
    """抽取出的研报元数据（各字段允许为空，置信度供 LLM 兜底决策）。"""

    title: str = ""
    org: str = ""
    report_type: str = ""
    stock_code: str = ""
    stock_name: str = ""
    rating: str = ""
    target_price: float | None = None
    publish_date: date | None = None
    # 哪些字段是低置信度（需要 LLM 补齐或人工复核）
    weak_fields: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "org": self.org,
            "report_type": self.report_type,
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "rating": self.rating,
            "target_price": self.target_price,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
        }


def parse_date_from_filename(filename: str) -> date | None:
    """东财直链命名：H3_AP202609201829672164_1.pdf -> 2026-09-20。"""
    m = re.search(r"AP(20\d{2})(\d{2})(\d{2})", filename)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def parse_meta(
    first_page_text: str,
    full_text_head: str = "",
    filename: str = "",
    title_candidates: list[str] | None = None,
) -> ReportMeta:
    """规则级元数据抽取。输入：首页文本、全文前 N 字、文件名、标题候选。"""
    meta = ReportMeta()
    head = (first_page_text or "")[:2000]
    corpus = head or full_text_head[:2000]

    # ---- 标题：字体最大候选优先，过滤掉明显不是标题的 ----
    for cand in title_candidates or []:
        if len(cand) >= 8 and not re.fullmatch(r"[\d\s\-/:.]+", cand):
            meta.title = cand
            break

    # ---- 发布日期：文件名 > 正文 ----
    meta.publish_date = parse_date_from_filename(filename)
    if meta.publish_date is None:
        m = _DATE_TEXT_RE.search(corpus)
        if m:
            try:
                meta.publish_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                meta.weak_fields.add("publish_date")
            except ValueError:
                pass
    else:
        meta.weak_fields.add("publish_date")  # 文件名日期是「上传日」，接近但非严格发布日

    # ---- 股票代码 ----
    m = _STOCK_CODE_RE.search(corpus)
    if m:
        meta.stock_code = m.group(1)
    else:
        meta.weak_fields.add("stock_code")

    # ---- 股票名称：代码附近或「XX（600519）」模式 ----
    if meta.stock_code:
        pat = re.compile(r"([\u4e00-\u9fa5]{2,8})\s*[\(（]\s*" + meta.stock_code)
        m = pat.search(corpus)
        if m:
            meta.stock_name = m.group(1)
        else:
            from autoreport.data_ingestion.storage import db as dbm

            name = BUILTIN_STOCKS.get(meta.stock_code) or dbm.known_stocks().get(meta.stock_code, "")
            meta.stock_name = name
    if not meta.stock_name:
        meta.weak_fields.add("stock_name")

    # ---- 评级 ----
    m = _RATING_RE.search(corpus)
    if m:
        meta.rating = m.group(1)
    else:
        # 宽松匹配：评级关键词首次出现（常见于首页评级框）
        m2 = _RATING_LOOSE_RE.search(corpus)
        if m2:
            meta.rating = m2.group(1)
            meta.weak_fields.add("rating")
    if not meta.rating:
        meta.weak_fields.add("rating")

    # ---- 目标价 ----
    m = _TARGET_PRICE_RE.search(corpus)
    if m:
        try:
            meta.target_price = float(m.group(1))
        except ValueError:
            meta.weak_fields.add("target_price")

    # ---- 研报类型：标题 + 正文判别 ----
    hay = f"{meta.title} {corpus[:600]}"
    for type_name, words in REPORT_TYPE_PATTERNS:
        if any(w in hay for w in words):
            meta.report_type = type_name
            break
    if not meta.report_type:
        meta.report_type = "点评"
        meta.weak_fields.add("report_type")

    # ---- 机构：在首页文本中找已知券商名 ----
    for org in KNOWN_ORGS:
        if org in corpus:
            meta.org = org
            break
    if not meta.org:
        meta.weak_fields.add("org")

    return meta


def refine_meta_with_llm(meta: ReportMeta, first_page_text: str) -> ReportMeta:
    """第二级：LLM 辅助抽取（仅当存在弱字段时调用；small 模型控制成本）。

    无 Key / 调用失败时原样返回 —— 元数据抽取永不阻塞主流程。
    """
    if not meta.weak_fields:
        return meta
    try:
        from pydantic import BaseModel, Field

        from autoreport.llm import get_chat_model

        class MetaOut(BaseModel):
            title: str = Field("", description="研报标题")
            org: str = Field("", description="券商/研究机构名称")
            stock_code: str = Field("", description="6位股票代码")
            stock_name: str = Field("", description="股票名称")
            rating: str = Field("", description="投资评级，如：买入/增持/中性/减持")
            report_type: str = Field("", description="类型：深度/点评/行业/宏观/投价")
            publish_date: str = Field("", description="发布日期 YYYY-MM-DD")

        llm = get_chat_model("small")
        prompt = (
            "从以下研报首页文本中抽取结构化元数据。只填有把握的字段，没把握的留空。\n"
            f"当前已抽取（供参考修正）：{meta.to_dict()}\n"
            f"弱字段：{sorted(meta.weak_fields)}\n"
            f"--- 研报首页文本 ---\n{first_page_text[:3000]}"
        )
        out = llm.with_structured_output(MetaOut).invoke(prompt)
        # LLM 结果只覆盖弱字段（强字段以规则结果为准，避免规则被模型覆盖）
        updates = out.model_dump()
        for key in list(meta.weak_fields):
            val = updates.get(key)
            if not val:
                continue
            if key == "target_price":
                meta.target_price = float(val)
            elif key == "publish_date":
                try:
                    meta.publish_date = date.fromisoformat(str(val)[:10])
                except ValueError:
                    continue
            else:
                setattr(meta, key, str(val).strip())
        meta.weak_fields.clear()
        logger.info("LLM 补齐元数据字段成功: %s", meta.to_dict())
    except Exception as e:
        logger.warning("LLM 元数据补齐失败（不影响入库）: %s", e)
    return meta
