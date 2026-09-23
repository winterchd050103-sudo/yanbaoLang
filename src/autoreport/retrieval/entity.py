"""实体抽取：股票代码 / 股票名称 / 财务指标。

领域特化设计（见方案 6.1）：
A 股研报检索的关键不是纯语义相似，而是「实体匹配」——
先解析出股票代码/名称，用代码精确过滤（向量库 where 条件），再语义召回相关内容。
这一步把"找比亚迪的信息"从「向量碰运气」变成「先圈定范围再精搜」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A 股代码前缀 -> 交易所（用于过滤误匹配的 6 位数字，如年份、金额）
_VALID_PREFIXES = (
    "600", "601", "603", "605", "688", "689",  # 沪市主板/科创板
    "000", "001", "002", "003", "300", "301", "302",  # 深市主板/创业板
    "430", "83", "87", "88", "92",  # 北交所
)

_CODE_RE = re.compile(r"(?<!\d)([0-9]{6})(?!\d)")

# 常见财务/业务指标词（查询改写与重排时的加权信号）
INDICATOR_TERMS = [
    "营业收入", "营收", "净利润", "归母净利润", "扣非", "毛利率", "净利率",
    "EPS", "每股收益", "PE", "市盈率", "PB", "市净率", "ROE", "ROA",
    "现金流", "经营现金流", "负债率", "存货", "订单", "产能", "出货量",
    "市场份额", "市占率", "目标价", "评级", "盈利预测", "业绩", "增速",
    "同比", "环比", "分红", "回购", "估值", "景气度",
]

# 股票简称常用词（内置兜底词典；运行时会叠加库中已有股票）
BUILTIN_STOCK_NAMES = {
    "600519": "贵州茅台", "300750": "宁德时代", "002594": "比亚迪",
    "601318": "中国平安", "600036": "招商银行", "000333": "美的集团",
    "601012": "隆基绿能", "603259": "药明康德", "603730": "岱美股份",
    "000858": "五粮液", "600900": "长江电力", "601899": "紫金矿业",
    "000651": "格力电器", "002415": "海康威视", "600030": "中信证券",
    "601138": "工业富联", "688411": "海博思创", "920438": "戈碧迦",
    "920392": "佳合科技", "920271": "邦德股份",
}

_NAME_STOPWORDS = {"股份", "科技", "证券", "银行", "集团", "公司"}


@dataclass
class QueryEntities:
    """从用户查询中抽取的实体。"""

    stock_codes: list[str] = field(default_factory=list)
    stock_names: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)

    @property
    def has_stock(self) -> bool:
        return bool(self.stock_codes or self.stock_names)


def is_valid_code(code: str) -> bool:
    return any(code.startswith(p) for p in _VALID_PREFIXES)


def extract_stock_codes(text: str) -> list[str]:
    """抽取 6 位股票代码，前缀校验过滤年份/金额等误匹配。"""
    out = []
    for m in _CODE_RE.finditer(text):
        code = m.group(1)
        if is_valid_code(code) and code not in out:
            out.append(code)
    return out


def match_stock_names(text: str, extra_names: dict[str, str] | None = None) -> list[str]:
    """在查询文本中匹配股票简称（支持「茅台」这类简称命中「贵州茅台」）。"""
    code2name = dict(BUILTIN_STOCK_NAMES)
    if extra_names:
        code2name.update(extra_names)
    name2code = {v: k for k, v in code2name.items()}

    matched: list[str] = []
    for name in sorted(name2code, key=len, reverse=True):
        if name in text and name not in matched:
            matched.append(name)
            continue
        # 简称启发式：A 股口语简称常取末两字（贵州茅台->茅台、工业富联->富联），
        # 排除「股份/科技/证券」这类通用后缀，避免噪声命中
        core = name[-2:]
        if len(name) >= 3 and core not in _NAME_STOPWORDS and core in text and name not in matched:
            matched.append(name)
    return matched


def extract_entities(query: str, known_stocks: dict[str, str] | None = None) -> QueryEntities:
    """查询实体抽取主入口。"""
    ent = QueryEntities()
    ent.stock_codes = extract_stock_codes(query)
    ent.stock_names = match_stock_names(query, known_stocks)
    # 代码 -> 补齐名称；名称 -> 补齐代码（known_stocks 优先）
    code2name = dict(BUILTIN_STOCK_NAMES)
    if known_stocks:
        code2name.update(known_stocks)
    name2code = {v: k for k, v in code2name.items()}
    for code in ent.stock_codes:
        n = code2name.get(code)
        if n and n not in ent.stock_names:
            ent.stock_names.append(n)
    for name in ent.stock_names:
        c = name2code.get(name)
        if c and c not in ent.stock_codes:
            ent.stock_codes.append(c)
    ent.indicators = [t for t in INDICATOR_TERMS if t in query]
    return ent


def rewrite_query(query: str) -> str:
    """轻量查询改写：剔除纯代码数字后保留语义词（供关键词检索用）。

    例："比亚迪 002594 2026年盈利预测" -> "比亚迪 2026年盈利预测"
    代码本身用于精确过滤，不进关键词通道（避免 6 位数字噪声）。
    """
    cleaned = _CODE_RE.sub(" ", query)
    return re.sub(r"\s+", " ", cleaned).strip()
