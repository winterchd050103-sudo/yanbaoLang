"""公告核对工具：「研报预测 vs 官方披露」交叉验证。

设计动机（见方案 6.3 防幻觉）：
- 研报是卖方观点，可能过于乐观；用官方公告（年报/季报）做第二信源交叉核对，
  是金融场景特有的「可信度增强」手段
- 官方数据是结构化强、位置可预期的文本（"营业收入 xxx,xxx,xxx,xxx 元"），
  正则抽取足够可靠；抽取失败时明确告诉 Agent 证据不足，而不是硬编一个数
"""

from __future__ import annotations

import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_NUM = r"([0-9][0-9,，]*(?:\.[0-9]+)?)"
# 「营业收入」等关键词后 240 字符内的第一个 带单位数字
_PATTERNS = [
    r"{kw}[^\d]{{0,40}}{_NUM}\s*(亿元|万元|元)",
    r"{kw}[^\d]{{0,240}}?{_NUM}\s*(亿元|万元|元)",
]


def _to_yi(value: float, unit: str) -> float:
    """统一换算成亿元。"""
    return value if unit == "亿元" else value / 1e4 if unit == "万元" else value / 1e8


def _parse_num(text: str) -> float | None:
    m = re.search(r"(-?[0-9][0-9,，]*(?:\.[0-9]+)?)", text.replace("，", ","))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_metric(doc_text: str, metric: str) -> tuple[float, str] | None:
    """从公告全文抽取指标数值（统一为亿元）。返回 (数值, 原文片段)。"""
    for tpl in _PATTERNS:
        pat = tpl.format(kw=re.escape(metric), _NUM=_NUM)
        m = re.search(pat, doc_text)
        if m:
            raw, unit = m.group(1).replace("，", "").replace(",", ""), m.group(2)
            try:
                return _to_yi(float(raw), unit), m.group(0)[:80]
            except ValueError:
                continue
    return None


def _verify_metric(doc_text: str, metric: str, predicted_yi: float) -> str:
    got = _extract_metric(doc_text, metric)
    if got is None:
        return (
            f"公告中未能定位「{metric}」的数值（可能是 PDF 版式特殊或指标口径不同），"
            "请勿声称已核对，如实说明证据不足。"
        )
    actual, snippet = got
    if predicted_yi <= 0:
        diff_txt = f"公告实际值 {actual:.2f} 亿元"
    else:
        diff = (actual - predicted_yi) / predicted_yi * 100
        verdict = "基本一致" if abs(diff) <= 5 else ("偏高" if diff > 0 else "偏低")
        diff_txt = f"公告实际值 {actual:.2f} 亿元，与预测偏差 {diff:+.1f}%（{verdict}）"
    return f"核对结果：{diff_txt}。公告原文片段：「{snippet}」"


@tool
def verify_with_announcement(
    stock_code: str, metric: str, predicted_value: str
) -> str:
    """用巨潮官方公告（年报/季报）核对研报中某项财务预测是否兑现。

    Args:
        stock_code: 6 位股票代码，如 600519
        metric: 指标名，如 营业收入 / 归属于上市公司股东的净利润
        predicted_value: 研报预测值文本，如 "128.5亿元" 或 "同比增长15%"
    """
    from pathlib import Path

    import pymupdf

    from autoreport.data_ingestion.storage import db

    anns = db.get_announcements(stock_code)
    if not anns:
        return (
            f"本地没有 {stock_code} 的公告文件。"
            "可提示用户运行: uv run autoreport crawl cninfo --code " + stock_code
        )
    ann = next((a for a in anns if a.local_path and Path(a.local_path).exists()), None)
    if ann is None:
        return f"{stock_code} 的公告记录存在，但本地 PDF 缺失。"

    try:
        with pymupdf.open(ann.local_path) as doc:
            doc_text = "\n".join(page.get_text() for page in doc)
    except Exception as e:  # noqa: BLE001
        return f"公告 PDF 读取失败: {e}"

    pred_num = _parse_num(predicted_value)
    # 预测值带单位则已按单位换算；"同比+15%" 这类无数值的预测跳过精确比对
    if pred_num is None:
        return (
            f"预测值「{predicted_value}」不含绝对数值，无法精确比对。"
            f"公告（{ann.title}）中「{metric}」相关内容需人工查看。"
        )
    unit = "亿元" if "亿" in predicted_value else "万元" if "万" in predicted_value else "元"
    pred_yi = _to_yi(pred_num, unit)
    header = f"核对 {ann.stock_name or stock_code}《{ann.title}》：\n"
    return header + _verify_metric(doc_text, metric, pred_yi)
