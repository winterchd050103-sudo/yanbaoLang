"""指标定义：客观指标程序判定，主观指标 LLM-as-judge（可复核）。

三个指标的设计依据：
1. 检索命中率@K —— RAG 的上限由检索决定；gold=标准研报是否进 Top-K
2. 引用正确率 —— 防幻觉的机制化度量：回答中的 [n] 是否都存在于
   引用清单（程序判定，零成本、完全客观）
3. 回答准确率 —— LLM-as-judge 按 criteria 打 1-5 分（用 small 模型控制
   成本）；judge 有噪声，所以目标值定 90% 而不是 100%，且留人工抽检
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# 回答中允许出现的引用编号模式：[12]
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def extract_cited_indexes(answer: str) -> set[int]:
    """从回答中抽取引用编号集合（[原文](path) 链接不会被误匹配——非数字）。"""
    return {int(m) for m in _CITE_RE.findall(answer)}


def extract_legal_indexes(citations_md: str) -> set[int]:
    """引用清单中存在的编号（行首 [n] 开头）。"""
    return {int(m.group(1)) for line in citations_md.splitlines() if (m := re.match(r"\s*\[(\d+)\]", line))}


def citation_accuracy(answer: str, citations_md: str) -> float | None:
    """引用正确率（单条）：回答引用编号中合法编号占比。

    无任何引用时返回 None（不计入统计——比如"库里没有资料"类回答）。
    """
    cited = extract_cited_indexes(answer)
    if not cited:
        return None
    legal = extract_legal_indexes(citations_md)
    return len(cited & legal) / len(cited)


def retrieval_hit(chunks: list, gold_report_ids: list[str], k: int = 5) -> bool:
    """检索命中率@K：gold 研报是否出现在 Top-K 检索结果中。"""
    if not gold_report_ids:
        return False
    top_ids = {c.report_id for c in chunks[:k]}
    return bool(set(gold_report_ids) & top_ids)


class JudgeVerdict(BaseModel):
    """LLM-as-judge 结构化输出。"""

    score: int = Field(ge=1, le=5, description="1=完全错误 3=部分正确 5=准确完整")
    reason: str = Field(description="一句话评分理由")


_JUDGE_PROMPT = """你是严格的评测员，给研报问答结果打分（1-5 分）。

评分标准：
5 = 准确完整回答了问题，数字/结论与要点一致
3 = 部分正确，或遗漏关键信息但无错误
1 = 答非所问 / 与要点矛盾 / 编造内容

## 问题
{question}

## 参考要点（gold）
{criteria}

## 待评回答
{answer}"""


def llm_judge(question: str, answer: str, criteria: str) -> dict | None:
    """LLM 打分。无 LLM 配置时返回 None（评测降级为纯客观指标）。"""
    if not criteria.strip():
        return None
    try:
        from autoreport.llm import get_chat_model, structured_invoke

        out = structured_invoke(
            get_chat_model("small"), JudgeVerdict,
            _JUDGE_PROMPT.format(question=question, criteria=criteria, answer=answer),
        )
        return {"score": out.score, "reason": out.reason}
    except Exception as e:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("judge 失败: %s", e)
        return None
