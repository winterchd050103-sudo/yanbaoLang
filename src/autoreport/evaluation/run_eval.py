"""评测执行器：跑评测集 -> 逐条记录 -> 本地报告（JSONL+Markdown）+ 可选 LangSmith。

输出产物（data/eval_output/）：
- results_YYYYmmdd_HHMMSS.jsonl   逐条明细（复现/对比优化前后）
- report_YYYYmmdd_HHMMSS.md       指标汇总 + 失败案例分析（README 数字来源）
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from autoreport.config import PROJECT_ROOT, get_settings
from autoreport.evaluation.dataset import load_evalset, upload_to_langsmith
from autoreport.evaluation.metrics import (
    citation_accuracy,
    llm_judge,
    retrieval_hit,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "eval_output"


def run_eval(
    limit: int = 0,
    top_k: int = 5,
    evalset_path: str | None = None,
) -> dict:
    """跑评测。返回汇总 {path_md, path_jsonl, summary}。"""
    items = load_evalset(evalset_path)
    if limit:
        items = items[:limit]

    settings = get_settings()
    use_llm = settings.llm_available
    results: list[dict] = []

    for it in items:
        t0 = time.perf_counter()
        record: dict = {"id": it["id"], "question": it["question"]}

        # ---- 检索层 ----
        from autoreport.retrieval import hybrid_retriever

        res = hybrid_retriever.retrieve(it["question"], top_k=top_k, with_context=False)
        chunks = res["chunks"]
        record["hit"] = retrieval_hit(chunks, it.get("gold_report_ids", []), k=top_k)
        record["top_report_ids"] = [c.report_id for c in chunks[:top_k]]

        # ---- 问答层（LLM 可用时）----
        if use_llm:
            from autoreport.qa import answer_question

            qa = answer_question(it["question"], top_k=top_k)
            answer, cites_md = qa["answer"], qa["citations_md"]
            record["answer"] = answer
            record["citations_md"] = cites_md
            ca = citation_accuracy(answer, cites_md)
            record["citation_accuracy"] = ca
            record["mode"] = qa["mode"]
            if ca is not None:
                verdict = llm_judge(it["question"], answer, it.get("criteria", ""))
                if verdict:
                    record["judge"] = verdict
        record["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        results.append(record)
        logger.info("[%s] hit=%s lat=%dms", it["id"], record["hit"], record["latency_ms"])

    summary = _summarize(results, top_k)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / f"results_{ts}.jsonl"
    md_path = OUTPUT_DIR / f"report_{ts}.md"
    _write_jsonl(results, jsonl_path)
    _write_markdown(summary, results, md_path, top_k)

    # 可选：评测集上传 LangSmith（tracing 已由全局环境变量接管）
    ds_name = upload_to_langsmith(items)
    summary["langsmith_dataset"] = ds_name

    return {"path_md": str(md_path), "path_jsonl": str(jsonl_path), "summary": summary}


def _summarize(results: list[dict], top_k: int) -> dict:
    n = len(results) or 1
    hits = sum(1 for r in results if r["hit"])
    cas = [r["citation_accuracy"] for r in results if r.get("citation_accuracy") is not None]
    scores = [r["judge"]["score"] for r in results if "judge" in r]
    return {
        "total": len(results),
        "top_k": top_k,
        "retrieval_hit_rate": round(hits / n, 3),
        "citation_accuracy": round(sum(cas) / len(cas), 3) if cas else None,
        "judge_avg_score": round(sum(scores) / len(scores), 2) if scores else None,
        "judge_score_rate": round(  # >=3 分视为合格
            sum(1 for s in scores if s >= 3) / len(scores), 3
        ) if scores else None,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n),
    }


def _write_jsonl(results: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_markdown(summary: dict, results: list[dict], path: Path, top_k: int) -> None:
    lines = [
        "# AutoReport 评测报告",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 样本数：{summary['total']}（Top-{top_k}）",
        f"- 检索命中率@{top_k}：**{summary['retrieval_hit_rate']:.1%}**（目标 ≥ 85%）",
        f"- 引用正确率：**{summary['citation_accuracy']:.1%}**" if summary["citation_accuracy"] is not None else "- 引用正确率：n/a（LLM 未配置或无引用输出）",
        f"- judge 平均分：{summary['judge_avg_score']}" if summary["judge_avg_score"] else "- judge 平均分：n/a",
        f"- 合格率（judge≥3）：{summary['judge_score_rate']:.1%}" if summary["judge_score_rate"] else "- 合格率：n/a",
        f"- 平均延迟：{summary['avg_latency_ms']} ms/条",
        "",
        "## 失败案例（优化重点）",
        "",
    ]
    fails = [r for r in results if not r["hit"]]
    if not fails:
        lines.append("（无）")
    for r in fails[:20]:
        lines.append(
            f"- `{r['id']}` {r['question']} -> 命中：{r['top_report_ids'] or '无'}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
