"""评测集构建：从真实入库研报自动生成种子 QA（gold report_id 天然可靠）。

设计取舍：
- 为什么自动生成而不是手标：gold 答案=「该研报是否被命中」，事实型问题的
  标准答案可以从元数据（评级/目标价/机构）直接推导，标注成本为零且零噪音
- 每条记录 gold_report_ids 支持多个（同一问题可能多份研报都算命中）
- evalset.json 落盘后允许人工补充/修改 —— 自动生成是种子，人工精选是上限
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from autoreport.config import PROJECT_ROOT, get_settings
from autoreport.data_ingestion.storage import db

logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVALSET_PATH = EVAL_DIR / "evalset.json"


def _mk(idx: int, question: str, gold_ids: list[str], stock_code: str = "",
        keywords: list[str] | None = None, criteria: str = "") -> dict:
    return {
        "id": f"q{idx:03d}",
        "question": question,
        "gold_report_ids": gold_ids,
        "expected_stock_code": stock_code,
        "keywords": keywords or [],
        "criteria": criteria,  # LLM-as-judge 的评分要点（可空）
    }


def build_seed_dataset(force: bool = False) -> dict:
    """从入库研报生成种子评测集并落盘。返回 {"path", "count"}。"""
    reports = db.list_reports(limit=500)
    items: list[dict] = []
    n = 0
    for r in reports:
        if r.status not in ("parsed", "indexed") or not r.stock_code:
            continue
        name = r.stock_name or r.stock_code
        code = r.stock_code
        date_s = r.publish_date.isoformat() if r.publish_date else ""
        # 1) 事实问答：评级
        n += 1
        items.append(_mk(
            n, f"{name}（{code}）的最新评级是什么？",
            [r.id], code,
            keywords=[name, "评级"],
            criteria=f"回答应给出评级「{r.rating or '见研报'}」并引用 {r.org} 的研报",
        ))
        # 2) 事实问答：目标价（有目标价才出题）
        if r.target_price:
            n += 1
            items.append(_mk(
                n, f"{name}（{code}）研报给出的目标价是多少？",
                [r.id], code,
                keywords=[name, "目标价"],
                criteria=f"回答应包含目标价 {r.target_price} 元",
            ))
        # 3) 机构归属
        if r.org:
            n += 1
            items.append(_mk(
                n, f"哪家机构发布的{name}研报？发布日期是？",
                [r.id], code,
                keywords=[name, r.org],
                criteria=f"回答应包含机构 {r.org}" + (f" 与日期 {date_s}" if date_s else ""),
            ))

    # 4) 对比/综合类模板（多 gold，检索考察面更广）
    stocks = {r.stock_code: r.stock_name or r.stock_code for r in reports if r.stock_code}
    if len(stocks) >= 2:
        codes = list(stocks)[:2]
        n += 1
        items.append(_mk(
            n, f"对比 {stocks[codes[0]]} 和 {stocks[codes[1]]} 研报的投资观点差异",
            [reports[0].id, reports[1].id], "",
            keywords=[stocks[codes[0]], stocks[codes[1]], "对比"],
            criteria="回答应分别给出两只股票的观点并做对比",
        ))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if EVALSET_PATH.exists() and not force:
        logger.info("评测集已存在（%s），跳过构建；--force 可重建", EVALSET_PATH)
        return {"path": str(EVALSET_PATH), "count": -1}
    EVALSET_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("评测集已生成：%d 条 -> %s", len(items), EVALSET_PATH)
    return {"path": str(EVALSET_PATH), "count": len(items)}


def load_evalset(path: str | Path | None = None) -> list[dict]:
    """加载评测集；不存在时现场生成种子。"""
    p = Path(path) if path else EVALSET_PATH
    if not p.exists():
        build_seed_dataset(force=True)
    return json.loads(p.read_text(encoding="utf-8"))


def upload_to_langsmith(items: list[dict], dataset_name: str | None = None) -> str:
    """评测集上传 LangSmith（可选能力：无 Key/失败时返回空串不阻断）。"""
    settings = get_settings()
    if not settings.langsmith_enabled:
        return ""
    try:
        from langsmith import Client

        client = Client()
        name = dataset_name or settings.langsmith_project + "-evalset"
        ds = client.create_dataset(dataset_name=name)
        for it in items:
            client.create_example(
                dataset_id=ds.id,
                inputs={"question": it["question"]},
                outputs={
                    "gold_report_ids": it["gold_report_ids"],
                    "criteria": it.get("criteria", ""),
                },
            )
        return name
    except Exception as e:  # noqa: BLE001
        logger.warning("评测集上传 LangSmith 失败（不影响本地评测）: %s", e)
        return ""
