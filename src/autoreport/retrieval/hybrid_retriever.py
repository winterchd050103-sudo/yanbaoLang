"""混合检索器：向量语义 + 关键词精确 双通道 + RRF 融合 + 业务重排。

链路（见方案 6.1）：
    查询 -> 实体抽取（代码/名称/指标） -> [向量通道(代码 where 过滤) + 关键词通道(FTS)]
         -> RRF 融合 -> 业务重排（实体命中加分、日期新衰减） -> 带引用的上下文

为什么不只用向量：研报问答里"600519 最新评级"这类查询是实体+事实型，
纯语义检索会把语义相近但股票不对的内容排前面 —— 先用代码过滤、
关键词通道兜底精确匹配，才能保证命中率。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from autoreport.config import get_settings
from autoreport.data_ingestion.storage import db, vector_store
from autoreport.retrieval import entity as entity_mod
from autoreport.retrieval.citations import CitationManager, build_context

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 常数（经验值，平衡头部与尾部排名贡献）


@dataclass
class RetrievedChunk:
    """融合后的检索单元（含得分明细，便于评测与排查）。"""

    chunk_id: str
    report_id: str
    page_no: int
    text: str
    title: str = ""
    org: str = ""
    publish_date: str = ""
    stock_code: str = ""
    stock_name: str = ""
    source_url: str = ""
    local_path: str = ""
    score_vector: float = 0.0
    score_keyword: float = 0.0
    score_final: float = 0.0
    meta: dict = field(default_factory=dict)


def retrieve(query: str, top_k: int | None = None,
             with_context: bool = True) -> dict:
    """混合检索主入口。

    返回 {"entities", "chunks"(top), "context", "citations"}；
    with_context=False 时省略 context/citations（评测检索层时用）。
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k

    # ---- 1. 实体抽取与查询改写 ----
    ent = entity_mod.extract_entities(query, db.known_stocks())
    cleaned = entity_mod.rewrite_query(query)

    # ---- 2a. 向量通道（实体过滤 + 语义召回） ----
    vector_hits: list[tuple[str, float]] = []  # (chunk_id, rrf-ready rank score 占位)
    vec_by_id: dict[str, dict] = {}
    try:
        from autoreport.llm import get_embeddings

        embedder = get_embeddings()
        where = {"stock_code": {"$in": ent.stock_codes}} if ent.stock_codes else None
        hits = vector_store.query(embedder.embed_query(cleaned), top_k=top_k * 2, where=where)
        for rank, h in enumerate(hits):
            vector_hits.append((h["chunk_id"], rank))
            vec_by_id[h["chunk_id"]] = h
    except Exception as e:
        logger.warning("向量通道不可用，仅关键词检索: %s", e)

    # ---- 2b. 关键词通道（FTS5 / LIKE 降级） ----
    kw_hits = db.keyword_search(cleaned or query, top_k=top_k * 2)
    kw_rank = {cid: rank for rank, (cid, _) in enumerate(kw_hits)}
    kw_score = dict(kw_hits)

    # ---- 3. RRF 融合 ----
    all_ids = {cid for cid, _ in vector_hits} | set(kw_rank)
    details = {c["chunk_id"]: c for c in db.get_chunks(chunk_ids=list(all_ids))}
    merged: dict[str, float] = {}
    for cid in all_ids:
        s = 0.0
        if cid in kw_rank:
            s += 1.0 / (RRF_K + kw_rank[cid])
        if cid in {c for c, _ in vector_hits}:
            rank = next(r for c, r in vector_hits if c == cid)
            s += 1.0 / (RRF_K + rank)
        merged[cid] = s

    # ---- 4. 业务重排：实体命中加分 + 日期新衰减 ----
    now = date.today()
    candidates: list[RetrievedChunk] = []
    for cid, base in merged.items():
        info = details.get(cid)
        if info is None:
            continue
        rc = RetrievedChunk(
            chunk_id=cid,
            report_id=info["report_id"],
            page_no=info["page_no"],
            text=info["text"],
            title=info["title"],
            org=info["org"],
            publish_date=info["publish_date"],
            stock_code=info["stock_code"],
            stock_name=info["stock_name"],
            source_url=info["source_url"],
            local_path=info["local_path"],
            score_keyword=kw_score.get(cid, 0.0),
            meta=info["meta"],
        )
        if cid in vec_by_id:
            rc.score_vector = 1.0 / (1.0 + vec_by_id[cid]["distance"])

        boost = 0.0
        if ent.stock_codes and rc.stock_code in ent.stock_codes:
            boost += 0.35  # 实体精确命中是硬信号
        for term in ent.indicators:
            if term in rc.text:
                boost += 0.05
        if rc.publish_date:
            try:
                age_days = max((now - date.fromisoformat(rc.publish_date)).days, 0)
                boost += min(0.15, age_days / 3650)  # 越新越好，上限 0.15
            except ValueError:
                pass
        rc.score_final = base + boost
        candidates.append(rc)

    candidates.sort(key=lambda c: -c.score_final)
    final = candidates[: (top_k or settings.retrieval_top_k)]

    out = {"entities": ent, "chunks": final}
    if with_context:
        context, cm = build_context(
            [
                {
                    "report_id": c.report_id, "page_no": c.page_no, "text": c.text,
                    "org": c.org, "title": c.title, "publish_date": c.publish_date,
                    "source_url": c.source_url, "local_path": c.local_path,
                }
                for c in final
            ]
        )
        out["context"] = context
        out["citations"] = cm
    return out


def format_answer(question: str, answer: str, citations: CitationManager) -> str:
    """问答结果格式化：答案 + 引用列表。"""
    cites = citations.render()
    return f"**Q: {question}**\n\n{answer}\n\n**引用来源**\n{cites}"
