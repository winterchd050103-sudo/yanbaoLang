"""向量库封装（Chroma 本地持久化）。

设计要点：
- 直接用 chromadb PersistentClient + langchain Embeddings 计算向量，
  少一层包装依赖，接口清晰
- metadata 冗余存储 stock_code / org / title / publish_date，
  支持检索期「实体过滤」与「日期重排」，无需回表
- 按 report_id 删除重建，支持重新索引
"""

from __future__ import annotations

import logging

import chromadb

from autoreport.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION = "report_chunks"

_client = None


def get_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(get_settings().chroma_dir_abs))
    return _client


def get_collection() -> chromadb.Collection:
    return get_client().get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(items: list[dict], embeddings: list[list[float]]) -> int:
    """写入/更新分块向量。

    items 元素字段：chunk_id, text, report_id, page_no, chunk_index,
                    stock_code, org, title, publish_date
    """
    if len(items) != len(embeddings):
        raise ValueError("items 与 embeddings 数量不一致")
    if not items:
        return 0
    col = get_collection()
    col.upsert(
        ids=[it["chunk_id"] for it in items],
        documents=[it["text"] for it in items],
        embeddings=embeddings,
        metadatas=[
            {
                "report_id": it["report_id"],
                "page_no": it["page_no"],
                "chunk_index": it["chunk_index"],
                "stock_code": it.get("stock_code", "") or "",
                "org": (it.get("org", "") or "")[:100],
                "title": (it.get("title", "") or "")[:200],
                "publish_date": it.get("publish_date", "") or "",
            }
            for it in items
        ],
    )
    return len(items)


def query(embedding: list[float], top_k: int = 6,
          where: dict | None = None) -> list[dict]:
    """向量检索，返回 [{chunk_id, text, distance, metadata}]。"""
    col = get_collection()
    res = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, max(col.count(), 1)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(ids)):
        out.append(
            {
                "chunk_id": ids[i],
                "text": docs[i],
                "distance": dists[i],
                "metadata": metas[i],
            }
        )
    return out


def delete_by_report(report_id: str) -> None:
    try:
        get_collection().delete(where={"report_id": report_id})
    except Exception as e:  # pragma: no cover
        logger.warning("向量删除失败 report_id=%s: %s", report_id, e)


def count() -> int:
    return get_collection().count()
