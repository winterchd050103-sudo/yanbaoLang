"""结构化存储层：SQLite（SQLAlchemy）+ FTS5 关键词索引。

表设计：
- reports        研报元数据（pdf_hash 去重是幂等采集的关键）
- chunks         文本分块（保留 report_id + page_no，页码是引用溯源的关键）
- chunk_fts      FTS5 全文索引（jieba 分词，解决中文分词问题）
- announcements  巨潮公告（年报等官方数据，用于「预测 vs 实际」校验）
- crawl_jobs     采集任务状态（断点续采、失败重试的依据）
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine, event, select)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            sessionmaker)

from autoreport.config import get_settings

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def gen_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Report(Base):
    """研报元数据表。"""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=gen_id)
    title: Mapped[str] = mapped_column(Text, default="")
    org: Mapped[str] = mapped_column(String(128), default="")  # 券商/机构
    report_type: Mapped[str] = mapped_column(String(32), default="")  # 深度/点评/行业/宏观
    stock_code: Mapped[str] = mapped_column(String(12), default="", index=True)
    stock_name: Mapped[str] = mapped_column(String(64), default="")
    rating: Mapped[str] = mapped_column(String(32), default="")
    target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    publish_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    source_site: Mapped[str] = mapped_column(String(32), default="inbox")
    source_url: Mapped[str] = mapped_column(Text, default="")
    local_path: Mapped[str] = mapped_column(Text, default="")
    pdf_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending/parsed/indexed/failed
    ocr_needed: Mapped[bool] = mapped_column(Boolean, default=False)
    license_note: Mapped[str] = mapped_column(
        Text, default="仅个人学习研究使用，版权归原机构所有，不二次分发、不商用"
    )
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Chunk(Base):
    """文本分块表：每块可追溯到 研报 + 页码。"""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=gen_id)
    report_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("reports.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    page_no: Mapped[int] = mapped_column(Integer, default=1)  # 1-based，引用溯源用
    text: Mapped[str] = mapped_column(Text)
    meta_json: Mapped[str] = mapped_column(Text, default="{}")


class Announcement(Base):
    """巨潮公告表（年报/半年报等官方文件）。"""

    __tablename__ = "announcements"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=gen_id)
    stock_code: Mapped[str] = mapped_column(String(12), default="", index=True)
    stock_name: Mapped[str] = mapped_column(String(64), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    ann_type: Mapped[str] = mapped_column(String(32), default="")  # 年报/半年报/一季报/三季报
    publish_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    source_site: Mapped[str] = mapped_column(String(32), default="cninfo")
    source_url: Mapped[str] = mapped_column(Text, default="")
    local_path: Mapped[str] = mapped_column(Text, default="")
    pdf_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class CrawlJob(Base):
    """采集任务表：断点续采与失败排查依据。"""

    __tablename__ = "crawl_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=gen_id)
    source: Mapped[str] = mapped_column(String(32), index=True)  # eastmoney/cninfo
    target: Mapped[str] = mapped_column(Text)  # 目标描述：股票代码/列表页参数
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending/running/done/failed
    items_found: Mapped[int] = mapped_column(Integer, default=0)
    items_done: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------------------
# 引擎与会话
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def get_engine():
    """全局单例引擎。FTS5 不可用时自动降级为 LIKE 检索（见 search 关键词部分）。"""
    global _engine, _SessionLocal
    if _engine is None:
        db_file = get_settings().db_file
        db_file.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(_engine)
        _init_fts(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    get_engine()
    return _SessionLocal()


def _raw_conn():
    return get_engine().raw_connection()


def _init_fts(engine) -> None:
    """创建 FTS5 虚拟表（jieba 空格分词方案）。失败则标记降级。"""
    global _FTS_AVAILABLE
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5("
                "chunk_id UNINDEXED, report_id UNINDEXED, tokens)"
            )
            conn.commit()
        _FTS_AVAILABLE = True
        logger.info("FTS5 关键词索引已启用")
    except sqlite3.Error as e:
        _FTS_AVAILABLE = False
        logger.warning("FTS5 不可用（%s），关键词检索降级为 LIKE", e)


_FTS_AVAILABLE = None


def fts_available() -> bool:
    global _FTS_AVAILABLE
    if _FTS_AVAILABLE is None:
        get_engine()
    return bool(_FTS_AVAILABLE)


# ---------------------------------------------------------------------------
# 数据访问（Repository 风格：上层只依赖这些函数）
# ---------------------------------------------------------------------------


def get_report_by_hash(pdf_hash: str) -> Report | None:
    with get_session() as s:
        return s.execute(select(Report).where(Report.pdf_hash == pdf_hash)).scalar_one_or_none()


def get_report(report_id: str) -> Report | None:
    with get_session() as s:
        return s.get(Report, report_id)


def list_reports(limit: int = 100) -> list[Report]:
    with get_session() as s:
        rows = s.execute(
            select(Report).order_by(Report.created_at.desc()).limit(limit)
        ).scalars().all()
        return list(rows)


def known_stocks() -> dict[str, str]:
    """从库中已知 股票代码->名称 映射（实体抽取的词典来源之一）。"""
    with get_session() as s:
        rows = s.execute(
            select(Report.stock_code, Report.stock_name).distinct().where(Report.stock_code != "")
        ).all()
        return {code: name for code, name in rows if code and name}


def upsert_report(**kwargs) -> Report:
    """按 pdf_hash 幂等插入/更新研报。"""
    with get_session() as s:
        rep = s.execute(select(Report).where(Report.pdf_hash == kwargs["pdf_hash"])).scalar_one_or_none()
        if rep is None:
            rep = Report(**kwargs)
            s.add(rep)
        else:
            for k, v in kwargs.items():
                setattr(rep, k, v)
        s.commit()
        s.refresh(rep)
        return rep


def delete_report(report_id: str) -> None:
    """删除研报及其分块（含 FTS 与向量，由调用方按需触发向量删除）。"""
    with get_session() as s:
        chunk_ids = s.execute(
            select(Chunk.id).where(Chunk.report_id == report_id)
        ).scalars().all()
        s.execute(select(Chunk).where(Chunk.report_id == report_id)).scalars().all()
        s.query(Chunk).filter(Chunk.report_id == report_id).delete()
        s.query(Report).filter(Report.id == report_id).delete()
        s.commit()
    if fts_available():
        try:
            conn = _raw_conn()
            conn.execute("DELETE FROM chunk_fts WHERE report_id = ?", (report_id,))
            conn.commit()
            conn.close()
        except Exception as e:  # pragma: no cover
            logger.warning("FTS 清理失败: %s", e)


def add_chunks(report_id: str, chunks: list[dict]) -> list[Chunk]:
    """批量写入分块并同步 FTS 索引。

    chunks 元素字段：text, page_no, chunk_index, meta(dict 可选)
    返回带数据库 id 的 Chunk 对象列表（顺序一致）。
    """
    with get_session() as s:
        rows = []
        for c in chunks:
            row = Chunk(
                report_id=report_id,
                chunk_index=c["chunk_index"],
                page_no=c["page_no"],
                text=c["text"],
                meta_json=json.dumps(c.get("meta", {}), ensure_ascii=False),
            )
            s.add(row)
            rows.append(row)
        s.commit()
        for r in rows:
            s.refresh(r)
        result = [(r.id, r.text) for r in rows]

    if fts_available():
        import jieba

        conn = _raw_conn()
        try:
            for cid, text in result:
                tokens = " ".join(t.strip() for t in jieba.cut_for_search(text) if t.strip())
                conn.execute(
                    "INSERT INTO chunk_fts (chunk_id, report_id, tokens) VALUES (?, ?, ?)",
                    (cid, report_id, tokens),
                )
            conn.commit()
        finally:
            conn.close()
    return rows


def get_chunks(report_ids: list[str] | None = None, chunk_ids: list[str] | None = None) -> list[dict]:
    """按条件取分块，附带研报元数据（引用溯源需要）。"""
    with get_session() as s:
        q = s.execute(
            select(Chunk, Report)
            .join(Report, Chunk.report_id == Report.id)
        ).all()
        out = []
        for chunk, rep in q:
            if report_ids and chunk.report_id not in report_ids:
                continue
            if chunk_ids and chunk.id not in chunk_ids:
                continue
            out.append(
                {
                    "chunk_id": chunk.id,
                    "report_id": chunk.report_id,
                    "page_no": chunk.page_no,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "meta": json.loads(chunk.meta_json or "{}"),
                    "title": rep.title,
                    "org": rep.org,
                    "publish_date": rep.publish_date.isoformat() if rep.publish_date else "",
                    "stock_code": rep.stock_code,
                    "stock_name": rep.stock_name,
                    "source_url": rep.source_url,
                    "local_path": rep.local_path,
                }
            )
        return out


def keyword_search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """FTS5 关键词检索（jieba 分词，OR 语义），返回 (chunk_id, score)。

    降级策略：FTS5 不可用时对 chunks.text 做 LIKE 匹配。
    score = 命中词数 * 词频权重，粗排足够（细排交给重排序层）。
    """
    import jieba

    terms = [t.strip() for t in jieba.cut_for_search(query) if len(t.strip()) >= 2][:8]
    if not terms:
        terms = [query.strip()] if query.strip() else []

    scores: dict[str, float] = {}
    if fts_available() and terms:
        match_expr = " OR ".join(f'"{t}"' for t in terms)
        conn = _raw_conn()
        try:
            rows = conn.execute(
                "SELECT chunk_id, tokens FROM chunk_fts WHERE chunk_fts MATCH ?",
                (match_expr,),
            ).fetchall()
        finally:
            conn.close()
        for chunk_id, tokens in rows:
            hit = sum(1 for t in terms if t in tokens)
            scores[chunk_id] = hit / len(terms)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    # LIKE 降级
    if not terms:
        return []
    with get_session() as s:
        rows = s.execute(select(Chunk.id, Chunk.text)).all()
    for cid, text in rows:
        hit = sum(1 for t in terms if t in text)
        if hit:
            scores[cid] = hit / len(terms)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def upsert_announcement(**kwargs) -> Announcement:
    with get_session() as s:
        ann = s.execute(
            select(Announcement).where(Announcement.pdf_hash == kwargs["pdf_hash"])
        ).scalar_one_or_none()
        if ann is None:
            ann = Announcement(**kwargs)
            s.add(ann)
        else:
            for k, v in kwargs.items():
                setattr(ann, k, v)
        s.commit()
        s.refresh(ann)
        return ann


def get_announcements(stock_code: str, ann_type: str | None = None) -> list[Announcement]:
    with get_session() as s:
        q = select(Announcement).where(Announcement.stock_code == stock_code)
        if ann_type:
            q = q.where(Announcement.ann_type == ann_type)
        q = q.order_by(Announcement.publish_date.desc().nullslast())
        return list(s.execute(q).scalars().all())


def upsert_crawl_job(**kwargs) -> CrawlJob:
    with get_session() as s:
        job = s.get(CrawlJob, kwargs.get("id", "")) if kwargs.get("id") else None
        if job is None:
            job = CrawlJob(**kwargs)
            s.add(job)
        else:
            for k, v in kwargs.items():
                setattr(job, k, v)
        s.commit()
        s.refresh(job)
        return job


def db_stats() -> dict:
    with get_session() as s:
        from sqlalchemy import func

        return {
            "reports": s.execute(select(func.count(Report.id))).scalar_one(),
            "indexed": s.execute(
                select(func.count(Report.id)).where(Report.status == "indexed")
            ).scalar_one(),
            "chunks": s.execute(select(func.count(Chunk.id))).scalar_one(),
            "announcements": s.execute(select(func.count(Announcement.id))).scalar_one(),
            "crawl_jobs": s.execute(select(func.count(CrawlJob.id))).scalar_one(),
        }
