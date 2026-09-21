"""AutoReport 命令行入口（typer）。

子命令随开发阶段逐步启用：
- ingest   研报导入（inbox 目录 / 指定目录 / 单文件）
- stats    数据库统计
- index    为未向量化的研报补建向量索引
- crawl    采集器（东财研报 / 巨潮公告）
- ask      带引用的研报问答
- report   多智能体生成完整研报
- eval     评测闭环
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from autoreport.config import apply_langsmith_env, get_settings

app = typer.Typer(help="AutoReport 多智能体自动研报系统", no_args_is_help=True,
                  add_completion=False)
console = Console()


def _setup_logging() -> None:
    settings = get_settings()
    apply_langsmith_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def ingest(
    path: str = typer.Option("", help="要导入的目录或 PDF 文件；留空则扫描 data/inbox/"),
    no_llm_meta: bool = typer.Option(False, "--no-llm-meta", help="禁用 LLM 辅助元数据抽取"),
) -> None:
    """导入研报 PDF：解析 -> 元数据抽取 -> 分块 -> 入库 -> 向量化。"""
    _setup_logging()
    from autoreport.data_ingestion.crawlers.inbox_watcher import watch_once
    from autoreport.data_ingestion.pipeline import ingest_directory, ingest_pdf
    from pathlib import Path as _P

    if path:
        p = _P(path)
        if p.is_dir():
            result = ingest_directory(p, use_llm_meta=not no_llm_meta)
        elif p.suffix.lower() == ".pdf":
            from autoreport.data_ingestion.pipeline import ingest_pdf as _ing

            r = _ing(p, source_site="manual", use_llm_meta=not no_llm_meta)
            result = {"total": 1, "ok": 0 if r["status"] == "failed" else 1,
                      "skipped": 1 if r["skipped"] else 0,
                      "failed": 1 if r["status"] == "failed" else 0}
        else:
            console.print(f"[red]路径不存在或不是 PDF：{path}[/red]")
            raise typer.Exit(1)
    else:
        result = watch_once(use_llm_meta=not no_llm_meta)

    console.print(
        f"[green]导入完成[/green] 共 {result['total']}："
        f"成功 {result['ok']}，跳过 {result['skipped']}，失败 {result['failed']}"
    )


@app.command()
def crawl(
    source: str = typer.Argument("eastmoney", help="采集源: eastmoney / cninfo"),
    code: str = typer.Option("", "--code", help="股票代码（留空抓取全市场）"),
    begin: str = typer.Option("2026-01-01", "--begin", help="开始日期 YYYY-MM-DD"),
    pages: int = typer.Option(2, "--pages", help="最多抓取列表页数"),
    year: int = typer.Option(0, "--year", help="cninfo 公告年份（默认上一年）"),
    ann_type: str = typer.Option("年报", "--ann-type", help="公告类型: 年报/半年报/一季报/三季报"),
) -> None:
    """运行采集器（东财研报直链 / 巨潮公告），下载后自动解析入库。"""
    _setup_logging()
    if source == "eastmoney":
        from autoreport.data_ingestion.crawlers.scheduler import crawl_eastmoney_reports

        result = crawl_eastmoney_reports(stock_code=code, begin=begin, max_pages=pages)
        console.print(
            f"[green]东财采集完成[/green] 发现 {result['found']}，"
            f"下载 {result['downloaded']}，入库 {result['ingested']}，"
            f"跳过 {result['skipped']}，失败 {result['failed']}"
        )
    elif source == "cninfo":
        from autoreport.data_ingestion.crawlers.cninfo import crawl_announcements

        out = crawl_announcements(code, ann_type=ann_type, year=year or None)
        console.print(f"[green]巨潮公告下载完成[/green] 共 {len(out)} 份")
        for it in out:
            console.print(f"  - {it['title']} -> {it['local_path']}")
    else:
        console.print(f"[red]未知采集源: {source}（支持 eastmoney / cninfo）[/red]")
        raise typer.Exit(1)


@app.command()
def stats() -> None:
    """查看数据库统计。"""
    _setup_logging()
    from autoreport.data_ingestion.storage import db, vector_store

    s = db.db_stats()
    table = Table(title="AutoReport 数据统计")
    table.add_column("指标", style="cyan")
    table.add_column("数量", justify="right")
    table.add_row("研报总数", str(s["reports"]))
    table.add_row("已向量化", str(s["indexed"]))
    table.add_row("文本分块", str(s["chunks"]))
    table.add_row("向量库分块", str(vector_store.count()))
    table.add_row("公告数", str(s["announcements"]))
    table.add_row("采集任务", str(s["crawl_jobs"]))
    console.print(table)

    reports = db.list_reports(limit=20)
    if reports:
        rt = Table(title="最近研报")
        rt.add_column("股票")
        rt.add_column("名称")
        rt.add_column("机构")
        rt.add_column("评级")
        rt.add_column("日期")
        rt.add_column("状态")
        rt.add_column("标题(截断)")
        for r in reports:
            rt.add_row(
                r.stock_code, r.stock_name, r.org, r.rating,
                r.publish_date.isoformat() if r.publish_date else "-",
                r.status, (r.title or "")[:36],
            )
        console.print(rt)


@app.command()
def index() -> None:
    """为「已解析但未向量化」的研报补建向量索引（需嵌入模型可用）。"""
    _setup_logging()
    from autoreport.data_ingestion.pipeline import index_report
    from autoreport.data_ingestion.storage import db

    reports = db.list_reports(limit=1000)
    todo = [r for r in reports if r.status == "parsed"]
    if not todo:
        console.print("没有待索引的研报")
        return
    for r in todo:
        n = index_report(r.id)
        console.print(f"[green]已索引[/green] {r.title[:40] or r.id}（{n} 块）")
    console.print("索引补建完成")


@app.command()
def ask(
    question: str = typer.Argument(..., help="研报相关问题"),
    top_k: int = typer.Option(6, "--top-k", help="检索条数"),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="附带展示命中片段"),
) -> None:
    """单轮研报问答（RAG，带引用溯源）。"""
    _setup_logging()
    from autoreport.qa import answer_question

    result = answer_question(question, top_k=top_k)
    md = f"**Q: {question}**\n\n{result['answer']}\n\n**引用来源**\n{result['citations_md']}"
    console.print(md)
    if show_chunks:
        t = Table(title="命中片段")
        t.add_column("报告", overflow="fold")
        t.add_column("页", justify="right")
        t.add_column("得分", justify="right")
        t.add_column("片段", overflow="fold")
        for c in result["chunks"]:
            t.add_row(
                f"{c.stock_name or c.stock_code} {c.org}",
                str(c.page_no),
                f"{c.score_final:.3f}",
                c.text[:80].replace("\n", " "),
            )
        console.print(t)


@app.command()
def report(
    question: str = typer.Argument(..., help="研报主题，如「生成 600519 的投资价值分析报告」"),
    auto: bool = typer.Option(False, "--auto", help="跳过人工计划确认（HITL 关闭）"),
) -> None:
    """多智能体生成完整研报（LangGraph 工作流）。"""
    import json
    from datetime import datetime

    _setup_logging()
    settings = get_settings()
    if auto:
        settings.hitl_enabled = False  # 本进程内生效，便于脚本化调用

    from autoreport.agents import build_graph, get_checkpointer
    from autoreport.agents.state import initial_state

    graph = build_graph(checkpointer=get_checkpointer())
    thread_id = f"cli-{datetime.now():%Y%m%d-%H%M%S}"
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]thread_id={thread_id}[/dim]")
    final_state: dict = {}
    for update in graph.stream(initial_state(question), config, stream_mode="updates"):
        for node, delta in update.items():
            if node == "__interrupt__":
                intr = delta[0]
                payload = intr.value if hasattr(intr, "value") else intr
                console.print("[yellow]图已在计划确认处挂起（HITL）[/yellow]")
                console.print(json.dumps(payload, ensure_ascii=False, indent=2))
                console.print(
                    "[yellow]CLI 下请改用 --auto 跳过确认，"
                    "或在 Web UI 中编辑计划后恢复[/yellow]"
                )
                raise typer.Exit(0)
            console.print(f"[cyan]-> {node}[/cyan]")
            final_state.update(delta or {})

    if final_state.get("error"):
        console.print(f"[red]失败：{final_state['error']}[/red]")
        raise typer.Exit(1)
    console.rule("[bold green]研报生成完成")
    console.print(final_state.get("final_report", "(空)"))


@app.command("eval")
def eval_cmd(
    build: bool = typer.Option(False, "--build", help="（重新）生成种子评测集"),
    limit: int = typer.Option(0, "--limit", help="只跑前 N 条（0=全部）"),
    top_k: int = typer.Option(5, "--top-k", help="检索命中判定的 K"),
) -> None:
    """评测闭环：检索命中率 / 引用正确率 / LLM-as-judge（输出 Markdown 报告）。"""
    _setup_logging()
    from autoreport.evaluation import build_seed_dataset, run_eval

    if build:
        r = build_seed_dataset(force=True)
        if r["count"] >= 0:
            console.print(f"[green]评测集已生成[/green] {r['count']} 条 -> {r['path']}")

    out = run_eval(limit=limit, top_k=top_k)
    s = out["summary"]
    cite_s = "n/a" if s["citation_accuracy"] is None else f"{s['citation_accuracy']:.1%}"
    console.print(
        f"[green]评测完成[/green] 样本 {s['total']}，"
        f"命中率@{s['top_k']} {s['retrieval_hit_rate']:.1%}，"
        f"引用正确率 {cite_s}，平均延迟 {s['avg_latency_ms']}ms"
    )
    console.print(f"报告: {out['path_md']}")


if __name__ == "__main__":
    app()
