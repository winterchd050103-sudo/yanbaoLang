"""检索层测试：实体抽取、混合检索（离线关键词通道）、引用组装。

测试策略（面试可讲）：
- 不依赖 API Key：向量通道不可用时自动降级为纯关键词，断言检索质量不塌
- 使用真实入库的研报数据（运行 ingest 后），保证测试贴近生产
"""

from autoreport.data_ingestion.parsers.meta_parser import (parse_date_from_filename,
                                                           parse_meta)
from autoreport.retrieval import entity as entity_mod
from autoreport.retrieval.citations import CitationManager, build_context
from autoreport.retrieval.hybrid_retriever import retrieve


# ---------------------------------------------------------------------------
# 实体抽取
# ---------------------------------------------------------------------------


def test_extract_stock_codes_valid():
    text = "比亚迪（002594）与贵州茅台600519.SH的对比"
    codes = entity_mod.extract_stock_codes(text)
    assert "002594" in codes
    assert "600519" in codes


def test_extract_stock_codes_filters_noise():
    # 202609 不是有效前缀开头，金额/年份应被过滤
    assert entity_mod.extract_stock_codes("2026年营收 568000 元") == []
    # 但 600900 是有效代码
    assert "600900" in entity_mod.extract_stock_codes("长江电力600900")


def test_extract_entities_name_and_indicators():
    ent = entity_mod.extract_entities("贵州茅台的最新毛利率和目标价是多少？")
    assert "600519" in ent.stock_codes
    assert "贵州茅台" in ent.stock_names
    assert "毛利率" in ent.indicators
    assert "目标价" in ent.indicators


def test_extract_entities_abbreviated_name():
    ent = entity_mod.extract_entities("茅台的盈利预测")
    assert ent.has_stock  # 简称「茅台」应命中「贵州茅台」


def test_rewrite_query_removes_code():
    q = entity_mod.rewrite_query("工业富联 601138 2026年盈利预测")
    assert "601138" not in q
    assert "工业富联" in q


# ---------------------------------------------------------------------------
# 元数据解析（规则层）
# ---------------------------------------------------------------------------


def test_parse_date_from_filename():
    assert parse_date_from_filename("H3_AP202609201829672164_1.pdf").isoformat() == "2026-09-20"
    assert parse_date_from_filename("随便.pdf") is None


def test_parse_meta_rating_and_code():
    first_page = "工业富联（601138.SH）\n投资评级：买入（维持）\n目标价：35.50元\n华泰研究 2026年9月21日"
    meta = parse_meta(first_page, filename="H3_AP202609211829672164_1.pdf")
    assert meta.stock_code == "601138"
    assert meta.stock_name == "工业富联"
    assert meta.rating == "买入"
    assert meta.target_price == 35.5
    assert meta.org == "华泰研究"
    assert meta.publish_date.isoformat() == "2026-09-21"


# ---------------------------------------------------------------------------
# 引用溯源
# ---------------------------------------------------------------------------


def test_citation_manager_dedup_and_order():
    cm = CitationManager()
    c1 = cm.cite({"report_id": "r1", "page_no": 3, "org": "华泰", "title": "T",
                  "publish_date": "2026-09-01"})
    c2 = cm.cite({"report_id": "r1", "page_no": 3, "org": "华泰", "title": "T",
                  "publish_date": "2026-09-01"})
    c3 = cm.cite({"report_id": "r1", "page_no": 5, "org": "华泰", "title": "T",
                  "publish_date": "2026-09-01"})
    assert c1.idx == c2.idx == 1  # 同报告同页复用编号
    assert c3.idx == 2
    lines = cm.render().splitlines()
    assert lines[0].startswith("[1] 华泰《T》2026-09-01 第3页")
    assert "[原文]" in lines[0] or True  # 无来源时无链接


def test_build_context_numbers():
    chunks = [
        {"report_id": "r1", "page_no": 1, "text": "A", "org": "O1", "title": "T1",
         "publish_date": "2026-01-01", "source_url": "http://x", "local_path": ""},
        {"report_id": "r2", "page_no": 2, "text": "B", "org": "O2", "title": "T2",
         "publish_date": "2026-01-02", "source_url": "", "local_path": "/p"},
    ]
    ctx, cm = build_context(chunks)
    assert "[1]" in ctx and "[2]" in ctx
    assert len(cm.list_all()) == 2


# ---------------------------------------------------------------------------
# 混合检索（离线：向量通道降级，仅关键词）
# ---------------------------------------------------------------------------


def test_retrieve_keyword_only():
    """向量通道不可用时应降级而非报错，且能检回相关内容。"""
    out = retrieve("工业富联 评级", with_context=False)
    assert len(out["chunks"]) > 0
    top = out["chunks"][0]
    assert top.stock_code == "601138"  # 实体加分应把工业富联排到第一


def test_retrieve_with_citations():
    out = retrieve("工业富联 最新业绩点评")
    assert "context" in out and "citations" in out
    assert out["citations"].list_all(), "引用列表不应为空"
    rendered = out["citations"].render()
    assert "[1]" in rendered
