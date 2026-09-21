"""分块器单元测试：页级分块的基本性质。"""

from autoreport.data_ingestion.parsers.chunker import chunk_report


def test_chunker_covers_all_pages():
    pages = [f"第{i}页内容 " + "研报正文。" * 50 for i in range(1, 4)]
    chunks = chunk_report(pages)
    page_nos = {c["page_no"] for c in chunks}
    assert page_nos == {1, 2, 3}
    assert all(c["text"].strip() for c in chunks)
    assert all(c["chunk_index"] >= 0 for c in chunks)


def test_chunker_splits_long_page():
    long_text = "指标数据很多。" * 500  # 远超单块上限
    chunks = chunk_report([long_text])
    assert len(chunks) >= 2  # 长页被切成多块
    # 拼起来不丢字（含 overlap 会多，不会少）
    assert len("".join(c["text"] for c in chunks)) >= len(long_text)
