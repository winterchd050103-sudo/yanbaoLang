"""多智能体层测试：工具安全性、指标正确性、图级降级与 HITL 流程。

全部不依赖真实 LLM Key —— 降级路径本身就是被测对象（graceful degradation）。
"""

import sqlite3

import pytest

from autoreport.agents.tools import safe_calc, web_search
from autoreport.evaluation.metrics import (
    citation_accuracy,
    extract_cited_indexes,
    retrieval_hit,
)
from autoreport.retrieval.citations import CitationManager


# ---------------- safe_calc：LLM 算术外置 + AST 安全边界 ----------------


def test_calc_basic_arithmetic():
    out = safe_calc.invoke({"expression": "(128.5-102.3)/102.3*100"})
    assert "25.6109" in out


def test_calc_rejects_call_and_import():
    assert "失败" in safe_calc.invoke({"expression": "__import__('os').system('1')"})
    assert "失败" in safe_calc.invoke({"expression": "len('abc')"})


def test_calc_rejects_division_by_zero():
    assert "失败" in safe_calc.invoke({"expression": "1/0"})


# ---------------- web_search：无 Key 优雅降级（工具永不抛异常） ----------------


def test_web_search_no_key_degrades(monkeypatch):
    monkeypatch.setattr("autoreport.config.Settings.tavily_api_key", "", raising=False)
    out = web_search.invoke({"query": "任何查询"})
    assert "未配置" in out


# ---------------- 评测指标 ----------------


def test_extract_cited_indexes():
    assert extract_cited_indexes("结论A [1] 结论B [12] 链接 [原文](x.pdf)") == {1, 12}


def test_citation_accuracy_flags_fake_citation():
    legal = "[1] 机构《标题》2026-01-01 第1页"
    assert citation_accuracy("结论 [1] [2]", legal) == 0.5  # [2] 是编造的
    assert citation_accuracy("无引用的回答", legal) is None  # 无引用不计入


def test_retrieval_hit_at_k():
    chunks = [type("C", (), {"report_id": f"r{i}"})() for i in range(8)]
    assert retrieval_hit(chunks, ["r3"], k=5) is True
    assert retrieval_hit(chunks, ["r7"], k=5) is False  # gold 只在 Top-5 之外
    assert retrieval_hit(chunks, [], k=5) is False


# ---------------- writer：无 LLM 降级（检索-引用能力仍可用） ----------------


def test_writer_fallback_without_llm(monkeypatch):
    import autoreport.agents.writer as writer

    monkeypatch.setattr(
        "autoreport.config.Settings.llm_available",
        property(lambda self: False),
        raising=False,
    )
    state = {
        "question": "测试问题",
        "findings": ["### 子任务：A\n结论 [1]"],
        "citations": [
            {
                "idx": 1,
                "report_id": "r1",
                "org": "测试机构",
                "title": "测试研报",
                "publish_date": "2026-08-17",
                "page_no": 2,
                "source_url": "",
                "local_path": "x.pdf",
            }
        ],
        "error": "",
    }
    out = writer.writer_node(state)
    assert "原始调研记录汇编" in out["final_report"]
    assert "测试机构" in out["final_report"]
    assert out["error"] == ""


# ---------------- 图级：无 LLM 降级路径 + HITL interrupt/resume ----------------


@pytest.fixture()
def graph():
    from autoreport.agents import build_graph
    from autoreport.agents.graph import make_sqlite_saver

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = make_sqlite_saver(conn)
    yield build_graph(checkpointer=saver)
    conn.close()


@pytest.fixture()
def no_hitl(monkeypatch):
    from autoreport.config import get_settings

    get_settings().hitl_enabled = False
    yield
    get_settings().hitl_enabled = True


def test_graph_without_llm_reaches_clean_error(graph, no_hitl):
    """无 Key：默认计划 -> 研究员报清晰错误 -> 图正常终止（不炸）。"""
    from autoreport.agents.state import initial_state

    final = graph.invoke(initial_state("贵州茅台"), {"configurable": {"thread_id": "t1"}})
    assert "LLM 未配置" in final["error"] or "无法启动" in final["error"]
    assert final["final_report"] == ""


def test_graph_hitl_cancel(graph):
    """HITL：interrupt 挂起 -> resume=False -> 取消路径。"""
    from autoreport.config import get_settings

    get_settings().hitl_enabled = True
    from autoreport.agents.state import initial_state

    result = graph.invoke(initial_state("测试"), {"configurable": {"thread_id": "t2"}})
    intrs = result.get("__interrupt__", [])
    assert len(intrs) == 1
    assert intrs[0].value["type"] == "plan_confirm"
    assert intrs[0].value["plan"], "挂起时应携带计划"
    final = graph.invoke(
        __import__("langgraph.types", fromlist=["Command"]).Command(resume=False),
        {"configurable": {"thread_id": "t2"}},
    )
    assert "取消" in final["error"]


def test_graph_hitl_edit_plan(graph):
    """HITL：人工改写计划后 resume -> 按新计划执行（无 LLM 时走 error 路径）。"""
    from langgraph.types import Command

    from autoreport.agents.state import initial_state

    result = graph.invoke(initial_state("测试"), {"configurable": {"thread_id": "t3"}})
    assert result.get("__interrupt__")
    edited = [{"type": "calc", "desc": "计算 1+1"}]
    final = graph.invoke(
        Command(resume=edited), {"configurable": {"thread_id": "t3"}}
    )
    # 无 LLM：tool_agent 报错终止 —— 证明新计划已被采用（走到 tool_agent 分支）
    assert final["plan"] == edited
    assert final["error"] != ""
