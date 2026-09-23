"""AutoReport Streamlit Demo UI。

启动：uv run streamlit run src/autoreport/ui/streamlit_app.py

两个页面：
1. 研报问答 —— RAG 轻链路：混合检索 + 带引用作答，可展开查看引用与命中片段
2. 研报生成 —— 多智能体重链路：LangGraph 节点进度实时展示；
   计划确认 interrupt 挂起后，可编辑子任务再 Command(resume) 恢复（HITL 核心卖点）

工程要点：
- @st.cache_resource 缓存 graph+checkpointer（SqliteSaver 连接常驻，跨 rerun 复用）
- thread_id 存 session_state：页面刷新后仍能对同一线程 resume
- 无 LLM Key 时问答降级可用、生成明确报错 —— 演示永不白屏
"""

from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from autoreport.config import apply_langsmith_env, get_settings

apply_langsmith_env()
st.set_page_config(page_title="AutoReport 自动研报系统", page_icon=None, layout="wide")

TAB_NAMES = ["研报问答", "研报生成", "数据统计"]


@st.cache_resource
def load_graph():
    """编译多智能体图（含 SqliteSaver），进程内只建一次。"""
    from autoreport.agents import build_graph, get_checkpointer

    return build_graph(checkpointer=get_checkpointer())


def stream_to_status(graph, state, cfg, status):
    """执行图并把节点进度实时写进 st.status。interrupt 时返回挂起信息。"""
    final: dict = {}
    for update in graph.stream(state, cfg, stream_mode="updates"):
        for node, delta in update.items():
            if node == "__interrupt__":
                final["__interrupt__"] = delta
                status.update(label="已挂起：等待人工确认计划", state="running")
                continue
            status.write(f"节点完成：**{node}**")
            final.update(delta or {})
    return final


def show_final_report(final: dict) -> None:
    st.session_state["report_stage"] = "done"
    if final.get("error"):
        st.session_state["report_stage"] = "error"
        st.session_state["report_error"] = final["error"]
        return
    st.session_state["report_result"] = final.get("final_report", "")
    st.session_state["report_citations"] = _render_citations(final.get("citations", []))


def _render_citations(citations: list[dict]) -> str:
    seen: dict[int, dict] = {}
    for c in citations:
        seen.setdefault(c["idx"], c)
    lines = []
    for idx in sorted(seen):
        c = seen[idx]
        src = c.get("source_url") or c.get("local_path", "")
        link = f"[原文]({src})" if src else ""
        lines.append(
            f"[{idx}] {c.get('org') or '未知机构'}《{c.get('title') or '未知标题'}》"
            f"{c.get('publish_date', '')} 第{c.get('page_no', '?')}页 {link}".strip()
        )
    return "\n".join(lines)


# ================= 侧栏：系统状态 =================
settings = get_settings()
with st.sidebar:
    st.header("AutoReport")
    st.caption("LangChain + LangGraph 多智能体自动研报系统")
    st.divider()
    st.markdown(
        f"- LLM：{'**可用** (' + settings.llm_provider + ')' if settings.llm_available else '**未配置**（降级模式）'}"
    )
    st.markdown(
        f"- LangSmith：{'已开启' if settings.langsmith_enabled else '未开启'}"
    )
    st.markdown("- 向量通道：请先 `autoreport ingest` 建索引")
    if not settings.llm_available:
        st.warning("未配置 API Key：问答返回检索原文，研报生成不可用。参考 .env.example 配置。")

tab_qa, tab_report, tab_stats = st.tabs(TAB_NAMES)

# ================= Tab 1：研报问答 =================
with tab_qa:
    st.subheader("带引用溯源的研报问答（RAG 轻链路）")
    q = st.text_input(
        "你的问题",
        placeholder="如：601138 的最新评级是什么？",
        key="qa_input",
    )
    top_k = st.slider("检索条数 Top-K", 3, 10, 6, key="qa_topk")
    if st.button("提问", type="primary", key="qa_btn") and q.strip():
        from autoreport.qa import answer_question

        with st.spinner("检索与生成中……"):
            res = answer_question(q.strip(), top_k=top_k)
        st.markdown(res["answer"])
        with st.expander("引用来源", expanded=True):
            st.markdown(res["citations_md"] or "（无）")
        with st.expander("命中片段明细"):
            if res["chunks"]:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "股票": c.stock_name or c.stock_code,
                                "机构": c.org,
                                "日期": c.publish_date,
                                "页": c.page_no,
                                "得分": round(c.score_final, 3),
                                "片段": c.text[:120].replace("\n", " "),
                            }
                            for c in res["chunks"]
                        ]
                    ),
                    width="stretch",
                )
            else:
                st.info("未命中任何片段。请先在命令行执行 `uv run autoreport ingest` 导入研报。")

# ================= Tab 2：研报生成（多智能体 + HITL） =================
with tab_report:
    st.subheader("多智能体研报生成（LangGraph 工作流）")
    topic = st.text_input(
        "报告主题", placeholder="如：生成 601138（工业富联）的投资价值分析报告", key="rp_topic"
    )
    auto = st.checkbox("跳过计划确认（关闭 HITL）", value=False, key="rp_auto")

    if st.button("开始生成", type="primary", key="rp_btn") and topic.strip():
        settings.hitl_enabled = not auto
        st.session_state["thread_id"] = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state["report_stage"] = "running"
        cfg = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        from autoreport.agents.state import initial_state

        with st.status("多智能体执行中……", expanded=True) as status:
            st.write(f"thread_id：`{st.session_state['thread_id']}`")
            final = stream_to_status(load_graph(), initial_state(topic.strip()), cfg, status)
        if "__interrupt__" in final:
            payload = final["__interrupt__"][0].value
            st.session_state["report_stage"] = "awaiting_plan"
            st.session_state["report_plan"] = [dict(p) for p in payload["plan"]]
        else:
            show_final_report(final)

    # ---- HITL 计划确认区 ----
    if st.session_state.get("report_stage") == "awaiting_plan":
        st.success("主管已产出研报计划，请确认或修改后继续（人机协同）")
        df = st.data_editor(
            pd.DataFrame(st.session_state["report_plan"]),
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "类型", options=["retrieve", "calc", "verify", "web"], required=True
                ),
                "desc": st.column_config.TextColumn("任务描述", width="large", required=True),
            },
            num_rows="dynamic",
            width="stretch",
            key="plan_editor",
        )
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("确认并继续", type="primary"):
            plan = [
                {"type": str(r["type"]), "desc": str(r["desc"]).strip()}
                for _, r in df.iterrows()
                if str(r.get("desc", "")).strip() and str(r.get("type", "")) in
                ("retrieve", "calc", "verify", "web")
            ]
            cfg = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            from langgraph.types import Command

            with st.status("按计划执行多智能体……", expanded=True) as status:
                final = stream_to_status(load_graph(), Command(resume=plan or True), cfg, status)
            show_final_report(final)
        if c2.button("取消生成"):
            cfg = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            from langgraph.types import Command

            with st.spinner("取消中……"):
                final = load_graph().invoke(Command(resume=False), cfg)
            show_final_report(final)

    # ---- 结果区 ----
    if st.session_state.get("report_stage") == "done":
        st.markdown(st.session_state.get("report_result", ""))
        cites = st.session_state.get("report_citations", "")
        if cites:
            with st.expander("引用来源", expanded=True):
                st.markdown(cites)
    elif st.session_state.get("report_stage") == "error":
        st.error(f"生成失败：{st.session_state.get('report_error')}")

# ================= Tab 3：数据统计 =================
with tab_stats:
    st.subheader("知识库统计")
    if st.button("刷新统计"):
        st.cache_data.clear()
    try:
        from autoreport.data_ingestion.storage import db

        s = db.db_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("研报总数", s["reports"])
        c2.metric("文本分块", s["chunks"])
        c3.metric("公告数", s["announcements"])
        reports = db.list_reports(limit=50)
        if reports:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "股票": r.stock_name or r.stock_code,
                            "代码": r.stock_code,
                            "机构": r.org,
                            "评级": r.rating,
                            "目标价": r.target_price,
                            "日期": r.publish_date,
                            "状态": r.status,
                        }
                        for r in reports
                    ]
                ),
                width="stretch",
            )
    except Exception as e:  # noqa: BLE001
        st.error(f"读取数据库失败: {e}")
