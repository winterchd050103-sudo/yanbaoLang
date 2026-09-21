"""研究员节点：带检索工具的 Agent（langchain create_agent）。

为什么用 create_agent 而不是手写 while 循环（面试可讲）：
- 检索质量差时模型需要「换关键词重试」「先查 A 再查 B」的自主性，
  这正是 ReAct 循环的价值；LangGraph 里一个节点封装整个循环，
  对外仍是一个 state->state 的普通节点，图结构保持简单
- 工具集刻意只给一个 search_reports —— 工具越少，模型选择越稳
"""

from __future__ import annotations

import logging

from autoreport.agents.state import AgentState
from autoreport.retrieval.citations import CitationManager

logger = logging.getLogger(__name__)

RESEARCHER_PROMPT = """你是资深证券研究员助理，负责在本地研报库中调研指定问题。

工作规范：
1. 用 search_reports 工具检索，必要时换关键词多次检索（如先查股票名，再查具体指标）
2. 结论必须基于检索到的内容，每个事实后面标注 [n] 引用编号（沿用检索结果的编号）
3. 检索不到的内容如实说明，禁止编造
4. 最后输出一段结构化中文总结：要点分条，每条带引用编号"""

MAX_STEPS = 6  # 防止 ReAct 循环失控（成本兜底）


def researcher_node(state: AgentState) -> dict:
    """执行当前子任务（retrieve 类型），产出带引用的调研结论。"""
    from langchain.agents import create_agent

    from autoreport.agents.tools import make_search_reports_tool
    from autoreport.llm import get_chat_model

    sub = state["plan"][state["task_index"]]
    existing = state.get("citations", [])
    cm = CitationManager(start_from=len(existing) + 1)  # 全局编号续接

    try:
        llm = get_chat_model("main")
    except RuntimeError as e:
        return {"error": f"研究员无法启动：{e}"}

    agent = create_agent(
        llm,
        tools=[make_search_reports_tool(cm)],
        system_prompt=RESEARCHER_PROMPT,
    )
    try:
        result = agent.invoke(
            {"messages": [("user", f"调研任务：{sub['desc']}")]},
            config={"recursion_limit": MAX_STEPS},
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"研究任务「{sub['desc']}」执行失败: {e}"}

    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)

    finding = f"### 子任务：{sub['desc']}\n\n{answer}"
    return {
        "findings": [finding],
        "citations": [c.__dict__ for c in cm.list_all()],
        "task_index": state["task_index"] + 1,
    }
