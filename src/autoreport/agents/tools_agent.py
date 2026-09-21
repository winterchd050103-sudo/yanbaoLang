"""工具员节点：执行 calc / verify / web 类子任务。

与 researcher 分开成两个节点而非塞进同一个 Agent（面试可讲）：
- 职责单一原则：检索型与计算/核对型的系统提示词、工具集完全不同，
  混在一起会稀释提示词约束、增加工具误选率
- 图上一目了然： recruiter 与 tool_agent 是两个可独立评测的单元
"""

from __future__ import annotations

import logging

from autoreport.agents.state import AgentState

logger = logging.getLogger(__name__)

TOOL_PROMPT = """你是数据分析助理，只负责执行给定的一项工具任务。

规范：
1. 按任务描述选择并调用合适的工具（计算/公告核对/联网搜索），可多步调用
2. 结论必须来自工具返回值；工具失败或证据不足时如实说明
3. 输出简洁的中文结论，涉及数字时保留工具给出的精确值"""

MAX_STEPS = 8

_TOOL_HINTS = {
    "calc": "使用 safe_calc 工具完成下面的计算任务，表达式要自己从描述中构造：",
    "verify": "使用 verify_with_announcement 工具核对下面的预测：",
    "web": "使用 web_search 工具查询：",
}


def tool_agent_node(state: AgentState) -> dict:
    """执行当前子任务（calc/verify/web 类型）。"""
    from langchain.agents import create_agent

    from autoreport.agents.tools import safe_calc, verify_with_announcement, web_search
    from autoreport.llm import get_chat_model

    sub = state["plan"][state["task_index"]]
    try:
        llm = get_chat_model("small")  # 单工具调用任务用小模型省成本
    except RuntimeError as e:
        return {"error": f"工具员无法启动：{e}"}

    agent = create_agent(
        llm,
        tools=[safe_calc, verify_with_announcement, web_search],
        system_prompt=TOOL_PROMPT,
    )
    prompt = f"{_TOOL_HINTS.get(sub['type'], '')}{sub['desc']}"
    try:
        result = agent.invoke(
            {"messages": [("user", prompt)]},
            config={"recursion_limit": MAX_STEPS},
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"工具任务「{sub['desc']}」执行失败: {e}"}

    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    finding = f"### 子任务：{sub['desc']}\n\n{answer}"
    return {
        "findings": [finding],
        "task_index": state["task_index"] + 1,
    }
