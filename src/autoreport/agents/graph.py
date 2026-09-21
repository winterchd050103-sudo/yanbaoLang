"""LangGraph 组装：主管 -> 计划确认(HITL) -> 循环调度 -> 写作。

图结构（面试可讲，建议对着代码画一遍）：

    supervisor(拆解计划)
        |
    plan_gate —— HITL: interrupt() 暂停，人工确认/修改计划后 Command(resume) 恢复
        |
    route <---------------+
    /            \\        |
researcher    tool_agent  （按 task_index 循环执行子任务）
    \\            /        |
      route --------------+
        |
     writer（汇编引用写作） -> END

- SqliteSaver checkpointer：状态持久化，进程重启后可从断点恢复；
  这也是 HITL 的前提 —— interrupt 挂起的图必须能「存起来等输入」
- 所有节点失败写 state["error"]，路由函数看到 error 直接终止，
  不让异常炸图（对批处理/长任务更友好）
"""

from __future__ import annotations

import logging
import sqlite3

from langgraph.graph import END, StateGraph

from autoreport.agents.state import AgentState
from autoreport.config import get_settings

logger = logging.getLogger(__name__)

_TASK_ROUTES = {
    "researcher": "researcher",
    "tool_agent": "tool_agent",
    "writer": "writer",
    "end": END,
}


def _next_node(state: AgentState) -> str:
    """统一路由：错误终止 / 任务做完去写作 / 按子任务类型分派。"""
    if state.get("error"):
        return "end"
    idx, plan = state.get("task_index", 0), state.get("plan", [])
    if idx >= len(plan):
        return "writer"
    ttype = plan[idx]["type"]
    return "researcher" if ttype in ("retrieve", "web") else "tool_agent"


# ---- 节点函数（薄包装，便于独立测试） ----


def supervisor_node(state: AgentState) -> dict:
    from autoreport.agents.supervisor import supervisor_node as fn

    return fn(state)


def plan_gate_node(state: AgentState) -> dict:
    """HITL 关卡：hitl_enabled 时 interrupt 挂起等待人工确认/修改计划。"""
    from langgraph.types import interrupt

    from autoreport.config import get_settings

    if not get_settings().hitl_enabled:
        return {}
    resumed = interrupt(
        {
            "type": "plan_confirm",
            "question": state["question"],
            "plan": state["plan"],
        }
    )
    # resume 约定：None/False 取消；list 视为人工修改后的计划；其他值视为确认
    if resumed is None or resumed is False:
        return {"error": "用户取消了对研报计划的确认", "plan": []}
    if isinstance(resumed, list) and resumed:
        return {"plan": resumed}
    return {}


def researcher_node(state: AgentState) -> dict:
    from autoreport.agents.researcher import researcher_node as fn

    return fn(state)


def tool_agent_node(state: AgentState) -> dict:
    from autoreport.agents.tools_agent import tool_agent_node as fn

    return fn(state)


def writer_node(state: AgentState) -> dict:
    from autoreport.agents.writer import writer_node as fn

    return fn(state)


def build_graph(checkpointer=None):
    """组装并编译多智能体图。"""
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("plan_gate", plan_gate_node)
    g.add_node("researcher", researcher_node)
    g.add_node("tool_agent", tool_agent_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "plan_gate")
    g.add_conditional_edges("plan_gate", _next_node, _TASK_ROUTES)
    g.add_conditional_edges("researcher", _next_node, _TASK_ROUTES)
    g.add_conditional_edges("tool_agent", _next_node, _TASK_ROUTES)
    g.add_edge("writer", END)
    return g.compile(checkpointer=checkpointer)


def make_sqlite_saver(conn):
    """SqliteSaver + metadata 序列化兼容层（生产与测试共用）。

    踩坑记录（面试可讲）：langgraph-checkpoint 4.x 移除了
    JsonPlusSerializer.dumps/loads（只留 *_typed），而
    langgraph-checkpoint-sqlite 2.0.10 的 metadata 序列化仍调用旧 API。
    这里替换 metadata 序列化器为基于 ormsgpack 的兼容实现
    （metadata 均为简单 dict，够用；channel 数据走的是新 *_typed API 不受影响）。
    """

    from langgraph.checkpoint.sqlite import SqliteSaver

    class _MetaSerdeCompat:
        """metadata 序列化兼容层（仅基础类型 dict）。"""

        def dumps(self, obj) -> bytes:
            import ormsgpack

            return ormsgpack.packb(obj, default=str, option=ormsgpack.OPT_NON_STR_KEYS)

        def loads(self, data: bytes):
            import ormsgpack

            return ormsgpack.unpackb(data, option=ormsgpack.OPT_NON_STR_KEYS)

    saver = SqliteSaver(conn)
    saver.jsonplus_serde = _MetaSerdeCompat()
    return saver


def get_checkpointer():
    """SqliteSaver：连接常驻（check_same_thread=False 支持多线程调用）。"""
    path = get_settings().abs_path(get_settings().checkpoint_path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return make_sqlite_saver(conn)
