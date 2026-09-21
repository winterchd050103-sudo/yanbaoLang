"""多智能体层：LangGraph 状态图驱动的「主管-执行-写作」协作。

模块结构（面试可讲，见方案 5.x）：
- state.py        共享状态定义（TypedDict + reducer）
- supervisor.py   主管节点：任务拆解（结构化输出）
- researcher.py   研究员节点：带检索工具的 Agent（create_agent）
- tools_agent.py  工具员节点：计算/公告核对/联网搜索
- writer.py       写作节点：模板化研报生成（强制引用编号）
- graph.py        StateGraph 组装 + SqliteSaver checkpointer + HITL interrupt
"""

from autoreport.agents.graph import build_graph, get_checkpointer

__all__ = ["build_graph", "get_checkpointer"]
