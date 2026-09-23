"""共享状态定义：整个多智能体图的数据总线。

设计要点：
- LangGraph 的核心心智模型是「状态机 + 共享黑板」：每个节点读 state、
  返回增量更新，框架负责合并与版本管理（checkpointer 可随时回放）
- 只有 findings/citations 用 operator.add 追加（多个执行节点各自产出），
  其余字段都是 last-value-wins，避免状态膨胀
- 不把任何不可序列化对象（如模型实例）放进 state —— 保证 SqliteSaver
  持久化与 HITL 中断恢复始终可用
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class SubTask(TypedDict):
    """一个子任务。type 决定由哪个执行节点处理。"""

    type: Literal["retrieve", "calc", "verify", "web"]
    desc: str


class AgentState(TypedDict):
    """多智能体共享状态。

    字段说明：
    - question:      用户原始问题 / 研报主题
    - plan:          主管拆解的子任务列表（interrupt 时可被人工改写）
    - task_index:    当前执行到的子任务下标（循环调度指针）
    - findings:      各子任务的调研结论（Markdown 片段，带 [n] 引用角标）
    - citations:     引用条目列表 [{idx, report_id, org, title, publish_date,
                     page_no, source_url, local_path}]，writer 统一渲染
    - final_report:  writer 产出的完整 Markdown 研报
    - error:         任一节点的失败原因；非空时路由直接终止
    """

    question: str
    plan: list[SubTask]
    task_index: int
    findings: Annotated[list[str], operator.add]
    citations: Annotated[list[dict[str, Any]], operator.add]
    final_report: str
    error: str


def initial_state(question: str) -> AgentState:
    """构造初始状态（CLI / UI 共用）。"""
    return AgentState(
        question=question,
        plan=[],
        task_index=0,
        findings=[],
        citations=[],
        final_report="",
        error="",
    )
