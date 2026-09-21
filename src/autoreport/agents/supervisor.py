"""主管节点：把用户问题拆解成可执行的计划。

设计要点（面试可讲）：
- 结构化输出（with_structured_output）而不是让模型自由发挥 JSON ——
  Pydantic schema 同时承担「提示词」和「校验器」两个角色
- LLM 不可用/拆解失败时降级为单步检索计划 —— 主管是「尽力优化」而非
  「必要依赖」，保证系统最低可用性（graceful degradation 一致原则）
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from autoreport.agents.state import AgentState, SubTask

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是研报研究主管。把用户的问题拆解成 1-4 个子任务。

子任务类型：
- retrieve: 从本地研报库检索资料（绝大多数问题只需要这个）
- calc: 需要精确算术计算时（增长率、估值对比等）
- verify: 需要用官方公告核对研报预测是否兑现时
- web: 需要联网查最新资讯时

要求：
1. 每个子任务的 desc 必须自包含（含股票名称/代码与具体关注点），
   执行者看不到你的拆解理由，只能看到 desc
2. 检索类问题保持简单：多数情况 1 个 retrieve 子任务即可
3. 不要虚构不存在的任务类型"""

RETRY_PROMPT = "上文没有成功产出结构化计划。请重新拆解，这次必须输出符合 schema 的 JSON。"


class SubTaskModel(BaseModel):
    """单个子任务的结构化 schema。"""

    type: Literal["retrieve", "calc", "verify", "web"] = Field(
        description="子任务类型"
    )
    desc: str = Field(description="自包含的任务描述（含股票名称/代码与关注点）")


class PlanModel(BaseModel):
    """完整计划的结构化 schema。"""

    subtasks: list[SubTaskModel] = Field(description="1-4 个子任务")


def default_plan(question: str) -> list[SubTask]:
    """降级计划：单步检索（无 LLM 时的保底路径）。"""
    return [SubTask(type="retrieve", desc=f"检索与「{question}」相关的研报内容并总结要点")]


def plan_question(question: str) -> tuple[list[SubTask], str]:
    """拆解问题。返回 (子任务列表, 备注)。"""
    from autoreport.llm import get_chat_model, structured_invoke

    llm = get_chat_model("main")
    plan: list[SubTask] | None = None
    note = ""
    for attempt, prompt in enumerate((question, RETRY_PROMPT), 1):
        try:
            out = structured_invoke(
                llm,
                PlanModel,
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", f"用户问题：{prompt if attempt > 1 else question}"),
                ],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("计划拆解第 %d 次失败: %s", attempt, e)
            continue
        subs = [SubTask(type=s.type, desc=s.desc.strip()) for s in out.subtasks if s.desc.strip()]
        if subs:
            plan = subs[:4]
            break
    if plan is None:
        plan = default_plan(question)
        note = "LLM 拆解失败，使用默认单步检索计划"
    return plan, note


def supervisor_node(state: AgentState) -> dict:
    """图节点包装：拆解计划写入 state（失败也不阻断流程）。"""
    try:
        plan, note = plan_question(state["question"])
        return {"plan": plan, "error": ""}
    except RuntimeError as e:  # LLM 未配置 —— 默认计划，让后续节点给出清晰错误
        logger.warning("LLM 不可用，使用默认计划: %s", e)
        return {"plan": default_plan(state["question"]), "error": ""}
