"""写作节点：把各子任务的调研结论汇编成结构化研报。

防幻觉的模板化写作（见方案 6.2）：
- prompt 中注入「已有引用清单」，要求模型只允许使用清单里的编号 ——
  引用编号在检索时确定，写作阶段无法新增，从机制上封死"编造来源"
- 温度 0.1 + 明确章节模板，保证输出结构稳定（便于 UI 渲染与评测打分）
"""

from __future__ import annotations

import logging

from autoreport.agents.state import AgentState

logger = logging.getLogger(__name__)

WRITER_PROMPT = """你是卖方分析师，根据研究团队提供的调研记录撰写正式研究报告。

可用引用清单（只能使用这些编号，禁止虚构其他编号）：
{citations}

写作要求：
1. 严格按以下 Markdown 结构输出：
   # {{标题（含股票与主题）}}
   ## 核心结论
   （3 条以内，每条给出依据与引用编号）
   ## 详细分析
   （整合调研记录的要点，事实必须带 [n] 引用编号）
   ## 数据与验证
   （若有计算/公告核对结果，列明精确数字）
   ## 风险提示
   （基于资料合理归纳，2-4 条）
   ## 引用来源
   （原样给出引用清单渲染结果）
2. 只使用调研记录中的事实，禁止补充外部知识或编造数字
3. 语言专业、克制，不做无依据的投资建议"""

MAX_REPORT_CHARS = 4000  # 生成截断保护


def render_citation_list(citations: list[dict]) -> str:
    """把 state 中的引用条目渲染成 Markdown 列表（按编号去重排序）。"""
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
    return "\n".join(lines) if lines else "（无引用）"


def writer_node(state: AgentState) -> dict:
    """汇编 findings -> 完整研报。"""
    from autoreport.llm import get_chat_model

    findings = state.get("findings", [])
    citations = render_citation_list(state.get("citations", []))

    if not findings:
        return {"final_report": "没有可用的调研结论（检索无结果或执行失败）。", "error": ""}

    try:
        llm = get_chat_model("main")
    except RuntimeError as e:
        # 无 LLM 降级：直接拼接调研记录，保证「检索-引用」能力仍然可用
        body = "\n\n---\n\n".join(findings)
        report = (
            f"# {state['question']}\n\n（LLM 未配置，以下为原始调研记录汇编）\n\n"
            f"{body}\n\n## 引用来源\n{citations}"
        )
        return {"final_report": report[:MAX_REPORT_CHARS], "error": ""}

    prompt = (
        f"{WRITER_PROMPT.format(citations=citations)}\n\n"
        f"## 用户问题\n{state['question']}\n\n"
        f"## 调研记录\n" + "\n\n---\n\n".join(findings)
    )
    try:
        report = llm.invoke(prompt).content
    except Exception as e:  # noqa: BLE001
        return {"error": f"报告写作失败: {e}"}
    if not isinstance(report, str):
        report = str(report)
    return {"final_report": report[:MAX_REPORT_CHARS], "error": ""}
