"""Agent 工具集：每个工具单一职责、返回自包含文本（LLM 可直接读）。"""

from autoreport.agents.tools.calc import safe_calc
from autoreport.agents.tools.search_reports import make_search_reports_tool
from autoreport.agents.tools.verify_announcement import verify_with_announcement
from autoreport.agents.tools.web_search import web_search

__all__ = [
    "make_search_reports_tool",
    "safe_calc",
    "verify_with_announcement",
    "web_search",
]
