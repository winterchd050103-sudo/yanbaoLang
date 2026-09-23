"""联网搜索工具（可选能力，无 Key 时优雅降级）。

设计原则：
- 工具永不抛异常 —— 把「没配置」「调用失败」作为普通文本结果返回给 Agent，
  让模型能自主决定换路径（如只用本地库），而不是让整张图崩溃
- 用 Tavily REST API 直调而非 SDK，减少一个依赖
"""

from __future__ import annotations

from langchain_core.tools import tool

_API_URL = "https://api.tavily.com/search"


@tool
def web_search(query: str) -> str:
    """联网搜索最新资讯（市场动态、公告快讯等），作为本地研报库的补充。

    检索本地库没有的信息（如最新股价、近期新闻）时使用。
    """
    from autoreport.config import get_settings

    key = get_settings().tavily_api_key
    if not key:
        return "联网搜索未配置（缺少 TAVILY_API_KEY），请改用本地研报库检索。"
    try:
        import requests

        resp = requests.post(
            _API_URL,
            json={"api_key": key, "query": query, "max_results": 5},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("results", [])
        if not items:
            return f"联网搜索「{query}」无结果。"
        lines = [
            f"- 【{it.get('title', '')}】{it.get('content', '')[:300]}（{it.get('url', '')}）"
            for it in items
        ]
        return "联网搜索结果：\n" + "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"联网搜索失败: {e}。请改用本地研报库检索。"
