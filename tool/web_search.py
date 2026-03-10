"""
网页搜索：请求 SearXNG，返回结构化结果。
提供可复用函数供应用内调用；MCP 工具委托同一逻辑。
"""

import asyncio
import logging
from typing import Any

import httpx

from core.config import SearXMHConfig

logger = logging.getLogger(__name__)

# 默认返回条数
DEFAULT_MAX_RESULTS = 5


def _parse_searxng_response(data: dict[str, Any], max_results: int) -> list[dict[str, Any]]:
    """
    解析 SearXNG JSON：通常含 results 数组，每项有 title、url、content。
    返回 list[dict]，每项含 content, source, score。
    """
    raw = data.get("results") or data.get("result") or []
    out = []
    for item in raw[:max_results]:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or item.get("title") or "").strip()
        if not content:
            content = (item.get("title") or "").strip()
        source = str(item.get("url") or item.get("link") or "")
        out.append({"content": content, "source": source, "score": 1.0})
    return out


async def search_web_async(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """
    异步：请求 SearXNG，返回 list[dict]，每项含 content, source, score。
    供 MCP 或异步调用方使用。
    """
    if not query or not query.strip():
        return []
    params = {"q": query.strip(), "language": "zh", "format": "json"}
    url = SearXMHConfig().SearXMG_URL
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.warning("web_search 请求失败 [%s]: %s", query[:50], e)
        return []
    return _parse_searxng_response(data, max_results)


def search_web_sync(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """
    同步：在应用内直接调用（如 node1 数据过少时补搜）。
    返回 list[dict]，每项含 content, source, score。
    """
    return asyncio.run(search_web_async(query, max_results))


# ---------------------------------------------------------------------------
# MCP 工具：委托给 search_web_async，供 Cursor 等 MCP 客户端调用
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("WebSearchUtils")

    @mcp.tool()
    async def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """根据 query 进行网页搜索，返回若干条结果（content、source、score）。"""
        return await search_web_async(query, max_results=max_results)
except ImportError:
    mcp = None
