"""
节点 2：数据过少时用 web_search 补搜并合并结果 (The Pantry / 采买补搜)

- 输入：节点 1 的 Node1Output（含 sub_query_results，部分为 no_answer/too_few）
- 对 status 为 no_answer 或 too_few 的子问题调用 search_web_sync，将网页结果合并进 hits
- 输出：更新后的 Node1Output（合并后可能 instruction 变为 use_local）
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.llm.chef.schema import LocalRecipeHit, Node1Output, SubQueryResult
from tool.web_search import search_web_sync
from typing import Literal, cast
logger = logging.getLogger(__name__)

WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_MAX_WORKERS = 5


def _safe_search_web_rows(sub_query: str) -> list[dict]:
    try:
        return search_web_sync(sub_query, max_results=WEB_SEARCH_MAX_RESULTS) or []
    except Exception as e:
        logger.warning("节点2 web 补搜失败 [%s]: %s", sub_query[:50], e)
        return []


def run_node2(node1_output: Node1Output) -> Node1Output:
    """
    对 node1 中 no_answer/too_few 的子问题做 web 补搜，合并进 hits，返回更新后的 Node1Output。
    """
    if not node1_output.sub_query_results:
        return node1_output

    final_results: list[SubQueryResult] = []
    has_any_results = any(r.status == "has_results" for r in node1_output.sub_query_results)

    need_web_search = [
        r for r in node1_output.sub_query_results if r.status in ("no_answer", "too_few")
    ]
    web_rows_map: dict[str, list[dict]] = {}
    if need_web_search:
        max_workers = min(WEB_SEARCH_MAX_WORKERS, len(need_web_search))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(_safe_search_web_rows, r.sub_query): r.sub_query
                for r in need_web_search
            }
            for future in as_completed(future_to_query):
                sub_query = future_to_query[future]
                web_rows_map[sub_query] = future.result()

    for r in node1_output.sub_query_results:
        if r.status not in ("no_answer", "too_few"):
            final_results.append(r)
            continue
        web_hits = [
            LocalRecipeHit(
                content=x["content"],
                source=x.get("source", ""),
                score=float(x.get("score", 1.0)),
            )
            for x in web_rows_map.get(r.sub_query, [])
        ]
        merged_hits = list(r.hits) + web_hits
        new_status = "has_results" if web_hits else r.status
        SubQueryStatus = Literal["has_results", "no_answer", "too_few"]
        if web_hits:
            has_any_results = True
        final_results.append(
            SubQueryResult(sub_query=r.sub_query, hits=merged_hits, status=cast(SubQueryStatus, new_status))
        )

    instruction = "use_local" if has_any_results else node1_output.instruction
    no_answer_count = sum(1 for r in final_results if r.status == "no_answer")
    too_few_count = sum(1 for r in final_results if r.status == "too_few")
    if no_answer_count or too_few_count:
        reason = (
            f"子问题共 {len(final_results)} 个："
            f"{len(final_results) - no_answer_count - too_few_count} 个有足够结果；"
            f"{no_answer_count} 个无回答、{too_few_count} 个结果过少（已尝试 web 补搜）。"
        )
    else:
        reason = "所有子问题均有足够结果（节点2 已对不足项做 web 补搜）。"
    if instruction == "search_new":
        reason = "本地与补搜后仍无足够匹配。 " + reason

    return Node1Output(
        instruction=instruction,
        sub_queries=node1_output.sub_queries,
        sub_query_results=final_results,
        reason=reason,
    )
