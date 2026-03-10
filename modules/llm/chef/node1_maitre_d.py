"""
节点 1：意图解析与环境感知 (The Maitre D' / 领班)

- 使用通用 rewrite 对问题做分解，得到子问题列表
- 对每个子问题调用 retrieval.pipeline 通用流水线：初检 → 按 parent 分组 → 扩展 ±1 → 合并 → 重排
- 对无回答或回答过少的子问题打标，供下一节点补搜
"""

import logging

from modules.llm.chef.schema import LocalRecipeHit, Node1Output, SubQueryResult
from modules.llm.schema import LLMQuery, NpcInfo
from retrieval.pipeline import (
    RERANK_MIN_SCORE,
    RERANK_TOP_K,
    run_retrieve_pipeline,
)
from tool.rewrite import rewrite_and_decompose
from typing import Literal, cast
logger = logging.getLogger(__name__)


def _collection_name(npc_role_type: str) -> str:
    key = (npc_role_type or "default").strip().lower() or "default"
    return f"{key}_collection"


# 至少几条结果视为「有足够回答」，不足则标为 too_few 供下一节点补搜
MIN_HITS_FOR_HAS_RESULTS = 2


def _retrieve_and_rerank_one(
    sub_query: str,
    npc_role_type: str,
    collection_name: str,
) -> tuple[list[LocalRecipeHit], str]:
    """
    对单个子问题执行通用检索流水线，返回 (LocalRecipeHit 列表, status)。
    """
    try:
        rows = run_retrieve_pipeline(
            query=sub_query,
            collection_name=collection_name,
            npc_role_type=npc_role_type,
            rerank_top_k=RERANK_TOP_K,
            rerank_min_score=RERANK_MIN_SCORE,
        )
    except Exception as e:
        logger.warning("子问题 pipeline 检索失败 [%s]: %s", sub_query[:50], e)
        return [], "no_answer"

    hits = [
        LocalRecipeHit(
            content=r["content"],
            source=r.get("source", ""),
            score=float(r.get("score", 0.0)),
        )
        for r in (rows or [])
    ]

    if len(hits) == 0:
        status = "no_answer"
    elif len(hits) < MIN_HITS_FOR_HAS_RESULTS:
        status = "too_few"
    else:
        status = "has_results"
    return hits, status


def run_node1(data: LLMQuery, npc_info: NpcInfo) -> Node1Output:
    """
    节点 1 全流程：rewrite 分解 → 对每个子问题检索(BM25+Dense IP)+RRF+BGE+阈值 → 打标供下一节点。
    """

    sub_queries = rewrite_and_decompose(data, npc_info)
    if not sub_queries:
        sub_queries = [data.query]

    coll = _collection_name(npc_info.npc_role_type)
    sub_query_results: list[SubQueryResult] = []
    has_any_results = False
    SubQueryStatus = Literal["has_results", "no_answer", "too_few"]
    for sub_q in sub_queries:
        hits, status = _retrieve_and_rerank_one(sub_q, npc_info.npc_role_type, coll)
        if status == "has_results":
            has_any_results = True
        sub_query_results.append(
            SubQueryResult(sub_query=sub_q, hits=hits, status=cast(SubQueryStatus, status))
        )

    instruction = "use_local" if has_any_results else "search_new"
    no_answer_count = sum(1 for r in sub_query_results if r.status == "no_answer")
    too_few_count = sum(1 for r in sub_query_results if r.status == "too_few")
    if no_answer_count or too_few_count:
        reason = f"子问题共 {len(sub_queries)} 个：{len(sub_queries) - no_answer_count - too_few_count} 个有足够结果；{no_answer_count} 个无回答、{too_few_count} 个结果过少，已打标供下一节点补搜。"
    else:
        reason = "所有子问题均在本地找到足够相关结果，可直接使用。"
    if instruction == "search_new":
        reason = "本地无足够匹配，建议搜索新菜谱。 " + reason

    return Node1Output(
        instruction=instruction,
        sub_queries=sub_queries,
        sub_query_results=sub_query_results,
        reason=reason,
    )
