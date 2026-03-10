"""
流水线编排：初检 → 分组/扩展/合并 → 重排。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from retrieval import bm25, rrf
from retrieval.bge import run_bge_rerank
from retrieval.dense import _build_dense_ip_ann_request

from retrieval.pipeline.config import (
    OUTPUT_FIELDS,
    RERANK_MIN_SCORE,
    RERANK_TOP_K,
    RRF_INITIAL_LIMIT,
)
from retrieval.pipeline.chunk_merge import (
    build_merged_segments,
    expand_and_fetch_neighbors,
    group_by_parent_id,
)
from retrieval.pipeline.models import MergedSegment, RawChunk

logger = logging.getLogger(__name__)


def _initial_retrieve(
    query: str,
    collection_name: str,
    npc_role_type: str,
    limit: int = RRF_INITIAL_LIMIT,
) -> list[RawChunk]:
    """初检：BM25 + Dense(IP) → RRF，返回带 parent_id/chunk_id/chunk_index 的 RawChunk 列表。"""
    try:
        dense_req = _build_dense_ip_ann_request(query, limit=limit)
        bm25_req = bm25.build_bm25_ann_request(query, limit=limit, npc_role_type=npc_role_type)
        raw = rrf.run_rrf_hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, bm25_req],
            output_fields=OUTPUT_FIELDS,
            limit=limit,
        )
    except Exception as e:
        logger.warning("pipeline 初检 RRF 失败 [%s]: %s", query[:50], e)
        return []

    if not raw or not raw[0]:
        return []

    chunks: list[RawChunk] = []
    for h in raw[0]:
        c = RawChunk.from_hit(h)
        if c is not None:
            chunks.append(c)
    return chunks


def _rerank_merged_segments(
    query: str,
    segments: list[MergedSegment],
    top_k: int = RERANK_TOP_K,
    min_score: float = RERANK_MIN_SCORE,
) -> list[tuple[MergedSegment, float]]:
    """对合并后的段落做 BGE 重排，返回 (MergedSegment, score) 列表，已按分数过滤。"""
    if not segments:
        return []
    docs = [
        Document(
            page_content=s.content,
            metadata={"source": s.source, "parent_id": s.parent_id, "chunk_indices": s.chunk_indices},
        )
        for s in segments
    ]
    try:
        reranked = run_bge_rerank(
            query=query,
            chunks=docs,
            top_k=min(top_k, len(docs)),
            min_score=min_score,
        )
    except Exception as e:
        logger.warning("pipeline BGE 重排失败: %s", e)
        return []
    result: list[tuple[MergedSegment, float]] = []
    for d in reranked:
        score = float(d.metadata.get("relevance_score", 0.0))
        meta = d.metadata or {}
        result.append(
            (
                MergedSegment(
                    content=d.page_content,
                    source=str(meta.get("source", "")),
                    parent_id=str(meta.get("parent_id", "")),
                    chunk_indices=list(meta.get("chunk_indices", [])),
                ),
                score,
            )
        )
    return result


def run_retrieve_pipeline(
    query: str,
    collection_name: str,
    npc_role_type: str,
    *,
    initial_limit: int = RRF_INITIAL_LIMIT,
    rerank_top_k: int = RERANK_TOP_K,
    rerank_min_score: float = RERANK_MIN_SCORE,
) -> list[dict[str, Any]]:
    """
    执行通用检索流水线：
    初检(多取) → 按 parent_id 分组 → 同 parent 内 ±1 扩展 → 合并重叠区间 → 对合并段重排。

    返回 list[dict]，每项含 "content", "source", "score"，便于转为 LocalRecipeHit 或直接使用。
    若任一步骤无结果，返回空列表。
    """
    chunks = _initial_retrieve(query, collection_name, npc_role_type, limit=initial_limit)
    if not chunks:
        logger.debug("pipeline 初检无结果 query=%s", query[:50])
        return []

    grouped = group_by_parent_id(chunks)
    if not grouped:
        return []

    expanded = expand_and_fetch_neighbors(grouped, collection_name)
    if not expanded:
        logger.debug("pipeline 扩展后无数据")
        return []

    segments = build_merged_segments(expanded)
    if not segments:
        return []

    reranked = _rerank_merged_segments(
        query, segments, top_k=rerank_top_k, min_score=rerank_min_score
    )
    return [
        {"content": seg.content, "source": seg.source, "score": score}
        for seg, score in reranked
    ]
