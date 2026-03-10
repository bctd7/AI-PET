"""
通用检索流水线包：初检(多取) → 按 parent_id 分组 → 同 parent 内 ±1 扩展 → 合并重叠区间 → 对合并段重排。

保证内容连续性与完整性，供 node1 或其它调用方复用。
"""

from retrieval.pipeline.config import (
    OUTPUT_FIELDS,
    RERANK_MIN_SCORE,
    RERANK_TOP_K,
    RRF_INITIAL_LIMIT,
)
from retrieval.pipeline.models import MergedSegment, RawChunk
from retrieval.pipeline.run import run_retrieve_pipeline

__all__ = [
    "MergedSegment",
    "OUTPUT_FIELDS",
    "RawChunk",
    "RERANK_MIN_SCORE",
    "RERANK_TOP_K",
    "RRF_INITIAL_LIMIT",
    "run_retrieve_pipeline",
]
