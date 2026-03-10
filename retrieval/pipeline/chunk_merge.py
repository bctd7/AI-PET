"""
按 parent_id 分组、同 parent 内 ±1 扩展、合并连续区间。
"""

from retrieval.pipeline.config import OUTPUT_FIELDS
from retrieval.pipeline.milvus_query import escape_milvus_string, query_by_filter
from retrieval.pipeline.models import MergedSegment, RawChunk


def group_by_parent_id(chunks: list[RawChunk]) -> dict[str, list[RawChunk]]:
    """按 parent_id 分组，同一文档的 chunk 归为一组。"""
    groups: dict[str, list[RawChunk]] = {}
    for c in chunks:
        pid = c.parent_id or "_empty_"
        if pid not in groups:
            groups[pid] = []
        groups[pid].append(c)
    return groups


def expand_and_fetch_neighbors(
    grouped: dict[str, list[RawChunk]],
    collection_name: str,
) -> dict[str, list[RawChunk]]:
    """
    对每个 parent 下已命中的 chunk_index 向上下各扩 1，从 Milvus 按条件查询补全。
    返回仍按 parent_id 分组，每组内为扩展后的 chunk 列表（已按 chunk_index 排序）。
    """
    expanded: dict[str, list[RawChunk]] = {}
    for parent_id, chunks in grouped.items():
        if not chunks:
            continue
        indices = {c.chunk_index for c in chunks}
        lo = min(indices) - 1
        hi = max(indices) + 1
        if lo < 0:
            lo = 0
        real_parent = parent_id if parent_id != "_empty_" else ""
        filter_expr = (
            f'parent_id == "{escape_milvus_string(real_parent)}" '
            f"&& chunk_index >= {lo} && chunk_index <= {hi}"
        )
        rows = query_by_filter(collection_name, filter_expr, OUTPUT_FIELDS)
        list_for_parent: list[RawChunk] = []
        seen_index: set[int] = set()
        for row in rows:
            c = RawChunk.from_hit(row)
            if c is None:
                continue
            if c.chunk_index in seen_index:
                continue
            seen_index.add(c.chunk_index)
            list_for_parent.append(c)
        if list_for_parent:
            expanded[parent_id] = sorted(list_for_parent, key=lambda x: x.chunk_index)
        else:
            expanded[parent_id] = sorted(chunks, key=lambda x: x.chunk_index)
    return expanded


def merge_consecutive_ranges(chunks_sorted: list[RawChunk]) -> list[list[RawChunk]]:
    """
    将已按 chunk_index 排序的 chunk 列表，按「连续下标」拆成多段。
    例如 [4,5,6,7] → 一段；[4,5,7] → [4,5] 与 [7] 两段。
    """
    if not chunks_sorted:
        return []
    segments: list[list[RawChunk]] = []
    current: list[RawChunk] = [chunks_sorted[0]]
    for i in range(1, len(chunks_sorted)):
        prev_idx = chunks_sorted[i - 1].chunk_index
        curr_idx = chunks_sorted[i].chunk_index
        if curr_idx == prev_idx + 1:
            current.append(chunks_sorted[i])
        else:
            segments.append(current)
            current = [chunks_sorted[i]]
    if current:
        segments.append(current)
    return segments


def build_merged_segments(expanded: dict[str, list[RawChunk]]) -> list[MergedSegment]:
    """将扩展后的分组，逐 parent 做连续区间合并，得到 MergedSegment 列表。"""
    out: list[MergedSegment] = []
    for parent_id, chunks in expanded.items():
        if not chunks:
            continue
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)
        runs = merge_consecutive_ranges(sorted_chunks)
        for run in runs:
            if not run:
                continue
            content = "\n".join(c.page_content for c in run).strip()
            if not content:
                continue
            source = run[0].source if run else ""
            real_parent = parent_id if parent_id != "_empty_" else ""
            out.append(
                MergedSegment(
                    content=content,
                    source=source,
                    parent_id=real_parent,
                    chunk_indices=[c.chunk_index for c in run],
                )
            )
    return out
