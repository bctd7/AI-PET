"""
流水线用到的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawChunk:
    """单条原始 chunk（来自 RRF 或按 filter 查询）。"""
    id: str
    page_content: str
    source: str
    parent_id: str
    chunk_id: str
    chunk_index: int

    @classmethod
    def from_hit(cls, h: dict[str, Any]) -> RawChunk | None:
        content = (h.get("page_content") or "").strip()
        if not content:
            return None
        try:
            idx = int(h.get("chunk_index", 0))
        except (TypeError, ValueError):
            idx = 0
        return cls(
            id=str(h.get("id") or ""),
            page_content=content,
            source=str(h.get("source") or ""),
            parent_id=str(h.get("parent_id") or ""),
            chunk_id=str(h.get("chunk_id") or ""),
            chunk_index=idx,
        )


@dataclass
class MergedSegment:
    """合并后的一段连续内容（用于重排与最终输出）。"""
    content: str
    source: str
    parent_id: str
    chunk_indices: list[int] = field(default_factory=list)
