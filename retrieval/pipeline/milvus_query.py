"""
Milvus 按条件查询（标量过滤，不涉及向量）。
用于扩展 ±1 chunk 时按 parent_id、chunk_index 区间拉取邻居。
"""

import logging
from typing import Any

from pymilvus import MilvusClient

from core.config import Vector_DataBaseConfig

logger = logging.getLogger(__name__)


def query_by_filter(
    collection_name: str,
    filter_expr: str,
    output_fields: list[str],
) -> list[dict[str, Any]]:
    """
    按表达式从 Milvus 查询标量字段，不涉及向量。
    若 collection 不存在或查询异常，返回空列表。
    """
    if not filter_expr or not output_fields:
        return []
    config = Vector_DataBaseConfig()
    client = MilvusClient(
        uri=config.uri,
        token=config.token or None,
        db_name=config.db_name,
    )
    try:
        res = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields,
        )
        if not res or not isinstance(res, list):
            return []
        return list(res) if res else []
    except Exception as e:
        logger.warning("Milvus query_by_filter 失败 [%s]: %s", filter_expr[:80], e)
        return []
    finally:
        client.close()


def escape_milvus_string(s: str) -> str:
    """对 Milvus expr 中的字符串字面量转义（双引号、反斜杠）。"""
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace('"', '\\"')
