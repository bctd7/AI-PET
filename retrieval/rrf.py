from typing import Any, List
from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient

from core.config import Vector_DataBaseConfig


def run_rrf_hybrid_search(
        collection_name: str,
        reqs: List[AnnSearchRequest],
        output_fields: List[str],
        limit: int,
) -> List[Any]:
    """
    执行纯 RRF 混合搜索：对多个 AnnSearchRequest 做排名融合并返回结果。
    """
    if not reqs:
        return []
    config = Vector_DataBaseConfig()
    client = MilvusClient(
        uri=config.uri,
        token=config.token or None,
        db_name=config.db_name,
    )
    try:
        ranker = RRFRanker()
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields,
        )
        return res
    except Exception as e:
        print(f"Hybrid search error: {e}")
        return []
    finally:
        client.close()