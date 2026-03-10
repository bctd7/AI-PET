from pymilvus import AnnSearchRequest

from common.dependencies import _get_embedding_model_qwen3


def _build_dense_cosine_ann_request(
    query: str,
    limit: int,
    filter_expr: str | None = "",
) -> AnnSearchRequest:
    """构建稠密向量 COSINE 检索的 AnnSearchRequest。"""
    embeddings = _get_embedding_model_qwen3()
    query_vector = embeddings.embed_query(query)
    return AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=limit,
        expr=filter_expr or None,
    )

def _build_dense_l2_ann_request(
    query: str,
    limit: int,
    filter_expr: str | None = "",
) -> AnnSearchRequest:
    """构建稠密向量 L2 检索的 AnnSearchRequest。"""
    embeddings = _get_embedding_model_qwen3()
    query_vector = embeddings.embed_query(query)
    return AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "L2"},
        limit=limit,
        expr=filter_expr or None,
    )


def _build_dense_ip_ann_request(
    query: str,
    limit: int,
    filter_expr: str | None = "",
) -> AnnSearchRequest:
    """构建稠密向量 IP 检索的 AnnSearchRequest。"""
    embeddings = _get_embedding_model_qwen3()
    query_vector = embeddings.embed_query(query)
    return AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "IP"},
        limit=limit,
        expr=filter_expr or None,
    )