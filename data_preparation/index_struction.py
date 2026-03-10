from pymilvus import MilvusClient


def _add_vector_index_params(client: MilvusClient):
    """统一向量索引：vector COSINE + bm25_vector SPARSE IP。"""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="bm25_vector",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    return index_params