from pymilvus import MilvusClient, DataType

VARCHAR_SOURCE = 2048
VARCHAR_LONG = 1024
VARCHAR_MED = 512
VARCHAR_SHORT = 256


def _create_schema(dimension: int):
    """
    主键与向量 + loader 与 splitter 的共有字段。
    """
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=VARCHAR_MED)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("bm25_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("page_content", DataType.VARCHAR, max_length=65535)
    # loader
    schema.add_field("source", DataType.VARCHAR, max_length=VARCHAR_SHORT)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=VARCHAR_MED)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=VARCHAR_SHORT)
    schema.add_field("npc_role_type", DataType.VARCHAR, max_length=VARCHAR_LONG)
    schema.add_field("relative_path", DataType.VARCHAR, max_length=VARCHAR_LONG)
    # splitter
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=VARCHAR_MED)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("chunk_size", DataType.INT64)
    return schema

