from core.config import EmbeddingConfig, Vector_DataBaseConfig
from pymilvus import MilvusClient

from data_preparation.index_struction import _add_vector_index_params
from data_preparation.model import _create_schema


def create_database() -> None:
    """
    确保固定数据库存在：不存在则创建，存在则复用。
    """
    config = Vector_DataBaseConfig()
    client = MilvusClient(uri=config.uri, token=config.token or None)

    try:
        db_name = config.db_name
        existing = client.list_databases()
        if db_name in existing:
            print(f"数据库 {db_name} 已存在，复用该数据库。")
        else:
            client.create_database(db_name=db_name)
            print(f"数据库 {db_name} 创建成功。")

    except Exception as e:
        print(f"操作数据库时出错: {e}")
    finally:
        client.close()


def create_vector_collection_base(
    collection_name: str,
) -> None:
    """按固定配置创建通用 schema 的 collection。"""
    config = Vector_DataBaseConfig()
    embedding_config = EmbeddingConfig()

    create_database()
    client = MilvusClient(
        uri=config.uri,
        token=config.token or None,
        db_name=config.db_name,
    )
    try:
        if client.has_collection(collection_name=collection_name):
            print(f"collection {collection_name} 已存在，正在删除并重建...")
            client.drop_collection(collection_name=collection_name)

        index_params = _add_vector_index_params(client)
        schema = _create_schema(dimension=embedding_config.embedding_dim)
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"collection {collection_name} 创建成功。")
    finally:
        client.close()