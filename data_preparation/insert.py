from typing import List

from langchain_core.documents import Document
from pymilvus import MilvusClient

from common.dependencies import _get_embedding_model_qwen3
from common.exceptions import DataPreparationError, ErrorCode
from core.config import Vector_DataBaseConfig
from data_preparation.create_vector_db_collection import create_vector_collection_base
from data_preparation.model import VARCHAR_LONG, VARCHAR_MED, VARCHAR_SHORT
from retrieval.bm25 import build_bm25_sparse_vectors_for_milvus

# Milvus 单次 insert 条数上限，超出易失败或截断；按批写入避免超限
INSERT_BATCH_SIZE = 2000


def insert_chunks_by_npc_role_type(
    chunks: List[Document],
    npc_role_type: str,
) -> None:
    """
    按职业写入：
    - collection 名：{npc_role_type}_collection
    - 稠密向量：vector
    - 稀疏向量：bm25_vector（基于 page_content）
    """
    if not chunks:
        raise DataPreparationError("chunks must not be empty", code=ErrorCode.EMPTY_CHUNKS)

    npc_key = (npc_role_type or "default").strip().lower()
    if not npc_key:
        npc_key = "default"
    collection_name = f"{npc_key}_collection"

    # 按职业自动确保 collection 存在（调用 data_preparation 下现有逻辑）
    create_vector_collection_base(collection_name)

    page_content = [c.page_content or "" for c in chunks]
    vectors = _get_embedding_model_qwen3().embed_documents(page_content)
    bm25_rows = build_bm25_sparse_vectors_for_milvus(page_content, npc_key)

    rows = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        meta = chunk.metadata or {}
        content = chunk.page_content or ""
        row = {
            "id": str(meta.get("chunk_id") or f"chunk_{i}")[:VARCHAR_MED],
            "vector": vec,
            "bm25_vector": bm25_rows[i],
            "page_content": content[:65535],
            "source": str(meta.get("source") or "")[:VARCHAR_SHORT],
            "parent_id": str(meta.get("parent_id") or "")[:VARCHAR_MED],
            "doc_type": str(meta.get("doc_type") or "child")[:VARCHAR_SHORT],
            "npc_role_type": str(meta.get("npc_role_type") or npc_key)[:VARCHAR_LONG],
            "relative_path": str(meta.get("relative_path") or "")[:VARCHAR_LONG],
            "chunk_id": str(meta.get("chunk_id") or "")[:VARCHAR_MED],
            "chunk_index": int(meta.get("chunk_index") or i),
            "chunk_size": int(meta.get("chunk_size") or len(content)),
        }
        rows.append(row)

    config = Vector_DataBaseConfig()
    client = MilvusClient(
        uri=config.uri,
        token=config.token or None,
        db_name=config.db_name,
    )
    try:
        for start in range(0, len(rows), INSERT_BATCH_SIZE):
            batch = rows[start : start + INSERT_BATCH_SIZE]
            client.insert(collection_name=collection_name, data=batch)
    finally:
        client.close()
