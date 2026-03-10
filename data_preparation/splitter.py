import hashlib
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.dependencies import _get_embedding_model_qwen3

def recursive_chunk(
    documents: List[Document],
) -> List[Document]:
    """批量：按分隔符递归分块"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "，", " ", ""],
        chunk_size=500,
        chunk_overlap=10,
    )
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        parent_id = doc.metadata.get("parent_id", str(uuid.uuid4()))
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:12]
            child_id = f"{parent_id}_{i}_{content_hash}"
            chunk.metadata.update(doc.metadata)
            chunk.metadata.update({
                "chunk_id": child_id,
                "parent_id": parent_id,
                "doc_type": "child",
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
            })
            all_chunks.append(chunk)
    return all_chunks

def semantic_chunk(
    documents: List[Document],
) -> List[Document]:
    """批量：按语义分块（依赖 embedding 模型）"""
    embeddings = _get_embedding_model_qwen3()
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="gradient",
    )
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        parent_id = doc.metadata.get("parent_id", str(uuid.uuid4()))
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:12]
            child_id = f"{parent_id}_{i}_{content_hash}"
            chunk.metadata.update(doc.metadata)
            chunk.metadata.update({
                "chunk_id": child_id,
                "parent_id": parent_id,
                "doc_type": "child",
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
            })
            all_chunks.append(chunk)
    return all_chunks