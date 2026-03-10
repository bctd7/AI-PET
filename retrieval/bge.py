from typing import Optional, List
import logging

from langchain_core.documents import Document
from pymilvus.model.reranker import CrossEncoderRerankFunction

_RERANKER: Optional[CrossEncoderRerankFunction] = None
logger = logging.getLogger(__name__)


def _get_reranker() -> CrossEncoderRerankFunction:
    """获取 BGE CrossEncoder 重排模型（懒加载单例）。"""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoderRerankFunction(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cuda",
        )
        logger.info("BGE CrossEncoder reranker loaded (singleton).")
    return _RERANKER


def run_bge_rerank(
    query: str,
    chunks: List[Document],
    top_k: int,
    min_score: float = 0.3,
) -> List[Document]:
    """
    使用 BGE CrossEncoder 对 (query, doc) 对打分，按分数降序返回 top_k 个文档。
    低于 min_score 的文档丢弃；若全部低于阈值则返回空列表。
    """
    if not chunks:
        return []
    reranker = _get_reranker()
    texts = [doc.page_content for doc in chunks]
    results = reranker(query=query, documents=texts, top_k=min(top_k, len(texts)))
    out = []
    for rank, r in enumerate(results):
        if r.score < min_score:
            continue
        orig = chunks[r.index]
        meta = dict(orig.metadata)
        meta["relevance_score"] = r.score
        meta["rank"] = rank
        out.append(Document(page_content=r.text, metadata=meta))
    logger.debug("BGE rerank returned %d chunks (min_score=%.2f).", len(out), min_score)
    return out