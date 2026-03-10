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
            model_name=r"C:\Users\27902\models\ms-marco-MiniLM-L6-v2",  # 本地目录
            device="cpu",
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
    # 便于直观观察：打出 query、条数、每条 rank/score 与内容摘要
    _log_rerank_result(query=query, input_count=len(chunks), output_count=len(out), min_score=min_score, out_docs=out)
    return out


def _log_rerank_result(
    query: str,
    input_count: int,
    output_count: int,
    min_score: float,
    out_docs: List[Document],
    snippet_len: int = 120,
) -> None:
    """打出 BGE 重排结果摘要，便于调试与观察。"""
    q = query[:80] + "..." if len(query) > 80 else query
    logger.info(
        "[BGE 重排] query=%s | 输入=%d 条, 输出=%d 条 (min_score=%.2f)",
        repr(q),
        input_count,
        output_count,
        min_score,
    )
    for i, doc in enumerate(out_docs):
        score = doc.metadata.get("relevance_score", 0.0)
        text = (doc.page_content or "").strip().replace("\n", " ")
        snippet = text[:snippet_len] + "..." if len(text) > snippet_len else text
        logger.info("[BGE 重排]  #%d score=%.4f | %s", i + 1, score, repr(snippet))