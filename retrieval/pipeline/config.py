"""
流水线常量配置。
"""

# 初检 RRF 阶段多取候选，保证合并后仍有足够内容做重排
RRF_INITIAL_LIMIT = 50
# 合并后重排保留条数
RERANK_TOP_K = 10
# BGE 重排分数阈值，低于丢弃
RERANK_MIN_SCORE = 0.35

# 初检与扩展阶段从 Milvus 返回的标量字段
OUTPUT_FIELDS = [
    "id",
    "page_content",
    "source",
    "parent_id",
    "chunk_id",
    "chunk_index",
]
