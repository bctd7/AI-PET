import logging
import os
from typing import Dict, List, Optional
from pymilvus import AnnSearchRequest
from pymilvus.model.sparse.bm25 import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

logger = logging.getLogger(__name__)

# 项目根目录（根据你的实际结构调整）
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
BM25_MODEL_DIR = os.path.join(_PROJECT_ROOT, "resource", "models")

# 按 npc_role_type 缓存的 BM25 模型单例
_BM25_FN_CACHE: Dict[str, BM25EmbeddingFunction] = {}


def get_bm25_model_path(npc_role_type: str) -> str:
    """
    根据 NPC 角色类型生成模型保存路径。
    例如: npc_role_type="chef" -> bm25_model_chef.json
    """
    safe_npc_type = (npc_role_type or "default").strip().lower()
    filename = f"bm25_model_{safe_npc_type}.json"
    return os.path.join(BM25_MODEL_DIR, filename)


def _get_bm25_fn(npc_role_type: str) -> BM25EmbeddingFunction:
    """懒加载：获取指定 NPC 角色的 BM25 模型。"""
    npc_key = (npc_role_type or "default").strip().lower()

    if npc_key not in _BM25_FN_CACHE:
        model_path = get_bm25_model_path(npc_key)
        analyzer = build_default_analyzer(language="zh")
        fn = BM25EmbeddingFunction(analyzer=analyzer)

        if os.path.exists(model_path):
            fn.load(model_path)
            logger.info(f"成功为角色 [{npc_key}] 加载 BM25 模型: {model_path}")
        else:
            logger.warning(f"角色 [{npc_key}] 的模型文件不存在，请确保先执行 fit 操作。")

        _BM25_FN_CACHE[npc_key] = fn
    return _BM25_FN_CACHE[npc_key]


def build_bm25_ann_request(
        query: str,
        limit: int,
        npc_role_type: str,
        filter_expr: str | None = "",
        anns_field: str = "bm25_vector",
) -> AnnSearchRequest:
    """构建搜索请求。通过 npc_role_type 确定使用哪个编码器。"""
    bm25_fn = _get_bm25_fn(npc_role_type)
    query_sparse = bm25_fn.encode_queries([query])

    return AnnSearchRequest(
        data=query_sparse,
        anns_field=anns_field,
        param={"metric_type": "IP"},
        limit=limit,
        expr=filter_expr or None,
    )


def build_bm25_sparse_vectors_for_milvus(
        texts: List[str],
        npc_role_type: str,
) -> List[Dict[int, float]]:
    """
    针对特定 NPC 角色的语料进行训练（Fit）、保存模型并返回稀疏向量。
    """
    analyzer = build_default_analyzer(language="zh")
    bm25_fn = BM25EmbeddingFunction(analyzer=analyzer)

    # 1. 训练模型
    bm25_fn.fit(texts)

    # 2. 编码文档
    bm25_sparse = bm25_fn.encode_documents(texts)

    # 3. 转换为 Milvus 需要的格式 [{idx: val}, ...]
    bm25_rows = []
    for i in range(bm25_sparse.shape[0]):
        row = bm25_sparse[i].tocoo()
        sparse_dict = {int(c): float(v) for c, v in zip(row.col, row.data)}
        bm25_rows.append(sparse_dict)

        # 4. 保存模型（基于 npc_role_type）
    model_path = get_bm25_model_path(npc_role_type)
    os.makedirs(BM25_MODEL_DIR, exist_ok=True)
    bm25_fn.save(model_path)

    # 更新缓存
    _BM25_FN_CACHE[npc_role_type.lower()] = bm25_fn

    logger.info(f"角色 [{npc_role_type}] 的 BM25 模型已保存至 {model_path}")
    return bm25_rows
