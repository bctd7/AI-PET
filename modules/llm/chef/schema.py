from typing import Literal

from pydantic import BaseModel, Field

from modules.llm.schema import LLMQuery, LLMResponse


class LocalRecipeHit(BaseModel):
    """本地检索到的一条菜谱/偏好记录。"""
    content: str = Field(..., description="文本内容，如 page_content")
    source: str = Field("", description="来源标识")
    score: float = Field(0.0, description="相关性分数，越高越相关")
    chunk_indices: list[int] = Field(default_factory=list, description="该段对应的 chunk_index 列表")


class SubQueryResult(BaseModel):
    """单个子问题对应的检索结果与状态，供下一节点判断是否需补搜。"""
    sub_query: str = Field(..., description="子问题原文")
    hits: list[LocalRecipeHit] = Field(default_factory=list, description="通过阈值后的检索结果")
    status: Literal["has_results", "no_answer", "too_few"] = Field(
        ...,
        description="has_results=有足够结果；no_answer=无结果；too_few=结果过少需补搜",
    )


class Node1Output(BaseModel):
    """节点 1 的输出：子问题列表 + 每个子问题的检索结果与状态 + 下一步指令。"""
    instruction: Literal["use_local", "search_new"] = Field(
        ...,
        description="use_local=用本地菜谱/偏好；search_new=去搜新菜谱",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="rewrite 分解后的子问题列表",
    )
    sub_query_results: list[SubQueryResult] = Field(
        default_factory=list,
        description="与 sub_queries 一一对应，含检索结果与状态（no_answer/too_few 供下一节点补搜）",
    )
    reason: str = Field("", description="做出该指令的简要原因")
    standard_answer_bank: str = Field(
        "",
        description="本地+网上结果经去伪存真、清洗、合并后的一整份标准答案库，供后续环节梳理最终答案",
    )


__all__ = [
    "LLMQuery",
    "LLMResponse",
    "LocalRecipeHit",
    "SubQueryResult",
    "Node1Output",
]
