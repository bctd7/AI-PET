from typing import Any

from pydantic import BaseModel, Field


class LLMQuery(BaseModel):
    npc_id: str
    query: str
    rewrite_query: list[str] = Field(default_factory=list, description="重写并拆解后的子问题列表")

class LLMResponse(BaseModel):
    npc_id: str
    response: str
    # chef 流程填入 Node1Output；其他 NPC 流程可为 None 或自有结构
    node1_output: Any | None = Field(None, description="节点 1 领班输出，供后续节点使用（chef 为 Node1Output）")

class PromptUpdateRequest(BaseModel):
    """更新 NPC 系统提示的请求体。"""
    npc_id: str = Field(..., description="NPC 唯一标识")
    prompt: str = Field(..., description="新的系统提示内容")


class PromptUpdateResponse(BaseModel):
    """更新 NPC 系统提示的响应。"""
    message: str = Field(..., description="提示信息")


class NpcInfo(BaseModel):
    npc_id: str
    npc_name: str
    npc_role_type: str
    npc_system_prompt: str | None = None
    npc_is_active: bool = True