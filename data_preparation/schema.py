from pydantic import BaseModel, Field
from typing import List

class DataUploadRequest(BaseModel):
    """数据上传请求体：npcId 与服务器端文件路径列表。"""
    npc_id: str = Field(..., description="NPC 唯一标识")
    file_paths: List[str] = Field(..., min_length=1, description="待加载文件的服务器路径列表")


class DataUploadResponse(BaseModel):
    """数据准备任务提交后的响应。"""
    message: str = Field(..., description="提示信息")

