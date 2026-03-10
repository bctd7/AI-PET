import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from common.dependencies import get_db
from common.exceptions import ErrorCode, NpcError
from modules.llm.chef import service  # noqa: F401 - trigger chef registration
from modules.llm.dispatcher import get_handler
from modules.llm.schema import LLMQuery, LLMResponse, PromptUpdateRequest, PromptUpdateResponse
from modules.llm.service import get_npc_info, update_npc_system_prompt
from tool.save_history import save_dialogue_summary

logger = logging.getLogger(__name__)
llmrouter = APIRouter()


@llmrouter.post("/llm/npc", response_model=LLMResponse)
def npc_chat(data: LLMQuery, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    logger.info("[controller] 收到请求 npc_id=%s query=%r", data.npc_id, data.query)

    npc_info = get_npc_info(data.npc_id, db)
    if npc_info is None:
        logger.error("[controller] npc_id=%s 不存在", data.npc_id)
        raise NpcError(f"NPC '{data.npc_id}' 不存在", code=ErrorCode.NPC_NOT_FOUND, status_code=404)
    logger.info("[controller] 查到 NPC: name=%s role_type=%s", npc_info.npc_name, npc_info.npc_role_type)

    handler = get_handler(npc_info.npc_role_type)
    if handler is None:
        logger.error("[controller] 未注册 role_type=%s 的 handler", npc_info.npc_role_type)
        raise NpcError(f"未找到 role_type='{npc_info.npc_role_type}' 的处理器", code=ErrorCode.HANDLER_NOT_FOUND)
    logger.info("[controller] 派发到 handler: %s", handler.__name__)

    try:
        result = handler(data, npc_info)
        logger.info("[controller] 请求处理完成 npc_id=%s", data.npc_id)
        background_tasks.add_task(save_dialogue_summary, data.query, result.response, data.npc_id)
        return result
    except Exception as e:
        logger.exception("[controller] handler 执行异常 npc_id=%s: %s", data.npc_id, e)
        raise

@llmrouter.post("/llm/prompt", response_model=PromptUpdateResponse)
def npc_prompt(body: PromptUpdateRequest, db: Session = Depends(get_db)) -> PromptUpdateResponse:
    """更新指定 NPC 的系统提示；NPC 不存在时返回 404。"""
    npc_info = get_npc_info(body.npc_id, db)
    if npc_info is None:
        logger.error("[controller] npc_prompt npc_id=%s 不存在", body.npc_id)
        raise NpcError(f"NPC '{body.npc_id}' 不存在", code=ErrorCode.NPC_NOT_FOUND, status_code=404)
    update_npc_system_prompt(body.npc_id, body.prompt, db)
    logger.info("[controller] npc_prompt 已更新 npc_id=%s", body.npc_id)
    return PromptUpdateResponse(message="NPC的提示词已更新")

