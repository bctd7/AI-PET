from typing import Callable

from modules.llm.schema import LLMQuery, LLMResponse, NpcInfo

# 按 npc_role_type 注册 handler，根据查出的 type 派发到对应 service；handler 接收 (data, npc_info)
npc_handlers: dict[str, Callable[[LLMQuery, NpcInfo], LLMResponse]] = {}


def register(role_type: str, handler: Callable[[LLMQuery, NpcInfo], LLMResponse]) -> None:
    npc_handlers[role_type] = handler


def get_handler(role_type: str) -> Callable[[LLMQuery, NpcInfo], LLMResponse] | None:
    return npc_handlers.get(role_type)
