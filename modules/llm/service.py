from sqlalchemy.orm import Session

from modules.llm.repository import Repository
from modules.llm.schema import NpcInfo


def get_npc_info(npc_id: str, db: Session) -> NpcInfo | None:
    """供 controller 调用：根据 npc_id 查 NPC 信息（含 npc_role_type）。"""
    repo = Repository(db)
    return repo._get_npc_info(npc_id)

def update_npc_system_prompt(npc_id: str, prompt: str, db: Session) -> None:
    """供 controller 调用：更新指定 NPC 的系统提示。"""
    repo = Repository(db)
    repo.update_npc_system_prompt(npc_id, prompt)
