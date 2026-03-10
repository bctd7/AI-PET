from typing import Optional

from sqlalchemy.orm import Session

from modules.llm.model import NpcInfoModel
from modules.llm.schema import NpcInfo


class Repository:
    def __init__(self, db: Session):
        self.db = db

    def _get_npc_info(self, npc_id: str) -> Optional[NpcInfo]:
        row = self.db.query(NpcInfoModel).filter(NpcInfoModel.npc_id == npc_id).first()
        if row is None:
            return None
        return NpcInfo(
            npc_id=str(row.npc_id),
            npc_name=str(row.npc_name),
            npc_role_type=str(row.npc_role_type),
            npc_system_prompt=str(row.npc_system_prompt) if row.npc_system_prompt is not None else None,
            npc_is_active=bool(row.npc_is_active),
        )

    def update_npc_system_prompt(self, npc_id: str, prompt: str) -> None:
        """按 npc_id 更新该 NPC 的 npc_system_prompt。"""
        row = self.db.query(NpcInfoModel).filter(NpcInfoModel.npc_id == npc_id).first()
        if row is None:
            return
        row.npc_system_prompt = prompt
        self.db.commit()

