"""NPC 相关 ORM 模型，供 Repository 层查询使用。"""
from datetime import datetime

from sqlalchemy import Boolean, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.mysql import DATETIME


class Base(DeclarativeBase):
    pass


class NpcInfoModel(Base):
    """NPC 信息表 ORM 模型，与建表语句一致。id 自增主键，npc_id 业务主键。"""
    __tablename__ = "npc"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    npc_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    npc_name: Mapped[str] = mapped_column(String(64), nullable=False)
    npc_role_type: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'default'"))
    npc_system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    npc_is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("1"))
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=3), nullable=False, server_default=text("CURRENT_TIMESTAMP(3)"))
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=3), nullable=False, server_default=text("CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3)"))

