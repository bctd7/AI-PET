"""
节点三：对「用户问题 + NPC 回答」生成摘要并写入 chat_history，供下一轮 get_latest_dialogue 使用。
"""
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text

from common.dependencies import _get_llm
from core.config import MySQLConfig

logger = logging.getLogger(__name__)

# chat_history.npc_id 表定义为 VARCHAR(15)
NPC_ID_MAX_LEN = 15


def save_dialogue_summary(
    query: str,
    response: str,
    npc_id: str,
) -> bool:
    """
    根据本轮用户问题与 NPC 回答，用 LLM 生成一条简短摘要并写入 chat_history。
    返回是否写入成功；失败时打日志并返回 False，不抛异常。
    """
    query = (query or "").strip()
    response = (response or "").strip()
    npc_id = (npc_id or "").strip()[:NPC_ID_MAX_LEN]
    if not npc_id:
        logger.warning("save_dialogue_summary: npc_id 为空，跳过写入")
        return False
    if not query and not response:
        logger.warning("save_dialogue_summary: query 与 response 均为空，跳过写入")
        return False

    # 1) LLM 生成摘要
    prompt_text = """请用一两句话概括以下「用户问题」与「NPC 回答」的核心要点，作为对话摘要，不要加引号或标题。
用户问题：{query}
NPC 回答：{response}
只输出摘要正文。"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | _get_llm() | StrOutputParser()
    try:
        summary = chain.invoke({"query": query or "(无)", "response": response or "(无)"}).strip()
    except Exception as e:
        logger.warning("save_dialogue_summary: LLM 生成摘要失败: %s", e)
        return False
    if not summary:
        summary = f"用户问：{query[:100]}；NPC 答：{response[:100]}"

    # 2) 写入 chat_history
    mysql_cfg = MySQLConfig()
    engine = create_engine(mysql_cfg.get_database_url(), pool_pre_ping=True)
    insert_sql = text(
        """
        INSERT INTO chat_history (npc_id, summary)
        VALUES (:npc_id, :summary)
        """
    )
    try:
        with engine.connect() as conn:
            conn.execute(insert_sql, {"npc_id": npc_id, "summary": summary})
            conn.commit()
    except Exception as e:
        logger.warning("save_dialogue_summary: 写入 chat_history 失败(npc_id=%s): %s", npc_id, e)
        return False
    return True
