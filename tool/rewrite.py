import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text

from common.dependencies import _get_llm, _get_llm_tongyi_xiaomi_analysis_flash
from core.config import MySQLConfig
from modules.llm.chef.schema import LocalRecipeHit
from modules.llm.schema import LLMQuery, NpcInfo

logger = logging.getLogger(__name__)


def get_latest_dialogue(
    query: str,
    npc_info: NpcInfo,
    limit: int = 7,
) -> str:
    """
    获取当前 NPC 最近的摘要记录（默认 7 条），按时间正序拼接返回。
    """
    _ = query  # 预留：后续可按 query 做相关性摘要检索
    npc_id = (npc_info.npc_id or "").strip()
    if not npc_id:
        return ""

    safe_limit = max(1, min(int(limit), 10))
    mysql_cfg = MySQLConfig()
    engine = create_engine(mysql_cfg.get_database_url(), pool_pre_ping=True)
    sql = text(
        """
        SELECT summary
        FROM chat_history
        WHERE npc_id = :npc_id
          AND summary IS NOT NULL
          AND summary <> ''
        ORDER BY created_at DESC, id DESC
        LIMIT :limit
        """
    )

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"npc_id": npc_id, "limit": safe_limit}).fetchall()
    except Exception as e:
        logger.warning("读取最近摘要失败(npc_id=%s): %s", npc_id, e)
        return ""

    summaries = [str(r[0]).strip() for r in rows if r and str(r[0]).strip()]
    if not summaries:
        return ""

    summaries.reverse()  # 转为旧->新，便于模型理解
    return "\n".join(f"- {s}" for s in summaries)


def rewrite_and_decompose(
    query: LLMQuery,
    npc_info: NpcInfo,
    latest_info: str = "",
) -> list[str]:
    logger.info("[rewrite] 开始 rewrite_and_decompose, npc_id=%s query=%r", npc_info.npc_id, query.query)
    llm_model = _get_llm_tongyi_xiaomi_analysis_flash()
    role_type = npc_info.npc_role_type
    query_content = query.query
    history_summary = (latest_info or "").strip() or get_latest_dialogue(query_content, npc_info)
    logger.debug("[rewrite] history_summary 长度=%d", len(history_summary))
    mixed_query = (
        f"[历史摘要]\n{history_summary}\n\n[当前问题]\n{query_content}"
        if history_summary
        else query_content
    )
    system_prompt = (
        f"你是{role_type}。\n"
        "任务：将“历史摘要+当前问题”融合重写为可检索表达，并拆解为子问题。\n"
        f"输入：{mixed_query}\n"
        "约束：\n"
        "1) 严禁推测，不得添加输入中未出现的新事实；\n"
        "2) 句意等值，重写后知识范畴与输入保持一致；\n"
        "3) 字数敏感，不做解释性扩写，保持凝练；\n"
        "4) 分解粒度：输出 2-3 条可独立检索/执行的子问题；\n"
        '5) 仅输出 JSON 数组字符串，如 ["子问题1","子问题2"]，禁止任何额外文本。'
    )
    logger.info("[rewrite] 调用 LLM 进行 rewrite, role_type=%s", role_type)
    try:
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | llm_model | StrOutputParser()
        raw = chain.invoke({"role_type": role_type, "query_content": query_content}).strip()
    except Exception as e:
        logger.exception("[rewrite] LLM rewrite 调用失败: %s", e)
        raise
    logger.info("[rewrite] LLM 原始返回: %r", raw[:200])

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = [str(item).strip() for item in parsed if str(item).strip()][:3]
            logger.info("[rewrite] JSON 解析成功，子问题列表: %s", result)
            return result
    except json.JSONDecodeError:
        logger.warning("[rewrite] JSON 解析失败，降级为按行拆分，raw=%r", raw[:200])

    # 兜底：当模型未严格返回 JSON 时，按行拆分
    lines = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    result = (lines[:3] if lines else [query_content])
    logger.info("[rewrite] 兜底子问题列表: %s", result)
    return result


def filter_and_clean_local_results(
    hits: list[LocalRecipeHit],
    query: str,
    npc_info: NpcInfo,
) -> str:
    """
    将本地 + 网上搜到的多段内容，经过去伪存真、清洗、合并，得到一整份文档（标准答案库）。
    供后续环节从中梳理最终答案，不直接给用户看。
    返回一整个文档内容（str）；无有效内容时返回空字符串。
    """
    logger.info("[rewrite] filter_and_clean_local_results: hits=%d query=%r", len(hits), query[:50])
    if not hits:
        logger.info("[rewrite] hits 为空，直接返回空字符串")
        return ""
    if not query or not query.strip():
        return "\n\n".join(h.content.strip() for h in hits if (h.content or "").strip())

    llm_model = _get_llm_tongyi_xiaomi_analysis_flash()
    system_desc = (npc_info.npc_system_prompt or "").strip() or "无"
    raw_content = "\n\n---\n\n".join(
        (h.content or "").strip()[:2000] for h in hits if (h.content or "").strip()
    )
    if not raw_content.strip():
        logger.info("[rewrite] raw_content 为空，返回空字符串")
        return ""

    logger.info("[rewrite] 调用 LLM 清洗合并内容, raw_content 长度=%d", len(raw_content))
    print("========== 给最终答案库清洗的数据（raw_content）==========")
    print(raw_content)
    print("========== 以上为给 LLM 的答案库清洗输入 ==========")
    prompt_text = """[NPC 人设/描述]
{system_desc}

[用户问题]
{query}

[检索到的多段内容]（含本地与网上）
{raw_content}

[任务]
请根据上述人设与用户问题，对以上内容去伪存真、清洗、合并：
- 剔除与问题无关或不可靠的内容；
- 去重、理顺逻辑、合并成一份连贯文档。
输出即为「标准答案库」，供后续环节从中梳理最终答案。
[输出]
直接输出合并后的整份文档正文，不要输出序号、列表或解释，只输出清洗合并后的连贯内容。"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm_model | StrOutputParser()
    try:
        doc = chain.invoke({
            "system_desc": system_desc,
            "query": query.strip(),
            "raw_content": raw_content,
        }).strip()
        logger.info("[rewrite] 标准答案库生成完成，长度=%d", len(doc))
        return doc if doc else ""
    except Exception as e:
        logger.warning("标准答案库 LLM 调用失败: %s", e)
        return raw_content[:8000]
