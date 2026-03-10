import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.dependencies import _get_llm
from modules.llm.chef.node1_maitre_d import run_node1
from modules.llm.chef.node2_web_fill import run_node2
from modules.llm.dispatcher import register
from modules.llm.schema import LLMQuery, LLMResponse, NpcInfo
from tool.rewrite import filter_and_clean_local_results

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """{system_prompt}

根据以下参考资料回答用户问题。如果资料中没有相关信息，请根据你的人设自行回答。

参考资料：
{bank_content}

用户问题：{query}
"""


def _chef_handle(data: LLMQuery, npc_info: NpcInfo) -> LLMResponse:
    logger.info("[chef] 开始处理 npc_id=%s query=%r", data.npc_id, data.query)

    llm_model = _get_llm()
    system_prompt = npc_info.npc_system_prompt or ""
    logger.debug("[chef] system_prompt 长度=%d", len(system_prompt))

    # 节点 1：意图解析与环境感知
    logger.info("[chef] >>> 节点1 开始：rewrite + 本地检索")
    try:
        node1_output = run_node1(data, npc_info)
    except Exception as e:
        logger.exception("[chef] 节点1 异常: %s", e)
        raise
    logger.info(
        "[chef] <<< 节点1 完成: instruction=%s sub_queries=%s reason=%s",
        node1_output.instruction, node1_output.sub_queries, node1_output.reason,
    )

    # 节点 2：数据过少时 web 补搜并合并结果
    logger.info("[chef] >>> 节点2 开始：web 补搜")
    try:
        node1_output = run_node2(node1_output)
    except Exception as e:
        logger.exception("[chef] 节点2 异常: %s", e)
        raise
    logger.info("[chef] <<< 节点2 完成: instruction=%s reason=%s", node1_output.instruction, node1_output.reason)

    # 标准答案库：清洗合并
    all_hits = []
    for r in node1_output.sub_query_results:
        all_hits.extend(r.hits)
    logger.info("[chef] 汇总 hits 共 %d 条，开始清洗合并", len(all_hits))

    try:
        bank = filter_and_clean_local_results(all_hits, data.query, npc_info)
    except Exception as e:
        logger.exception("[chef] filter_and_clean_local_results 异常: %s", e)
        raise
    logger.info("[chef] 标准答案库 bank 长度=%d", len(bank))

    node1_output = node1_output.model_copy(update={"standard_answer_bank": bank})
    bank_content = bank if bank else "暂无参考资料"

    # 最终 LLM 生成回复
    logger.info("[chef] >>> 最终 LLM 调用开始")
    try:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | llm_model | StrOutputParser()
        response = chain.invoke({
            "query": data.query,
            "system_prompt": system_prompt,
            "bank_content": bank_content,
        })
    except Exception as e:
        logger.exception("[chef] 最终 LLM 调用异常: %s", e)
        raise
    logger.info("[chef] <<< 最终 LLM 调用完成，response 长度=%d", len(response))

    out = LLMResponse(
        response=response,
        npc_id=data.npc_id,
        node1_output=node1_output,
    )
    # 节点三：写入对话历史由 controller 通过 BackgroundTasks 在返回响应后异步执行
    return out


register("chef", _chef_handle)
