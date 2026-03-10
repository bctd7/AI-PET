import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from common.dependencies import get_db
from common.exceptions import ErrorCode, NpcError
from data_preparation.insert import insert_chunks_by_npc_role_type
from data_preparation.loader import loader_data
from data_preparation.pdf_loader import load_pdfs_as_documents
from data_preparation.schema import DataUploadRequest, DataUploadResponse
from data_preparation.splitter import recursive_chunk
from modules.llm.service import get_npc_info

logger = logging.getLogger(__name__)
uploadrouter = APIRouter()


def _run_data_preparation_task(file_paths: List[str], npc_role_type: str) -> None:
    """在后台执行：加载、分块、写入向量库。"""
    try:
        suffixes = {Path(p).suffix.lower() for p in file_paths}
        only_pdf = bool(suffixes) and all(s == ".pdf" for s in suffixes)
        if only_pdf:
            logger.info("[dataUpload] 检测到纯 PDF 输入，执行 PDF -> Document")
            documents = load_pdfs_as_documents(file_paths, npc_role_type)
            logger.info("[dataUpload] pdf_loader 文档数=%d", len(documents))
        else:
            documents = loader_data(file_paths, npc_role_type)
            logger.info("[dataUpload] loader_data 文档数=%d", len(documents))
        chunks = recursive_chunk(documents)
        logger.info("[dataUpload] recursive_chunk 分片数=%d", len(chunks))
        if not chunks:
            logger.warning("[dataUpload] 未生成有效文本块，跳过写入 file_paths=%s", file_paths)
            return
        insert_chunks_by_npc_role_type(chunks, npc_role_type)
        logger.info("[dataUpload] 任务完成 npc_role_type=%s chunks=%d", npc_role_type, len(chunks))
    except Exception as e:
        logger.exception("[dataUpload] 后台任务执行失败: %s", e)


@uploadrouter.post("/dataUpload", response_model=DataUploadResponse, status_code=202)
def run_data_preparation(
    body: DataUploadRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> DataUploadResponse:
    """提交数据准备任务：按 npc_id 校验 NPC 存在后，在后台执行加载、分块与写入。"""
    npc_info = get_npc_info(body.npc_id, db)
    if npc_info is None:
        logger.error("[dataUpload] npc_id=%s 不存在", body.npc_id)
        raise NpcError(f"NPC '{body.npc_id}' 不存在", code=ErrorCode.NPC_NOT_FOUND, status_code=404)

    background_tasks.add_task(_run_data_preparation_task, body.file_paths, npc_info.npc_role_type)
    return DataUploadResponse(message="数据准备任务已提交")


