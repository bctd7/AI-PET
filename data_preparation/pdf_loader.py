from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def _pdf_to_text(path: Path) -> str:
    """使用 PyPDF 将单个 PDF 文件转换为纯文本。"""
    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("[pdf_loader] 提取 PDF 页面失败 path=%s page=%d err=%s", path, i, e)
            text = ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def load_pdfs_as_documents(file_paths: List[str], npc_role_type: str) -> List[Document]:
    """将 PDF 路径列表加载为 Document 列表。"""
    docs: List[Document] = []
    for path_str in file_paths:
        p = Path(path_str).resolve()
        if not p.is_file():
            logger.warning("[pdf_loader] 路径不存在或不是文件: %s", p)
            continue
        if p.suffix.lower() != ".pdf":
            logger.debug("[pdf_loader] 跳过非 PDF 文件: %s", p)
            continue

        try:
            content = _pdf_to_text(p)
            if not content.strip():
                logger.warning("[pdf_loader] PDF 转文本后内容为空: %s", p)
                continue

            parent_id = hashlib.md5(str(p).encode("utf-8")).hexdigest()
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(p),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                    "npc_role_type": npc_role_type,
                    "relative_path": p.name,
                },
            )
            docs.append(doc)
        except Exception as e:
            logger.exception("[pdf_loader] 解析 PDF 失败 path=%s err=%s", p, e)
    return docs
