import hashlib
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def loader_data(file_paths: List[str], npc_role_type: str) -> List[Document]:
    """
    根据上传的物理路径直接加载文件内容并封装为 Document。
    """
    documents = []
    for path_str in file_paths:
        p = Path(path_str).resolve()
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            parent_id = hashlib.md5(str(p).encode("utf-8")).hexdigest()

            # 3. 构建文档对象
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(p),  # 完整物理地址: C:\Users\...\xxx.txt
                    "parent_id": parent_id,  # 溯源 ID
                    "doc_type": "parent",  # 原始文档标识
                    "npc_role_type": npc_role_type,
                    "relative_path": p.name,  # 对于直接上传的文件，相对路径即文件名
                },
            )
            documents.append(doc)
        except Exception as e:
            print(f"解析文件 {p} 出错: {e}")
    return documents




