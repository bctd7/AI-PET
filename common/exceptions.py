"""
项目统一异常定义。

- AppError: 项目级基类，所有业务异常继承它
- NpcError: NPC 交互相关的业务错误（NPC 不存在、处理器不存在等）
- DataPreparationError: 数据准备/上传相关的错误（chunks 为空、文件加载失败等）
"""


class AppError(Exception):
    """项目级异常基类，所有业务异常都继承它。"""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)


class NpcError(AppError):
    """NPC 交互相关的业务错误。"""

    status_code = 400
    code = "NPC_ERROR"


class DataPreparationError(AppError):
    """数据准备/上传相关的错误。"""

    status_code = 400
    code = "DATA_PREP_ERROR"


# 常用错误码常量，便于统一管理
class ErrorCode:
    NPC_NOT_FOUND = "NPC_NOT_FOUND"
    HANDLER_NOT_FOUND = "HANDLER_NOT_FOUND"
    EMPTY_CHUNKS = "EMPTY_CHUNKS"
    FILE_LOAD_FAILED = "FILE_LOAD_FAILED"
