import logging
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from common.exceptions import AppError
from data_preparation.controller import uploadrouter as upload_router
from modules.llm.controller import llmrouter as llm_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = FastAPI()


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """统一处理 AppError 及其子类，返回结构化 JSON 响应。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message},
    )


app.include_router(llm_router)
app.include_router(upload_router)
