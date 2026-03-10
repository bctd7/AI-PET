from typing import Generator

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import LLMConfig, MySQLConfig, EmbeddingConfig


def get_db() -> Generator[Session, None, None]:
    """供 FastAPI Depends 注入：每个请求一个 Session，用毕关闭。"""
    mysql_config = MySQLConfig()
    engine = create_engine(
        mysql_config.get_database_url(),
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def _get_llm():
    """使用 ALIYUNCS（OpenAI 兼容）Chat 模型，模型与 base_url 从 config 读取。"""
    llm_config = LLMConfig()
    return ChatOpenAI(
        base_url=llm_config.llm_base_url,
        model=llm_config.llm_model,
        api_key=llm_config.llm_api_key,
        temperature=llm_config.temperature,
        max_tokens=min(llm_config.max_tokens, 4096),
    )

def _get_embedding_model_qwen3():
    embedding_config = EmbeddingConfig()
    return OllamaEmbeddings(
        base_url = embedding_config.ollama_base_url,
        model = embedding_config.embedding_model
    )

def _get_llm_tongyi_xiaomi_analysis_flash():
    llm_config = LLMConfig()
    llm_config.llm_model = "tongyi-xiaomi-analysis-flash"
    llm_config.temperature = 0.1
    return ChatOpenAI(
        base_url=llm_config.llm_base_url,
        model=llm_config.llm_model,
        api_key=llm_config.llm_api_key,
        temperature=llm_config.temperature,
        max_tokens=min(llm_config.max_tokens, 256),
    )