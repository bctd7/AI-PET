from pydantic import BaseModel, Field, SecretStr
from typing import Optional, List
import os
from dotenv import load_dotenv
load_dotenv()



class MySQLConfig(BaseModel):
    mysql_host: str = Field(
        default_factory=lambda: os.getenv("MYSQL_HOST"),
        description="MySQL 主机地址",
    )
    mysql_user: str = Field(
        default_factory=lambda: os.getenv("MYSQL_USER"),
        description="MySQL 用户名",
    )
    mysql_password: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MYSQL_PASSWORD")),
        description="MySQL 密码",
    )
    mysql_db: str = Field(
        default_factory=lambda: os.getenv("MYSQL_DATABASE"),
        description="MySQL 数据库名",
    )
    mysql_port: int = Field(
        default_factory=lambda: int(os.getenv("MYSQL_PORT")),
        description="MySQL 端口号"
    )
    def get_database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password.get_secret_value()}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            f"?charset=utf8mb4"
        )


class LLMConfig(BaseModel):
    llm_model: str = Field(
        default="qwen3.5-plus",
        description="默认模型",
    )
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="LLM API 地址",
    )
    llm_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ALIYUNCS_API_KEY", "")),
        description="LLM API Key",
    )
    temperature: float = Field(default=0.7, description="生成温度")
    max_tokens: int = Field(default=4096, description="最大生成长度")

class SearXMHConfig(BaseModel):
    SearXMG_URL: str = Field(
        default="http://localhost:8080",
        description="搜索引擎"
    )



class EmbeddingConfig(BaseModel):
    """Ollama Embedding 相关配置"""
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama 服务地址",
    )
    embedding_model: str = Field(
        default="qwen3-embedding:0.6b",
        description="Ollama embedding 模型名",
    )
    embedding_dim: int = Field(
        default=1024,
        description="Embedding 向量维度（需与当前 embedding 模型输出一致，如 qwen3-embedding 为 1024）",
    )

class Vector_DataBaseConfig(BaseModel):
    """Milvus 相关配置（库名、连接；collection 名在使用处显式指定）"""
    uri: str = Field(
        default="http://localhost:19530",
        description="Milvus URI",
    )
    token: str = Field(
        default="",
        description="Milvus Token",
    )
    db_name: str = Field(
        default="ai_pet",
        description="Milvus 数据库名",
    )
