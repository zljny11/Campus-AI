"""
配置管理和异常定义
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================================
# 异常类
# ============================================================================

class AIServiceError(Exception):
    """AI 服务基础异常"""
    pass


class ConfigError(AIServiceError):
    """配置错误"""
    pass


class VectorStoreError(AIServiceError):
    """向量存储错误"""
    pass


class ModelMismatchError(VectorStoreError):
    """模型不匹配错误"""
    pass


class LLMError(AIServiceError):
    """LLM 调用错误"""
    pass


# ============================================================================
# 配置类
# ============================================================================

class AppConfig:
    """应用配置"""

    # FastAPI 服务配置
    FASTAPI_HOST: str = os.getenv('FASTAPI_HOST', '0.0.0.0')
    FASTAPI_PORT: int = int(os.getenv('FASTAPI_PORT', '8000'))
    FASTAPI_RELOAD: bool = os.getenv('FASTAPI_RELOAD', 'false').lower() == 'true'

    # 日志配置
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', 'json')  # json 或 text

    # 向量索引配置
    INDEX_ROOT_DIR: Path = Path(os.getenv('INDEX_ROOT_DIR', 'vector_indexes'))
    USE_LATEST_INDEX: bool = os.getenv('USE_LATEST_INDEX', 'true').lower() == 'true'
    INDEX_PATH: Optional[Path] = Path(os.getenv('INDEX_PATH')) if os.getenv('INDEX_PATH') else None

    # Ollama Embedding 配置
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'mxbai-embed-large')
    OLLAMA_BASE_URL: Optional[str] = os.getenv('OLLAMA_BASE_URL')

    # DeepSeek LLM 配置
    DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY', '')
    DEEPSEEK_API_BASE: str = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
    DEEPSEEK_MODEL: str = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    DEEPSEEK_TEMPERATURE: float = float(os.getenv('DEEPSEEK_TEMPERATURE', '0'))

    # RAG 配置
    DEFAULT_TOP_K: int = int(os.getenv('DEFAULT_TOP_K', '5'))
    MAX_TOP_K: int = int(os.getenv('MAX_TOP_K', '20'))

    # 系统提示词
    SYSTEM_PROMPT: str = os.getenv(
        'SYSTEM_PROMPT',
        """你是 UniTicket 助手。请根据以下检索到的上下文信息回答用户问题。

上下文信息：
{context}

请严格基于上述上下文回答问题。如果上下文中没有相关信息，请明确告知用户上下文中未包含相关内容。

用户问题：{question}"""
    )

    @classmethod
    def validate(cls) -> None:
        """验证配置"""
        errors = []

        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY 未设置")

        if cls.INDEX_PATH is None and not cls.INDEX_ROOT_DIR.exists():
            errors.append(f"索引根目录不存在: {cls.INDEX_ROOT_DIR}")

        if errors:
            raise ConfigError(f"配置验证失败: {'; '.join(errors)}")

    @classmethod
    def get_index_path(cls) -> Path:
        """获取向量索引路径"""
        if cls.INDEX_PATH:
            return cls.INDEX_PATH

        if cls.USE_LATEST_INDEX:
            latest_link = cls.INDEX_ROOT_DIR / 'latest'
            if latest_link.exists():
                # 如果是符号链接，解析真实路径
                if latest_link.is_symlink():
                    return cls.INDEX_ROOT_DIR / latest_link.readlink()
                return latest_link

        # 如果没有 latest 链接，查找最新的索引目录
        if cls.INDEX_ROOT_DIR.exists():
            index_dirs = sorted(
                [d for d in cls.INDEX_ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith('faiss_v')],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if index_dirs:
                return index_dirs[0]

        raise ConfigError(f"未找到可用的向量索引: {cls.INDEX_ROOT_DIR}")


# 获取配置实例
config = AppConfig()
