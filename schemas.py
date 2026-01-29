"""
FastAPI 请求/响应模型定义
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# 请求模型
# ============================================================================

class AskRequest(BaseModel):
    """AI 问答请求"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="检索的文档数量")


class HealthCheckRequest(BaseModel):
    """健康检查请求（预留）"""
    pass


# ============================================================================
# 响应模型
# ============================================================================

class AskResponse(BaseModel):
    """AI 问答响应"""
    answer: str = Field(..., description="AI 回答")
    latency_ms: float = Field(..., description="响应延迟（毫秒）")
    retrieval_count: int = Field(..., description="检索到的文档数量")
    index_version: str = Field(..., description="向量索引版本")
    model_version: str = Field(..., description="Embedding 模型版本")


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态: healthy/unhealthy")
    index_loaded: bool = Field(..., description="向量索引是否已加载")
    index_version: Optional[str] = Field(None, description="索引版本")
    embedding_model: Optional[str] = Field(None, description="Embedding 模型")
    llm_model: Optional[str] = Field(None, description="LLM 模型")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误信息")
