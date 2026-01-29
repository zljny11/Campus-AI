"""
UniTicket FastAPI AI Service

提供 RAG 问答服务，使用 FAISS 向量检索 + DeepSeek LLM
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import AIServiceError, LLMError, config
from vector_store import vector_store_manager
from schemas import (
    AskRequest,
    AskResponse,
    HealthCheckResponse,
    ErrorResponse
)


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging() -> logging.Logger:
    """配置结构化日志"""
    logger = logging.getLogger("uniticket-ai")
    handler = logging.StreamHandler(sys.stdout)

    if config.LOG_FORMAT == 'json':
        # JSON 格式日志（生产环境推荐）
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
        except ImportError:
            # 如果 pythonjsonlogger 不可用，回退到文本格式
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    else:
        # 文本格式日志（开发环境）
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    return logger


logger = setup_logging()


# ============================================================================
# 全局状态
# ============================================================================

class GlobalState:
    """全局应用状态"""

    def __init__(self):
        self.vector_store_loaded = False
        self.llm: ChatOpenAI = None
        self.rag_chain = None

    def is_ready(self) -> bool:
        """服务是否就绪"""
        return self.vector_store_loaded and self.llm is not None


state = GlobalState()


# ============================================================================
# 异常处理器
# ============================================================================

async def aiservice_exception_handler(request: Request, exc: AIServiceError) -> JSONResponse:
    """AI 服务异常处理器"""
    error_type = exc.__class__.__name__
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=error_type,
            message=str(exc),
            detail=repr(exc) if config.LOG_LEVEL == 'DEBUG' else None
        ).model_dump()
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP 异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPError",
            message=exc.detail
        ).model_dump()
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    logger.exception(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="服务器内部错误",
            detail=str(exc) if config.LOG_LEVEL == 'DEBUG' else None
        ).model_dump()
    )


# ============================================================================
# 应用生命周期
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info("UniTicket AI Service 启动中...")
    logger.info("=" * 60)

    try:
        # 1. 验证配置
        logger.info("[1/3] 验证配置...")
        config.validate()

        # 2. 加载向量存储
        logger.info("[2/3] 加载向量存储...")
        index_path = config.get_index_path()
        vector_store_manager.load(index_path)
        metadata = vector_store_manager.metadata
        logger.info(f"向量存储加载成功: {index_path}")
        logger.info(f"索引版本: {metadata.get('version')}")
        logger.info(f"索引模型: {metadata.get('model')}")
        state.vector_store_loaded = True

        # 3. 初始化 LLM
        logger.info("[3/3] 初始化 LLM...")
        state.llm = ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_API_BASE,
            temperature=config.DEEPSEEK_TEMPERATURE
        )
        logger.info(f"LLM 初始化成功: {config.DEEPSEEK_MODEL}")

        logger.info("=" * 60)
        logger.info("服务启动完成，准备接受请求")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

    yield

    # 关闭时执行
    logger.info("服务关闭中...")


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="UniTicket AI Service",
    description="基于 RAG 的智能问答服务",
    version="1.0.0",
    lifespan=lifespan
)

# 注册异常处理器
app.add_exception_handler(AIServiceError, aiservice_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# ============================================================================
# 辅助函数
# ============================================================================

def format_docs(docs) -> str:
    """格式化检索到的文档"""
    return "\n\n".join([
        f"【文档{i+1}】\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])


def create_rag_chain(retriever, question: str):
    """创建 RAG 链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | state.llm
        | StrOutputParser()
    )


# ============================================================================
# API 路由
# ============================================================================

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="健康检查",
    description="检查服务状态和组件健康情况"
)
async def health_check():
    """健康检查接口"""
    metadata = None
    if state.vector_store_loaded:
        try:
            metadata = vector_store_manager.metadata
        except Exception:
            pass

    return HealthCheckResponse(
        status="healthy" if state.is_ready() else "unhealthy",
        index_loaded=state.vector_store_loaded,
        index_version=metadata.get('version') if metadata else None,
        embedding_model=metadata.get('model') if metadata else None,
        llm_model=config.DEEPSEEK_MODEL if state.llm else None
    )


@app.post(
    "/ai/ask",
    response_model=AskResponse,
    summary="AI 问答",
    description="使用 RAG 进行智能问答"
)
async def ask_ai(request: AskRequest):
    """
    AI 问答接口

    - **question**: 用户问题
    - **top_k**: 检索的文档数量（默认 5）
    """
    if not state.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务未就绪，请稍后重试"
        )

    start_time = time.time()

    try:
        # 获取检索器
        retriever = vector_store_manager.get_retriever(
            search_kwargs={"k": request.top_k}
        )

        # 先检索以便记录
        search_results = retriever.invoke(request.question)
        logger.info(f"检索到 {len(search_results)} 条相关文档")

        # 执行 RAG
        rag_chain = create_rag_chain(retriever, request.question)
        answer = rag_chain.invoke(request.question)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"问答完成，耗时 {latency_ms:.2f}ms")

        return AskResponse(
            answer=answer,
            latency_ms=round(latency_ms, 2),
            retrieval_count=len(search_results),
            index_version=vector_store_manager.metadata.get('version'),
            model_version=vector_store_manager.metadata.get('model')
        )

    except Exception as e:
        logger.exception(f"问答处理失败: {e}")
        raise LLMError(f"问答处理失败: {e}") from e


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.FASTAPI_HOST,
        port=config.FASTAPI_PORT,
        reload=config.FASTAPI_RELOAD,
        log_config=None  # 使用自定义日志配置
    )
