"""
向量存储管理 - 加载和验证 FAISS 索引
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from config import AppConfig, ModelMismatchError, VectorStoreError


# ============================================================================
# 模型元数据
# ============================================================================

def load_model_metadata(index_path: Path) -> Dict[str, Any]:
    """
    加载索引的模型元数据

    Args:
        index_path: 索引目录路径

    Returns:
        模型元数据字典

    Raises:
        VectorStoreError: 元数据文件不存在或格式错误
    """
    metadata_file = index_path / 'model_metadata.json'

    if not metadata_file.exists():
        raise VectorStoreError(f"模型元数据文件不存在: {metadata_file}")

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise VectorStoreError(f"元数据文件格式错误: {e}") from e


def validate_model_compatibility(
    metadata: Dict[str, Any],
    current_model: str,
    current_base_url: Optional[str] = None
) -> None:
    """
    验证当前模型与索引是否匹配

    Args:
        metadata: 索引的模型元数据
        current_model: 当前使用的模型
        current_base_url: 当前模型的 base_url

    Raises:
        ModelMismatchError: 模型不匹配
    """
    index_model = metadata.get('model')
    index_base_url = metadata.get('base_url')

    if index_model != current_model:
        raise ModelMismatchError(
            f"模型不匹配: 索引使用 '{index_model}'，当前使用 '{current_model}'"
        )

    # base_url 为 None 时不进行比较
    if current_base_url and index_base_url and index_base_url != current_base_url:
        raise ModelMismatchError(
            f"模型 base_url 不匹配: 索引使用 '{index_base_url}'，当前使用 '{current_base_url}'"
        )


# ============================================================================
# 向量存储加载器
# ============================================================================

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self, config: AppConfig = None):
        """
        初始化向量存储管理器

        Args:
            config: 应用配置
        """
        self.config = config or AppConfig()
        self._vectorstore: Optional[FAISS] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._embeddings: Optional[OllamaEmbeddings] = None

    @property
    def is_loaded(self) -> bool:
        """向量存储是否已加载"""
        return self._vectorstore is not None

    @property
    def metadata(self) -> Dict[str, Any]:
        """获取索引元数据"""
        if self._metadata is None:
            raise VectorStoreError("向量存储未加载")
        return self._metadata

    def _create_embeddings(self) -> OllamaEmbeddings:
        """
        创建 Embedding 模型

        Returns:
            OllamaEmbeddings 实例
        """
        if self._embeddings is None:
            if self.config.OLLAMA_BASE_URL:
                self._embeddings = OllamaEmbeddings(
                    model=self.config.OLLAMA_MODEL,
                    base_url=self.config.OLLAMA_BASE_URL
                )
            else:
                self._embeddings = OllamaEmbeddings(model=self.config.OLLAMA_MODEL)
        return self._embeddings

    def load(self, index_path: Path = None) -> FAISS:
        """
        加载向量存储

        Args:
            index_path: 索引路径，如果不指定则使用配置

        Returns:
            FAISS 向量存储实例

        Raises:
            VectorStoreError: 加载失败
            ModelMismatchError: 模型不匹配
        """
        # 确定索引路径
        if index_path is None:
            index_path = self.config.get_index_path()

        if not index_path.exists():
            raise VectorStoreError(f"索引路径不存在: {index_path}")

        # 加载模型元数据
        self._metadata = load_model_metadata(index_path)

        # 创建 Embedding 模型
        embeddings = self._create_embeddings()

        # 验证模型兼容性
        validate_model_compatibility(
            self._metadata,
            self.config.OLLAMA_MODEL,
            self.config.OLLAMA_BASE_URL
        )

        # 加载 FAISS 索引
        try:
            self._vectorstore = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            return self._vectorstore
        except Exception as e:
            raise VectorStoreError(f"加载 FAISS 索引失败: {e}") from e

    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        """
        获取检索器

        Args:
            search_kwargs: 检索参数

        Returns:
            FAISS 检索器
        """
        if not self.is_loaded:
            raise VectorStoreError("向量存储未加载")

        kwargs = search_kwargs or {"k": self.config.DEFAULT_TOP_K}
        return self._vectorstore.as_retriever(search_kwargs=kwargs)


# 全局向量存储管理器实例
vector_store_manager = VectorStoreManager()
