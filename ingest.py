"""
UniTicket 向量库构建脚本

功能：
1. 从 MySQL 数据库提取活动数据
2. 使用 Ollama Embedding 模型向量化
3. 构建 FAISS 索引并保存
4. 版本管理：索引文件名包含版本号和时间戳
5. 模型绑定：索引与 embedding 模型信息绑定
6. 增量更新：基于内容哈希只写入新增文档
"""

import json
import os
import shutil
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pymysql
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()


# ============================================================================
# 配置类
# ============================================================================

class Config:
    """配置管理类"""

    # 向量库版本
    INDEX_VERSION = os.getenv('INDEX_VERSION', '1.0.0')

    # 向量库根目录
    INDEX_ROOT_DIR = Path(os.getenv('INDEX_ROOT_DIR', 'vector_indexes'))

    # 是否保留历史索引
    KEEP_HISTORY = os.getenv('KEEP_HISTORY', 'true').lower() == 'true'

    # 最大保留的历史索引数量
    MAX_HISTORY_COUNT = int(os.getenv('MAX_HISTORY_COUNT', '5'))

    # 文档切分配置
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '80'))


# ============================================================================
# 异常类
# ============================================================================

class IngestError(Exception):
    """数据导入基础异常"""
    pass


class DatabaseError(IngestError):
    """数据库相关异常"""
    pass


class EmbeddingError(IngestError):
    """向量化相关异常"""
    pass


class VersionMismatchError(IngestError):
    """版本不匹配异常"""
    pass


# ============================================================================
# 配置获取函数
# ============================================================================

def get_db_config() -> Dict[str, Any]:
    """
    从环境变量获取数据库配置

    Returns:
        数据库配置字典

    Raises:
        ValueError: 缺少必需的环境变量
    """
    config = {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME'),
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }

    # 验证必需的环境变量
    required_fields = ['user', 'password', 'database']
    missing = [field for field in required_fields if not config.get(field)]
    if missing:
        raise ValueError(f"缺少必需的环境变量: {', '.join(missing)}")

    return config


def get_embedding_config() -> Dict[str, str]:
    """
    从环境变量获取 Embedding 模型配置

    Returns:
        Embedding 配置字典
    """
    return {
        'model': os.getenv('OLLAMA_MODEL', 'mxbai-embed-large'),
        'base_url': os.getenv('OLLAMA_BASE_URL')
    }


# ============================================================================
# 数据库操作
# ============================================================================

def fetch_events_from_db(db_config: Dict[str, Any]) -> list[Document]:
    """
    从数据库提取活动数据

    Args:
        db_config: 数据库配置

    Returns:
        Document 列表

    Raises:
        DatabaseError: 数据库操作失败
    """
    db = None
    try:
        db = pymysql.connect(**db_config)
        cursor = db.cursor(pymysql.cursors.DictCursor)

        query = """
            SELECT event_name, category, venue, event_time, description, tags
            FROM events
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            print("警告: 数据库中没有找到任何活动数据")
            return []

        documents = []
        for row in rows:
            content = (
                f"活动名称: {row['event_name']}\n"
                f"分类: {row['category']}\n"
                f"地点: {row['venue']}\n"
                f"时间: {row['event_time']}\n"
                f"详情介绍: {row['description']}\n"
                f"相关标签: {row['tags']}"
            )
            doc_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            doc_key = f"{row['event_name']}|{row['event_time']}|{row['venue']}"
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "mysql",
                        "event_name": row["event_name"],
                        "category": row["category"],
                        "venue": row["venue"],
                        "event_time": str(row["event_time"]),
                        "tags": row["tags"],
                        "doc_key": doc_key,
                        "doc_hash": doc_hash
                    }
                )
            )

        print(f"从数据库成功提取 {len(documents)} 条活动数据")
        return documents

    except pymysql.Error as e:
        raise DatabaseError(f"数据库操作失败: {e}") from e
    finally:
        if db:
            db.close()


# ============================================================================
# 向量化操作
# ============================================================================

def create_embeddings(embedding_config: Dict[str, str]) -> OllamaEmbeddings:
    """
    创建 Embedding 模型

    Args:
        embedding_config: Embedding 配置

    Returns:
        OllamaEmbeddings 实例
    """
    model = embedding_config['model']
    base_url = embedding_config['base_url']

    if base_url:
        return OllamaEmbeddings(model=model, base_url=base_url)
    return OllamaEmbeddings(model=model)


def create_vectorstore(
    documents: list[Document],
    embeddings: OllamaEmbeddings,
    embedding_config: Dict[str, str]
) -> FAISS:
    """
    创建向量库

    Args:
        documents: 文档列表
        embeddings: Embedding 模型
        embedding_config: Embedding 配置

    Returns:
        FAISS 向量库

    Raises:
        EmbeddingError: 向量化失败
    """
    try:
        if not documents:
            raise EmbeddingError("没有文档可以向量化")

        print(f"开始向量化 {len(documents)} 条文档...")
        print(f"使用模型: {embedding_config['model']}")

        vectorstore = FAISS.from_documents(documents, embeddings)

        print("向量化完成")
        return vectorstore

    except Exception as e:
        raise EmbeddingError(f"向量化失败: {e}") from e


def get_ingest_state_path() -> Path:
    """获取增量导入状态文件路径"""
    return Config.INDEX_ROOT_DIR / 'ingest_state.json'


def load_ingest_state() -> Dict[str, Any]:
    """加载已导入文档的状态"""
    state_path = get_ingest_state_path()
    if not state_path.exists():
        return {"documents": {}, "legacy_hashes": set()}

    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    documents = state.get('documents', {})
    legacy_hashes = set(state.get('hashes', []))

    return {
        "documents": documents,
        "legacy_hashes": legacy_hashes
    }


def save_ingest_state(documents: Dict[str, str], latest_index_dir: Path) -> None:
    """保存已导入文档的状态"""
    Config.INDEX_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = get_ingest_state_path()
    payload = {
        'documents': documents,
        'latest_index': latest_index_dir.name,
        'chunk_config': {
            'chunk_size': Config.CHUNK_SIZE,
            'chunk_overlap': Config.CHUNK_OVERLAP
        }
    }
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def get_latest_index_dir() -> Path | None:
    """获取最新索引目录（优先 latest 链接）"""
    if not Config.INDEX_ROOT_DIR.exists():
        return None

    latest_link = Config.INDEX_ROOT_DIR / 'latest'
    if latest_link.exists() and latest_link.is_dir():
        return latest_link

    index_dirs = sorted(
        [d for d in Config.INDEX_ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith('faiss_v')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    return index_dirs[0] if index_dirs else None


def analyze_documents(
    documents: list[Document],
    state: Dict[str, Any]
) -> tuple[list[Document], list[Document], Dict[str, str]]:
    """判断新增/变更文档并返回最新的文档哈希映射"""
    existing = state.get("documents", {})
    legacy_hashes = state.get("legacy_hashes", set())

    new_docs = []
    updated_docs = []
    latest_mapping: Dict[str, str] = {}

    for doc in documents:
        doc_key = doc.metadata.get("doc_key")
        doc_hash = doc.metadata.get("doc_hash")
        if not doc_key or not doc_hash:
            continue

        latest_mapping[doc_key] = doc_hash

        if doc_key in existing:
            if existing[doc_key] != doc_hash:
                updated_docs.append(doc)
        elif doc_hash in legacy_hashes:
            # 兼容旧状态（只有 hash 列表）
            continue
        else:
            new_docs.append(doc)

    return new_docs, updated_docs, latest_mapping


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """创建文本切分器"""
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """将文档切分为 chunk，并保留元数据"""
    splitter = create_text_splitter()
    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        chunk_total = len(chunks)
        for index, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata.update({
                "chunk_index": index,
                "chunk_total": chunk_total
            })
            chunked_docs.append(Document(page_content=chunk, metadata=metadata))
    return chunked_docs


def build_metadata_summary(documents: list[Document], chunked_docs: list[Document]) -> Dict[str, Any]:
    """构建元数据统计摘要"""
    categories = Counter()
    venues = Counter()
    for doc in documents:
        if "category" in doc.metadata:
            categories[doc.metadata["category"]] += 1
        if "venue" in doc.metadata:
            venues[doc.metadata["venue"]] += 1

    return {
        "total_documents": len(documents),
        "total_chunks": len(chunked_docs),
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "top_categories": categories.most_common(10),
        "top_venues": venues.most_common(10)
    }


def load_or_create_vectorstore(
    new_documents: list[Document],
    all_documents: list[Document],
    embeddings: OllamaEmbeddings,
    embedding_config: Dict[str, str]
) -> FAISS:
    """优先加载最新索引并增量写入，否则全量重建"""
    latest_dir = get_latest_index_dir()
    if latest_dir:
        try:
            vectorstore = FAISS.load_local(
                str(latest_dir),
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"已加载最新索引: {latest_dir}")
            vectorstore.add_documents(new_documents)
            print(f"新增写入 {len(new_documents)} 条文档")
            return vectorstore
        except Exception as e:
            print(f"警告: 加载最新索引失败，将重新构建索引: {e}")

    return create_vectorstore(all_documents, embeddings, embedding_config)


# ============================================================================
# 版本管理和索引保存
# ============================================================================

def get_index_dir() -> Path:
    """
    生成带版本号和时间戳的索引目录名

    Returns:
        索引目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"faiss_v{Config.INDEX_VERSION}_{timestamp}"
    return Config.INDEX_ROOT_DIR / dir_name


def save_model_metadata(
    index_dir: Path,
    embedding_config: Dict[str, str],
    summary: Dict[str, Any]
) -> None:
    """
    保存模型元数据到索引目录

    Args:
        index_dir: 索引目录
        embedding_config: Embedding 配置
    """
    metadata = {
        'version': Config.INDEX_VERSION,
        'model': embedding_config['model'],
        'base_url': embedding_config['base_url'],
        'created_at': datetime.now().isoformat(),
        'description': 'UniTicket 活动向量库',
        'data_summary': summary
    }

    metadata_file = index_dir / 'model_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"模型元数据已保存到: {metadata_file}")


def cleanup_old_indexes() -> None:
    """
    清理旧版本的索引，保留最近的 N 个
    """
    if not Config.KEEP_HISTORY:
        return

    if not Config.INDEX_ROOT_DIR.exists():
        return

    # 获取所有索引目录
    index_dirs = sorted(
        [d for d in Config.INDEX_ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith('faiss_v')],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    # 保留最新的 N 个，删除其余的
    if len(index_dirs) > Config.MAX_HISTORY_COUNT:
        for old_dir in index_dirs[Config.MAX_HISTORY_COUNT:]:
            try:
                shutil.rmtree(old_dir)
                print(f"已删除旧索引: {old_dir.name}")
            except Exception as e:
                print(f"警告: 删除旧索引失败 {old_dir.name}: {e}")


def save_vectorstore(
    vectorstore: FAISS,
    embedding_config: Dict[str, str],
    summary: Dict[str, Any]
) -> Path:
    """
    保存向量库到带版本信息的目录

    Args:
        vectorstore: FAISS 向量库
        embedding_config: Embedding 配置

    Returns:
        保存的索引目录路径

    Raises:
        IngestError: 保存失败
    """
    try:
        index_dir = get_index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        vectorstore.save_local(str(index_dir))

        # 保存模型元数据
        save_model_metadata(index_dir, embedding_config, summary)

        # 创建最新版本的符号链接
        latest_link = Config.INDEX_ROOT_DIR / 'latest'
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.is_dir():
                shutil.rmtree(latest_link)

        try:
            latest_link.symlink_to(index_dir.name)
        except OSError:
            # Windows 可能需要管理员权限创建符号链接，使用副本代替
            shutil.copytree(index_dir, latest_link, dirs_exist_ok=True)

        print(f"\n向量库已保存到: {index_dir}")
        print(f"最新版本链接: {latest_link}")

        # 清理旧索引
        cleanup_old_indexes()

        return index_dir

    except Exception as e:
        raise IngestError(f"保存向量库失败: {e}") from e


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数"""
    print("=" * 60)
    print("UniTicket 向量库构建工具")
    print(f"版本: {Config.INDEX_VERSION}")
    print("=" * 60)

    try:
        # 1. 获取配置
        print("\n[1/6] 加载配置...")
        db_config = get_db_config()
        embedding_config = get_embedding_config()
        print(f"数据库: {db_config['host']}:{db_config['port']}/{db_config['database']}")
        print(f"Embedding 模型: {embedding_config['model']}")

        # 2. 从数据库提取数据
        print("\n[2/6] 从数据库提取数据...")
        documents = fetch_events_from_db(db_config)

        if not documents:
            print("警告: 没有数据可以向量化，程序退出")
            return

        # 3. 增量分析
        print("\n[3/6] 增量分析...")
        ingest_state = load_ingest_state()
        new_documents, updated_documents, latest_mapping = analyze_documents(documents, ingest_state)
        print(
            f"本次新增 {len(new_documents)} 条，"
            f"变更 {len(updated_documents)} 条，"
            f"已存在 {len(documents) - len(new_documents) - len(updated_documents)} 条"
        )

        if not new_documents and not updated_documents:
            print("没有新增或变更数据，向量库无需更新")
            return

        # 4. 文档切分
        print("\n[4/6] 文档切分...")
        all_chunks = split_documents(documents)
        summary = build_metadata_summary(documents, all_chunks)

        # 5. 创建/增量更新向量库
        print("\n[5/6] 创建/更新向量库...")
        embeddings = create_embeddings(embedding_config)
        if updated_documents:
            print("检测到变更文档，执行全量重建")
            vectorstore = create_vectorstore(all_chunks, embeddings, embedding_config)
        else:
            new_doc_keys = {doc.metadata.get("doc_key") for doc in new_documents}
            new_chunks = [doc for doc in all_chunks if doc.metadata.get("doc_key") in new_doc_keys]
            vectorstore = load_or_create_vectorstore(
                new_chunks,
                all_chunks,
                embeddings,
                embedding_config
            )

        # 6. 保存向量库
        print("\n[6/6] 保存向量库...")
        index_dir = save_vectorstore(vectorstore, embedding_config, summary)
        save_ingest_state(latest_mapping, index_dir)

        print("\n" + "=" * 60)
        print("向量库构建完成!")
        print(f"索引位置: {index_dir}")
        print("=" * 60)

    except IngestError as e:
        print(f"\n错误: {e}")
        exit(1)
    except Exception as e:
        print(f"\n未预期的错误: {e}")
        exit(1)


if __name__ == '__main__':
    main()
