import pymysql
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. 数据库配置
# ingest.py 关键片段
db = pymysql.connect(
    host='127.0.0.1',    # 强制使用 IPv4 地址，避开 localhost 解析问题
    port=3306,           # 显式指定端口，确保没有端口冲突
    user='root', 
    password='MyPassword!', 
    database='uniticketAI',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor # 确保返回字典格式
)

try:
    cursor = db.cursor(pymysql.cursors.DictCursor)

    # 2. 提取业务数据
    query = "SELECT event_name, category, venue, event_time, description, tags FROM events"
    cursor.execute(query)
    rows = cursor.fetchall()

    documents = []
    for row in rows:
        # --- 拼接模板放在这里 ---
        # 这种格式化方式最利于 LLM 理解上下文关系
        content = (
            f"活动名称: {row['event_name']}\n"
            f"分类: {row['category']}\n"
            f"地点: {row['venue']}\n"
            f"时间: {row['event_time']}\n"
            f"详情介绍: {row['description']}\n"
            f"相关标签: {row['tags']}"
        )
        # -----------------------
        
        # 将拼接后的字符串封装进 Document 对象
        documents.append(Document(page_content=content, metadata={"source": "mysql"}))

    # 3. 向量化并保存
    # 注意：确保你已经设置了环境变量 OPENAI_API_KEY 或在这里显式传入
    # 修改后的 Embedding 初始化
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = FAISS.from_documents(documents, embeddings)
    # 这一步会在本地生成 faiss_index 文件夹
    vectorstore.save_local("faiss_index")
    print(f"成功处理 {len(documents)} 条数据，向量库构建完成！")

finally:
    db.close()