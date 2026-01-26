from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import uvicorn
import time

app = FastAPI()

# 1. 加载本地向量索引（必须与 ingest 保持一致）
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. 初始化 DeepSeek API (对话不吃内存)
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key="sk-849ad16616534e628f0498952b93cc6c", # 填入你申请的 Key
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)

# 3. 定义文档格式化函数
def format_docs(docs):
    return "\n\n".join([f"【文档{i+1}】\n{doc.page_content}" for i, doc in enumerate(docs)])

# 4. RAG 链配置 (使用 LCEL 构建更清晰的链)
system_prompt = """你是 UniTicket 助手。请根据以下检索到的上下文信息回答用户问题。

上下文信息：
{context}

请严格基于上述上下文回答问题。如果上下文中没有相关信息，请明确告知用户上下文中未包含相关内容。

用户问题：{question}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 构建完整的 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/ai/ask")
async def ask_ai(question: str):
    start = time.time()

    # 先检索以便打印调试信息
    search_results = retriever.invoke(question)
    print(f"\n[DEBUG] 检索到的文档数量: {len(search_results)}")
    for i, doc in enumerate(search_results):
        print(f"[DEBUG] 匹配到的第 {i+1} 条内容:\n{doc.page_content}\n")

    # 执行 RAG
    answer = rag_chain.invoke(question)

    elapsed = (time.time() - start) * 1000  # 转换为毫秒
    print(f"[PERF] 总响应时间: {elapsed:.2f}ms\n")

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)