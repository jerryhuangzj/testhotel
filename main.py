# 🔧 简易 RAG 问答系统原型 - 云端部署版（Streamlit + LangChain）

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# === 配置区域 ===
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # 替换成你的真实 API Key
PDF_PATH = "./docs/einvoice.pdf"         # 上传的 PDF 文件路径
VECTOR_DB_PATH = "./vector_store"

# === 初始化模型与向量库 ===
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

@st.cache_resource
def load_qa_chain():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=db.as_retriever(),
        return_source_documents=True
    )
    return qa

qa_chain = load_qa_chain()

# === Streamlit UI ===
st.set_page_config(page_title="E-Invoice 问答助手", page_icon="🧾", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧾 E-Invoice 问答助手（酒店版）")
st.markdown("""
欢迎使用本系统。您可以输入与马来西亚电子发票系统（MyInvois）相关的问题，AI 会从文档中提取相关信息并为您解答。

📘 例如：
- 如何处理未索取发票的客人？
- Consolidated Invoice 的填写要求是什么？
- TIN 号码格式是怎样的？

👉 本系统基于您上传的 PDF 文档自动构建，无需联网查询。
""")

query = st.text_input("💬 请输入你的问题：")

if query:
    with st.spinner("AI 正在阅读文档中..."):
        result = qa_chain(query)
        st.success("✅ AI 回答已生成")
        st.write("### ✨ 答案：")
        st.markdown(f"{result['result']}")

        with st.expander("📄 点击查看原始参考内容"):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f"**文档段落 {i+1}:**\n\n" + doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))
