# faiss_llama3_chat_app.py

import os
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# --- Configuration ---
EMBEDDING_MODEL = "llama3"
LLM_MODEL = "llama3"
DOCUMENT_DIR = "docs"
INDEX_PATH = "faiss_index"

# --- Load and Embed Documents (with update check) ---
def build_or_update_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    os.makedirs(DOCUMENT_DIR, exist_ok=True)  # ✅ Ensure docs folder exists

    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        processed_files = set(os.listdir(os.path.join(INDEX_PATH, "processed")) if os.path.exists(os.path.join(INDEX_PATH, "processed")) else [])
    else:
        vectorstore = None
        processed_files = set()

    new_docs = []
    for fname in os.listdir(DOCUMENT_DIR):
        if fname.endswith(".txt") and fname not in processed_files:
            loader = TextLoader(os.path.join(DOCUMENT_DIR, fname))
            new_docs.extend(loader.load())
            processed_files.add(fname)

    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        docs = splitter.split_documents(new_docs)
        if vectorstore:
            vectorstore.add_documents(docs)
        else:
            vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_PATH)
        os.makedirs(os.path.join(INDEX_PATH, "processed"), exist_ok=True)
        for doc in new_docs:
            open(os.path.join(INDEX_PATH, "processed", os.path.basename(doc.metadata['source'])), "w").close()

    return vectorstore

# --- Initialize Retriever and LLM ---
def create_qa_chain():
    vectorstore = build_or_update_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = Ollama(model=LLM_MODEL)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- Streamlit UI ---
st.set_page_config(page_title="LLaMA 3 RAG Chat", layout="wide")
st.title("🦙 LLaMA 3 + FAISS Chatbot")

qa_chain = create_qa_chain()

user_question = st.text_input("Ask a question about your documents:", placeholder="What is this about?")

if user_question:
    with st.spinner("Thinking..."):
        result = qa_chain.run(user_question)
    st.markdown("---")
    st.markdown("### 💬 Answer")
    st.write(result)
