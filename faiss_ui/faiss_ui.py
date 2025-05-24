import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./faiss_index")

# Load FAISS
embedding_model = OllamaEmbeddings(model="llama3")
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# UI
st.title("🔍 FAISS Semantic Search")
query = st.text_input("Enter your question:")

if query:
    results = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(results):
        st.write(f"**Result {i+1}:**")
        st.write(doc.page_content)
