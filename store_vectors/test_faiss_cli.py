import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import pickle

# Path to your FAISS index directory
VECTOR_STORE_PATH = "./faiss_index"

# Load embedding model
embedding_model = OllamaEmbeddings(model="llama3")

# Load the FAISS vector store
try:
    vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH,
    embedding_model,
    allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS index from {VECTOR_STORE_PATH}")
except Exception as e:
    print(f"❌ Error loading FAISS index: {e}")
    exit(1)

# CLI loop
while True:
    query = input("\n🔍 Enter your query (or type 'exit' to quit): ").strip()
    if query.lower() in ['exit', 'quit']:
        break

    try:
        results = vectorstore.similarity_search(query, k=3)
        if results:
            for i, doc in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(doc.page_content.strip())
        else:
            print("⚠️ No results found.")
    except Exception as err:
        print(f"❌ Error during search: {err}")
