import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import pickle

VECTOR_STORE_PATH = "./faiss_index"

print("🔁 Checking embedding model...")
embedding_model = OllamaEmbeddings(model="llama3")
print("✅ Embedding model ready.")

def store_file(file_path, vectorstore):
    if not file_path.endswith(".md"):
        print(f"⚠️ Skipping non-Markdown file: {file_path}")
        return vectorstore

    print(f"📄 Loading file: {file_path}")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"📚 Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs)} chunks.")

    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embedding_model)
    else:
        vectorstore.add_documents(docs)

    print(f"✅ File {file_path} inserted into FAISS!")
    return vectorstore

def process_all_md_files(folder_path="../MS"):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    print(f"📁 Scanning folder: {folder_path}")
    vectorstore = None
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        print(f"🔍 Checking file: {full_path}")
        if os.path.isfile(full_path) and filename.endswith(".md"):
            vectorstore = store_file(full_path, vectorstore)

    if vectorstore:
        print("💾 Saving FAISS index to disk...")
        vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"✅ FAISS index saved to {VECTOR_STORE_PATH}")
    else:
        print("⚠️ No Markdown files processed, FAISS index not created.")

def load_vectorstore():
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings=embedding_model)
    return None

if __name__ == "__main__":
    try:
        process_all_md_files("/home/frea/MS")
    except Exception as e:
        print(f"❌ Uncaught error: {e}")
