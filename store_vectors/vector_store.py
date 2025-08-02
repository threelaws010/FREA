import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR", "./input")
VECTOR_STORE_PATH = "./faiss_index"

print("🔁 Checking embedding model...")
embedding_model = OllamaEmbeddings(model="llama3")
print("✅ Embedding model ready.")

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def has_been_processed(file_path, marker_dir):
    hash_val = get_file_hash(file_path)
    marker_path = os.path.join(marker_dir, hash_val + ".done")
    return os.path.exists(marker_path), marker_path

def store_file(file_path, vectorstore):
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

def process_all_txt_files(folder_path=INPUT_DIR):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    marker_dir = os.path.join(VECTOR_STORE_PATH, "processed")
    os.makedirs(marker_dir, exist_ok=True)

    print(f"📁 Scanning folder: {folder_path}")
    vectorstore = load_vectorstore() or None

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if not os.path.isfile(full_path) or not filename.lower().endswith(".txt"):
            continue

        already_done, marker_path = has_been_processed(full_path, marker_dir)
        if already_done:
            print(f"⏩ Already processed (by hash): {filename}")
            continue

        vectorstore = store_file(full_path, vectorstore)

        # Write marker
        with open(marker_path, "w") as m:
            m.write("processed")

    if vectorstore:
        print("💾 Saving FAISS index to disk...")
        vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"✅ FAISS index saved to {VECTOR_STORE_PATH}")
    else:
        print("⚠️ No new TXT files processed.")

def load_vectorstore():
    if os.path.exists(VECTOR_STORE_PATH):
        print("✅ FAISS index found — loading.")
        return FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )

    print("⚠️ FAISS index not found.")
    return None

def query_loop():
    vectorstore = load_vectorstore()
    if not vectorstore:
        print("⚠️ No vectorstore to query.")
        return

    print("\n💬 Enter query (or 'exit' to quit):")
    while True:
        q = input("> ")
        if q.strip().lower() in ("exit", "quit"):
            break
        results = vectorstore.similarity_search(q, k=3)
        for i, r in enumerate(results):
            print(f"[{i+1}] {r.page_content[:300]}\n---")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        query_loop()
    else:
        try:
            process_all_txt_files()
        except Exception as e:
            print(f"❌ Uncaught error: {e}")
