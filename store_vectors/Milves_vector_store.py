import os
import json
import hashlib
from pathlib import Path
from typing import List, Any

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus

# Backends
# - ST: local SentenceTransformers
# - LM_STUDIO: LM Studio's OpenAI-compatible embeddings API
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ST").upper()
EMBED_MODEL   = os.getenv("EMBED_MODEL", "allenai/specter2")   # pick your HF model or LM Studio embedding model name
INPUT_DIR     = os.getenv("INPUT_DIR", "./input")
VECTOR_BASE_PATH = Path(os.getenv("VECTOR_BASE_PATH", "./faiss_indexes"))

# LM Studio server (only used when EMBED_BACKEND=LM_STUDIO)
# Make sure LM Studio Developer server is ON and the embedding model is loaded.
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1234/v1")
os.environ.setdefault("OPENAI_API_KEY", "lm-studio")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

load_dotenv()

# ---------- Embedding wrappers ----------

class STEmbeddings:
    """Thin wrapper so FAISS can call a SentenceTransformers model."""
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        # convert_to_numpy=True is faster; convert to list for LangChain
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


class LMStudioEmbeddings:
    """Use LM Studio's OpenAI-compatible embeddings endpoint via langchain_openai."""
    def __init__(self, model_name: str):
        from langchain_openai import OpenAIEmbeddings
        self.model_name = model_name
        self.inner = OpenAIEmbeddings(model=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.inner.embed_query(text)

def get_embedding_impl():
    if EMBED_BACKEND == "LM_STUDIO":
        print(f"🔗 Using LM Studio embeddings: {EMBED_MODEL}")
        return LMStudioEmbeddings(EMBED_MODEL)
    # default ST
    print(f"💻 Using local SentenceTransformers embeddings: {EMBED_MODEL}")
    return STEmbeddings(EMBED_MODEL)

# ---------- FAISS index dir per embedding ----------

def index_dir_for_embedding() -> Path:
    """Unique directory per (backend, model)."""
    safe_model = EMBED_MODEL.replace("/", "__")
    dir_path = VECTOR_BASE_PATH / f"{EMBED_BACKEND}__{safe_model}"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def write_embedding_meta(dir_path: Path, extra: dict | None = None):
    meta = {
        "backend": EMBED_BACKEND,
        "model": EMBED_MODEL,
    }
    if extra:
        meta.update(extra)
    with open(dir_path / "embedding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def read_embedding_meta(dir_path: Path) -> dict | None:
    p = dir_path / "embedding_meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return None

# ---------- Helpers ----------

def get_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def has_been_processed(file_path: str, marker_dir: Path) -> tuple[bool, Path]:
    marker_dir.mkdir(parents=True, exist_ok=True)
    hash_val = get_file_hash(file_path)
    marker_path = marker_dir / f"{hash_val}.done"
    return marker_path.exists(), marker_path

# ---------- Core pipeline ----------

def store_file(file_path: str, vectorstore: Any, embeddings_impl: Any):
    print(f"📄 Loading file: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"📚 Loaded {len(documents)} docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs)} chunks")

    if vectorstore is None:
        vectorstore = Milvus.from_documents(
            docs,
            embeddings_impl,
            connection_args={"host": "localhost", "port": "19530"},
            collection_name="text_collection"
        )
    else:
        vectorstore.add_documents(docs)

    print(f"✅ Inserted into Milvus: {file_path}")
    return vectorstore

def load_vectorstore(embeddings_impl: Any):
    try:
        return Milvus(
            embeddings_impl,
            connection_args={"host": "localhost", "port": "19530"},
            collection_name="text_collection"
        )
    except Exception as e:
        print(f"⚠️ Could not connect to Milvus: {e}")
        return None

def save_vectorstore(vectorstore: Any):
    dir_path = index_dir_for_embedding()
    print(f"💾 Saving FAISS to {dir_path} …")
    vectorstore.save_local(dir_path)
    # Drop a meta file so we know which embedding created it
    write_embedding_meta(dir_path)
    print("✅ Saved.")

def process_all_txt_files(folder_path: str = INPUT_DIR):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return

    dir_path = index_dir_for_embedding()
    marker_dir = dir_path / "processed"
    marker_dir.mkdir(parents=True, exist_ok=True)

    embeddings_impl = get_embedding_impl()
    vectorstore = load_vectorstore(embeddings_impl)

    print(f"📁 Scanning: {folder}")
    any_new = False
    for name in sorted(os.listdir(folder)):
        full = folder / name
        if not full.is_file() or full.suffix.lower() != ".txt":
            continue
        already, marker_path = has_been_processed(str(full), marker_dir)
        if already:
            print(f"⏩ Skipping (already processed): {name}")
            continue
        vectorstore = store_file(str(full), vectorstore, embeddings_impl)
        marker_path.write_text("processed")
        any_new = True

    if vectorstore and any_new:
        print("✅ Indexed new documents into Milvus.")
    elif not vectorstore:
        print("⚠️ Nothing indexed yet. Add .txt files and rerun.")
    else:
        print("ℹ️ No new files; index unchanged.")

def query_loop():
    embeddings_impl = get_embedding_impl()
    vectorstore = load_vectorstore(embeddings_impl)
    if not vectorstore:
        print("⚠️ No Milvus collection to query.")
        return

    print("\n💬 Enter query (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        hits = vectorstore.similarity_search(q, k=4)
        for i, d in enumerate(hits, 1):
            src = d.metadata.get("source", "?")
            print(f"[{i}] {src} → {d.page_content[:220]}…\n---")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        query_loop()
    else:
        try:
            process_all_txt_files()
        except Exception as e:
            print(f"❌ Uncaught error: {e}")
