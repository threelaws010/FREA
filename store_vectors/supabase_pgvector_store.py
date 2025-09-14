# Supabase_pgvector_store.py
import os
import json
import hashlib
from pathlib import Path
from typing import List, Any

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ PGVector (Supabase Postgres with pgvector)
from sqlalchemy import create_engine
from langchain_community.vectorstores import PGVector
from langchain.schema import Document

#from anythingllm_read_embed_model import read_embed_model_from_api

load_dotenv()

# -------------------- Config --------------------

BASE = os.getenv("ANYLLM_BASE", "http://localhost:3001")
KEY  = os.getenv("ANYLLM_API_KEY", "")
SLUG = os.getenv("ANYLLM_WORKSPACE", "default")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

#EMBED_MODEL = read_embed_model_from_api(BASE, KEY, SLUG) or "allenai/specter2"
print("Using embed model:", EMBED_MODEL)

# Backends
# - ST: local SentenceTransformers
# - LM_STUDIO: LM Studio's (or LocalAI/OpenAI-compatible) embeddings API
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ST").upper()
INPUT_DIR     = os.getenv("INPUT_DIR", "./input")

# Postgres / Supabase (inside Docker network use service DNS, e.g., "supabase-db")
PGHOST     = os.getenv("PGHOST", "localhost")
PGPORT     = os.getenv("PGPORT", "5432")
PGUSER     = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")
PGDATABASE = os.getenv("PGDATABASE", "postgres")

# Collection (aka “index”/namespace) name
COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION", "text_collection")

# LM Studio / LocalAI server (only used when EMBED_BACKEND=LM_STUDIO)
# Ensure the server is up and the embedding model is loaded/exposed.
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"))
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "lm-studio"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---------------- Embedding wrappers ----------------

class STEmbeddings:
    """Thin wrapper so LangChain vector stores can call a SentenceTransformers model."""
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


class LMStudioEmbeddings:
    """Use an OpenAI-compatible embeddings endpoint (LM Studio / LocalAI) via langchain_openai."""
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
        print(f"🔗 Using OpenAI-compatible embeddings: {EMBED_MODEL}")
        return LMStudioEmbeddings(EMBED_MODEL)
    # default: SentenceTransformers
    print(f"💻 Using local SentenceTransformers embeddings: {EMBED_MODEL}")
    return STEmbeddings(EMBED_MODEL)

# ---------------- Helpers ----------------

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

def get_engine():
    # SQLAlchemy 2.x engine for PGVector
    conn = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    return create_engine(conn)

# ---------------- Core pipeline (PGVector) ----------------

def ensure_vectorstore(embeddings_impl: Any) -> PGVector | None:
    """
    Connect to (or lazily create) a PGVector collection. Assumes pgvector extension is installed:
      CREATE EXTENSION IF NOT EXISTS vector;
    """
    try:
        engine = get_engine()
        # Bare init (does not create tables); will create on first .add_documents if needed
        vs = PGVector(
            embedding_function=embeddings_impl,
            collection_name=COLLECTION_NAME,
            connection=engine,
            use_jsonb=True,
        )
        return vs
    except Exception as e:
        print(f"⚠️ Could not connect to Postgres/PGVector: {e}")
        return None

def store_file(file_path: str, vectorstore: PGVector, embeddings_impl: Any):
    print(f"📄 Loading file: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"📚 Loaded {len(documents)} docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs)} chunks")

    if vectorstore is None:
        # Create a new collection and insert docs
        engine = get_engine()
        vectorstore = PGVector.from_documents(
            documents=docs,
            embedding=embeddings_impl,
            collection_name=COLLECTION_NAME,
            connection=engine,
            use_jsonb=True,
        )
        print(f"✅ Created PGVector collection '{COLLECTION_NAME}' and inserted chunks.")
    else:
        vectorstore.add_documents(docs)
        print(f"✅ Appended {len(docs)} chunks to PGVector collection '{COLLECTION_NAME}'.")

    return vectorstore

def process_all_txt_files(folder_path: str = INPUT_DIR):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return

    # Marker dir still useful to avoid re-ingesting the same files
    marker_dir = Path("./pgvector_processed") / COLLECTION_NAME
    marker_dir.mkdir(parents=True, exist_ok=True)

    embeddings_impl = get_embedding_impl()
    vectorstore = ensure_vectorstore(embeddings_impl)

    if not vectorstore:
        print("⚠️ Vector store not available. Check Postgres connection and pgvector extension.")
        return

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

    if any_new:
        print(f"✅ Indexed new documents into PGVector collection '{COLLECTION_NAME}'.")
    else:
        print("ℹ️ No new files; index unchanged.")

def query_loop():
    embeddings_impl = get_embedding_impl()
    vectorstore = ensure_vectorstore(embeddings_impl)
    if not vectorstore:
        print("⚠️ No PGVector collection to query.")
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
