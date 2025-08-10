"""
LM Studio + FAISS RAG — API + CLI
---------------------------------
This single file lets you:
  • Ingest documents into FAISS (LM Studio embeddings or local sentence‑transformers)
  • Chat with RAG via CLI
  • Run a small local **HTTP API** so any client (or browser app) can ask questions with retrieval

Quick install
-------------
python -m pip install \
  faiss-cpu langchain-community langchain-openai python-dotenv pypdf sentence-transformers \
  openai fastapi uvicorn pydantic

LM Studio notes
---------------
• In LM Studio: Settings → Developer → Enable server. Default base URL: http://localhost:1234/v1
• Load a chat model (for generation). Optionally load an embedding model if you want LM Studio to embed.

Run examples
------------
# Ingest a folder
python lmstudio_faiss_rag.py --ingest ./docs

# Ask a question (CLI)
python lmstudio_faiss_rag.py --ask "What are the Enzmann fusion notes about?" --k 6

# Serve an HTTP API (see endpoints below)
python lmstudio_faiss_rag.py --serve --port 8055

HTTP API
--------
GET  /health                              → { status, lm_studio_ok }
GET  /indexes                             → { indexes: [ {name, path, meta} ] }
POST /ingest  { folder, lm_embed_model? } → { saved_to }
POST /ask     { question, index?, k? }    → { answer, citations: [ {doc, source} ] }

Supported file types: .txt, .md, .pdf (easy to extend).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

# FastAPI server bits
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# LangChain I/O
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# OpenAI-compatible client (LM Studio server)
from openai import OpenAI

META_FILENAME = "meta.json"
SUPPORTED_EXTS = {".txt", ".md", ".pdf"}

# ------------------------------
# Utilities & config
# ------------------------------

def die(msg: str):
    print(f"[FATAL] {msg}")
    sys.exit(1)

@dataclass
class Config:
    lm_base_url: str
    lm_api_key: str
    lm_chat_model: str
    embedding_mode: str  # "lmstudio" or "local-st"
    st_model_name: str
    index_root: str
    chunk_size: int
    chunk_overlap: int
    host: str
    port: int


def load_config() -> Config:
    load_dotenv()
    lm_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    lm_chat_model = os.getenv("LMSTUDIO_CHAT_MODEL", "local-chat-model")
    embedding_mode = os.getenv("EMBEDDING_MODE", "lmstudio")  # or "local-st"
    st_model_name = os.getenv("ST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    index_root = os.getenv("INDEX_ROOT", "indexes")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    host = os.getenv("RAG_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_SERVER_PORT", "8055"))
    return Config(
        lm_base_url, lm_api_key, lm_chat_model,
        embedding_mode, st_model_name, index_root,
        chunk_size, chunk_overlap, host, port
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def index_path(index_root: str, emb_model_name: str) -> str:
    safe = emb_model_name.replace("/", "_").replace(" ", "-")
    p = os.path.join(index_root, safe)
    ensure_dir(p)
    return p


# ------------------------------
# Embedding backends
# ------------------------------

class LMStudioEmbeddings:
    """LangChain-compatible wrapper for LM Studio OpenAI-style embeddings."""
    def __init__(self, base_url: str, api_key: str, model: Optional[str] = None):
        self._emb = OpenAIEmbeddings(
            openai_api_base=base_url,
            openai_api_key=api_key,
            model=model or "text-embedding-3-large"
        )
        self.model_name = model or "lmstudio-current-embedding"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)


class LocalSentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0].tolist()


# ------------------------------
# Loading & splitting
# ------------------------------

def load_docs_from_dir(path: str) -> List[Document]:
    docs: List[Document] = []
    for root, _, files in os.walk(path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            fp = os.path.join(root, fn)
            if ext in {".txt", ".md"}:
                loader = TextLoader(fp, encoding="utf-8")
                docs.extend(loader.load())
            elif ext == ".pdf":
                try:
                    loader = PyPDFLoader(fp)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[WARN] Failed to load PDF {fp}: {e}")
    return docs


def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # fixed: escaped newlines
    )
    return splitter.split_documents(docs)


# ------------------------------
# Metadata persistence for indexes
# ------------------------------

def write_index_meta(folder: str, meta: Dict[str, Any]):
    with open(os.path.join(folder, META_FILENAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def read_index_meta(folder: str) -> Optional[Dict[str, Any]]:
    fp = os.path.join(folder, META_FILENAME)
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ------------------------------
# Build / load FAISS vectorstore
# ------------------------------

def build_embeddings_for_ingest(cfg: Config, lm_embed_model_name: Optional[str]) -> (Any, str, Dict[str, Any]):
    if cfg.embedding_mode.lower() == "lmstudio":
        embeddings = LMStudioEmbeddings(cfg.lm_base_url, cfg.lm_api_key, lm_embed_model_name)
        emb_name = embeddings.model_name
        meta = {
            "embedding_backend": "lmstudio",
            "lm_embed_model": emb_name,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        return embeddings, emb_name, meta
    else:
        embeddings = LocalSentenceTransformerEmbeddings(cfg.st_model_name)
        emb_name = embeddings.model_name
        meta = {
            "embedding_backend": "local-st",
            "st_model": emb_name,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        return embeddings, emb_name, meta


def embeddings_from_meta_or_env(cfg: Config, idx_folder: str):
    meta = read_index_meta(idx_folder)
    if meta and meta.get("embedding_backend") == "lmstudio":
        return LMStudioEmbeddings(cfg.lm_base_url, cfg.lm_api_key, meta.get("lm_embed_model"))
    if meta and meta.get("embedding_backend") == "local-st":
        return LocalSentenceTransformerEmbeddings(meta.get("st_model", cfg.st_model_name))
    # fallback to env
    if cfg.embedding_mode.lower() == "lmstudio":
        return LMStudioEmbeddings(cfg.lm_base_url, cfg.lm_api_key)
    return LocalSentenceTransformerEmbeddings(cfg.st_model_name)


def ingest(folder: str, cfg: Config, lm_embed_model_name: Optional[str] = None) -> str:
    print(f"[INGEST] Scanning {folder} ...")
    docs = load_docs_from_dir(folder)
    if not docs:
        die("No documents found. Supported: .txt, .md, .pdf")

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source", d.metadata.get("source", "unknown"))

    chunks = split_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
    print(f"[INGEST] {len(docs)} docs → {len(chunks)} chunks")

    embeddings, emb_name, meta = build_embeddings_for_ingest(cfg, lm_embed_model_name)
    print(f"[INGEST] Using embeddings: {meta}")

    vs = FAISS.from_documents(chunks, embeddings)
    out_dir = index_path(cfg.index_root, emb_name)
    vs.save_local(out_dir)
    write_index_meta(out_dir, meta)
    print(f"[INGEST] Saved index → {out_dir}")
    return out_dir


def load_index(idx_folder: str, cfg: Config) -> FAISS:
    embeddings = embeddings_from_meta_or_env(cfg, idx_folder)
    return FAISS.load_local(idx_folder, embeddings, allow_dangerous_deserialization=True)


def retrieve_context(vs: FAISS, query: str, k: int = 4) -> List[Document]:
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)


def format_context(docs: List[Document]):
    blocks = []
    cites = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[DOC {i}] Source: {src}\n{d.page_content}")  # fixed: \n instead of raw newline
        cites.append({"doc": i, "source": src})
    return "\n\n".join(blocks), cites  # fixed: escaped newlines


def call_lmstudio_chat(cfg: Config, system_prompt: str, user_msg: str) -> str:
    client = OpenAI(base_url=cfg.lm_base_url, api_key=cfg.lm_api_key)
    resp = client.chat.completions.create(
        model=cfg.lm_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return resp.choices[0].message.content or ""


def rag_answer(cfg: Config, idx_path: str, question: str, k: int = 4):
    vs = load_index(idx_path, cfg)
    ctx_docs = retrieve_context(vs, question, k=k)
    if not ctx_docs:
        return {
            "answer": "I didn't find anything relevant in the vector index.",
            "citations": []
        }
    context_text, cites = format_context(ctx_docs)
    system_prompt = (
        "You are a precise research assistant. Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know. Cite sources by their [DOC n] labels."
    )
    user_msg = f"Question: {question}

Context:
{context_text}

Answer:"
    answer = call_lmstudio_chat(cfg, system_prompt, user_msg)
    return {"answer": answer, "citations": cites}


# ------------------------------
# FastAPI app
# ------------------------------

app = FastAPI(title="LM Studio + FAISS RAG")
CFG = load_config()

class IngestReq(BaseModel):
    folder: str
    lm_embed_model: Optional[str] = None

class AskReq(BaseModel):
    question: str
    index: Optional[str] = None
    k: int = 4

@app.get("/health")
async def health():
    # check LM Studio server quickly
    ok = True
    try:
        client = OpenAI(base_url=CFG.lm_base_url, api_key=CFG.lm_api_key)
        # a tiny no-op call (list models may be heavy on some versions; skip). Assume OK if client init didn't raise.
        _ = client  # noqa
    except Exception:
        ok = False
    return {"status": "ok", "lm_studio_ok": ok}

@app.get("/indexes")
async def list_indexes():
    items = []
    if os.path.isdir(CFG.index_root):
        for name in sorted(os.listdir(CFG.index_root)):
            p = os.path.join(CFG.index_root, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "index.pkl")):
                items.append({"name": name, "path": p, "meta": read_index_meta(p)})
    return {"indexes": items}

@app.post("/ingest")
async def api_ingest(req: IngestReq):
    out_dir = ingest(req.folder, CFG, lm_embed_model_name=req.lm_embed_model)
    return {"saved_to": out_dir, "meta": read_index_meta(out_dir)}

@app.post("/ask")
async def api_ask(req: AskReq):
    idx = req.index
    if not idx:
        # auto-pick first available index
        candidates = await list_indexes()
        if not candidates["indexes"]:
            return {"answer": "No indexes found. Ingest first.", "citations": []}
        idx = candidates["indexes"][0]["path"]
    return rag_answer(CFG, idx, req.question, k=req.k)


# ------------------------------
# CLI
# ------------------------------

def list_indexes_cli(cfg: Config):
    print(f"[INDEXES] under {cfg.index_root}")
    if not os.path.isdir(cfg.index_root):
        print("(none)")
        return
    for name in sorted(os.listdir(cfg.index_root)):
        p = os.path.join(cfg.index_root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "index.pkl")):
            meta = read_index_meta(p)
            print(f"- {name}  → {p}
  meta: {json.dumps(meta, indent=2)}")


def main():
    cfg = CFG
    p = argparse.ArgumentParser(description="LM Studio + FAISS RAG (API + CLI)")
    p.add_argument("--ingest", metavar="FOLDER", help="Folder to ingest into FAISS index", default=None)
    p.add_argument("--ask", metavar="QUESTION", help="Ask a RAG question", default=None)
    p.add_argument("--index", metavar="INDEX_PATH", help="Path to FAISS index (folder)", default=None)
    p.add_argument("--k", metavar="K", type=int, default=4, help="# chunks to retrieve")
    p.add_argument("--lm-embed-model", metavar="NAME", default=None, help="Hint for LM Studio embedding model name")
    p.add_argument("--list-indexes", action="store_true", help="List discovered indexes")
    p.add_argument("--serve", action="store_true", help="Run HTTP API server")
    p.add_argument("--host", default=None, help="Server host override")
    p.add_argument("--port", type=int, default=None, help="Server port override")
    args = p.parse_args()

    if args.list_indexes:
        list_indexes_cli(cfg)
        return

    if args.ingest:
        ingest(args.ingest, cfg, lm_embed_model_name=args.lm_embed_model)
        return

    if args.ask:
        if args.index is None:
            # Try to pick a sensible default index
            # Prefer an index that has meta.json, else the first folder with index.pkl
            chosen = None
            if os.path.isdir(cfg.index_root):
                for name in sorted(os.listdir(cfg.index_root)):
                    pth = os.path.join(cfg.index_root, name)
                    if os.path.isdir(pth) and os.path.isfile(os.path.join(pth, "index.pkl")):
                        chosen = pth
                        # prioritize those with meta.json
                        if os.path.isfile(os.path.join(pth, META_FILENAME)):
                            chosen = pth
                            break
            if not chosen:
                die("No index found. Run --ingest first or specify --index.")
            args.index = chosen
        out = rag_answer(cfg, args.index, args.ask, k=args.k)
        print("
=== Answer ===
" + out["answer"]) 
        if out["citations"]:
            print("
Citations:")
            for c in out["citations"]:
                print(f"  [DOC {c['doc']}] {c['source']}")
        return

    if args.serve:
        host = args.host or cfg.host
        port = args.port or cfg.port
        print(f"[SERVE] Starting RAG API on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        return

    print("Nothing to do. Use --ingest, --ask, --list-indexes, or --serve. -h for help.")


if __name__ == "__main__":
    main()


# ------------------------------
# .env TEMPLATE (copy next to this file)
# ------------------------------
"""
# LM Studio server (enable in LM Studio: Settings → Developer → Enable server)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio

# Chat model name exactly as shown in LM Studio (e.g., Llama-3.1-8B-Instruct-GGUF)
LMSTUDIO_CHAT_MODEL=Llama-3.1-8B-Instruct-GGUF

# Embedding mode: "lmstudio" to use LM Studio embeddings; or "local-st" to use sentence-transformers
EMBEDDING_MODE=lmstudio

# Local sentence-transformers model (used when EMBEDDING_MODE=local-st)
ST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Where to store FAISS indexes
INDEX_ROOT=indexes

# Chunking
CHUNK_SIZE=1200
CHUNK_OVERLAP=200

# API server defaults
RAG_SERVER_HOST=127.0.0.1
RAG_SERVER_PORT=8055
"""