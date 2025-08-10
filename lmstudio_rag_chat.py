#!/usr/bin/env python3
"""
LM Studio RAG Chat (Streamlit)
--------------------------------
Single-file Streamlit app that chats with a **locally running LM Studio API** and
optionally uses a **FAISS RAG microservice** (your `lmstudio_faiss_rag.py --serve --port 8055`).

Features
- Chat UI backed by LM Studio (`/v1/chat/completions`) or your RAG service (`/ask`).
- Upload and ingest files (pdf, txt, md, images, etc.) into your FAISS index folder.
- Use the **system prompt configured in LM Studio** (Althea tab) by default; optionally override.
- One-click **knowledge graph** of the latest answer or the entire conversation (opens in new tab).

Quick start
1) Start LM Studio local server (e.g. http://127.0.0.1:1234) with a model running.
2) Start your RAG server: `python lmstudio_faiss_rag.py --serve --port 8055`.
3) `pip install streamlit requests pyvis neo4j` (neo4j only needed if you later wire to a live DB).
4) Run: `streamlit run lmstudio_rag_chat.py`.

Notes
- By default we **do not** send a system message; LM Studio will use the one you set in the Althea tab.
  Use the sidebar toggle to add/override a system prompt locally if desired.
- For file ingestion: we save uploads under the FAISS index folder, then (optionally) call `POST /ingest` on
  your RAG service if that endpoint exists in your implementation. If not, you can trigger your own reindexing
  flow externally.
"""

import os
import io
import json
import time
import uuid
import pathlib
import traceback
from typing import List, Dict, Any

import requests
import streamlit as st
from pyvis.network import Network
from streamlit.components.v1 import html

# =====================
# Page config FIRST
# =====================
st.set_page_config(page_title="LM Studio RAG Chat", layout="wide")

# =====================
# Constants / Defaults
# =====================
DEFAULT_LMS_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
DEFAULT_LMS_API_KEY  = os.getenv("LMSTUDIO_API_KEY", "")  # usually not required locally
DEFAULT_LMS_MODEL    = os.getenv("LMSTUDIO_MODEL", "local-model")

DEFAULT_RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8055")
DEFAULT_FAISS_DIR    = os.getenv("FAISS_INDEX_DIR", str(pathlib.Path("faiss_index").absolute()))
UPLOADS_SUBDIR       = "uploads"  # will be created inside FAISS dir

# =====================
# Helpers
# =====================

def ensure_dirs(base_dir: str) -> str:
    p = pathlib.Path(base_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    (p / UPLOADS_SUBDIR).mkdir(parents=True, exist_ok=True)
    return str(p)


def lmstudio_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call LM Studio's OpenAI-compatible /v1/chat/completions endpoint.
    Returns the raw JSON response.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def rag_ask(rag_base_url: str, question: str, k: int = 5, timeout: int = 180) -> Dict[str, Any]:
    """POST to your RAG service /ask endpoint. Expected response shape:
    {"answer": str, "sources": [ {"source": "...", "snippet": "..."}, ... ]}
    """
    url = rag_base_url.rstrip("/") + "/ask"
    payload = {"question": question, "k": int(k)}
    resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def try_rag_ingest(rag_base_url: str, file_path: str) -> Dict[str, Any]:
    """Optional: call your RAG server ingestion route if implemented.
    Returns a dict with status.
    """
    url = rag_base_url.rstrip("/") + "/ingest"
    try:
        data = {"path": file_path}
        resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=300)
        if resp.status_code == 404:
            return {"ok": False, "message": "RAG /ingest not implemented (404). File saved only."}
        resp.raise_for_status()
        return {"ok": True, "message": resp.text}
    except Exception as e:
        return {"ok": False, "message": f"Ingest failed: {e}"}


def extract_triples_with_lmstudio(base_url: str, api_key: str, model: str, text: str, timeout: int = 120) -> List[Dict[str, str]]:
    """Ask LM Studio to extract (s, p, o) triples from text. Returns a list of dicts.
    We use a strict output format request to reduce parsing failures.
    """
    system_prompt = (
        "You are an information extraction tool. Given TEXT, extract semantic triples as a JSON array. "
        "Each item is an object with keys 's', 'p', and 'o' for subject, predicate, and object. "
        "Return ONLY valid JSON. If no triples, return []."
    )
    user_prompt = f"""
    TEXT:\n{text}

    Return JSON like: [{{"s":"Subject","p":"predicate","o":"Object"}}]
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        res = lmstudio_chat(base_url, api_key, model, messages, temperature=0.0, max_tokens=800, timeout=timeout)
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Strip code fences if the model wrapped JSON
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`\n ")
            # remove a possible language hint like ```json
            if content.lower().startswith("json"):
                content = content[4:].strip()
        triples = json.loads(content)
        # validate
        cleaned = []
        for t in triples if isinstance(triples, list) else []:
            s = str(t.get("s", "")).strip()
            p = str(t.get("p", "")).strip()
            o = str(t.get("o", "")).strip()
            if s and p and o:
                cleaned.append({"s": s, "p": p, "o": o})
        return cleaned
    except Exception:
        return []


def build_kg_html(triples: List[Dict[str, str]], title: str = "Knowledge Graph") -> str:
    """Build a PyVis HTML graph and return the file path."""
    net = Network(height="700px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # add nodes + edges
    def add_node(nid: str):
        net.add_node(nid, label=nid, title=nid)

    for t in triples:
        s, p, o = t.get('s'), t.get('p'), t.get('o')
        if not (s and p and o):
            continue
        add_node(s)
        add_node(o)
        net.add_edge(s, o, label=p, title=p)

    # save
    out_dir = pathlib.Path("kg_html").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"kg_{uuid.uuid4().hex}.html"
    out_path = str(out_dir / fname)
    net.set_options("""
    var options = {
      nodes: { shape: 'dot', size: 12, font: { size: 14 } },
      edges: { arrows: { to: { enabled: true } }, font: { size: 12, align: 'top' } },
      physics: { stabilization: true }
    };
    """)
    net.show(out_path)
    return out_path


# =====================
# Sidebar Controls
# =====================
with st.sidebar:
    st.header("⚙️ Settings")
    lms_base = st.text_input("LM Studio Base URL", DEFAULT_LMS_BASE_URL)
    lms_model = st.text_input("LM Studio Model", DEFAULT_LMS_MODEL)
    lms_key   = st.text_input("LM Studio API Key (optional)", DEFAULT_LMS_API_KEY, type="password")

    use_rag = st.toggle("Use RAG (FAISS via lmstudio_faiss_rag.py)", value=True)
    rag_base = st.text_input("RAG Base URL", DEFAULT_RAG_BASE_URL)
    k_docs   = st.slider("Top-K documents", 1, 20, 6)

    st.divider()
    st.caption("System prompt")
    use_local_system = st.toggle("Override LM Studio system prompt here", value=False, help="If off, the system prompt you set in LM Studio (Althea tab) is used.")
    local_system_prompt = ""
    if use_local_system:
        local_system_prompt = st.text_area("System prompt override", value="You are a helpful research assistant.")

    st.divider()
    faiss_dir = st.text_input("FAISS index folder", DEFAULT_FAISS_DIR)
    ensure_dirs(faiss_dir)

    st.divider()
    st.caption("Uploads → saved under FAISS index folder /uploads")
    up_files = st.file_uploader("Add files to index", accept_multiple_files=True)
    if up_files:
        saved_paths = []
        for uf in up_files:
            try:
                # Save preserving original filename
                dest = pathlib.Path(faiss_dir) / UPLOADS_SUBDIR / uf.name
                with open(dest, 'wb') as f:
                    f.write(uf.getbuffer())
                saved_paths.append(str(dest))
            except Exception as e:
                st.error(f"Failed saving {uf.name}: {e}")
        if saved_paths:
            st.success(f"Saved {len(saved_paths)} file(s).")
            if st.button("Notify RAG to ingest (if supported)"):
                msgs = []
                for p in saved_paths:
                    res = try_rag_ingest(rag_base, p)
                    msgs.append(f"{os.path.basename(p)} → {res['message']}")
                st.info("\n".join(msgs))

# =====================
# Session State
# =====================
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {role, content}
if "triples" not in st.session_state:
    st.session_state.triples = []  # accumulate triples across turns

# =====================
# Header / Layout
# =====================
st.title("🧭 LM Studio RAG Chat")
st.caption("Chat with LM Studio locally. Toggle FAISS RAG. Build knowledge graphs from answers.")

# Display transcript
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input area
prompt = st.chat_input("Ask something…")

# Handle new user message
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for LM Studio if not using RAG
    messages = []
    if use_local_system and local_system_prompt.strip():
        messages.append({"role": "system", "content": local_system_prompt.strip()})
    # Include prior conversation for context
    for m in st.session_state.chat:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    # Call backend
    answer_text = ""
    sources = []
    error_text = ""
    try:
        if use_rag:
            res = rag_ask(rag_base, prompt, k=k_docs)
            answer_text = res.get("answer", "") or res.get("result", "")
            sources = res.get("sources", []) or res.get("docs", [])
        else:
            res = lmstudio_chat(lms_base, lms_key, lms_model, messages, temperature=0.2, max_tokens=1024)
            answer_text = res.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        error_text = f"Request failed: {e}\n\n{traceback.format_exc()}"

    if error_text:
        with st.chat_message("assistant"):
            st.error(error_text)
        st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {error_text}"})
    else:
        # Show answer + optional sources
        with st.chat_message("assistant"):
            st.markdown(answer_text or "(no content)")
            if sources:
                with st.expander("Sources / retrieved context"):
                    for s in sources:
                        # Allow various shapes
                        src = s.get("source") if isinstance(s, dict) else None
                        snippet = s.get("snippet") if isinstance(s, dict) else None
                        st.markdown(f"**{src or 'source'}**\n\n{snippet or s}")
        st.session_state.chat.append({"role": "assistant", "content": answer_text})

        # Try to extract triples for the answer and store them (best effort)
        triples = extract_triples_with_lmstudio(lms_base, lms_key, lms_model, answer_text)
        if triples:
            st.session_state.triples.extend(triples)

# =====================
# Actions row
# =====================
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    if st.button("🧠 Graph latest answer"):
        # find the last assistant msg
        last = next((m for m in reversed(st.session_state.chat) if m["role"] == "assistant"), None)
        if not last:
            st.warning("No assistant answer yet.")
        else:
            triples = extract_triples_with_lmstudio(lms_base, lms_key, lms_model, last["content"]) or []
            if not triples:
                st.info("No triples found in the last answer.")
            else:
                html_path = build_kg_html(triples, title="Answer Graph")
                st.success("Graph built.")
                st.markdown(f"<a href='file://{html_path}' target='_blank'>Open graph in new tab</a>", unsafe_allow_html=True)

with colB:
    if st.button("🌐 Graph conversation"):
        triples = st.session_state.triples
        if not triples:
            st.info("No triples collected yet.")
        else:
            html_path = build_kg_html(triples, title="Conversation Graph")
            st.success("Conversation graph built.")
            st.markdown(f"<a href='file://{html_path}' target='_blank'>Open graph in new tab</a>", unsafe_allow_html=True)

with colC:
    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.session_state.triples = []
        st.experimental_rerun()

with colD:
    st.write("")
    st.write("")
    st.caption("Tip: Use sidebar to toggle RAG and manage uploads.")

# =====================
# Footer
# =====================
st.divider()
st.caption("Local-only app. Keep your LM Studio server and RAG service running while you chat.")
