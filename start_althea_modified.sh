#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Start LM Studio, Neo4j, FAISS-backed RAG service, and Streamlit chat UI
# - Starts LM Studio (if not already running)
# - Verifies/notes FAISS index path
# - Starts Neo4j service (best effort, no sudo prompts)
# - Starts lmstudio_faiss_rag.py as a background API service
# - Launches Streamlit app: /home/frea/FREA/lmstudio_rag_chat.py
#
# All warnings and messages are preserved and expanded for clarity.
###############################################################################

# --- CONFIG ---
STREAMLIT_APP="/home/frea/FREA/lmstudio_rag_chat.py"
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-/home/frea/FREA/faiss_index}"
RAG_API_PY="/home/frea/FREA/lmstudio_faiss_rag.py"
RAG_API_PORT="${RAG_API_PORT:-8055}"
LMSTUDIO_BASE_URL="${LMSTUDIO_BASE_URL:-http://127.0.0.1:1234/v1}"

# Log dir
LOG_DIR="${HOME}/.cache/local_althea/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
LMSTUDIO_LOG="${LOG_DIR}/lmstudio-${TS}.log"
RAG_API_LOG="${LOG_DIR}/lmstudio_faiss_rag-${TS}.log"
STARTUP_LOG="${LOG_DIR}/startup-${TS}.log"

# --- helpers ---
info(){ echo "ℹ️  $*"; }
ok(){ echo "✅ $*"; }
warn(){ echo "⚠️  $*"; }
err(){ echo "❌ $*" >&2; }

# Record to a startup log (tee keeps messages on screen)
exec > >(tee -a "${STARTUP_LOG}") 2>&1

info "Starting Local Althea stack…"

# -----------------------------
# 1) Start LM Studio (best effort)
# -----------------------------
is_llm_studio_running() {
  pgrep -f "LM[- ]?Studio|lmstudio" >/dev/null 2>&1
}

start_lmstudio() {
  # Try common install locations
  CANDIDATES=(
    "${HOME}"/Downloads/LM-Studio-*.AppImage
    "${HOME}"/LM-Studio-*.AppImage
    "${HOME}"/.local/bin/LM\ Studio
    "/usr/bin/LM Studio"
    "/opt/LM Studio/LM Studio"
  )
  for c in "${CANDIDATES[@]}"; do
    # Expand globs safely
    for bin in $c; do
      [[ -e "$bin" ]] || continue
      info "Attempting to start LM Studio from: $bin"
      nohup "$bin" >> "${LMSTUDIO_LOG}" 2>&1 & disown || continue
      sleep 5
      if is_llm_studio_running; then
        ok "LM Studio started successfully. Logs: ${LMSTUDIO_LOG}"
        return 0
      fi
    done
  done
  warn "Could not auto-start LM Studio. Please start it manually. Logs: ${LMSTUDIO_LOG}"
  return 1
}

if is_llm_studio_running; then
  ok "LM Studio already running."
else
  info "LM Studio not running — starting…"
  start_lmstudio || warn "LM Studio may not be running yet. Continuing…"
fi

info "Waiting 5 seconds for LM Studio API to be ready at ${LMSTUDIO_BASE_URL}…"
sleep 5

# -----------------------------
# 2) Check FAISS index directory
# -----------------------------
if [[ -d "${FAISS_INDEX_DIR}" ]]; then
  ok "FAISS index directory found: ${FAISS_INDEX_DIR}"
else
  warn "FAISS index directory not found at: ${FAISS_INDEX_DIR}"
  warn "Proceeding anyway — the RAG service may build or require this path."
fi

# -----------------------------
# 3) Start Neo4j (best effort, no sudo prompt)
# -----------------------------
start_neo4j() {
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl --user is-active neo4j >/dev/null 2>&1; then
      ok "Neo4j (user) is already active."
      return 0
    fi
    if systemctl is-active neo4j >/dev/null 2>&1; then
      ok "Neo4j (system) is already active."
      return 0
    fi
    info "Attempting to start Neo4j (user)…"
    if systemctl --user start neo4j >/dev/null 2>&1; then
      sleep 2
      systemctl --user is-active neo4j >/dev/null 2>&1 && ok "Neo4j (user) started." && return 0
    fi
    info "Attempting to start Neo4j (system, no-ask-password)…"
    if systemctl --no-ask-password start neo4j >/dev/null 2>&1; then
      sleep 2
      systemctl is-active neo4j >/dev/null 2>&1 && ok "Neo4j (system) started." && return 0
    fi
    warn "Failed to auto-start Neo4j (no sudo used). If needed, run: sudo systemctl start neo4j"
    return 1
  else
    warn "systemctl not available; skipping Neo4j auto-start."
    return 1
  fi
}

start_neo4j || warn "Neo4j may not be running. The Streamlit app will try to connect and report errors."

# -----------------------------
# 4) Start lmstudio_faiss_rag.py API (background)
# -----------------------------
if [[ -f "${RAG_API_PY}" ]]; then
  if pgrep -f "python.*${RAG_API_PY}.*--serve" >/dev/null 2>&1; then
    ok "lmstudio_faiss_rag.py service already running."
  else
    info "Starting lmstudio_faiss_rag.py API on port ${RAG_API_PORT}…"
    # Keep same warnings/messages style
    nohup python3 "${RAG_API_PY}" --serve --port "${RAG_API_PORT}" >> "${RAG_API_LOG}" 2>&1 & disown || {
      err "Failed to start lmstudio_faiss_rag.py API. See logs: ${RAG_API_LOG}"
    }
    sleep 2
    if pgrep -f "python3.*${RAG_API_PY}.*--serve" >/dev/null 2>&1; then
      ok "lmstudio_faiss_rag.py API started. Logs: ${RAG_API_LOG}"
    else
      warn "lmstudio_faiss_rag.py API may not have started correctly. Logs: ${RAG_API_LOG}"
    fi
    
  fi
else
  warn "RAG API script not found at: ${RAG_API_PY}"
fi

# -----------------------------
# 5) Launch Streamlit app (foreground)
# -----------------------------
if [[ -f "${STREAMLIT_APP}" ]]; then
  info "🚀 Launching Streamlit app: ${STREAMLIT_APP}"
  info "📜 Startup log: ${STARTUP_LOG}"
  # Streamlit runs in the foreground; when you Ctrl+C, only Streamlit stops.
  # Background services (LM Studio / RAG API) keep running.
  exec streamlit run "${STREAMLIT_APP}"
else
  err "Streamlit app not found: ${STREAMLIT_APP}"
  exit 1
fi


python3 auto_run_if_idle.py && echo "✅ All services started."
else
  err "Failed to start all services. Check logs for details."
  exit 1
fi

# Run auto_run_if_idle.py visibly in terminal
python3 auto_run_if_idle.py && echo "✅ All services started."
else
  echo "❌ Failed to start all services. Check logs for details." >&2
  exit 1
fi
