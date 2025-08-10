#!/bin/bash

# --- CONFIG ---
APP_PATH="/home/frea/FREA/lmstudio_rag_chat.py
FAISS_INDEX_DIR="faiss_index"
NEO4J_SERVICE="neo4j"

# Function to check if LLM Studio is running
is_llm_studio_running() {
  pgrep -f "lmstudio" > /dev/null
  return $?
}

# Start LLM Studio if not running
if is_llm_studio_running; then
  echo "✅ LLM Studio is already running."
else
  echo "🚀 Starting LLM Studio..."
  nohup lmstudio > /dev/null 2>&1 &
  sleep 5
  if is_llm_studio_running; then
    echo "✅ LLM Studio started successfully."
  else
    echo "❌ Failed to start LLM Studio."
  fi
fi

echo "🔌 Ensuring Neo4j is running..."
sudo systemctl start neo4j

sleep 5

if systemctl is-active --quiet neo4j; then
  echo "✅ Neo4j is running."
else
  echo "❌ Failed to start Neo4j."
fi

echo "📦 Checking FAISS index..."
if [ ! -d "$FAISS_INDEX_DIR" ]; then
    echo "🧠 Building FAISS index..."
    python3 -c "
from faiss_llama3_chat_app import build_or_update_vectorstore
build_or_update_vectorstore()
"
else
    echo "✅ FAISS index already exists."
fi

# Start auto_run_if_idle.py in its own tmux session
if ! tmux has-session -t idlecheck 2>/dev/null; then
  echo "Starting auto_run_if_idle.py in tmux session 'idlecheck'..."
  tmux new-session -d -s idlecheck "python3 /home/frea/FREA/auto_run_if_idle.py"
else
  echo "auto_run_if_idle.py already running in tmux session 'idlecheck'."
fi

echo "🌐 Launching Streamlit app..."
streamlit run "$APP_PATH"

# === Added functionality ===
# Run lmstudio_faiss_rag.py in the background on port 8055
echo "Starting lmstudio_faiss_rag.py on port 8055 in the background..."
nohup python lmstudio_faiss_rag.py --serve --port 8055 > lmstudio_faiss_rag.log 2>&1 &
echo "Process started with PID $!"



set -euo pipefail

# --- Set this if you know the exact file name (recommended) ---
# LM_STUDIO_APPIMAGE="$HOME/Downloads/LM-Studio-0.3.0-x86_64.AppImage"

DOWNLOADS_DIR="$HOME/Downloads"

# If not hardcoded, pick the newest matching AppImage in Downloads
if [ -z "${LM_STUDIO_APPIMAGE:-}" ]; then
  echo "Searching for LM Studio AppImage in $DOWNLOADS_DIR ..."
  LM_STUDIO_APPIMAGE="$(
    find "$DOWNLOADS_DIR" -maxdepth 1 -type f -iname "*lm*studio*.AppImage" \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-
  )"
fi

if [ -z "${LM_STUDIO_APPIMAGE:-}" ] || [ ! -f "$LM_STUDIO_APPIMAGE" ]; then
  echo "❌ Error: No LM Studio AppImage found in $DOWNLOADS_DIR."
  echo "Tip: put it in $DOWNLOADS_DIR or set LM_STUDIO_APPIMAGE to the exact path."
  exit 1
fi

echo "✅ Using: $LM_STUDIO_APPIMAGE"

# Make sure it’s executable
chmod +x "$LM_STUDIO_APPIMAGE"

# Avoid duplicate instances
if pgrep -f "$(basename "$LM_STUDIO_APPIMAGE")" >/dev/null 2>&1; then
  echo "⚠️  LM Studio already appears to be running."
  exit 0
fi

# Launch in background with minimal logging
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/lm-studio"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/lm-studio.log"

echo "🚀 Launching LM Studio..."
nohup "$LM_STUDIO_APPIMAGE" >>"$LOG_FILE" 2>&1 & disown

sleep 2
if pgrep -f "$(basename "$LM_STUDIO_APPIMAGE")" >/dev/null 2>&1; then
  echo "✅ LM Studio started successfully."
else
  echo "❌ Failed to start LM Studio. Check logs at: $LOG_FILE"
fi

echo "📜 Logs: $LOG_FILE"

echo "⏳ Waiting 5 seconds for LM Studio API to be ready..."
sleep 5

# --- Launch Streamlit app ---
echo "🚀 Launching Streamlit Neo4j KG Answer app..."
streamlit run neo4j_kg_answer.py
