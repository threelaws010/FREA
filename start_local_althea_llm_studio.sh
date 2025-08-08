#!/bin/bash

# --- CONFIG ---
APP_PATH="/home/frea/FREA/faiss_LLM_Studio_chat_app.py"
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
