#!/bin/bash

# --- CONFIG ---
APP_PATH="/home/frea/FREA/faiss_llama3_chat_app.py"
FAISS_INDEX_DIR="faiss_index"
NEO4J_SERVICE="neo4j"

echo "🔍 Checking Neo4j status..."
if ! systemctl is-active --quiet $NEO4J_SERVICE; then
    echo "🚀 Starting Neo4j service..."
    sudo systemctl start $NEO4J_SERVICE
else
    echo "✅ Neo4j is already running."
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

echo "🌐 Launching Streamlit app..."
streamlit run "$APP_PATH"
