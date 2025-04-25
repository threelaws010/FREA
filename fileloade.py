import os
import sys
import json
import re
import time
import threading
from pathlib import Path
from PIL import Image
from flask import Flask
import streamlit as st
import cv2
import numpy as np
import pytesseract
import torch
from transformers import pipeline
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_community.document_loaders import UnstructuredFileLoader
from functions.yolo8.segmenter import segment_image

# Constants
INPUT_DIR = Path("/data/input")
PROCESSED_TRACKER = Path("/data/.processed_files.json")
NEO4J_URL = "bolt://host.docker.internal:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Initialize services
describer = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", use_fast=True)
embedding_model = OpenAIEmbeddings()
vectorstore = Neo4jVector(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    embedding=embedding_model,
    database="neo4j",
    index_name="documents"
)

# Load processed tracker
processed_files = set()
if PROCESSED_TRACKER.exists():
    with open(PROCESSED_TRACKER, "r") as f:
        processed_files = set(json.load(f))

def save_processed(filepath):
    processed_files.add(filepath)
    with open(PROCESSED_TRACKER, "w") as f:
        json.dump(list(processed_files), f)

def document_already_processed(source_path):
    results = vectorstore.similarity_search("", k=1, filter={"source": source_path})
    return any(doc.metadata.get("source") == source_path for doc in results)

def classify_text(text):
    patterns = {
        "math_equation": [r'\\?int', r'sin\\(|cos\\(|tan\\(', r'=\\s*[^ ]+', r'\\d+\\s*[+\-*/^]\\s*\\d+'],
        "chemical_equation": [r'[A-Z][a-z]?[0-9]*', r'\\+|\\->|\\(|\\)'],
    }
    keywords = {
        "diagram": ["diagram", "flowchart", "chart", "illustration"],
        "handwriting": ["handwriting", "handwritten", "pen", "ink"]
    }
    scores = {k: sum(bool(re.search(p, text)) for p in v) for k, v in patterns.items()}
    text_lower = text.lower()
    scores.update({k: sum(kw in text_lower for kw in v) for k, v in keywords.items()})
    return max(scores, key=scores.get, default="unknown")

def process_image(filepath):
    seg_result = segment_image(str(filepath))
    original_image = cv2.imread(str(filepath))

    for i, box in enumerate(seg_result["boxes"]):
        x1, y1, x2, y2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
        cropped = original_image[y1:y2, x1:x2]
        segment_path = filepath.with_name(f"{filepath.stem}_seg_{i}.jpg")
        overlay_path = filepath.with_name(f"{filepath.stem}_overlay_{i}.jpg")
        cv2.imwrite(str(segment_path), cropped)
        mask = np.array(seg_result["masks"][i], dtype=np.uint8)
        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
        overlay = cv2.addWeighted(cropped, 0.8, np.stack([mask_resized * 255] * 3, axis=-1), 0.2, 0)
        cv2.imwrite(str(overlay_path), overlay)

        pil_img = Image.fromarray(cropped)
        ocr_text = pytesseract.image_to_string(pil_img).strip()
        if len(ocr_text) < 20:
            try:
                ocr_text = describer(pil_img)[0]['generated_text']
            except:
                ocr_text = "No readable content."

        doc = {
            "page_content": ocr_text,
            "metadata": {
                "source": str(filepath),
                "segment_path": str(segment_path),
                "overlay_path": str(overlay_path),
                "bounding_box": [x1, y1, x2, y2],
                "confidence": box["confidence"],
                "class_name": seg_result["class_names"][box["class_id"]],
                "classification": classify_text(ocr_text)
            }
        }
        vectorstore.add_documents([doc])

def process_document(filepath):
    if str(filepath) in processed_files or document_already_processed(str(filepath)):
        return
    if filepath.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        process_image(filepath)
    else:
        loader = UnstructuredFileLoader(str(filepath))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(filepath)
        vectorstore.add_documents(docs)
    save_processed(str(filepath))

def run_idle_vectorization():
    while True:
        for filepath in INPUT_DIR.glob("**/*"):
            if filepath.is_file():
                try:
                    print(f"Processing {filepath}")
                    process_document(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
        time.sleep(60)

def show_dashboard():
    st.set_page_config(page_title="Vectorization Dashboard", layout="wide")
    st.title("📊 Vectorization Control Panel")
    st.metric("✅ Files Vectorized", len(processed_files))
    if processed_files:
        with st.expander("📁 View Vectorized Files"):
            for f in sorted(processed_files):
                st.write(f)
        if st.button("🗑️ Clear Vector Cache"):
            processed_files.clear()
            if PROCESSED_TRACKER.exists(): PROCESSED_TRACKER.unlink()
            st.experimental_rerun()
    else:
        st.info("No files vectorized yet.")

# Healthcheck
health_app = Flask(__name__)
@health_app.route("/health")
def health():
    return "ok", 200

def start_health_server():
    threading.Thread(target=lambda: health_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True).start()

if __name__ == "__main__":
    # Only run idle mode here
    start_health_server()
    run_idle_vectorization()
