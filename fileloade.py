import os
import sys
import json
from pathlib import Path
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredImageLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OpenAIEmbeddings
from PIL import Image
import torch
import cv2
import numpy as np
from transformers import pipeline
import pytesseract
import re
import streamlit as st
import threading
import time
from functions.yolo8.segmenter import segment_image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Setup paths and environment
INPUT_DIR = Path("E:/Astronomy/Envelope102")
PROCESSED_TRACKER = Path(".processed_files.json")
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Initialize LLM image captioning
describer = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Initialize vector DB and embedding model
embedding_model = OpenAIEmbeddings()
vectorstore = Neo4jVector(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    embedding=embedding_model,
    database="neo4j",
    index_name="documents"
)

# Load or initialize processed file tracker
if PROCESSED_TRACKER.exists():
    with open(PROCESSED_TRACKER, "r") as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()

def save_processed(filepath):
    processed_files.add(filepath)
    with open(PROCESSED_TRACKER, "w") as f:
        json.dump(list(processed_files), f)

def document_already_processed(source_path: str):
    results = vectorstore.similarity_search("", k=1, filter={"source": source_path})
    return any(doc.metadata.get("source") == source_path for doc in results)

def classify_text(text):
    math_patterns = [r'\\?int', r'sin\\(|cos\\(|tan\\(', r'=\\s*[^ ]+', r'\\d+\\s*[+\-*/^]\\s*\\d+']
    chem_patterns = [r'[A-Z][a-z]?[0-9]*', r'\\+|\\->|\\(|\\)']
    diagram_keywords = ["diagram", "flowchart", "chart", "illustration"]
    handwriting_keywords = ["handwriting", "handwritten", "pen", "ink"]

    math_score = sum(bool(re.search(p, text)) for p in math_patterns)
    chem_score = sum(bool(re.search(p, text)) for p in chem_patterns)
    text_lower = text.lower()
    diagram_score = sum(kw in text_lower for kw in diagram_keywords)
    handwriting_score = sum(kw in text_lower for kw in handwriting_keywords)

    if math_score > max(chem_score, diagram_score, handwriting_score):
        return "math_equation"
    elif chem_score > max(math_score, diagram_score, handwriting_score):
        return "chemical_equation"
    elif diagram_score > 0:
        return "diagram"
    elif handwriting_score > 0:
        return "handwriting"
    return "unknown"

def process_document(filepath: Path):
    if str(filepath) in processed_files:
        print(f"Skipping {filepath}, already marked processed.")
        return
    if document_already_processed(str(filepath)):
        print(f"Skipping {filepath}, already in vectorstore.")
        save_processed(str(filepath))
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

def process_image(filepath: Path):
    seg_result = segment_image(str(filepath))
    original_image = cv2.imread(str(filepath))

    for i, box in enumerate(seg_result["boxes"]):
        x1, y1, x2, y2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
        cropped = original_image[y1:y2, x1:x2]

        segment_path = filepath.with_name(f"{filepath.stem}_seg_{i}.jpg")
        cv2.imwrite(str(segment_path), cropped)

        # Save mask overlay
        mask = np.array(seg_result["masks"][i], dtype=np.uint8)
        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        color_mask = np.stack([mask_resized * 255] * 3, axis=-1)
        overlay = cv2.addWeighted(cropped, 0.8, color_mask, 0.2, 0)
        overlay_path = filepath.with_name(f"{filepath.stem}_overlay_{i}.jpg")
        cv2.imwrite(str(overlay_path), overlay)

        pil_img = Image.fromarray(cropped)
        ocr_text = pytesseract.image_to_string(pil_img).strip()

        if len(ocr_text) < 20:
            try:
                ocr_text = describer(pil_img)[0]['generated_text']
            except Exception:
                ocr_text = "No readable content."

        classification = classify_text(ocr_text)

        metadata = {
            "source": str(filepath),
            "segment_path": str(segment_path),
            "overlay_path": str(overlay_path),
            "bounding_box": [x1, y1, x2, y2],
            "confidence": box["confidence"],
            "class_name": seg_result["class_names"][box["class_id"]],
            "classification": classification
        }

        langchain_doc = [{"page_content": ocr_text, "metadata": metadata}]
        vectorstore.add_documents(langchain_doc)

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
            if PROCESSED_TRACKER.exists():
                PROCESSED_TRACKER.unlink()
            st.experimental_rerun()
    else:
        st.info("No files vectorized yet.")

def run_idle_vectorization():
    while True:
        if not any([cv2.waitKey(1) & 0xFF == ord('q')]):
            for filepath in INPUT_DIR.glob("**/*"):
                if filepath.is_file():
                    try:
                        print(f"Processing {filepath}")
                        process_document(filepath)
                    except Exception as e:
                        print(f"Failed to process {filepath}: {e}")
        time.sleep(60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        show_dashboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "idle":
        run_idle_vectorization()
    else:
        for filepath in INPUT_DIR.glob("**/*"):
            if filepath.is_file():
                try:
                    print(f"Processing {filepath}")
                    process_document(filepath)
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")
