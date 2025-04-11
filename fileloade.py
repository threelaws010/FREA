import os
import json
from pathlib import Path
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredImageLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OpenAIEmbeddings
from PIL import Image
import torch
import cv2
import numpy as np
from yolov5 import YOLOv5
from transformers import pipeline
import pytesseract
import re

# Setup paths and environment
INPUT_DIR = Path("./data")
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Initialize YOLOv5 model
yolo = YOLOv5("./yolov5s.pt", device="cuda" if torch.version.hip else "cpu")

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

def document_already_processed(source_path: str):
    """Check Neo4j if document with source path already exists."""
    results = vectorstore.similarity_search("", k=1, filter={"source": source_path})
    return any(doc.metadata.get("source") == source_path for doc in results)

def classify_text(text):
    math_patterns = [r'\\?int', r'sin\\(|cos\\(|tan\\(', r'=\\s*[^ ]+', r'\\d+\\s*[+\-*/^]\\s*\\d+']
    chem_patterns = [r'[A-Z][a-z]?[0-9]*', r'\\+|\\->|\\(|\\)']

    math_score = sum(bool(re.search(p, text)) for p in math_patterns)
    chem_score = sum(bool(re.search(p, text)) for p in chem_patterns)

    if math_score > chem_score and math_score > 0:
        return "math_equation"
    elif chem_score > math_score and chem_score > 0:
        return "chemical_equation"
    return "unknown"

def process_document(filepath: Path):
    if document_already_processed(str(filepath)):
        print(f"Skipping {filepath}, already in vectorstore.")
        return

    if filepath.suffix.lower() == ".jpg":
        return process_image(filepath)
    else:
        loader = UnstructuredFileLoader(str(filepath))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(filepath)
        vectorstore.add_documents(docs)

def process_image(filepath: Path):
    image = cv2.imread(str(filepath))
    results = yolo.predict(image)

    for i, det in enumerate(results.xyxy[0]):
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cropped = image[y1:y2, x1:x2]
        segment_path = filepath.with_name(f"{filepath.stem}_seg_{i}.jpg")
        cv2.imwrite(str(segment_path), cropped)

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
            "bounding_box": [x1, y1, x2, y2],
            "classification": classification
        }
        langchain_doc = [{"page_content": ocr_text, "metadata": metadata}]

        vectorstore.add_documents(langchain_doc)

if __name__ == "__main__":
    for filepath in INPUT_DIR.glob("**/*"):
        if filepath.is_file():
            try:
                print(f"Processing {filepath}")
                process_document(filepath)
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")
