import os
from dotenv import load_dotenv
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import csv
from collections import Counter
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2
import easyocr
from PIL import Image
from transformers import pipeline

# Load environment variables
INPUT_DIR = os.getenv('INPUT_DIR', 'E:/test/sample')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'E:/test/MD')
UNKNOWN_CSV_FILENAME = os.getenv('UNKNOWN_CSV', 'unknown_segments.csv')
STATUS_CSV_FILENAME = os.getenv('STATUS_CSV', 'status.csv')

# Setup
import torch

# Smart device selection (Nvidia, AMD ROCm, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.version, "hip") and torch.version.hip:
    device = torch.device("cuda")  # AMD ROCm shows as "cuda"
else:
    device = torch.device("cpu")

print(f"Selected device: {device}")

# Initialize models
yolo_model = YOLO('yolov8l-seg.pt')
yolo_model.to(device)

ocr_reader = easyocr.Reader(
    ['en'],
    recog_network='english_g2',
    gpu=(device.type == "cuda")
)

captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=0 if device.type == "cuda" else -1
)


# Initialize models
yolo_model = YOLO('yolov8l-seg.pt')
yolo_model.to(device)
ocr_reader = easyocr.Reader(['en'], recog_network='english_g2', gpu=torch.cuda.is_available())
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=0 if torch.cuda.is_available() else -1
)

# Help Menu
if '--help' in sys.argv:
    print("""
Usage: python make_dm_input.py [OPTIONS]

Options:
  --force             Reprocess all files even if Markdown already exists.
  --date YYYY-MM-DD   Only reprocess files older than given date.
  --y-thresh VALUE    Vertical merge threshold (default: 30 pixels).
  --x-thresh VALUE    Horizontal merge threshold (default: 50 pixels).
  --help              Show this help message and exit.
""")
    sys.exit(0)

# Parse command line arguments
force = '--force' in sys.argv
if '--date' in sys.argv:
    date_index = sys.argv.index('--date')
    try:
        cutoff_date = datetime.strptime(sys.argv[date_index + 1], "%Y-%m-%d")
    except (IndexError, ValueError):
        print("Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)
else:
    cutoff_date = None

y_thresh = 30
x_thresh = 50
if '--y-thresh' in sys.argv:
    y_thresh_index = sys.argv.index('--y-thresh')
    try:
        y_thresh = int(sys.argv[y_thresh_index + 1])
    except (IndexError, ValueError):
        print("Invalid y-thresh value. Must be an integer.")
        sys.exit(1)
if '--x-thresh' in sys.argv:
    x_thresh_index = sys.argv.index('--x-thresh')
    try:
        x_thresh = int(sys.argv[x_thresh_index + 1])
    except (IndexError, ValueError):
        print("Invalid x-thresh value. Must be an integer.")
        sys.exit(1)

# Initialize global lists
unknown_segments = []
status_log_path = None

# Helper Functions
def preprocess_for_ocr(image_crop):
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def describe_image(image_crop):
    image_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    description = captioner(image_pil)[0]['generated_text']
    return description

def additional_description(segment_type, content):
    if segment_type == "Diagram":
        return "This segment is likely a hand-drawn diagram or illustration."
    elif segment_type == "Text or Handwriting":
        return "This segment appears to contain textual notes, either printed or handwritten."
    elif segment_type == "Chemical Formula":
        return "This segment likely contains chemical notation or structures."
    elif segment_type == "Math Formula":
        return "This segment likely contains mathematical equations or formulas."
    else:
        return "This segment type is unknown and may require manual review."

def summarize_segments(segments_info):
    type_counts = Counter(segment['type'] for segment in segments_info)
    summary = "Summary:"
    for seg_type, count in type_counts.items():
        summary += f"- {count} {seg_type}(s)"
    return summary

def create_summary_chart(segments_info):
    chart_lines = ["| Type | Confidence | Coordinates |", "|:-----|:-----------|:------------|"]
    for seg in segments_info:
        seg_type = seg['type']
        conf = f"{seg['confidence']:.2f}"
        coords = str(seg['coordinates'])
        chart_lines.append(f"| {seg_type} | {conf} | {coords} |")
    return "".join(chart_lines)

def merge_text_block(block):
    merged_content = "".join([seg['content'] for seg in block])
    x1 = min(seg['coordinates'][0] for seg in block)
    y1 = min(seg['coordinates'][1] for seg in block)
    x2 = max(seg['coordinates'][2] for seg in block)
    y2 = max(seg['coordinates'][3] for seg in block)
    return {
        "type": "Text or Handwriting",
        "content": merged_content,
        "coordinates": (x1, y1, x2, y2),
        "confidence": sum(seg['confidence'] for seg in block) / len(block),
        "additional_description": "Merged block of closely located textual notes."
    }

def merge_close_text_segments(segments_info, y_thresh=30, x_thresh=50):
    merged_segments = []
    current_text = []
    for segment in segments_info:
        if segment['type'] == "Text or Handwriting" and segment['coordinates']:
            if current_text and (
                abs(segment['coordinates'][1] - current_text[-1]['coordinates'][3]) <= y_thresh or
                abs(segment['coordinates'][0] - current_text[-1]['coordinates'][2]) <= x_thresh
            ):
                current_text.append(segment)
            else:
                if current_text:
                    merged_segments.append(merge_text_block(current_text))
                current_text = [segment]
        else:
            if current_text:
                merged_segments.append(merge_text_block(current_text))
                current_text = []
            merged_segments.append(segment)
    if current_text:
        merged_segments.append(merge_text_block(current_text))
    return merged_segments

def write_unknown_segments(unknowns_log_path):
    if unknown_segments:
        unknowns_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(unknowns_log_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['markdown_file', 'original_image', 'coordinates']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in unknown_segments:
                writer.writerow(item)

def write_status(image_name, status, note=""):
    global status_log_path
    if status_log_path:
        status_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not status_log_path.exists():
            with open(status_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'status', 'note'])
        with open(status_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([image_name, status, note])

# Process a single image
def process_image(image_path, output_dir, unknowns_log_path):
    img = cv2.imread(str(image_path))
    results = yolo_model(img, conf=0.25)

    segments_info = []
    text_found = False

    for idx, result in enumerate(results):
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            if cls_id == 0:
                crop_processed = preprocess_for_ocr(crop)
                text_results = ocr_reader.readtext(crop_processed, detail=1)
                texts = [r[1] for r in text_results]
                content = '\n'.join(texts)
                type_detected = "Text or Handwriting"
                text_found = True
            elif cls_id == 1:
                description = describe_image(crop)
                content = f"[Diagram Description]: {description}"
                type_detected = "Diagram"
            elif cls_id == 2:
                content = "[Chemical Formula detected]"
                type_detected = "Chemical Formula"
            elif cls_id == 3:
                content = "[Math Formula detected]"
                type_detected = "Math Formula"
            else:
                content = "[Unknown segment]"
                type_detected = "Unknown"
                unknown_segments.append({
                    "markdown_file": str(output_dir / (image_path.stem + ".md")),
                    "original_image": str(image_path),
                    "coordinates": (x1, y1, x2, y2)
                })

            segments_info.append({
                "type": type_detected,
                "content": content,
                "coordinates": (x1, y1, x2, y2),
                "confidence": conf,
                "additional_description": additional_description(type_detected, content)
            })

    if not text_found:
        full_text_results = ocr_reader.readtext(img, detail=1)
        full_texts = [r[1] for r in full_text_results]
        content = '\n'.join(full_texts)
        segments_info.append({
            "type": "Full Image Text or Handwriting (Fallback)",
            "content": content,
            "coordinates": None,
            "confidence": 1.0,
            "additional_description": "This fallback captures any missed handwritten or printed notes."
        })

    segments_info.sort(key=lambda s: (s['coordinates'][1] if s['coordinates'] else float('inf'), s['coordinates'][0] if s['coordinates'] else float('inf')))
    segments_info = merge_close_text_segments(segments_info, y_thresh=y_thresh, x_thresh=x_thresh)

    md_path = output_dir / (image_path.stem + ".md")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Analyzed Output for {image_path.name}\n\n")
        f.write(f"Original Image: ![]({image_path})\n\n")
        f.write(summarize_segments(segments_info) + "\n\n")
        f.write(create_summary_chart(segments_info) + "\n\n")
        for i, segment in enumerate(segments_info):
            f.write(f"## Segment {i+1} ({segment['type']})\n")
            f.write(f"**Confidence**: {segment['confidence']:.2f}\n\n")
            f.write(f"**Coordinates**: {segment['coordinates']}\n\n")
            f.write(f"**Description:** {segment['additional_description']}\n\n")
            f.write(f"**Content:**\n\n{segment['content']}\n\n")
            f.write("---\n")

    write_unknown_segments(unknowns_log_path)
def process_directory(root_dir, output_dir):
    global status_log_path
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    unknowns_log_path = root_path / UNKNOWN_CSV_FILENAME
    status_log_path = root_path / STATUS_CSV_FILENAME

    for path in root_path.rglob('*.jpg'):
        relative_path = path.relative_to(root_path).parent
        output_subdir = output_path / relative_path
        md_path = output_subdir / (path.stem + ".md")

        if md_path.exists() and not force:
            if cutoff_date:
                md_mtime = datetime.fromtimestamp(md_path.stat().st_mtime)
                if md_mtime > cutoff_date:
                    print(f"Skipping (recently processed): {path}")
                    write_status(path.name, "skipped", "Recently processed")
                    continue
            else:
                print(f"Skipping (already processed): {path}")
                write_status(path.name, "skipped", "Already processed")
                continue

        print(f"Processing: {path}")
        write_status(path.name, "processing", "Started")
        process_image(path, output_subdir, unknowns_log_path)
        write_status(path.name, "completed", "Finished MD generation")
# Start processing with file watching
import time

def get_all_jpg_files(directory):
    return {str(f): os.path.getmtime(f) for f in Path(directory).rglob('*.jpg')}

previous_files = get_all_jpg_files(INPUT_DIR)

while True:
    print("Checking for new or updated files...")
    current_files = get_all_jpg_files(INPUT_DIR)

    # Compare previous and current
    new_or_updated = [f for f in current_files if f not in previous_files or current_files[f] != previous_files[f]]

    if new_or_updated:
        print(f"✅ Found {len(new_or_updated)} new/updated file(s). Running processing...")
        process_directory(INPUT_DIR, OUTPUT_DIR)
        previous_files = current_files.copy()
    else:
        print("No new files found.")

    print("🕑 Sleeping for 2 minutes...")
    time.sleep(120)  # Sleep for 2 minutes
