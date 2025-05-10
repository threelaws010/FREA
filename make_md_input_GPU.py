import os
from dotenv import load_dotenv
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import csv
import hashlib
from collections import Counter
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2
import easyocr
from PIL import Image
from transformers import pipeline
import time
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

load_dotenv()
# Normalize to absolute paths
INPUT_DIR = os.getenv('INPUT_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
UNKNOWN_CSV_FILENAME = os.getenv('UNKNOWN_CSV')
STATUS_CSV_FILENAME = os.getenv('STATUS_CSV')

#print(f"Loaded INPUT_DIR={INPUT_DIR}")
#print(f"Directory exists? {Path(INPUT_DIR).exists()}")
#print(f"Contains files? {[f for f in Path(INPUT_DIR).rglob('*') if f.is_file()]}")

force = '--force' in sys.argv
cutoff_date = None
y_thresh = 30
x_thresh = 50
sleep_minutes = 2
use_rocm = '--use-rocm' in sys.argv

if '--date' in sys.argv:
    try:
        cutoff_date = datetime.strptime(sys.argv[sys.argv.index('--date') + 1], "%Y-%m-%d")
    except (IndexError, ValueError):
        print("Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)
if '--y-thresh' in sys.argv:
    y_thresh = int(sys.argv[sys.argv.index('--y-thresh') + 1])
if '--x-thresh' in sys.argv:
    x_thresh = int(sys.argv[sys.argv.index('--x-thresh') + 1])
if '--sleep' in sys.argv:
    try:
        sleep_minutes = int(sys.argv[sys.argv.index('--sleep') + 1])
    except (IndexError, ValueError):
        print("Invalid sleep value. It must be an integer number of minutes.")
        sys.exit(1)

if use_rocm and hasattr(torch.version, "hip") and torch.version.hip:
    print("🟣 ROCm detected.")
    device = torch.device("cuda")
elif torch.cuda.is_available():
    print("🟢 CUDA GPU detected.")
    device = torch.device("cuda")
else:
    print("⚪ No GPU available. Falling back to CPU.")
    device = torch.device("cpu")

print(f"Selected device: {device}")


yolo_model = YOLO('yolov8l-seg.pt')
yolo_model.to(device)
#ocr_reader = easyocr.Reader(['en'], recog_network='english_g2', gpu=(device.type == "cuda"))
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
model = model.to(device)
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if device.type == "cuda" else -1)

def trocr_ocr(image_crop):
    image_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription




def ensure_csv_exists(filename, headers):
    path = Path(filename)
    if not path.exists():
        print(f"Creating missing CSV file: {filename}")
        path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

ensure_csv_exists(os.path.join(OUTPUT_DIR, UNKNOWN_CSV_FILENAME), ["image", "segment", "text"])
ensure_csv_exists(os.path.join(OUTPUT_DIR, STATUS_CSV_FILENAME), ["image", "hash", "status", "note"])

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

unknown_segments = []
status_log_path = None

def file_hash(filepath):
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def read_status_csv(csv_path):
    if not csv_path.exists():
        return {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return {row['image']: row for row in reader}

def write_status(image_name, file_hash, status, note=""):
    global status_log_path
    if status_log_path:
        status_log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not status_log_path.exists()
        with open(status_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(['image', 'hash', 'status', 'note'])
            writer.writerow([image_name, file_hash, status, note])

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

def summarize_segments(segments_info):
    type_counts = Counter(segment['type'] for segment in segments_info)
    summary = "Summary:\n"
    for seg_type, count in type_counts.items():
        summary += f"- {count} {seg_type}(s)\n"
    return summary

def create_summary_chart(segments_info):
    chart_lines = ["| Type | Confidence | Coordinates |", "|:-----|:-----------|:------------|"]
    for seg in segments_info:
        seg_type = seg['type']
        conf = f"{seg['confidence']:.2f}"
        coords = str(seg['coordinates'])
        chart_lines.append(f"| {seg_type} | {conf} | {coords} |")
    return "\n".join(chart_lines)

def write_unknown_segments(unknowns_log_path):
    if unknown_segments:
        unknowns_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(unknowns_log_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['markdown_file', 'original_image', 'coordinates']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in unknown_segments:
                writer.writerow(item)

def process_image(image_path, output_dir, unknowns_log_path):
    import numpy as np

    img = cv2.imread(str(image_path))
    results = yolo_model(img, conf=0.20, iou=0.5)  # lowered conf, tuned IoU

    segments_info = []
    text_found = False

    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            for idx, mask in enumerate(result.masks.xy):
                mask_np = np.array(mask, dtype=np.int32)
                mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_img, [mask_np], 255)

                # apply mask to image
                masked_crop = cv2.bitwise_and(img, img, mask=mask_img)

                # find bounding box for cropping
                x, y, w, h = cv2.boundingRect(mask_np)
                crop = masked_crop[y:y+h, x:x+w]

                # Skip very small crops
                if w < 5 or h < 5:
                    continue

                cls_id = int(result.boxes.cls[idx])
                conf = float(result.boxes.conf[idx])

                if cls_id == 0:
                    content = trocr_ocr(crop)
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
                        "coordinates": (x, y, x+w, y+h)
                    })

                segments_info.append({
                    "type": type_detected,
                    "content": content,
                    "coordinates": (x, y, x+w, y+h),
                    "confidence": conf,
                    "additional_description": additional_description(type_detected, content)
                })

    if not text_found:
        # fallback OCR over full image
        content = trocr_ocr(img)
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


def trocr_ocr(image_crop):
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_new_tokens=512)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if not transcription.strip():
            return "[Empty Text Detected]"
        return transcription
    except Exception as e:
        print(f"⚠️ TrOCR OCR failed: {e}")
        return "[OCR Failed]"

def process_directory(root_dir, output_dir):
    global status_log_path
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    unknowns_log_path = root_path / UNKNOWN_CSV_FILENAME
    status_log_path = root_path / STATUS_CSV_FILENAME

    status_data = read_status_csv(status_log_path)

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for path in root_path.rglob(ext):

            relative_path = path.relative_to(root_path).parent
            output_subdir = output_path / relative_path
            md_path = output_subdir / (path.stem + ".md")

            current_hash = file_hash(path)
            prev = status_data.get(path.name)
            if prev and prev['hash'] == current_hash and not force:
                print(f"Skipping (already processed and hash matched): {path}")
                write_status(path.name, current_hash, "skipped", "Hash match")
                continue

            print(f"Processing: {path}")
            write_status(path.name, current_hash, "processing", "Started")
            process_image(path, output_subdir, unknowns_log_path)
            write_status(path.name, current_hash, "completed", "Finished MD generation")

def get_all_jpg_files(directory):
    return {str(f): os.path.getmtime(f) for f in Path(directory).rglob('*.jpg')}

previous_files = get_all_jpg_files(INPUT_DIR)
print("previous_files")
print(previous_files)

while True:
    print("Checking for new or updated files...")
    current_files = get_all_jpg_files(INPUT_DIR)

    new_or_updated = [f for f in current_files if f not in previous_files or current_files[f] != previous_files[f]]

    if new_or_updated or force:
        if force:
            print(f"⚡ Force mode enabled: Reprocessing all files.")
            files_to_process = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                files_to_process.extend([str(p) for p in Path(INPUT_DIR).rglob(ext)])

        else:
            print(f"✅ Found {len(new_or_updated)} new/updated file(s). Running processing...")
            files_to_process = new_or_updated

        for file_path in files_to_process:
            path = Path(file_path)
            relative_path = path.relative_to(INPUT_DIR).parent
            output_subdir = Path(OUTPUT_DIR) / relative_path

            print(f"Processing file: {path}")
            process_image(path, output_subdir, Path(INPUT_DIR) / UNKNOWN_CSV_FILENAME)

            current_hash = file_hash(path)
            write_status(path.name, current_hash, "completed", "Finished MD generation")

        previous_files = current_files.copy()

        # ➡️ Exit if we were running in --force mode
        if force:
            print("✅ Force mode finished. Exiting program.")
            break

    else:
        print("No new files found.")

    print(f"🕑 Sleeping for {sleep_minutes} minutes...")
    time.sleep(sleep_minutes * 60)


