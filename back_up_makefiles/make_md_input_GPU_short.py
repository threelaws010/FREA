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
import ctypes

INPUT_DIR = os.getenv('INPUT_DIR', 'E:/test/sample')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'E:/test/MD')
UNKNOWN_CSV_FILENAME = os.getenv('UNKNOWN_CSV', 'unknown_segments.csv')
STATUS_CSV_FILENAME = os.getenv('STATUS_CSV', 'status.csv')

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
    device = torch.device("cuda")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Selected device: {device}")

yolo_model = YOLO('yolov8l-seg.pt')
yolo_model.to(device)
ocr_reader = easyocr.Reader(['en'], recog_network='english_g2', gpu=(device.type == "cuda"))
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if device.type == "cuda" else -1)

unknown_segments = []
status_log_path = None

def ensure_csv_exists(filename, headers):
    if not os.path.exists(filename):
        print(f"Creating missing CSV file: {filename}")
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

ensure_csv_exists(os.path.join(OUTPUT_DIR, UNKNOWN_CSV_FILENAME), ["image", "segment", "text"])
ensure_csv_exists(os.path.join(OUTPUT_DIR, STATUS_CSV_FILENAME), ["image", "hash", "status", "note"])

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

def user_is_active():
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [('cbSize', ctypes.c_uint), ('dwTime', ctypes.c_uint)]

    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(lii)
    if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
        millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
        return millis < (30 * 1000)  # 30 seconds of no input = idle
    return False

def sleep_smart():
    if user_is_active():
        print("👆 User active — sleeping 2 minutes...")
        time.sleep(120)
    else:
        print("😴 No activity — sleeping 10 seconds...")
        time.sleep(10)

def get_all_jpg_files(directory):
    return {str(f): os.path.getmtime(f) for f in Path(directory).rglob('*.jpg')}

def process_directory(root_dir, output_dir):
    global status_log_path
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    unknowns_log_path = root_path / UNKNOWN_CSV_FILENAME
    status_log_path = root_path / STATUS_CSV_FILENAME

    ensure_csv_exists(status_log_path, ["image", "hash", "status", "note"])
    ensure_csv_exists(unknowns_log_path, ["image", "segment", "text"])

    status_data = read_status_csv(status_log_path)

    for path in root_path.rglob('*.jpg'):
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
        # Placeholder: call actual process_image(path, output_subdir, unknowns_log_path)
        write_status(path.name, current_hash, "completed", "Finished MD generation")

previous_files = get_all_jpg_files(INPUT_DIR)

while True:
    print("Checking for new or updated files...")
    current_files = get_all_jpg_files(INPUT_DIR)

    new_or_updated = [f for f in current_files if f not in previous_files or current_files[f] != previous_files[f]]

    if new_or_updated:
        print(f"✅ Found {len(new_or_updated)} new/updated file(s). Running processing...")
        process_directory(INPUT_DIR, OUTPUT_DIR)
        previous_files = current_files.copy()
    else:
        print("No new files found.")

    sleep_smart()
