import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import json
import traceback  # NEW: for detailed error output

from processors.segmentation import ImageSegmenter
from processors.ocr import OCRReader
from processors.captioning import ImageCaptioner
from processors.utils import ImageUtils

load_dotenv()

INPUT_DIR = os.getenv('INPUT_DIR', 'home/test/sample')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'home/test/MD')

INPUT_DIR = os.path.abspath(INPUT_DIR)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

mode = 'full'
if len(sys.argv) > 1:
    if '--ocr-only' in sys.argv:
        mode = 'ocr'

force = '--force' in sys.argv
no_segmentation = '--no-segmentation' in sys.argv

# === LOAD CONFIG WITH ERROR HANDLING ===
try:
    with open('config.json') as f:
        config = json.load(f)
except Exception as e:
    print("❌ Error loading config.json:")
    traceback.print_exc()
    sys.exit(1)

conf_threshold = config.get('conf_threshold', 0.25)
max_segments = config.get('max_segments', None)
min_size = config.get('min_size', 0)

# Override config from CLI args
for arg in sys.argv:
    if arg.startswith('--conf='):
        try:
            conf_threshold = float(arg.split('=')[1])
        except ValueError:
            print("⚠️ Invalid --conf value.")
    if arg.startswith('--max-segments='):
        try:
            max_segments = int(arg.split('=')[1])
        except ValueError:
            print("⚠️ Invalid --max-segments value.")
    if arg.startswith('--min-size='):
        try:
            min_size = int(arg.split('=')[1])
        except ValueError:
            print("⚠️ Invalid --min-size value.")

device = "cuda" if torch.cuda.is_available() else "cpu"
segmenter = ImageSegmenter(device=device, conf_threshold=conf_threshold, max_segments=max_segments, min_size=min_size)
ocr_reader = OCRReader(device=device)
captioner = ImageCaptioner(device=device)

all_images = list(Path(INPUT_DIR).rglob("*.jpg"))

for img_path in tqdm(all_images, desc="Processing Images"):
    output_md = os.path.join(OUTPUT_DIR, f"{img_path.stem}.md")
    output_debug = os.path.join(OUTPUT_DIR, f"{img_path.stem}_debug.jpg")

    if os.path.exists(output_md) and not force:
        print(f"⏩ Skipping {img_path.name} (already processed)")
        continue

    print(f"\n📷 Processing: {img_path}")
    try:
        image = ImageUtils.load_image(str(img_path))
        if no_segmentation:
            print("🛑 Skipping segmentation; using full image.")
            segments = [[0, 0, image.width, image.height]]
        else:
            segments = segmenter.segment(str(img_path))
        if not segments:
            print("⚠️ No segments found, defaulting to whole image.")
            segments = [[0, 0, image.width, image.height]]

        segments_data = []
        for idx, box in enumerate(tqdm(segments, desc=f"Segments for {img_path.name}", leave=False)):
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))

            try:
                text = ocr_reader.read_text(cropped)
                if text:
                    segments_data.append({"type": "Text", "content": text, "box": (x1, y1, x2, y2)})
                    continue
            except Exception:
                print(f"⚠️ OCR error for segment {idx + 1}")
                traceback.print_exc()

            if mode == 'full':
                try:
                    caption = captioner.caption(cropped)
                    segments_data.append({"type": "Caption", "content": caption, "box": (x1, y1, x2, y2)})
                except Exception:
                    print(f"⚠️ Captioning error for segment {idx + 1}")
                    traceback.print_exc()
                    segments_data.append({"type": "Unknown", "content": "[Failed to analyze segment]", "box": (x1, y1, x2, y2)})
            else:
                segments_data.append({"type": "Unknown", "content": "[OCR Failed, no captioning in OCR-only mode]", "box": (x1, y1, x2, y2)})

        try:
            ImageUtils.save_markdown(output_md, segments_data)
            ImageUtils.save_debug_image(str(img_path), segments, output_debug)
            print(f"✅ Saved: {output_md} and {output_debug}")
        except Exception:
            print("❌ Error saving markdown or debug image.")
            traceback.print_exc()

    except Exception:
        print(f"❌ Failed to process {img_path}")
        traceback.print_exc()
