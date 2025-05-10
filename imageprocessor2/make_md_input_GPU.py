import os
import sys
from processors.segmentation import ImageSegmenter
from processors.ocr import OCRReader
from processors.captioning import ImageCaptioner
from processors.utils import ImageUtils
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import torch

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

device = "cuda" if torch.cuda.is_available() else "cpu"
segmenter = ImageSegmenter(device=device)
ocr_reader = OCRReader(device=device)
captioner = ImageCaptioner(device=device)

all_images = list(Path(INPUT_DIR).rglob("*.jpg"))

for img_path in tqdm(all_images, desc="Processing Images"):
    print(f"\nProcessing: {img_path}")
    try:
        segments = segmenter.segment(str(img_path))
        image = ImageUtils.load_image(str(img_path))

        segments_data = []
        for idx, box in enumerate(tqdm(segments, desc=f"Segments for {img_path.name}", leave=False)):
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))

            try:
                text = ocr_reader.read_text(cropped)
                if text:
                    segments_data.append({"type": "Text", "content": text})
                    continue
            except Exception as e:
                print(f"⚠️ OCR error: {e}")

            if mode == 'full':
                try:
                    caption = captioner.caption(cropped)
                    segments_data.append({"type": "Caption", "content": caption})
                except Exception as e:
                    print(f"⚠️ Captioning error: {e}")
                    segments_data.append({"type": "Unknown", "content": "[Failed to analyze segment]"})
            else:
                segments_data.append({"type": "Unknown", "content": "[OCR Failed, no captioning in OCR-only mode]"})

        output_md = os.path.join(OUTPUT_DIR, f"{img_path.stem}.md")
        ImageUtils.save_markdown(output_md, segments_data)
        print(f"✅ Saved: {output_md}")

    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")
