import json
import os
from PIL import Image
from processors.segmentation import ImageSegmenter
from processors.typed_ocr import TypedOCRReader
from processors.utils import ImageUtils

# === Setup Paths ===
image_path = os.path.join("processors", "your_test_image.jpg")
config_path = os.path.join("config.json")

# === Load Base Config ===
with open(config_path) as f:
    base_config = json.load(f)

conf_thresholds = [0.1, 0.25, 0.4]
min_sizes = [10, 20, 30]
max_segments_list = [5, 10, 15]

best_result = {"count": 0, "text": [], "config": {}}

# === Load Image ===
image = ImageUtils.load_image(image_path)

# === OCR Setup ===
ocr_reader = TypedOCRReader()

print("🧪 Testing config combinations...\n")

for conf_threshold in conf_thresholds:
    for min_size in min_sizes:
        for max_segments in max_segments_list:
            print(f"🔍 Config: conf={conf_threshold}, min_size={min_size}, max_segments={max_segments}")

            segmenter = ImageSegmenter(
                conf_threshold=conf_threshold,
                min_size=min_size,
                max_segments=max_segments
            )

            segments = segmenter.segment(image_path)

            count = 0
            texts = []

            for idx, box in enumerate(segments):
                x1, y1, x2, y2 = map(int, box)
                cropped = image.crop((x1, y1, x2, y2))
                try:
                    text = ocr_reader.read_text(cropped)
                    if text.strip():
                        count += 1
                        texts.append(text.strip())
                except Exception as e:
                    continue

            print(f"✅ Text segments: {count}\n")

            if count > best_result["count"]:
                best_result = {
                    "count": count,
                    "text": texts,
                    "config": {
                        "conf_threshold": conf_threshold,
                        "min_size": min_size,
                        "max_segments": max_segments
                    }
                }

# === Output Best Result ===
print("🏆 Best Configuration Found:")
print(json.dumps(best_result["config"], indent=2))
print("\n📝 Sample Extracted Text:")
for t in best_result["text"][:3]:
    print(f"• {t}")
