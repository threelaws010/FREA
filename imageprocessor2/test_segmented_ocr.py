from PIL import Image
from processors.segmentation import ImageSegmenter
from processors.typed_ocr import TypedOCRReader
import os

# Paths
image_path = "/home/frea/FREA/imageprocessor2/processors/your_test_image.jpg"
output_md_path = image_path.replace(".jpg", ".md")

# Load image
image = Image.open(image_path).convert("RGB")

# Init segmenter and OCR
segmenter = ImageSegmenter()
ocr_reader = TypedOCRReader()

# ✅ FIXED: Call correct method
segments = segmenter.segment(image)
print(f"🔍 Found {len(segments)} segments")

# OCR results
ocr_results = []
for idx, (bbox, cropped) in enumerate(segments):
    try:
        text = ocr_reader.read_text(cropped)
        clean_text = text.strip()
        if clean_text and clean_text != "***":
            ocr_results.append(f"### Box {idx + 1}:\n{clean_text}\n")
        else:
            ocr_results.append(f"### Box {idx + 1}:\n⚠️ No readable text.\n")
    except Exception as e:
        ocr_results.append(f"### Box {idx + 1}:\n❌ Error: {e}\n")

# Write to markdown
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write("# OCR Output for Segments\n\n")
    f.writelines("\n".join(ocr_results))
    print(f"📄 Segmented OCR results saved to {output_md_path}")
