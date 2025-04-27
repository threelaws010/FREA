import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from ultralytics import YOLO
import cv2
import easyocr
from PIL import Image

# Setup
yolo_model = YOLO('yolov8l-seg.pt')  # Load your trained YOLOv8 model
ocr_reader = easyocr.Reader(['en'])  # OCR reader, adjust languages if needed

def preprocess_for_ocr(image_crop):
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def process_image(image_path, output_dir):
    img = cv2.imread(str(image_path))
    results = yolo_model(img)
    
    segments_info = []
    text_found = False

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = img[y1:y2, x1:x2]

            if cls_id == 0:  # text
                crop_processed = preprocess_for_ocr(crop)
                text_results = ocr_reader.readtext(crop_processed, detail=1)
                texts = [r[1] for r in text_results]
                content = '\n'.join(texts)
                type_detected = "Text"
                text_found = True
            elif cls_id == 1:
                content = "[Diagram detected]"
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

            segments_info.append({
                "type": type_detected,
                "content": content,
                "coordinates": (x1, y1, x2, y2),
                "confidence": conf
            })

    # Fallback: no text found
    if not text_found:
        full_text_results = ocr_reader.readtext(img, detail=1)
        full_texts = [r[1] for r in full_text_results]
        content = '\n'.join(full_texts)
        segments_info.append({
            "type": "Full Image Text (Fallback)",
            "content": content,
            "coordinates": None,
            "confidence": 1.0
        })

    # 2. Write to Markdown
    md_path = output_dir / (image_path.stem + ".md")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Analyzed Output for {image_path.name}\n\n")
        f.write(f"Original Image: ![]({image_path})\n\n")
        for i, segment in enumerate(segments_info):
            f.write(f"## Segment {i+1} ({segment['type']})\n")
            f.write(f"**Confidence**: {segment['confidence']:.2f}\n\n")
            f.write(f"**Coordinates**: {segment['coordinates']}\n\n")
            f.write(f"**Content:**\n\n{segment['content']}\n\n")
            f.write("---\n")

def process_directory(root_dir, output_dir):
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    for path in root_path.rglob('*.jpg'):
        relative_path = path.relative_to(root_path).parent
        output_subdir = output_path / relative_path
        print(f"Processing: {path}")
        process_image(path, output_subdir)

# Example usage
process_directory('E:/test/sample', 'E:/test/MD')
