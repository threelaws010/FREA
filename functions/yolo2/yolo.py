import torch
import cv2
import pytesseract
from sympy import sympify, SympifyError
import re

# Load YOLO model (assumes YOLOv5 from ultralytics)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# Check for mathematical expressions
def is_math(text):
    try:
        sympify(text)
        return True
    except (SympifyError, TypeError):
        return False

# Check for chemical formulas (simple regex for chemical notation)
def is_chemistry(text):
    chem_pattern = r"^[A-Z][a-z]?\d*([A-Z][a-z]?\d*)*$"
    return bool(re.match(chem_pattern, text))

# Check if the image is an illustration (basic heuristic: low text density)
def is_illustration(segment):
    text = pytesseract.image_to_string(segment)
    text_length = len(text)
    area = segment.shape[0] * segment.shape[1]
    text_density = text_length / area
    return text_density < 0.001  # Threshold for illustration

# Main function to process image
def segment_and_classify(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    segments = []

    for box in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        segment = image[int(y1):int(y2), int(x1):int(x2)]
        text = pytesseract.image_to_string(segment)

        if text.strip():
            if is_math(text):
                label = "Math"
            elif is_chemistry(text):
                label = "Chemistry"
            else:
                label = "Text"
        else:
            if is_illustration(segment):
                label = "Illustration"
            else:
                label = "Unknown"

        segments.append((segment, label))
        print(f"Segment classified as: {label}")

    return segments

# Example usage
# segments = segment_and_classify('image.jpg')
