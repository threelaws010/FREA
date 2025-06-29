from PIL import Image, ImageOps, ImageEnhance
from processors.typed_ocr import TypedOCRReader

# Initialize OCR model
ocr_reader = TypedOCRReader()

# Load image from correct path
image_path = "/home/frea/FREA/imageprocessor2/processors/your_test_image.jpg"
image = Image.open(image_path)

# Apply image preprocessing to help OCR
image = ImageOps.autocontrast(image)
image = ImageEnhance.Contrast(image).enhance(2.0)
image = ImageEnhance.Sharpness(image).enhance(2.0)
image = ImageEnhance.Brightness(image).enhance(1.5)

# Run OCR
text = ocr_reader.read_text(image)

# Output result
print("🧠 Full Image OCR Result:\n")
print(repr(text))

# Optionally save result to .md file
output_path = image_path.replace(".jpg", ".md")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text + "\n")
    print(f"📄 OCR result written to {output_path}")
