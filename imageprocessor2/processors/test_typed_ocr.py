from PIL import Image
from typed_ocr import TypedOCRReader

ocr = TypedOCRReader(device="cpu")  # Force CPU
img = Image.open("your_test_image.jpg")
print("📝 OCR Output:", ocr.read_text(img))