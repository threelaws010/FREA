from PIL import Image,ImageEnhance, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Load and process image
image_path = "/home/frea/FREA/imageprocessor2/test_images/handwriting.jpeg"
image = Image.open(image_path).convert("RGB")
image = image.resize((384, 384))  
image = image.filter(ImageFilter.SHARPEN)
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2.0)

# Load processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare input
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model.device)

# Generate text
generated_ids = model.generate(pixel_values, max_new_tokens=512)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("✅ OCR Result:", text.strip())
