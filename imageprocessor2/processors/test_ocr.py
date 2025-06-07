from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

image = Image.open("your_test_image.jpg").convert("RGB")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to("cuda" if torch.cuda.is_available() else "cpu")

pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model.device)
generated_ids = model.generate(pixel_values, max_new_tokens=512)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("✅ OCR Result:", text.strip())
