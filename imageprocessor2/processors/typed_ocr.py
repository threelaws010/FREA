from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

class TypedOCRReader:
    def __init__(self, model_name='microsoft/trocr-base-printed', device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"🔍 Using Typed OCR model on {self.device}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

    def read_text(self, cropped_image):
        pixel_values = self.processor(images=cropped_image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=512)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
