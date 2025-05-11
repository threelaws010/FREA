from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class ImageCaptioner:
    def __init__(self, model_name='Salesforce/blip-image-captioning-base', device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def caption(self, cropped_image):
        inputs = self.processor(cropped_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption