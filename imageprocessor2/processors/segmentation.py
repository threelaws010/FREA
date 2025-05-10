from ultralytics import YOLO
import torch

class ImageSegmenter:
    def __init__(self, model_path='yolov8n-seg.pt', device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.device = device

    def segment(self, image_path):
        results = self.model.predict(image_path, device=self.device)
        segments = []
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                segments.append(box)
        return segments