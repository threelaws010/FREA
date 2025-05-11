from ultralytics import YOLO
import torch

class ImageSegmenter:
    def __init__(self, model_path='yolov8n-seg.pt', device=None, conf_threshold=0.25, max_segments=None, min_size=0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_segments = max_segments
        self.min_size = min_size

    def segment(self, image_path):
        results = self.model.predict(image_path, device=self.device, conf=self.conf_threshold)
        segments = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                if width >= self.min_size and height >= self.min_size:
                    segments.append(box)
        if self.max_segments:
            segments = segments[:self.max_segments]
        return segments
