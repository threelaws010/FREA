from ultralytics import YOLO
import torch

# ROCm device setup
device = "cuda" if torch.version.hip else "cpu"

model = YOLO("yolov8n-seg.pt")  # Smallest model for segmentation
results = model("test.jpg", task="segment", device=device)
results[0].save()  # Save the segmented image
