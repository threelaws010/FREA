from ultralytics import YOLO
import torch

# ROCm device setup
device = "cuda" if torch.version.hip else "cpu"

model = YOLO("yolov8l-seg.pt")  # Smallest model for segmentation
results = model("C:/Users/three/FREA/functions/yolo8/test.jpg", task="segment", device=device)
results[0].save()  # Save the segmented image


print(results[0].boxes)        # bounding boxes
print(results[0].masks)        # segmentation masks
print(results[0].names)        # class names
print(results[0].probs)  

results[0].show()
