import tempfile
import os
import shutil
from ultralytics import YOLO
import torch
import numpy as np

device = "cuda" if torch.version.hip else "cpu"
model = YOLO("yolov8l-seg.pt")

def segment_image(image_path: str) -> dict:
    results = model(image_path, task="segment", device=device)
    result = results[0]

    boxes = []
    scores = []
    if result.boxes is not None and result.boxes.xyxy is not None:
        for i, box in enumerate(result.boxes.xyxy.tolist()):
            boxes.append({
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "class_id": int(result.boxes.cls[i]),
                "confidence": float(result.boxes.conf[i])
            })
            scores.append(float(result.boxes.conf[i]))

    masks = result.masks.data.cpu().numpy().astype(np.uint8).tolist() if result.masks is not None else []

    # Use a temporary directory to save the segmented image
    temp_dir = tempfile.mkdtemp(prefix="segmented_")

    # This saves to the temp_dir using the default filename
    result.save(save_dir=temp_dir)

    # Get the name of the saved image file (it matches the original filename)
    base_filename = os.path.basename(image_path)
    segmented_image_path = os.path.join(temp_dir, base_filename)

    return {
        "image_path": image_path,
        "segmented_image_path": segmented_image_path,
        "temp_dir": temp_dir,
        "boxes": boxes,
        "masks": masks,
        "class_names": result.names,
        "probabilities": result.probs.tolist() if result.probs is not None else [],
        "scores": scores,
    }
