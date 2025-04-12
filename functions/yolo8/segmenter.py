from ultralytics import YOLO
import torch
import numpy as np

# Setup device (ROCm-aware)
device = "cuda" if torch.version.hip else "cpu"

# Load the larger segmentation model
model = YOLO("yolov8l-seg.pt")


def segment_image(image_path: str) -> dict:
    """
    Segments the given image and returns the results in a structured dictionary.
    
    Args:
        image_path (str): Path to the image to segment.

    Returns:
        dict: {
            'image_path': str,
            'boxes': list of dict,
            'masks': list of binary mask arrays,
            'class_names': list of str,
            'probabilities': list of float
        }
    """
    results = model(image_path, task="segment", device=device)
    result = results[0]
    
    boxes = []
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

    masks = result.masks.data.cpu().numpy().astype(np.uint8).tolist() if result.masks is not None else []

    return {
        "image_path": image_path,
        "boxes": boxes,
        "masks": masks,
        "class_names": result.names,
        "probabilities": result.probs.tolist() if result.probs is not None else [],
    }
