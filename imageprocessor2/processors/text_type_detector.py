import cv2
import numpy as np

def _calculate_contour_complexity(image_path: str) -> float:
    """Internal helper to calculate average contour complexity from image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    complexities = [
        len(c) / max(1.0, cv2.contourArea(c))
        for c in contours
        if cv2.contourArea(c) > 20
    ]

    if not complexities:
        return 0.0

    return float(np.mean(complexities))


def is_handwritten(image_path: str, threshold: float = 0.12) -> bool:
    """
    Returns True if the image is likely to contain handwritten text.
    """
    complexity = _calculate_contour_complexity(image_path)
    return complexity > threshold


def is_typed(image_path: str, threshold: float = 0.12) -> bool:
    """
    Returns True if the image is likely to contain typed text.
    """
    return not is_handwritten(image_path, threshold)


# Optional CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python text_type_detector.py /path/to/image.jpg")
    else:
        img_path = sys.argv[1]
        print("Handwritten:", is_handwritten(img_path))
        print("Typed:", is_typed(img_path))
