
import os
from text_type_detector import is_handwritten, is_typed

def run_tests(image_dir="/home/frea/FREA/imageprocessor2/test_images"):
    supported_extensions = (".jpg", ".jpeg", ".png")
    
    if not os.path.isdir(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(supported_extensions)
    ]

    if not image_files:
        print("⚠️ No images found in the directory.")
        return

    for filename in image_files:
        path = os.path.join(image_dir, filename)
        try:
            if is_handwritten(path):
                result = "🖋 Handwritten"
            elif is_typed(path):
                result = "⌨️ Typed"
            else:
                result = "❓ Unknown"
            print(f"{filename}: {result}")
        except Exception as e:
            print(f"{filename}: ❌ Error - {e}")

if __name__ == "__main__":
    run_tests("/home/frea/FREA/imageprocessor2/test_images")