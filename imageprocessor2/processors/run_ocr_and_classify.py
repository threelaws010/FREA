import os
from text_type_detector import is_handwritten, is_typed
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Load models once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Handwriting OCR
hand_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
hand_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

# Typed OCR
typed_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
typed_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").to(device)

def run_tests(image_dir="/home/frea/FREA/imageprocessor2/test_images",
              output_dir="/home/frea/FREA/imageprocessor2/test_output"):

    supported_extensions = (".jpg", ".jpeg", ".png")

    os.makedirs(output_dir, exist_ok=True)

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
            image = Image.open(path).convert("RGB")

            if is_handwritten(path):
                result = "🖋 Handwritten"
                processor = hand_processor
                model = hand_model
            elif is_typed(path):
                result = "⌨️ Typed"
                processor = typed_processor
                model = typed_model
            else:
                result = "❓ Unknown"
                print(f"{filename}: {result}")
                continue

            # Run OCR
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_new_tokens=512)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            print(f"{filename}: {result} → ✅ OCR Complete")

            # Save result as .md
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.md")
            with open(output_path, "w") as f:
                f.write(f"# OCR Result for {filename}\n\n")
                f.write(f"**Type:** {result}\n\n")
                f.write(text)

        except Exception as e:
            print(f"{filename}: ❌ Error - {e}")

if __name__ == "__main__":
    run_tests()
