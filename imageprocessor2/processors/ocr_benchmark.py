import os
import time
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Set your test images directory
IMAGE_DIR = "/home/frea/FREA/imageprocessor2/test_images"
SUPPORTED_EXT = (".jpg", ".jpeg", ".png")


def load_images(image_dir):
    return [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(SUPPORTED_EXT)
    ]


def run_ocr(model, processor, device, image_paths):
    model.to(device)
    times = []

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        start = time.time()
        _ = model.generate(pixel_values, max_new_tokens=512)
        end = time.time()

        times.append(end - start)

    return times


def benchmark_ocr():
    image_paths = load_images(IMAGE_DIR)
    if not image_paths:
        print("❌ No images found.")
        return

    print(f"📂 Found {len(image_paths)} images for benchmarking.")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Benchmark on CPU
    print("\n🔍 Running OCR on CPU...")
    cpu_times = run_ocr(model, processor, device="cpu", image_paths=image_paths)
    cpu_avg = sum(cpu_times) / len(cpu_times)
    print(f"🧠 CPU avg time per image: {cpu_avg:.4f} seconds")

    # Benchmark on GPU
    if torch.cuda.is_available():
        print("\n⚡ Running OCR on GPU...")
        gpu_times = run_ocr(model, processor, device="cuda", image_paths=image_paths)
        gpu_avg = sum(gpu_times) / len(gpu_times)
        print(f"🚀 GPU avg time per image: {gpu_avg:.4f} seconds")

        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else float('inf')
        print(f"📈 Speedup (CPU → GPU): {speedup:.2f}x")
    else:
        print("\n⚠️ CUDA/ROCm device not available — skipping GPU benchmark.")


if __name__ == "__main__":
    benchmark_ocr()
