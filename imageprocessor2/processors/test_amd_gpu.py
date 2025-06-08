import torch

def test_gpu():
    print("Torch version:", torch.__version__)
    print("Is ROCm available:", torch.backends.mps.is_available() or torch.version.hip is not None)
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Simple test tensor operation
        x = torch.rand(3, 3).to(device)
        y = torch.mm(x, x)
        print("Matrix multiplication result on device:\n", y)
    except Exception as e:
        print("❌ Error running tensor operation on GPU:")
        print(e)

if __name__ == "__main__":
    test_gpu()
