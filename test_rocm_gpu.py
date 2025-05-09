import torch

print("=== ROCm + PyTorch AMD GPU Test ===")

# Check if HIP backend is available
hip_version = getattr(torch.version, "hip", None)
print(f"HIP version: {hip_version}")

# Check for CUDA (ROCm uses 'cuda' as a label)
cuda_available = torch.cuda.is_available()
print(f"GPU available via torch.cuda: {cuda_available}")

if cuda_available:
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Try allocating a tensor on GPU
    try:
        x = torch.rand(3, 3, device='cuda')
        print("Tensor on GPU:")
        print(x)
    except Exception as e:
        print(f"❌ Failed to create tensor on GPU: {e}")
else:
    print("❌ No GPU available to PyTorch. ROCm is not working.")
