import torch

# 1. Check version
print(f"PyTorch version: {torch.__version__}")

# 2. Check for GPU (CUDA) availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# 3. Perform a quick calculation
# Create two random tensors and multiply them
try:
    device = "cuda" if cuda_available else "cpu"
    x = torch.rand(3, 3).to(device)
    y = torch.rand(3, 3).to(device)
    z = torch.matmul(x, y)

    print("\nSuccess! Matrix multiplication result:")
    print(z)
except Exception as e:
    print(f"\nSomething went wrong: {e}")
