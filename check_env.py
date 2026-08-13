import sys

print("Python Version:", sys.version)

packages = [
    "torch",
    "torchvision",
    "torchinfo",
    "sklearn",
    "matplotlib",
    "seaborn",
    "kaggle"
]

missing = []

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "N/A")
        print(f"[OK] {pkg}: {ver}")
    except ImportError:
        print(f"[MISSING] {pkg}")
        missing.append(pkg)

print("\n--- GPU Check ---")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is NOT available in PyTorch build or GPU not detected.")
except Exception as e:
    print(f"Error checking CUDA: {e}")
