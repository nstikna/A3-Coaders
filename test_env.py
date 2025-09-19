import torch

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU only")
