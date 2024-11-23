import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Free GPU Memory: {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
else:
    print("CUDA is not available.")
