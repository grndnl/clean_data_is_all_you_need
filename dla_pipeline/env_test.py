import torch
print("")
print("*"*100)

is_available = torch.cuda.is_available()

print(f"CUDA AVAILABLE: {is_available}")

if is_available:
    print(f"NUMBER OF DEVICES: {torch.cuda.device_count()}")
    print(f"CURRENT DEVICE: {torch.cuda.get_device_name(0)}")
    
print("*"*100)
print("")