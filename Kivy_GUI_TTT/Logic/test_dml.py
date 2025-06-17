import torch

try:
    import torch_directml
    print("torch_directml module imported successfully.")
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"DirectML is AVAILABLE! Using device: {device}")
        # Simple test operation
        a = torch.tensor([1.0, 2.0]).to(device)
        b = torch.tensor([3.0, 4.0]).to(device)
        c = a + b
        print(f"Test calculation on DirectML device: {c}")
    else:
        print("DirectML module imported, but it is NOT AVAILABLE. Falling back to CPU.")
        device = torch.device("cpu")
except ImportError:
    print("torch_directml module not found. DirectML will not be used.")
    device = torch.device("cpu")
except Exception as e:
    print(f"An unexpected error occurred during DirectML check: {e}")
    device = torch.device("cpu")

print(f"Final device selected: {device}")