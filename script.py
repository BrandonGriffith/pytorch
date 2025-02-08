import torch
import numpy as np


def get_device() -> torch.device:
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Demonstrates basic tensor operations using PyTorch with device support.
    The function:
    1. Determines the best device to use.
    2. Creates tensors directly on the target device.
    3. Calculates a sum of tensors.
    4. Transfers tensor data back to CPU if needed.
    5. Converts the final result into a NumPy array.
    """
    device: torch.device = get_device()
    print(f"Using device: {device}")

    # Create tensors directly on the chosen device.
    x: torch.Tensor = torch.ones(2, 2, device=device)
    y: torch.Tensor = torch.ones(2, 2, device=device)
    z: torch.Tensor = x + y
    print(f"Tensor z on {device}:")
    print(z)

    # If using CUDA, transfer result back to CPU for NumPy conversion.
    if device.type == "cuda":
        z = z.to("cpu")
        print("Tensor z on CPU:")
        print(z)

    # Convert tensor to NumPy array and display.
    a: np.ndarray = z.numpy()
    print("Converted to NumPy array:")
    print(a)

if __name__ == "__main__":
    main()
