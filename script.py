import torch
import numpy as np


def main():
    """
    Main function to demonstrate basic tensor operations using PyTorch with CUDA support.
    This function performs the following steps:
    1. Checks if CUDA is available.
    2. If CUDA is available:
        - Sets the device to CUDA.
        - Creates a tensor `x` with ones on the CUDA device.
        - Creates a tensor `y` with ones on the CPU and transfers it to the CUDA device.
        - Adds tensors `x` and `y` to get tensor `z`.
        - Prints tensor `z` on the CUDA device.
        - Transfers tensor `z` back to the CPU.
        - Prints tensor `z` on the CPU.
        - Converts tensor `z` to a NumPy array `a` and prints it.
    3. If CUDA is not available, prints a message indicating that CUDA is not available.
    """
    if torch.cuda.is_available():
        # Check if CUDA is available
        device: torch.device = torch.device("cuda")
        # Set the device to CUDA
        x: torch.Tensor = torch.ones(2, 2, device=device)
        # Create a tensor `x` with ones on the CUDA device
        y: torch.Tensor = torch.ones(2, 2)
        # Create a tensor `y` with ones on the CPU
        y = y.to(device)
        # Transfer tensor `y` to the CUDA device
        z: torch.Tensor = x + y
        # Add tensors `x` and `y` to get tensor `z`
        print(z)
        # Print tensor `z` on the CUDA device
        z = z.to("cpu")
        # Transfer tensor `z` back to the CPU
        print(z)
        # Print tensor `z` on the CPU
        a: np.ndarray = z.numpy()
        # Convert tensor `z` to a NumPy array `a`
        print("numpy", a)
        # Print the NumPy array `a`
    else:
        # If CUDA is not available
        print("cuda not available")
        # Print a message indicating that CUDA is not available


if __name__ == "__main__":
    main()
    # Call the main function if this script is executed
