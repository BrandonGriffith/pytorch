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
        device = torch.device("cuda")
        x = torch.ones(2, 2, device=device)
        y = torch.ones(2, 2)
        y = y.to(device)
        z = x + y
        print(z)
        z = z.to("cpu")
        print(z)
        a = z.numpy()
        print("numpy",a)

    else:
        print("cuda not available")

if __name__ == "__main__":
    main()
