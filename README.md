# PyTorch Tensor Operations Example

This project demonstrates basic tensor operations using [PyTorch](https://pytorch.org/) with device support. It showcases how to:

- Determine the best available device (CUDA or CPU).
- Create and manipulate tensors directly on the target device.
- Transfer tensors from CUDA to CPU when needed.
- Convert the final tensor result to a NumPy array for further processing.

## Using CUDA for Accelerated Computations

For users with a CUDA-enabled GPU, PyTorch can leverage GPU acceleration to significantly speed up tensor operations. To use CUDA:

- Ensure that your system has a compatible NVIDIA GPU.
- Install the appropriate NVIDIA drivers along with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
- Install the CUDA-enabled version of PyTorch. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select the CUDA version matching your toolkit.
- Verify CUDA availability in your code:
  
  ```python
  import torch
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device:", device)
  ```

- When performing tensor operations, create tensors directly on the GPU and transfer them as needed using `.to(device)`.

## Project Structure

- [README.md](README.md) – This file.
- [script.py](script.py) – Main script demonstrating tensor operations and device management.

## Prerequisites

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) installed (choose the CUDA version if applicable)
- [NumPy](https://numpy.org/) installed

You can install the required packages using `pip`:

```sh
pip install torch numpy
```