import torch
import numpy as np

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