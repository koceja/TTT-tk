import torch
import time

print("Start")

test = torch.randn(4,16,4096,4096, device='cuda')


for i in range(10000):
    test = test @ test
    
    print(i)

