import torch

x = torch.randint(0,3,(3,2))

y = x> 0.5
print(y)
print(y*1)