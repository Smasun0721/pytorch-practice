import torch

t1 = torch.zeros(2, 3, 4)
t3 = torch.full_like(t1, fill_value=4)
print(t1)
print(t3)
