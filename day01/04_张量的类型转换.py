import torch

t1 = torch.tensor([1, 2, 3],dtype=torch.float)
print(t1,t1.dtype)

t2 = t1.type(torch.long)
print(t2,t2.dtype)

t3 = t1.double()
print(t3)

