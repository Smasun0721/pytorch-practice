import numpy as np
import torch

# numpy 转 tensor
t1 = torch.tensor([1])
n1 = t1.numpy().copy()
print(n1, type(n1))

# numpy 转 tensor
n2 = np.arange(5)
t2 = torch.tensor(n1)
print(t2, type(t2))

# 从tensor中提取值
val = t1.item()
print(val, type(val))
