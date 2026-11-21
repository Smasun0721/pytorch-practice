import torch
import numpy as np

torch.manual_seed(0)

# range
t1 = torch.arange(0, 10, 2)
print(t1, type(t1))

# 等差数列
t2 = torch.linspace(0, 10, 7)
print(t2, type(t2))

# 0-1随机张量
t3 = torch.rand(5, 5)
print(t3, type(t3))

# 符合正态分布的随机张量
t4 = torch.randn(2, 3)
print(t4, type(t4))

# 指定范围内的整数随机张量
t5 = torch.randint(low=0, high=10, size=(2, 3,5))
print(t5, type(t5))

