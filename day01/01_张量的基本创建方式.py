import torch
import numpy as np
from pandas.core.dtypes import astype

t1 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
print(t1, type(t1))

T1 = torch.Tensor(3, 3)
print(T1, type(T1))

t2 = torch.tensor(np.random.uniform(2,425,(5,23)), dtype=torch.int64)
print(t2, type(t2))
