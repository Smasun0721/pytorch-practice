import torch  # noqa: I001
import numpy as np

t1 = torch.tensor([1., 2., 3.])
n1 = t1.numpy()
print(n1, type(n1))

t2=torch.tensor([1., 2., 3.],requires_grad=True)
n2 = t2.detach().numpy()
print(n2, type(n2))
