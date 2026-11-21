import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression  # 创建数据集
from torch import (
    nn,  # nn模块中有平方根损失函数和假设函数
    optim,  # optim模块中有优化器函数
)
from torch.utils.data import DataLoader, TensorDataset  # 构造数据集对象，数据集加载器

# 1.创建数据集
x, y, coef = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    coef=True,
    bias=14.5,
    random_state=0,
)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# 数据可视化
plt.figure(figsize=(20,10))
plt.scatter(x, y)
plt.plot(x, x * coef)
plt.show()