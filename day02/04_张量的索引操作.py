import torch
from sympy.tensor import tensor

t = torch.tensor(
    [
        [
            [1, 2, 3, 4, 5, 6],
            [4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21],
        ],
        [
            [10, 20, 30, 40, 50, 60],
            [40, 50, 60, 70, 80, 90],
            [10, 11, 12, 130, 14, 15],
            [16, 17, 18, 19, 20, 21],
        ],
    ],
    dtype=torch.float32,
)

# 简单索引
print(t[1][2][4])

# 获取(0,1) (1,3)的张量
# 0维，1维，2维
print(t[[0, 1], [0, 2], [1, 3]])

# 范围索引
print(t[:, 1::2, ::2])

# 布尔索引
print(t[torch.tensor([[True, False, True, False],
                     [False, True, False, True]])])
mask=t[:,:,2]>10
print(mask)
print(t[mask])
# 多维索引
