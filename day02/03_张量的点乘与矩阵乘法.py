import torch

t1 = torch.tensor([1, 2, 3, 4, 5, 6])
t2 = torch.tensor(
    [
        [
            [1, 2, 3, 4, 5, 6],
            [4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21],
        ],
        [
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21],
        ],
    ],
    dtype=torch.float32,
)

# 矩阵乘法
# t3 = t1 @ t2
# print(t3)

# 求和
print(t2.sum())
print(t2.sum(dim=2))
print(t2.sum(dim=1))
print(t2.sum(dim=0))
print("-" * 36)
# 求最大值
print(t2.max())
print(t2.max(dim=2))
print(t2.max(dim=1))
print(t2.max(dim=0))
print("-" * 36)
# 求平均值
print(t2.mean())
print(t2.mean(dim=2))
print(t2.mean(dim=1))
print(t2.mean(dim=0))
print("-" * 36)

print(t2.exp())
print("-" * 36)
print(t2**3)
print("-" * 36)
print(t2.log())
print("-" * 36)
print(t2.log2())
print("-" * 36)
print(t2.log10())
print("-" * 36)
print(t2.log1p())
print("-" * 36)
