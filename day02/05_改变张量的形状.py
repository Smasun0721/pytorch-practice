import torch

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

t = t.permute(1, 0, 2)
print(t)
print(t.shape)
print(t.is_contiguous())

t1 = t.contiguous().view(6,4,2)
print(t1)
