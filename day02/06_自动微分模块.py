import torch

w = torch.tensor(10, requires_grad=True, dtype=torch.float)
loss = w ** 2 + 20
print(f'初始loss:{loss}')
for i in range(1, 501):
    # 反向传播
    loss.sum().backward()
    # 更新w值
    w.data = w.data - 0.01 * w.grad
    # 计算loss值
    loss = w ** 2 + 20
    # 打印数据
    print(f'第 {i} 次迭代: w:{w:.5f}, loss:{loss:.5f}')
    # 清空梯度值
    w.grad.zero_()
