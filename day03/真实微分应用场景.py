import torch
torch.manual_seed(1)

x = torch.ones(2, 5)
print(f'x: {x}')
y = torch.zeros(2, 3)
print(f'y: {y}')
w = torch.randn(5, 3, requires_grad=True)
print(f'w: {w}')
b = torch.randn(1, 3, requires_grad=True)
print(f'b: {b}')
z = x @ w + b
print(f'z: {z}')

criterion = torch.nn.MSELoss()
loss = criterion(z, y)
print(f'loss: {loss}')

loss.sum().backward()

w.data = w.data - 0.01 * w.grad
b.data = b.data - 0.01 * b.grad

print(f'w: {w}')
print(f'b: {b}')

z = x @ w + b
loss=criterion(z, y)
loss.sum().backward()
print(f'loss: {loss}')
