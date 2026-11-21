import jieba
import torch
from torch import nn

# 创建RNN层
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
# 定义变量
x = torch.randn(size=(5, 32, 128))
h0 = torch.randn(size=(1, 32, 256))
# 预测
output, h1 = rnn(x, h0)
print(output.shape)  # torch.Size([5, 32, 256])
print(h1.shape)  # torch.Size([1, 32, 256])
print(output)
print(h1)
