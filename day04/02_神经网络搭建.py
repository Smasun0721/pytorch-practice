import torch
from torch import nn
from torchsummary import summary


# 1.搭建神经网络，继承自nn.Module
class ModelDemo(nn.Module):
    # 1.1.初始化 父类成员 和 神经网络
    def __init__(self):
        # 初始化父类成员
        super().__init__()
        # 搭建神经网络 -> 隐藏层 + 输出层
        # 隐藏层1：输入特征 3，输出特征 3
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层2：输入特征 3，输出特征 2
        self.linear2 = nn.Linear(3, 2)
        # 输出层：输入特征 2，输出特征 2
        self.output = nn.Linear(2, 2)

        # 初始化隐藏层参数
        # 隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # 1.2.前向传播：输入层 -> 隐藏层 -> 输出层
    def forward(self, x):  # 该函数在类对象调用时会自动调用
        # 加权求和 + 激活函数
        # 隐藏层1
        x = torch.sigmoid(self.linear1(x))
        # 隐藏层2
        x = torch.relu(self.linear2(x))
        # 输出层   dim=-1 表示按行计算
        x = torch.softmax(self.output(x), dim=1)
        # 返回预测值
        return x


# 2.模型训练
def train():
    # 1.创建模型对象
    model = ModelDemo()

    # 2.创建数据样本
    data = torch.randn(size=(5, 3))
    print(f"data :{data}")
    print(f"data.shape: {data.shape}")
    print(f"data.requires_grad: {data.requires_grad}\n")

    # 3.调用神经网络模型
    output = model(data)
    print(f"output: {output}")
    print(f"output.shape: {output.shape}")
    print(f"output.requires_grad: {output.requires_grad}\n")

    # 4.计算 和 查看模型参数
    print("=" * 36 + "计算模型参数" + "=" * 36 + "\n")
    summary(model, input_size=(5, 3))

    print("=" * 36 + "查看模型参数" + "=" * 36 + "\n")
    for name, param in model.named_parameters():
        print(f"name: {name}")
        print(f'param: {param}\n')


# 测试
if __name__ == "__main__":
    model = ModelDemo()
    train()
