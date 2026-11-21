# %%
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

torch.manual_seed(0)
np.random.seed(0)


# %%
def preprocessing():
    # 1.导入数据集
    data = pd.read_csv("./data/手机价格预测.csv")
    # 提取特征数据
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # 转换数据类型
    x = x.astype(np.float32)
    # 切割数据
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )
    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 封装成张量数据集
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))
    # 记录 输入特征数 与 分类数量
    input_dim = x.shape[1]
    output_dim = len(np.unique(y))

    print(f"输入特征数: {input_dim}")
    print(f"分类数量: {output_dim}")

    return train_dataset, test_dataset, input_dim, output_dim


# %%
# 2.搭建神经网络
class PhonePriceModel(nn.Module):
    # 初始化神经网络
    def __init__(self, input_dim, output_dim):
        # 初始化父类成员
        super().__init__()
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 256)
        # 隐藏层2
        self.linear2 = nn.Linear(256, 128)
        # 输出层
        self.output = nn.Linear(128, output_dim)

    # 前向传播
    def forward(self, x):
        # 隐藏层1
        x = torch.relu(self.linear1(x))
        # 隐藏层2
        x = torch.relu(self.linear2(x))
        # 输出层
        x = self.output(x)

        return x


# %%
# 3.模型训练
def train(train_dataset, input_dim, output_dim):
    # 加载数据集
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 创建神经网络
    model = PhonePriceModel(input_dim, output_dim)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练轮数
    epochs = 100
    # 记录loss值
    loss_list = []

    # 更改模型状态
    model.train()
    # 模型训练
    start_time = time.time()  # 记录开始时间
    for epoch in range(epochs):
        total_loss = 0  # 记录当前轮次的总loss值
        for x, y in train_loader:
            optimizer.zero_grad()  # 梯度清零
            y_pred = model(x)  # 模型预测
            loss = criterion(y_pred, y)  # 损失计算
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新
            total_loss += loss.item()  # 累加批次loss值
        loss_list.append(total_loss / len(train_loader))  # 记录当前轮次的平均loss值
        print(
            f"epoch: {epoch + 1}, loss: {total_loss / len(train_loader):.4f}, time: {time.time() - start_time:.2f}s"
        )

    # 保存模型
    torch.save(model.state_dict(), "./model/phonePrice.pth")

    # 训练可视化
    plt.plot(loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(alpha=0.3)
    plt.show()


# %%
# 4.模型测试
def evaluate(test_dataset, input_dim, output_dim):
    # 创建模型对象
    model = PhonePriceModel(input_dim, output_dim)
    # 加载模型参数
    model.load_state_dict(torch.load("./model/phonePrice.pth"))
    # 加载测试集
    test_loader = DataLoader(test_dataset, batch_size=8)

    correct_num = 0  # 统计总正确个数
    model.eval()  # 切换模型状态
    # 模型评估
    for x, y in test_loader:
        y_pred = torch.argmax(model(x), dim=1)
        # print(y_pred)
        correct_num += (y_pred == y).sum()

    print(f"准确率: {correct_num * 100 / len(test_dataset):.2f}%")


# %%
# 5.测试
if __name__ == "__main__":
    train_dataset, test_dataset, input_dim, output_dim = preprocessing()
    model = PhonePriceModel(input_dim, output_dim)
    summary(model, input_size=(input_dim,))
    train(train_dataset, input_dim, output_dim)
    evaluate(test_dataset, input_dim, output_dim)
