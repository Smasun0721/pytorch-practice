# 导包
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,  # 组合多个操作
    Normalize,  # 标准化
    RandomCrop,  # 随机裁剪
    RandomErasing,
    RandomHorizontalFlip,  # 随机翻转
    ToTensor,  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/  # 转为张量
)

# 每批次样本数
BATCH_SIZE = 512


# 1.创建数据集
def create_dataset():
    # 1. 训练集：数据增强 + 标准化
    train_transform = Compose(
        [
            RandomCrop(32, padding=4),  # 随机裁剪，增强泛化
            RandomHorizontalFlip(),  # 随机翻转，增强泛化
            ToTensor(),  # 转为张量，[0, 1]
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化到 [-1, 1]
            RandomErasing(p=0.5, scale=(0.02, 0.1)),  # 50%概率随机遮挡一小块
        ]
    )

    # 2. 测试集：只需要标准化，绝对不能做数据增强！
    test_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载训练集
    train_dataset = CIFAR10(
        root="./data", train=True, transform=train_transform, download=True
    )
    # 加载测试集
    test_dataset = CIFAR10(
        root="./data", train=False, transform=test_transform, download=True
    )
    # 返回数据集
    return train_dataset, test_dataset


# 2.搭建神经网络
class ImgModel(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        # 第1个卷积层    输入3通道 输出16通道 卷积核3*3 默认 padding=0 stride=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3)
        # 第1个池化层    池化核2*2 默认 padding=0 stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第2个卷积层    输入16通道 输出64通道 卷积核3*3 默认 padding=0 stride=1
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3)
        # 第2个池化层    池化核2*2 默认 pa1dding=0 stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 第3个卷积层    输入64通道 输出256通道 卷积核3*3 默认 padding=1 stride=1
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, padding=1, kernel_size=3
        )
        # 第3个池化层    池化核2*2 默认 pa1dding=0 stride=2
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout_conv = nn.Dropout2d(0.3)  # 在卷积层后丢弃30%的通道

        # # 第1个全连接层   输入：2304 输出：4096
        # self.linear1 = nn.Linear(2304, 4096)
        # self.dropout1 = nn.Dropout(0.5)
        # # 第2个全连接层   输入：4096 输出：2048
        # self.linear2 = nn.Linear(4096, 2048)
        # self.dropout2 = nn.Dropout(0.5)
        # # 第3个全连接层   输入：2048 输出：512
        # self.linear3 = nn.Linear(2048, 1024)
        # self.dropout3 = nn.Dropout(0.5)
        # # 输出层   输入：512 输出：10
        # self.output = nn.Linear(1024, 10)

        # 第1个全连接层
        self.linear1 = nn.Linear(2304, 1024)  # 砍到 1024（之前是4096，上次是512）
        self.dropout1 = nn.Dropout(0.5)
        # 第2个全连接层
        self.linear2 = nn.Linear(1024, 256)  # 砍到 256
        self.dropout2 = nn.Dropout(0.5)
        # 输出层
        self.output = nn.Linear(256, 10)

    # 前向传播
    def forward(self, x):
        # 第1层：卷积层 + 激励层(激活函数) + 池化层
        x = self.pool1(torch.relu(self.conv1(x)))
        # 第2层：卷积层 + 激励层(激活函数) + 池化层
        x = self.pool2(torch.relu(self.conv2(x)))
        # 第3层：卷积层 + 激励层(激活函数) + 池化层
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.dropout_conv(x)  # 在卷积层后丢弃30%的通道

        # 将数据拉平(展开成2维)
        x = x.reshape(x.size(0), -1)

        # # 第4层：全连接层 + 激励层
        # x = torch.relu(self.linear1(x))
        # x = self.dropout1(x)  # 丢弃 50%
        # # 第5层：全连接层 + 激励层
        # x = torch.relu(self.linear2(x))
        # x = self.dropout2(x)  # 丢弃 50%
        # # 第6层：全连接层 + 激励层
        # x = torch.relu(self.linear3(x))
        # x = self.dropout3(x)  # 丢弃 50%
        # # 输出层
        # x = self.output(x)
        # # 返回数据
        # return x
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout2(x)
        # 【彻底删掉 linear3 和 dropout3 的两行】
        x = self.output(x)
        return x


# 3.模型训练
def train(model, train_dataset):
    # 定义设备（自动检测 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用计算设备: {device}")

    # 加载数据集
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # 避免每轮重新创建进程
        prefetch_factor=2,  # 每个进程提前预取2个batch
    )
    total_samples = len(train_dataset)  # 总样本数
    print(f"当前一轮训练的数据集样本数: {total_samples}")

    # 将模型移至 GPU
    model = model.to(device)
    # model = torch.compile(model)
    scaler = torch.amp.GradScaler("cuda")

    # 定义变量
    epochs = 300  # 总训练轮数
    loss_list = []  # 每轮的平均损失
    total_time = 0  # 总耗时
    acc_list = []  # 准确率列表

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 切换模型到训练状态
    model.train()

    # 轮次训练
    for epoch in range(epochs):
        # 定义记录变量
        total_loss = 0  # 该轮的总损失
        total_correct = 0  # 该轮的总正确个数
        start_time = time.time()  # 本轮开始时间
        for x, y in dataloader:
            # 将数据移至 GPU
            x, y = x.to(device), y.to(device)
            # 【修改】前向传播开启 autocast
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # 模型预测
                y_pred = model(x)
                # 损失计算
                loss = criterion(y_pred, y)
            # 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            # 【修改】反向传播用 scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 每批统计
            total_loss += loss.detach() * len(y)
            total_correct += (torch.argmax(y_pred, dim=1) == y).sum().detach()

        scheduler.step()
        # 每轮统计
        loss_list.append(total_loss.item() / total_samples)
        acc_list.append(total_correct.item() / total_samples)
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(
            f"第{epoch + 1}轮，loss: {loss_list[-1]:.2f} , accuracy: {acc_list[-1] * 100:.2f}% , 该轮耗时: {epoch_time:.2f}s , 总耗时: {total_time:.2f}s"
        )

    # 保存模型
    os.makedirs("./model", exist_ok=True)  # 文件夹若不存在则创建
    torch.save(model.cpu().state_dict(), "./model/ImgModel9.pth")
    model.to(device)  # 放回GPU

    # 可视化训练结果 (Loss 和 Accuracy 建议分开画)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), loss_list, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), acc_list, marker="o", color="orange")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.show()


# 4.模型评估
def evaluate(model, test_dataset):
    # 定义设备（自动检测 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model.load_state_dict(torch.load("./model/ImgModel9.pth", map_location=device))
    model.to(device)
    # 加载数据集
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    total_correct = 0  # 总正确个数

    # 切换模型状态
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            total_correct += (torch.argmax(y_pred, dim=1) == y).sum().detach()

    print(f"accuracy: {total_correct.item() / len(test_dataset) * 100:.2f}%")


# 测试
if __name__ == "__main__":
    print("CUDA是否可用:", torch.cuda.is_available())
    print("PyTorch版本:", torch.__version__)
    print("CUDA版本:", torch.version.cuda)

    train_dataset, test_dataset = create_dataset()
    model = ImgModel()
    train(model, train_dataset)
    evaluate(model, test_dataset)
