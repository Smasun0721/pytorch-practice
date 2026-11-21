import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
matplotlib.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 1. 定义和训练时完全一致的网络结构（必须完全一样！）
class ImgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, padding=1, kernel_size=3
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.3)

        self.linear1 = nn.Linear(2304, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.dropout_conv(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x


# 2. CIFAR-10 的10个类别（顺序绝对不能错！）
CLASSES = [
    "plane 飞机",
    "car 汽车",
    "bird 鸟",
    "cat 猫",
    "deer 鹿",
    "dog 狗",
    "frog 青蛙",
    "horse 马",
    "ship 船",
    "truck 卡车",
]


def predict_image(image_path, model_path="./model/ImgModel9.pth"):
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")

    # 加载模型
    model = ImgModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"❌ 找不到模型文件: {model_path}，请确认路径是否正确！")
        return
    model.eval()  # 切换到评估模式（关闭Dropout）

    # 3. 图片预处理（必须和测试集完全一致：Resize到32*32 + 标准化）
    transform = Compose(
        [
            Resize((32, 32)),  # 核心：把任何尺寸压缩成 32x32
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # 打开图片，并强制转为 RGB（防止你传 png 透明图或灰度图报错）
    original_img = Image.open(image_path).convert("RGB")

    # 预处理后增加一个 Batch 维度: [3, 32, 32] -> [1, 3, 32, 32]
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 4. 开始预测
    with torch.no_grad():
        output = model(input_tensor)
        # 使用 Softmax 计算概率（可选）
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        # 找出最大概率的类别索引
        pred_idx = torch.argmax(output, dim=1).item()
        pred_class = CLASSES[pred_idx]
        confidence = prob[pred_idx].item() * 100

    print(f"🎯 预测结果: {pred_class} (置信度: {confidence:.2f}%)")

    # 5. 可视化展示
    plt.figure(figsize=(8, 4))

    # 左图：原始图片
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    # 右图：模型实际看到的 32x32 图片（逆标准化才能正常显示颜色）
    plt.subplot(1, 2, 2)
    img_show = input_tensor[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    plt.imshow(img_show)
    plt.title(
        f"Model Input (32x32)\nPred: {pred_class} ({confidence:.1f}%)", color="red"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 在这里填入你要测试的图片路径（绝对路径或相对路径都可以）
    # 比如：test_img = "C:/Users/Smasun/Desktop/test_cat.jpg"
    test_img = "./data/img/plane.png"  # 默认测试一张图，请改成你自己的图片路径

    predict_image(test_img)
