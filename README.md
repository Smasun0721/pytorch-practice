# PyTorch 深度学习练习合集

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Type](https://img.shields.io/badge/Type-Learning%20Notes-6A5ACD)

按天整理的 PyTorch 课程练习，从张量基础到 CNN 图像分类和 RNN 歌词生成。

## Requirements

- Python 3.10+
- PyTorch 2.0+ 与 torchvision
- 依赖见 requirements.txt

## Installation

~~~bash
git clone https://github.com/Smasun0721/pytorch-practice.git
cd pytorch-practice
pip install -r requirements.txt
~~~

## Quick start

~~~bash
cd day07
python 04_AI歌词生成器.py
~~~

部分脚本依赖本地数据，需要先把对应数据集放到脚本约定的 data 目录；CIFAR-10 压缩包和模型权重未上传。

## Contents

| 目录 | 主题 | 主要内容 |
| --- | --- | --- |
| day01 | 张量基础 | 张量创建、类型转换、随机张量 |
| day02 | 张量运算 | NumPy 互转、矩阵乘法、索引、reshape、自动微分 |
| day03 | 线性回归 | 手写线性回归、真实微分应用 |
| day04 | 神经网络基础 | 参数初始化、网络搭建、损失函数 |
| day05 | 训练优化 | 动量法、学习率衰减、Dropout、BatchNorm、手机价格预测 |
| day06 | CNN 基础 | 图像绘制、卷积层与池化层 API |
| day07 | 实战案例 | CIFAR-10 图像分类、词嵌入、RNN、AI 歌词生成器 |

## 数据样例

day07 的 CIFAR-10 图像分类使用猫、狗、飞机三类图片作为演示数据：

![CIFAR-10 类别样例](docs/class_samples.png)

## Project structure

~~~text
pytorch-practice/
├── day01/ ~ day07/
├── day05/data/
├── day07/data/img/
├── docs/
├── requirements.txt
└── README.md
~~~

## Notes

本仓库是学习练习合集，保留原课程代码和目录结构；大体积数据和模型权重未上传。数据和课程素材版权归原始提供方所有。
