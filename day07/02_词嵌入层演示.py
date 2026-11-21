import jieba
import torch
from torch import nn


def dm01():
    # 1.定义一句话
    text = "综合来看，在当前阶段，DeepSeek V4.1 Flash 是更务实、性价比更高的选择。"
    # 2.使用jieba分词器分词
    words = [w for w in jieba.lcut(text) if w.strip()]
    print(words)
    # 3.创建词嵌入层
    embed = nn.Embedding(len(words), 4)
    for i, word in enumerate(words):
        word_vector = embed(torch.tensor(i))
        print(f"{word}: {word_vector}")


if __name__ == "__main__":
    dm01()
