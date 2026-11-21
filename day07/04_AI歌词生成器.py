import os
import time

import jieba
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


# 构建词典
def build_vocab():
    unique_words = set()
    all_words = []
    # 读取文件
    with open("./data/jaychou_lyrics.txt", "r", encoding="utf-8") as f:
        for line in f:
            words = jieba.lcut(line)
            all_words.append(words)
            for word in words:
                unique_words.add(word)
    # 排序，保证每次运行词表索引一致
    unique_words = sorted(unique_words)

    # 创建词表
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    # 歌词文本用词索引表示
    corpus_idx = []
    for words in all_words:
        tmp = [word_to_idx[word] for word in words]
        tmp.append(word_to_idx[" "])  # 添加空格索引
        corpus_idx.extend(tmp)  # 展平加入

    return unique_words, word_to_idx, len(unique_words), corpus_idx


# 数据集
class LyricDataset(Dataset):
    def __init__(self, corpus_idx, seq_len):
        # 文本数据中词的索引
        self.corpus_idx = torch.tensor(corpus_idx)
        # 序列长度
        self.seq_len = seq_len
        # 文本数据中词的总数
        self.word_count = len(corpus_idx)
        # 序列数量
        self.seq_size = self.word_count // self.seq_len

    def __len__(self):
        # 总词数减去序列长度，就是能产生的滑动窗口总数
        return self.word_count - self.seq_len

    def __getitem__(self, index):
        # idx: 词的索引
        start = index
        end = start + self.seq_len
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start + 1 : end + 1]
        return x, y


# 神经网络模型
class TextModel(nn.Module):
    def __init__(self, unique_words_count):
        super().__init__()
        # 初始化词嵌入层
        self.ebd = nn.Embedding(unique_words_count, 128)
        # 循环网络层
        self.rnn = nn.RNN(128, 256, 1, batch_first=True)
        # 输出层(全连接层)
        self.output = nn.Linear(256, unique_words_count)

    def forward(self, input, hidden):
        # x 的格式: [batch, sql_len]
        # 词嵌入层输出格式: [batch, sql_len, dim]
        embd = self.ebd(input)
        output, hidden = self.rnn(embd, hidden)
        # 输出层
        output = self.output(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)


# 模型训练
def train(model):
    # 定义设备（自动检测 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用计算设备: {device}")

    # 将模型移至 GPU
    model.to(device)

    # 1.构建词典
    unique_words, word_to_idx, unique_words_count, corpus_idx = build_vocab()
    # 2.获取数据集
    dataset = LyricDataset(corpus_idx, 32)
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
    # 3.损失函数
    criterion = nn.CrossEntropyLoss()
    # 4.优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5.模型训练
    # 切换模型到训练状态
    model.train()
    # 定义变量
    epochs = 10
    loss_list = []
    total_time = 0
    total_samples = len(dataset)
    print(f"总样本数量: {total_samples}")
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataLoader):
            x, y = x.to(device), y.to(device)
            # 获得隐藏层初始值
            batch_size = x.size(0)
            hidden = model.init_hidden(batch_size, device)
            # 模型预测
            output, hidden = model(x, hidden)
            # 将 y, output 转成一样的形状
            output = output.reshape(-1, unique_words_count)  # (batch * seq, vocab_size)
            y = y.reshape(-1)  # (batch * seq)
            # 损失计算
            loss = criterion(output, y)
            # 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.detach() * len(y)

        # 本轮统计
        loss_list.append(total_loss.item() / total_samples)
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(
            f"第{epoch + 1}轮，loss: {loss_list[-1]:.2f} , 该轮耗时: {epoch_time:.2f}s , 总耗时: {total_time:.2f}s"
        )

    # 保存模型参数
    os.makedirs("./model", exist_ok=True)  # 文件夹若不存在则创建
    torch.save(model.cpu().state_dict(), "./model/TextModel.pth")
    model.to(device)  # 放回GPU

    # 可视化训练结果 (Loss 和 Accuracy 建议分开画)
    plt.figure(figsize=(10, 4))

    plt.plot(range(1, epochs + 1), loss_list, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.show()


# 模型评估
def evaluate(start_word, seq_len):
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建词典
    unique_words, word_to_idx, unique_words_count, corpus_idx = build_vocab()
    # 反向映射字典，索引 -> 词
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    # 获取模型
    model = TextModel(unique_words_count)
    # 加载模型参数
    model.load_state_dict(torch.load("./model/TextModel.pth", map_location=device))
    model.to(device)
    model.eval()
    # 初始化隐藏层
    hidden = model.init_hidden(1, device)
    # 将 输入词 转换成 词表索引
    start_idx = word_to_idx[start_word]
    word_idx = torch.tensor([[start_idx]], dtype=torch.long).to(device)
    # 定义列表记录产生的词的索引
    generate_sentence = [start_word]
    for i in range(seq_len):
        output, hidden = model(torch.tensor([[word_idx]], dtype=torch.long).to(device), hidden)
        word_idx = output.argmax(dim=-1)[:, -1].item()
        generate_sentence.append(unique_words[word_idx])

    for word in generate_sentence:
        print(word,end='')


if __name__ == "__main__":
    # unique_words, word_to_idx, unique_words_count, corpus_idx = build_vocab()
    # dataset = LyricDataset(corpus_idx, 32)
    # print(len(dataset))
    # x, y = dataset[0]
    # print(x)
    # print(y)
    # model = TextModel(unique_words_count)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # train(model)
    evaluate('星星',32)
