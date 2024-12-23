---
layout: post
title: "词向量-word2vec之cbow和skipgram"
date: 2024-06-03
author: cora Liu
categories: [LLM]
---
## 词向量

词向量 **（Word Embedding）** 是一种将词语表示为实数向量的方法。这种方法通过将词语映射到一个高维空间中，使得在该空间中具有相似意义的词语的向量彼此接近。

如果不用向量的方法来表示，那么词与词之间的相关关系，就没法很好地表示了。

### 最基础的表示方法—one-hot

在`one-hot`编码中，每个类别都被表示为一个与类别数量相同长度的二进制向量。在这个向量中，只有一个位置的值为1，其余位置的值均为0。

下面举个简单的 🌰

> 假设我们有一个包含三个类别的分类特征：["猫", "狗", "兔子"]。使用one-hot编码，可以将这三个类别表示为如下的二进制向量 ⬇️

- 猫 -> [1, 0, 0]
- 狗 -> [0, 1, 0]
- 兔子 -> [0, 0, 1]

从上面例子可以看到，向量与向量之间是互相垂直的，它们之间的相关性无法体现。

同时，当类别数量较多时，`one-hot`编码会产生非常高维度的稀疏向量，可能会增加计算和存储成本。

那么向量与向量之间的相关性在哪体现呢，我们可以从点乘距离和欧氏距离的角度来看看～

#### 1、点乘距离
点乘距离，其实就是计算两个向量的内积（也叫点积）。你可以把它想象成我们在比较两个向量在同一个方向上的相似程度。如果两个向量的方向相似，点乘的结果会比较大；如果方向相反，结果会比较小，甚至是负数。

#### 2、欧氏距离

欧氏距离就是我们平常说的“直线距离”。它测量的是两点之间的直线距离。可以想象一下，你在地图上看两个地点之间的最短距离，那就是欧氏距离。

#### 3、举个小例子 🌰

假设你和朋友在二维平面上，你在(1, 2)点，他在(4, 6)点。

- **欧氏距离**：可以用勾股定理计算，$\sqrt{(4-1)^2 + (6-2)^2}$，也就是$\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$。所以你们之间的直线距离是5。
- **点乘距离**：假设你们的向量分别是$[1, 2]$和$[4, 6]$。点乘距离就是 $1*4 + 2*6 = 4 + 12 = 16$。这个值表示你们在“逛商场偏好”上的相似程度。


### 静态词向量 vs 动态词向量

#### 1、静态词向量
静态词向量，就像每个词都有一张固定的名片。无论在什么语境下，这张名片上的信息（向量）都是不变的。换句话说，每个词都有一个唯一的表示方法，这个表示方法不会随着上下文的变化而变化。

> 比如，“苹果”这个词在所有的句子里都是用同一个向量来表示，无论它是在“我吃了一个苹果”还是“苹果公司发布了新产品”这样的句子里，向量都是一样的。

常见的静态词向量模型有**Word2Vec**、**GloVe**等。

#### 2、动态词向量
动态词向量则更聪明，它会根据词在句子中的具体语境来调整自己。

>比如，“苹果”这个词在“我吃了一个苹果”里表示水果；而在“苹果公司发布了新产品”里，“苹果”指的是公司。

常见的动态词向量模型有**BERT**、**GPT**等。


下面主要介绍 `Word2Vec` 中两种模型：`CBOW` 和 `skip-gram`。


## CBOW
`BOW： bag of word`，统计词频。

`CBOW（Continuous Bag of Words）`是`Word2Vec`中的一种模型，它的目标是根据上下文中的词汇来预测当前词汇。如果采用的是  `trigram策略`，那么根据前后两个词来预测当前词汇，如下图所示 ⬇️

<img src="/assets/imgs/ai/llm/cbow.png" />


## skip-gram

实现`Word2Vec`模型除了`CBOW`， 还有 `Skip-gram`。它跟`CBOW`是反过来的，即通过当前词汇去预测上下文中的词汇，如下图所示 ⬇️

<img src="/assets/imgs/ai/llm/skip-gram.png" />

接下来演示一下如何使用PyTorch实现Word2Vec的Skip-gram模型。

首先，我们需要准备数据和定义一些超参数：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义语料库
corpus = [
    'I like deep learning',
    'I like NLP',
    'I enjoy coding',
    'I enjoy writing'
]

# 构建词汇表
word_list = ' '.join(corpus).split()
vocab = list(set(word_list))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

# 超参数
embedding_dim = 100
window_size = 2
batch_size = 1
num_epochs = 100
learning_rate = 0.001
```

接下来，我们需要定义一个数据集类来处理数据：

```python
class Word2VecDataset(Dataset):
    def __init__(self, corpus, word_to_idx, window_size):
        self.data = []
        for sentence in corpus:
            words = sentence.split()
            for i, target_word in enumerate(words):
                context_words = [words[j] for j in range(max(i - window_size, 0), min(i + window_size + 1, len(words))) if j != i]
                for context_word in context_words:
                    self.data.append((word_to_idx[context_word], word_to_idx[target_word]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

然后，我们定义Word2Vec模型：

```python
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embedded = self.embeddings(inputs)
        output = self.linear(embedded)
        return output
```

接着，我们可以准备数据并训练模型：

```python
# 创建数据集和数据加载器
dataset = Word2VecDataset(corpus, word_to_idx, window_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型、损失函数和优化器
model = SkipGramModel(len(vocab), embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for context_idx, target_idx in data_loader:
        optimizer.zero_grad()
        outputs = model(context_idx)
        loss = criterion(outputs, target_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
```

在训练完成后，可以使用模型的词嵌入层来获取词向量：

```python
# 获取词嵌入矩阵
embeddings = model.embeddings.weight.data.numpy()

# 查看词向量
for word, idx in word_to_idx.items():
    print(f'Word: {word}, Embedding: {embeddings[idx]}')
```

以上是一个简单的`Word2Vec` `Skip-gram`模型的`PyTorch`实现示例。在实际中，可以根据需要调整模型结构、超参数等，以适应不同的数据集和任务。


## 小结
总的来说，词向量是一种将词语映射到高维连续空间的技术，使得每个词语都可以用一个向量表示。这些向量捕捉了词语之间的语义关系，使得机器能够理解和处理自然语言。

常见的词向量技术有`Word2Vec`，`GloVe`（Global Vectors for Word Representation）等。本文主要介绍`Word2Vec`，包含`CBOW`和`Skip-gram`两种模型。