---
layout: post
title: "了解 Word Embedding"
date: 2023-06-25
author: Cola Liu
categories: [机器学习]
---

## Word Embedding
`Word embedding`是一种将词汇表中的单词映射到实数向量空间的技术。它是自然语言处理中的重要概念，可以将离散的文本单词转换为连续的向量表示，以捕捉单词之间的语义关系。


假设我们有以下这个句子：
> `the cloud is in the sky.`

按照机器学习对于文本的常用处理方法，我们先把这个句子中的所有词汇编成一个词汇表如下：

|id |word|
| -- | -- |
| the |0 |
| cloud | 1|
| is | 2|
| in | 3|
| sky | 4|

那么，`the cloud is in the sky.`可以用数字编码`[0, 1, 2, 3, 0, 4]`来表示。接着我们用`one-hot`表示法来处理数字编码 `[0, 1, 2, 3, 0, 4]`，如下所示 ⬇️
- 0表示为：[1,0,0,0,0]
- 1表示为：[0,1,0,0,0]
- 2表示为：[0,0,1,0,0]
- 3表示为：[0,0,0,1,0]
- 4表示为：[0,0,0,0,1]


相信细心的伙伴可以看出来，在这里的数字编码，并没有体现出词与词之间的语义关系。如上面句子中的 `cloud` 和 `sky`，这两个单词之间有较大的语义联系。

为了解决这个问题，我们提出了 `Word Embedding`。

`Word embedding`可以应用于各种自然语言处理任务，如`文本分类`、`命名实体识别`、`情感分析`等。它不仅提供了更紧凑、更具语义的单词表示，还能够捕捉单词之间的语义关系，从而提升自然语言处理任务的性能。


## 常见的word embedding 方法

### Skip-gram
`Skip-gram` 模型通过给定一个单词来预测其周围的上下文单词。

比如我们有下面这个句子：
> “The man who passes the sentence should swing the sword.” – Ned Stark

当我们的 sliding window 是5的时候，可以得到目标单词与上下文之间的关系如下：

| Sliding window (size = 5) | Target word | Context|
|--|--|--|
|[The man who] |the |man, who |
|[The man who passes]|man|the, who, passes |
|[The man who passes the]|who|the, man, passes, the |
|[man who passes the sentence]|passes|man, who, the, sentence |
|…|…|…|
|[sentence should swing the sword]|swing|sentence, should, the, sword |
|[should swing the sword]|the|should, swing, sword |
|[swing the sword]|sword|swing, the |

在这里，使用`Skip-gram` 模型我们就可以用单词来预测其周围的上下文单词。例如，`sword`的上下文单词是`swing`以及`the`。

而`CBOW`模型则相反，通过给定上下文单词来预测目标单词，在这里不加赘述。


### GloVe
`GloVe（Global Vectors for Word Representation）`是一种基于全局词频统计的`word embedding`方法。

假设我们有一个包含以下句子的文本语料库：

"I enjoy playing soccer."
"I love playing basketball."
"I like watching soccer matches."

首先，我们需要构建一个词汇表，将语料库中的单词映射到索引。假设我们的词汇表如下：
```python
{
    "I": 0,
    "enjoy": 1,
    "playing": 2,
    "soccer": 3,
    "love": 4,
    "basketball": 5,
    "like": 6,
    "watching": 7,
    "matches": 8
}

```
接下来，我们需要计算每对单词的共现次数。假设我们使用一个context窗口大小为1，那么可以得到以下共现矩阵：

```python
              I  enjoy  playing  soccer  love  basketball  like  watching  matches
I             0      1        1       1     0           0     0         0        0
enjoy         1      0        1       1     0           0     0         0        0
playing       1      1        0       2     0           1     0         0        0
soccer        1      1        2       0     0           0     1         1        1
love          0      0        0       0     0           1     1         0        0
basketball    0      0        1       0     1           0     0         0        0
like          0      0        0       1     1           0     0         1        0
watching      0      0        0       1     0           0     1         0        1
matches       0      0        0       1     0           0     0         1        0

```
共现矩阵的每个元素表示两个单词在一起出现的次数。接下来，我们通过对共现矩阵进行矩阵分解，得到每个单词的向量表示。

最终，我们可以得到每个单词的向量表示，如：
```python
I             [0.2, 0.4, -0.1, 0.7]
enjoy         [0.1, -0.3, 0.5, -0.2]
playing       [-0.4, 0.2, 0.6, -0.5]
soccer        [0.3, -0.1, -0.2, 0.6]
love          [-0.5, 0.2, 0.4, -0.3]
basketball    [-0.2, 0.6, 0.1, -0.4]
like          [0.4, -0.5, 0.3, -0.2]
watching      [-0.3, 0.1, 0.2, 0.5]
matches       [0.1, -0.2, 0.4, -0.3]
```

这些向量表示每个单词在嵌入空间中的位置，捕捉了单词之间的语义关系。例如，"soccer"和"playing"的向量表示在嵌入空间中更接近，表明它们在语义上更相关。

`GloVe`利用全局统计信息来建立一个单词共现矩阵，然后通过矩阵分解的方法学习得到每个单词的向量表示。`GloVe`既考虑了上下文信息，又考虑了全局统计信息，因此在一些任务上表现较好。


## 小结 
`Word embedding`可以应用于各种自然语言处理任务，如文本分类、命名实体识别、情感分析等。它不仅提供了更紧凑、更具语义的单词表示，还能够捕捉单词之间的语义关系，从而提升自然语言处理任务的性能。



参考文章：
1、<https://lilianweng.github.io/posts/2017-10-15-word-embedding/>
