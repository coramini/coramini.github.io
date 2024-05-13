---
layout: post
title: "详细了解transformer"
date: 2023-07-03
author: Cola Liu
categories: [编程篇,BaseLine,模型设计]
---

## 写在前面
`Transformer`是一种用于自然语言处理和其他序列到序列任务的机器学习模型。它于2017年由Google的研究人员提出，并在机器翻译任务中取得了重大突破。

`Transformer`模型由编码器`（Encoder）`和解码器`（Decoder）`组成。
- 编码器负责将输入序列进行编码，解码器负责根据编码器的输出生成目标序列。
- 编码器和解码器都由 **多层堆叠的自注意力层和前馈神经网络层** 组成。

自注意力机制 `Self-Attention` 是 `Transformer` 的核心组件之一，本文先从 `Self-Attention` 开始。

## Self-Attention

### Self-Attention 结构
自注意力机制`（Self-Attention）`是 `Transformer` 模型的核心组件之一，用于捕捉输入序列中各个位置之间的依赖关系。

`Self-Attention` 结构图如下 ⬇️

<img src="/assets/imgs/ai/transformer/self-attention.png" />

假设我们有一个输入矩阵：`embedded_input`。 形状为 `(seq_len x hidden_dim)`

其中， `(seq_len x hidden_dim)`是什么意思呢？ 比如，我们有一个序列：`i saw a cat`，维度为 6。那我们可以表示为：
<img src="/assets/imgs/ai/transformer/input-shape.png" />

这里的`seq_len`为4（4个单词，我们也可以用其他的划分方法），每一个单词用一个6维的向量表示，即`hidden_dim`为6。

### 1、计算查询矩阵Q、键矩阵K和值矩阵V
<img src="/assets/imgs/ai/transformer/q.png" />

<img src="/assets/imgs/ai/transformer/k.png" />

<img src="/assets/imgs/ai/transformer/v.png" />

那么问题来了，`Wk`、`Wq`、`Wv` 怎么初始化呢？这个问题跟 `num_heads` 有关。

在多头 `Self-Attention` 中，假设输入的 `hidden_dim` 为 6，有2个头 `num_heads`，那么每个权重矩阵的维度`head_dim`为 3。它们之间的关系如下 ⬇️

<img src="/assets/imgs/ai/transformer/hidden-dim-cal.png" />

`Wk`、`Wq`、`Wv` 即可以初始化为`hidden_dim x head_dim`的权重矩阵。

> 需要注意的是，这里的 `head_dim`就是我们在计算 `attention score` 开根号的时候的维度 `d`。下文会继续讲解，在此先简单提及。

```python
# 代码示例
query = self.query_linear(query).view(batch_size. seq_len, self.num_heads, self.head_dim).transpose(1, 2)
key = self.key_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
value = self.value_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
```

### 2、对查询矩阵Q和键矩阵K进行相似度计算
对查询矩阵`Q`和键矩阵`K`进行相似度计算，可以使用点积注意力`（Dot-Product Attention）`或其他相似度度量方法，如`缩放点积注意力（Scaled Dot-Product Attention）`等。

<img src="/assets/imgs/ai/transformer/qkt.png" />

一般为了防止数值差别过大，我们会把结果除以跟维度有关的数值。（图中省略）

<img src="/assets/imgs/ai/transformer/qkt-d.png" width="50" />

这里下方开根号的d，就是`head_dim`。即权重矩阵`Wk`、`Wq`、`Wv`的**列数**。

```python
# 代码示例
scores = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
```


### 3、softmax

对相似度进行`softmax`归一化，得到注意力权重矩阵A。

<img src="/assets/imgs/ai/transformer/soft-max.png" />

```python
attention_weights = F.softmax(scores, dim=-1)
```
### 4、注意力加权求和

将注意力权重矩阵与值矩阵`V`相乘，得到加权求和的表示。

<img src="/assets/imgs/ai/transformer/matmul-v.png" />

这里的加权求和相当于对每个位置的值进行加权平均，权重由注意力权重矩阵确定。

以上过程总结一下，就是下方公式所示：

<img src="/assets/imgs/ai/transformer/attention-score.png" />

### 5、输出表示计算
将加权求和的表示进行线性变换，得到最终的output。


<img src="/assets/imgs/ai/transformer/linear-output.png" />


细心的童鞋可以发现，我们用的 `head_dim = 3` 进行计算，也就是在上面只计算了其中一个头。

但是，我们输入的有 2 个头（`num_heads = 2`）多个头的怎么计算的呢？

## Multi-Head Attention

### Multi-Head Attention 结构
在传统的 `Self-Attention` 中，通过计算每个位置与其他位置的相似度，获得每个位置对其他位置的注意力矩阵。

而`Multi-Head Attention` 则进一步扩展了这一机制，通过并行地使用`多个独立的注意力头`来处理输入序列。
<img src="/assets/imgs/ai/transformer/multihead-attention.png" />


### 1、分头计算 Attention 矩阵

<img src="/assets/imgs/ai/transformer/mul-z1-z2.png" />


在上面例子中，我们计算了第一个头，输入为`embedded_input`，输出为`Z1`。

按照同样的方式，计算出第二个头，输入为`embedded_input`，输出为`Z2`。

### 2、将每个注意力头的加权求和结果进行拼接

将每个注意力头的加权求和结果进行`拼接`，得到多头注意力的输出表示 `Z1Z2`。

### 3、输出线性变换

对多头注意力的输出表示进行线性变换，得到最终的注意力输出。

<img src="/assets/imgs/ai/transformer/mul-linear-output.png" />

可以看到，经过`multihead-attention`或者`self-attention`计算之后，输入矩阵的形状与输出矩阵的形状相同，即 `seq_len x hidden_dim`。

<img src="/assets/imgs/ai/transformer/self-attention-output.png" />


## Encoder
`Transformer`的结构包括 `Encoder` 和 `Decoder` 两个部分，每个部分的核心组件为 `Multihead-Attention`，先来看看`Encoder`。

### Encoder 结构
`Encoder`由多个相同结构的编码器层堆叠而成，每个编码器层包含两个子层：多头自注意力层`（Multi-Head Attention）`和前馈神经网络层`（Feed-Forward Neural Network）`。
<img src="/assets/imgs/ai/transformer/encoder-layer.png" />

### 1、Input Embedding
首先，从下往上看，最下面是 `Input Embedding` 嵌入层。

输入嵌入层将离散输入序列映射为`连续向量`，方便模型处理和理解输入，提高任务性能。

<img src="/assets/imgs/ai/transformer/embed.png" />
 
```python
# 代码示例
self.embedding = nn.Embedding(embed_dim, hidden_dim)
```
其中，`embed_dim`为原始输入序列的维度，这里假设是 5，`hidden_dim`为经过嵌入层变换之后的维度，这里假设为6，它将参与到后续计算中。

### 2、Positional-Encoding
在原来的`Attention`计算中，由于矩阵并行计算，位置信息丢失了。因此，研究人员们提出了位置编码`(Positional-Encoding)`来解决这个问题。

位置编码`（Positional Encoding）`用于在Transformer模型中为输入序列中的每个位置添加位置信息，以帮助模型理解和捕捉序列中的顺序关系。


`Position Embedding` 用 `PE`表示，`PE` 的维度与`Word Embedding` 是一样的。`PE` 可以通过训练得到，也可以使用某种公式计算得到。在 `Transformer` 中采用了后者，计算公式如下：

<img src="/assets/imgs/ai/transformer/positional-encoding.png" />

其中，`pos` 表示单词在句子中的位置，`d` 表示 `PE`的维度 (与`Word Embedding` 一样)，2i 表示偶数的维度，`2i+1` 表示奇数维度 (即 `2i≤d`, `2i+1≤d`)。

经过位置编码之后，最终我们得到一个 `seq_len x hidden_dim` 的输入矩阵 `embedded_input`。

<img src="/assets/imgs/ai/transformer/position-embedding.png" />

```python
# 代码示例
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self,max_seq_len = max_seq_len

        position_encoding = torch.zeros(max_seq_len, hidden_dim)
        positions = torch.range(0, max_seq_len).unsequeeze(1)
        div_term = torch.exp(torch.arrange(0, hidden_dim, 2))  * -(math.log(10000.0) / hidden_dim))
        position_encoding[:, 0::2] = torch.sin(positions * div_term)
        positions_encoding[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('position_encoding', position_encoding)

    def forward(self, input):
        seq_len = input.size(1)
        if seq_len > self.max_seq_len:
            raise RuntimeError("Input sequence length exceeds maximum sequence length for positional encoding")
        position_encoding = self.positional_encoding[:seq_len, :].unsequeeze(0)
        return input + position_encoding
```
### 3、ADD & Norm
在这个地方，`ADD`是残差连接操作。残差连接`（Residual Connection）`是在`Transformer`模型中的编码器和解码器层之间引入的一种跳跃连接，将输入与输出相加，用于缓解梯度消失问题。

残差连接操作很简单，它的过程如下图所示 ⬇️

<img src="/assets/imgs/ai/transformer/residual.png" />

如下图所示，我们把 `Attention` 输出与输入相加，得到新的向量矩阵 `residual_output`。计算过程中矩阵形状依旧保持不变。
<img src="/assets/imgs/ai/transformer/residual-output.png" />

```python
# 代码示例
self_attention_output, _ = self.self_attention(embedded_input, embedded_input, embedded_input)
# ADD
residual_output = embedded_input + self_attention_output
```

`Norm` 则简单一点，将向量进行归一化操作。

向量归一化（`Vector Normalization`）可以消除向量的尺度影响，使其具有单位长度，有助于提高模型的稳定性和泛化能力，加速收敛速度，减少模型训练中的梯度爆炸和梯度消失问题。
<img src="/assets/imgs/ai/transformer/norm.png" />

```python
# 代码示例
normalized_output = torch.layer_norm(residual_output, normalize_shape=[hidden_dim])
```

### 4、Feed-Forward
前馈神经网络`(Feed-Forward Neural Network)`通过多层线性变换和非线性激活函数对输入进行逐层处理，以提取特征和进行非线性变换。

<img src="/assets/imgs/ai/transformer/ff.png" />

经过多层线性变换后，输出矩阵的形状与输入矩阵的形状保持不变。

```python
# 代码示例
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, input):
        hidden = self.fc1(input):
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)
        return output
```

## Decoder

### Decoder 结构
`Decoder` 跟 `Encoder` 的关键操作基本相同。它比`Encoder`多了一个`Multihead-Attention`层+`ADD/Norm`操作，其他基本保持不变。

<img src="/assets/imgs/ai/transformer/decoder-layer.png" />

在这里，我们重点介绍一下 `Attention Mask`。

### 1、Attention Mask

从`Attention`的计算原理来看，我们一般会关注到整个序列上前后位置全部的输入信息。但是有时候我们需要对信息进行遮挡，或者改变注意力的范围，这就用到了`mask`。

**在`Decoder`中，为了避免解码器在生成每个单词时依赖于后面的单词，可以使遮挡掩码，只允许模型在解码过程中依赖于前面的位置。**

`mask`通常是放在 `Self-Attention` 层中的一个操作。 位于 注意力矩阵计算之后， `soft-max` 函数之前。
<img src="/assets/imgs/ai/transformer/where-mask.jpg" />

```python
# 代码示例
scores = scores.masked_fill(attention_mask == 0, float("-inf"))
```

在这里，我们把原始输入矩阵与`Attention Mask`做按位相乘操作（逐位相乘）。
<img src="/assets/imgs/ai/transformer/mask.png" />

通过注意力掩码，可以控制模型对输入序列中不需要关注的位置进行屏蔽，确保模型在计算注意力权重时只关注有意义的信息，提高模型的精确性和泛化能力。

接下来，就是`soft-max`操作以及`按位加权`操作，跟前面一样一样的hhh，在此不再重复展开。

<img src="/assets/imgs/ai/transformer/mask-relu.png" />
<img src="/assets/imgs/ai/transformer/mask-matmul-v.png" />


⚠️ 注意：上文介绍的 `Transformer Encoder` 和 `Decoder` 的结构，只是其中一个层。
我们可以重复堆叠多个 `Encoder/Decoder`层，通常是6层或更多，以逐步建模和提取输入序列的特征。

## 小结

传统的循环神经网络`（RNN）`在处理序列数据时，需要按顺序逐个元素进行处理，这导致计算效率较低。

而`Transformer`采用了一种并行计算的方式，它能够同时处理整个输入序列，使得训练和推理的速度大大提高。

`Transformer`模型由编码器`（Encoder）`和解码器`（Decoder）`组成。编码器负责将输入序列进行编码，解码器负责根据编码器的输出生成目标序列。编码器和解码器都由多层堆叠的自注意力层和前馈神经网络层组成。

## 参考资料
https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=61
https://zhuanlan.zhihu.com/p/338817680