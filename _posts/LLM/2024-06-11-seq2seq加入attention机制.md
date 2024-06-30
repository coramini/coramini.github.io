---
layout: post
title: "seq2seq模型和attention机制"
date: 2024-06-11
author: Cola Liu
categories: [LLM]
---
`Seq2Seq` 模型是一种用于处理序列数据的深度学习模型架构。

它的基本思想是将一个变长的输入序列转换为一个变长的输出序列。这个过程通常包括两个主要组件：编码器（`Encoder`）和解码器（`Decoder`）。
<img src="/assets/imgs/ai/seq2seq/seq2seq-attention-1.png" />



编码器（`Encoder`）将输入序列编码为一个上下文向量 `context C`。解码器（`Decoder`）接收上下文向量 `context C` 和前一步的输出。


下面进一步看看它的具体运行机制吧～

### 1、运行机制

以下是`Seq2Seq`模型的运行机制：

1. **输入序列**：编码器接收输入序列 $ X = (x_1, x_2, ..., x_T) $。
2. **编码过程**：编码器将输入序列编码为一个上下文向量 $ C $，同时生成一系列隐藏状态 $ H = (h_1, h_2, ..., h_T) $。
3. **解码过程**：解码器接收上下文向量 $ C $ 和前一步的输出（或初始时接收特殊的起始符号），逐步生成输出序列 $ Y = (y_1, y_2, ..., y_T') $。
4. **输出序列**：解码器生成完整的输出序列。

具体过程可以看下图的直观展示 ⬇️


<img src="/assets/imgs/ai/seq2seq/seq2seq-attention-2.png" />

其中，中间变量`C`可以是`Encoder`直接输出的 $h_T$，也可以由$h_T$ 变换得到。

### 2、另一种模型

如果将`C`当作`Decoder`的每一时刻输入，则是`Seq2Seq`模型的第二种模型，如下图所示 ⬇️

<img src="/assets/imgs/ai/seq2seq/seq2seq-attention-3.png" />
<img src="/assets/imgs/ai/seq2seq/seq2seq-attention-4.png" />

### 3、引入attention机制

在传统的`Seq2Seq`模型中，编码器`（Encoder）`将输入序列编码成一个固定大小的上下文向量`context C`，接着解码器`（Decoder）`基于这个上下文向量生成输出序列。

然而，对于长序列来说，`context C` 可能不足以捕捉整个输入序列的信息，这会导致性能下降。

注意力机制的引入解决了这个问题。简单的结构如下图所示 ⬇️

<img src="/assets/imgs/ai/seq2seq/seq2seq-attention-5.png" />

从上图中可以看出，引入attention机制后，它允许解码器在生成每个输出时，都能动态地选择输入序列中最相关的信息。

我们进一步将上图细化，把计算过程画出来如下 ⬇️
<img src="/assets/imgs/ai/seq2seq/seq2seq-attention-6.png" />

在seq2seq模型中，注意力机制（Attention Mechanism）是通过动态地计算解码器每一步生成输出时对编码器输出的关注度来实现的。具体来说，注意力机制允许解码器在生成每个输出时，根据输入序列的不同部分计算一个加权和，而不仅仅依赖于一个固定的上下文向量。这大大缓解了传统seq2seq模型在处理长序列时信息丢失的问题。

### 注意力机制在seq2seq模型中的实现步骤

以下是seq2seq模型中注意力机制的详细实现步骤：

1. **计算注意力权重（Attention Weights）**：
   - 对于解码器的每个时间步 $ t $，计算当前隐藏状态 $ s_t $（即下图的 $h_{i-1}^\prime$) 和所有编码器隐藏状态 $ h_i $ 之间的相似度得分 $ \alpha_{t,i} $。

2. **计算上下文向量（Context Vector）**：
   - 根据注意力权重对编码器隐藏状态进行加权求和，得到当前时间步的上下文向量 $ c_t $。

3. **生成解码器输出（Decoder Output）**：
   - 解码器结合当前时间步的隐藏状态 $ s_t $(即下图的 $h_{i-1}^\prime$)) 和上下文向量 $ c_t $ 生成输出。

具体实现细节如图所示 ⬇️

<img src="/assets/imgs/ai/seq2seq/seq2seq-attention-7.png" />


## 小结
通过引入注意力机制，解码器在生成每个输出时，可以动态地关注输入序列的不同部分，从而提升了seq2seq模型的性能，尤其是在处理长序列和复杂依赖关系的任务中。