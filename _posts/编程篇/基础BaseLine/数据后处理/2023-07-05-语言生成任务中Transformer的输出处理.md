---
layout: post
title: "语言生成任务中Transformer的输出处理"
date: 2023-06-01
author: cora Liu
categories: [编程篇,BaseLine,数据后处理]
---


## 写在前面
我们知道，`Transformer` 的输入是一个张量，经过计算之后输出也是一个张量。而我们的输入是一个序列（可以是文本也可以是其他序列），最终想要的目标也是一个序列。

而将输入转换成`tensor张量`是比较简单的事情。但是如何利用输出矩阵输出目标的序列呢？本文将以**语言生成任务**为例，介绍如何将`Transformer`的输出矩阵转换成我们想要的结果。


## Transformer 输入矩阵
`Transformer`可以应用于很多任务中。在不同任务中对于输出有不同的处理方式。

假设我们输入一个`(batch_size, seq_size, vocab_size)`的矩阵。对于`batch_size`，`seq_size`，`vocab_size`的理解，下面举个栗子：

比如我们有一个长度为`8`的词汇表，`vocab_size`为`8`。

<img src="/assets/imgs/ai/transformer/vocab-match.png" width="400" />

那么`“i like cats”`， `“i like dogs”`(忽略单复数)可以表示为：
<img src="/assets/imgs/ai/transformer/transformer-input.png" width="500" />

其中，每个词为一行，在词汇表中对应的位置设置为`1`，其他位置设置为`0`（或者可以采用其他的编码方式）。


## Transformer 输出矩阵
那么上面的输入矩阵经过`Transformer`处理之后，输出矩阵长什么样呢？

在**语言生成/序列生成**任务中，可以通过一个`Linear变换`（通常是一个全连接层）将其映射到词汇表的维度上。然后，可以使用生成模型（如softmax）将这些向量转换为概率分布，从而生成下一个词或字符。

经过`Transformer`处理之后，输出矩阵形状同输入矩阵一样，也是`(batch_size, seq_size, vocab_size)`，不过矩阵中的值同输入矩阵不同，输出矩阵中的值是概率分布。每一列代表对应的词的概率。

<img src="/assets/imgs/ai/transformer/transformer-output.png" width="500" />

对每一行的概率取**最大值所在的索引**如下 ⬇️
<img src="/assets/imgs/ai/transformer/transformer-output-max.png" width="500" />

最后，对照词汇表找到索引所对应的词，得到目标序列。

<img src="/assets/imgs/ai/transformer/vocab-match.png" width="400" />

以上简单举例介绍了在语音生成任务中`Transformer`对输出矩阵的处理。
