---
layout: post
title: "self-attention模型源码实现"
date: 2023-06-01
author: Cola Liu
categories: [编程篇,BaseLine,模型设计]
---


在`self-attention`模型源码中，直接调用Pytorch封装好的`TransformerEncoderLayer`函数是很简单的。不过模型里面还有诸如`Linear`、`permute`、`transpose`、`mean`、`sigmoid`、`argmax`等操作，我们也需要知道它们到底发挥了什么作用。

> 本文主要介绍`self-attention`模型源码中，输入的tensor发生的变化，以及介绍`Linear`、`permute`、`transpose`、`mean`、`sigmoid`、`argmax`等处理tensor变换的操作。

## model
话不多说直接上源码 ⬇️

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
 def __init__(self, d_model=80, n_spks=600, dropout=0.1):
  super().__init__()
  # Project the dimension of features from that of input into d_model.
  self.prenet = nn.Linear(40, d_model)
  # TODO:
  #   Change Transformer to Conformer.
  #   https://arxiv.org/abs/2005.08100
  self.encoder_layer = nn.TransformerEncoderLayer(
   d_model=d_model, dim_feedforward=256, nhead=2
  )
  # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

  # Project the the dimension of features from d_model into speaker nums.
  self.pred_layer = nn.Sequential(
   nn.Linear(d_model, d_model),
   nn.Sigmoid(),
   nn.Linear(d_model, n_spks),
  )

 def forward(self, mels):
  """
  args:
   mels: (batch size, length, 40)
  return:
   out: (batch size, n_spks)
  """
  # out: (batch size, length, d_model)
  out = self.prenet(mels)
  # out: (length, batch size, d_model)
  out = out.permute(1, 0, 2)
  # The encoder layer expect features in the shape of (length, batch size, d_model).
  out = self.encoder_layer(out)
  # out: (batch size, length, d_model)
  out = out.transpose(0, 1)
  # mean pooling
  stats = out.mean(dim=1)

  # out: (batch, n_spks)
  out = self.pred_layer(stats)
  return out
```
源码看似很简单，但是每一步`tensor`发生了什么变化，我们很难看出来。在这里我们用一张图来展示上面的`model`。
（点击看大图 ⬇️）

<img src="/assets/imgs/ai/self-attention/model.png" />

## 输入矩阵

首先，经过我们前面对于数据的预处理。假设我们输入的`batch_size=32`，即每个batch中有32条数据。每条数据格式为`[128,40]`，则我们的输入矩阵`shape`为：`torch.Size([32,128,40])`。

<img src="/assets/imgs/ai/self-attention/input.png" />


## prenet
在本模型中，第一步操作是`prenet`，`prenet`中是一个全联接操作`(nn.Linear)`。输入维度为40，输出维度为80。
```python
# 定义
self.prenet = nn.Linear(40, d_model) # 已知 de_model=80
# forward
out = self.prenet(mels)
```

<img src="/assets/imgs/ai/self-attention/prenet.png" />

由上图可以看出，经过`prenet`操作后，tensor由 `torch.Size([32,128,40])` 转换成 `torch.Size([32,128,80])`。

## permute
接着是`permute`函数操作。`permute`函数在本blog另一篇文章里面有详细介绍，这里不再赘述。
```python
# forward
 out = out.permute(1, 0, 2) # 把维度0和维度1掉换
```
<img src="/assets/imgs/ai/self-attention/permute.png" />

`permute(1, 0, 2)`的作用是把维度0和维度1掉换，那么这里维度0是`batch_size`=32，维度1是` length`=128。

tensor形状由 `torch.Size([32,128,80])` 转换成 `torch.Size([128,32,80])`。

## encoder_layer
接下来是`encoder_layer`，也就是self-attention的核心代码。
```python
# 定义
self.encoder_layer = nn.TransformerEncoderLayer(
   d_model=d_model, dim_feedforward=256, nhead=2
  ) # d_model=80

# forward
# out: (batch size, length, d_model)
out = self.prenet(mels)
# out: (length, batch size, d_model)
out = out.permute(1, 0, 2)
# The encoder layer expect features in the shape of (length, batch size, d_model).
out = self.encoder_layer(out)
# out: (batch size, length, d_model)
```
<img src="/assets/imgs/ai/self-attention/encoder_layer.png" />

在这个环节中，tensor的形状没有发生变化。

## transpose
接下来是`tranpose`函数操作。同样，`transpose`函数在本blog另一篇文章里面有详细介绍，这里不再赘述。

```python
# out: (batch size, length, d_model)
out = out.transpose(0, 1)
```

<img src="/assets/imgs/ai/self-attention/transpose.png" />


从上图可以看出来，`tranpose(0, 1)`把维度0和维度1掉换。tensor形状由 `torch.Size([128,32,80])` 转换成 `torch.Size([32,128,80])`。恢复到`permute`之前的shape。

## mean
`mean`函数是对某个维度求均值，这里`dim=1`。

```python
# mean pooling
stats = out.mean(dim=1)
```

<img src="/assets/imgs/ai/self-attention/mean.png" />

对维度1求均值，tensor形状由 `torch.Size([32,128,80])` 转换成 `torch.Size([32,80])`。


## pred_layer

`pred_layer`为结果预测层。里面包含三个操作：Linear、sigmoid、Linear。

通过第一个全联接层后，tensor的形状没有发生变化。
<img src="/assets/imgs/ai/self-attention/pred_layer_1.png" />

仔细看看最后一个全联接层。由于我们有600个分类（n_spks），即类别1，类别2，类别3，...，类别600。因此，输出的结果应该是`[batch_size,600]`，600里面对应是该条数据是类别1，类别2，类别3....类别600的概率。
<img src="/assets/imgs/ai/self-attention/pred_layer_2.png" />

这个理解起来有点抽象，我们画个表帮助理解一下，也就是如下表（32 * 600） 所示 ⬇️

对于batch中每一条数据，这条数据属于类别1-600的概率分别展示出来，就是到目前为止，这个模型的输出结果。

| |类别1 | 类别2|类别3 |类别... | 类别600|
|--|--|--|--|--|--|
|数据1|0.1|0.3|1|--|0.4|
|数据2|1|0.2|0.1|--|0.8|
|数据3|0.7|0.8|0.2|--|0.1|
|...|--|--|--|--|--|
|数据32|0.5|0.2|0.8|--|0.9|


## argmax
`argmax`就更好理解了，求出当概率最大时的自变量（也就是类别）
<img src="/assets/imgs/ai/self-attention/argmax.png" />
还是用表格来表示一下比较好理解 ⬇️

我们在上面的表格中，对所有类别的概率进行比较，找到概率最大的值所属类别，则认为该数据属于该类别。

| | | 
|--|--|
|数据1|类别3(数据1为类别3时概率为1最高，取类别3)|
|数据2|类别1(数据2为类别1时概率为1最高，取类别1)|
|数据3|类别2(数据3为类别2时概率为0.8最高，取类别2)|
|...|--|
|数据32|类别600|数据32为类别600时概率为0.9最高，取类别600|

### argmax与max的区别
那`argmax`和`max`是什么区别呢，当函数f(x)最大时，`argmax`取x自变量的取值（也就argument的意思）`max`则是取f(x)最大值。

同样也是这个例子，我们来比较一下`argmax`和`max`

| |argmax | max |
|--|--|--|
|数据1|类别3|1|
|数据2|类别1|1|
|数据3|类别2|0.8
|...|--|
|数据32|类别600|0.9|

从定义上来说，`argmax`取值可以有多个，`max`取值一般只有1个。


## 小结
在 `self-attention`源码中，除了关键的encoder操作，还有很多细节需要考虑。我们需要清楚其中每一步tensor的变化。