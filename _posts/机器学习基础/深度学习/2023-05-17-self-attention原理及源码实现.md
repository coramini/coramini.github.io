---
layout: post
title: "self-attention原理及源码实现"
date: 2023-05-17
author: Cola Liu
categories: [机器学习基础]
pin: true
---

`self-attention`是一个新的模型架构。对待新的模型架构，一般是从已有的知识入手，分析新旧模型之间的异同，提出新的架构可以帮助以前的模型解决什么痛点。

## self-attention的提出

### 从简单的模型架构开始

我们先来看常规的模型架构，通常，我们只有一个输入输出（偶尔会有多个输出，比如给出一张动物的图片，判断这张图片中的小动物是猫还是狗狗，但是本质上还是一个输出，输出的是这张图片是猫或狗的概率。）

<img src="/assets/imgs/ai/self-attention/单输入输出.png" width="300" />

输入的`x`经常是一个`vector`或者一整个表格，然后我们把它拼成一个长`vector`。

### 处理多输入

那么问题来了，如果我们有多个输入呢？也就是输入有多个`vector`呢？如下图所示，我们有4个输入，然后对应有4个输出 ⬇️

<img src="/assets/imgs/ai/self-attention/多输入输出-1.png" width="500" />

很简单，我们把这4个输入拆开成单个，逐一输入计算就完事～或者干脆作为一个矩阵丢进去计算也行！这个问题暂且就解决了。

<img src="/assets/imgs/ai/self-attention/多输入输出-2.png" width="500" />

### 输入数据之间有联系

等等，我们再考虑其他的情况：

如果这4个输入彼此之间是有一定关系的呢？比如，`i saw a saw`，此`saw`非彼`saw`。

特别是在语言分析中，字与字之间的关系不同，含义就不同。"小心地滑"里面的地，可以解释为“地面”或者“得的地”中的“地”，后加动词。

<img src="/assets/imgs/ai/self-attention/多输入输出-3.png" width="500" />

显然，这里考虑了输入与输入之间的“先后”关系，可为“位置关系”，也可以理解为“时间关系”。

这里有一个常规的解决方法是，把前面`x1`的计算结果`a1`,作为下一个`x2`的输入。这样一来，“先后”关系得以保存下来。

<img src="/assets/imgs/ai/self-attention/rnn架构.png" width="500" />

这就是我们常常听到的RNN架构，原理非常简单，即把前面的输出结果作为后面计算过程的输入。这是一种常规的串行的解决方法。

但是由于串行效率过低，并且越前面的信息由于计算次数越多，会导致信息的逐级递减。因此，有人提出了`self-attention`这个模型架构。

<img src="/assets/imgs/ai/self-attention/self-attention-1.png" width="500" />

`self-attention`能够很好地解决RNN中串行计算的问题，它的原理是计算向量与向量之间的位置关系/相似度。

### self-attention 具体架构

我们来看一下`self-attention`的架构。

- 1、将输入`vector` `x1` 与一个权重矩阵 `Wq` 相乘，得到 `q1`；
- 2、将输入`vector` `x1` 与另一个权重矩阵 `Wk` 相乘，得到 `k1`；
- 3、以此类推得到`q2`、`k2`、`q2`、`k3`等；

<img src="/assets/imgs/ai/self-attention/self-attention-2.png" width="400" />

> 到此为止，我们的参数只有 **Wq** 和 **Wk**

- 4、对于每一个`q1`，我们用`k1`、`k2`、`k3`、`k4`与之做点乘（dot product）

> 这里就可以体现出向量与向量之间的关系了。何以见得呢？

#### 向量点乘的几何意义

我们知道，向量点乘的结果是一个标量，即一个数。先来看看向量点乘是怎么定义的：

<img src="/assets/imgs/ai/self-attention/向量点乘的计算过程.png" width="500"/>

用数学式子表示还是比较抽象，回到向量点乘的几何意义。可以看到，向量点乘的几何意义主要有以下几个点：

- **投影**：向量的点乘可以用来计算一个向量在另一个向量方向上的投影长度。

- **夹角**：向量的点乘还可以用来计算两个向量之间的夹角。
    这个夹角θ表示了两个向量之间的方向一致程度，当两个向量平行时夹角为0°，当两个向量垂直时夹角为90°。

<img src="/assets/imgs/ai/self-attention/向量点乘的几何意义.png" width="300"/>

以上简单介绍了点乘这个操作如何体现输入vector之间的位置关系以及相似度的。接着我们继续看self-attention的计算过程。

- 5、将输入`vector` `x1` 与一个权重矩阵 `Wv` 相乘，得到 `v1`；以此类推

> 到此为止，我们的参数只有 **Wq** 、**Wk** 和 **Wv**

- 6、将步骤4得到的点乘结果`α11`（q1·k1）乘于`v1`，`α12`（q1·k2）乘于`v2`,...
- 7、将上述结果 **累加** 可以得到`y1`

<img src="/assets/imgs/ai/self-attention/self-attention-3.png" />

- 8、关于`y2`、`y3`、`y4`的计算过程跟`y1`的计算过程相同。


可以看出，`self-attention`架构的设计，主要利用了向量（矩阵）的几何意义来保存相关的位置关系的。这是一个很好的解决问题的思路之一。

<img src="/assets/imgs/ai/self-attention/self-attention原理.png" />

`self-attention`与 传统的 `RNN` 相比，从串行计算变成并行计算，可以大大提升整个模型的运行效率。


## 计算过程

1、首先，我们来看 `α11` 的计算过程。

<img src="/assets/imgs/ai/self-attention/self-attention-2.png" width="300" style="display:block;"/>

2、从上图可知，**α11 = q1 · k1**，点乘 `dot product` 可以写成 `k1 的逆` 与 `q1` 做矩阵相乘。

<img src="/assets/imgs/ai/self-attention/self-attention-4.png" width="300" />

3、同理，**α12 = q1 · k2**，即可以写成 `k2 的逆` 与 `q1` 做矩阵相乘。

<img src="/assets/imgs/ai/self-attention/self-attention-5.png" width="300" />

4、计算完 `α1*`（输入的x1与其他vector*的关系计算），我们来计算`α2*`（输入的x2与其他vector* 的关系计算）。

可以得到， **α21 = q2 · k1**，可以写成 `k1 的逆` 与 `q2` 做矩阵相乘。

**α22 = q2 · k2**，可以写成 `k2 的逆` 与 `q2` 做矩阵相乘。

<img src="/assets/imgs/ai/self-attention/self-attention-6.png" width="400" />

5、我们把所有的`q1、q2、q3、q4` 以及 `k1、k2、k3、k4` 加进来计算。可以分别得到 `α11-α14`，`α21-α24`、`α31-α34`、`α41-α44`...

<img src="/assets/imgs/ai/self-attention/self-attention-7.png" width="500" />

6、在这里，我们可以把输入`x1、x2、x3、x4`这四个vector拼成一个矩阵 `I`。

那么q1、q2、q3、q4组成的矩阵 `Q` 则为 `I` 与 `Wq` 矩阵相乘的结果。k1、k2、k3、k4组成的矩阵 `K` 则为 `I` 与 `Wk` 矩阵相乘的结果。

从图中，可以看到我们把 `K` 倒转过来了，即计算 `K` 的逆。

<img src="/assets/imgs/ai/self-attention/self-attention-8.png" width="500" />

7、最后，我们来看看 `v1、v2、v3、v4`。由于 **y1 =  α11 ✖️ v1 + α12 ✖️ v2 + α13 ✖️ v3 + α14 ✖️ v4**，刚好也可以写成矩阵相乘的结果！

也即是将上面计算的结果，再与`v1、v2、v3、v4`组成的矩阵 `V` 的**逆** 做矩阵相乘的计算。

<img src="/assets/imgs/ai/self-attention/self-attention-9.png" width="600" />

8、把`self-attention`的计算过程总结一下，可以得到 ⬇️

原理就是矩阵的运算，通过矩阵取逆、矩阵相乘等操作来实现相关的计算。
<img src="/assets/imgs/ai/self-attention/self-attention-10.png" width="700" />


在`self-attention`模型源码中，直接调用Pytorch封装好的`TransformerEncoderLayer`函数是很简单的。不过模型里面还有诸如`Linear`、`permute`、`transpose`、`mean`、`sigmoid`、`argmax`等操作，我们也需要知道它们到底发挥了什么作用。

## 源码实现
下面主要介绍`self-attention`模型源码中，输入的tensor发生的变化，以及介绍`Linear`、`permute`、`transpose`、`mean`、`sigmoid`、`argmax`等处理tensor变换的操作。

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

### 输入矩阵

首先，经过我们前面对于数据的预处理。假设我们输入的`batch_size=32`，即每个batch中有32条数据。每条数据格式为`[128,40]`，则我们的输入矩阵`shape`为：`torch.Size([32,128,40])`。

<img src="/assets/imgs/ai/self-attention/input.png" />


### prenet
在本模型中，第一步操作是`prenet`，`prenet`中是一个全联接操作`(nn.Linear)`。输入维度为40，输出维度为80。
```python
# 定义
self.prenet = nn.Linear(40, d_model) # 已知 de_model=80
# forward
out = self.prenet(mels)
```

<img src="/assets/imgs/ai/self-attention/prenet.png" />

由上图可以看出，经过`prenet`操作后，tensor由 `torch.Size([32,128,40])` 转换成 `torch.Size([32,128,80])`。

### permute
接着是`permute`函数操作。`permute`函数在本blog另一篇文章里面有详细介绍，这里不再赘述。
```python
# forward
 out = out.permute(1, 0, 2) # 把维度0和维度1掉换
```
<img src="/assets/imgs/ai/self-attention/permute.png" />

`permute(1, 0, 2)`的作用是把维度0和维度1掉换，那么这里维度0是`batch_size`=32，维度1是` length`=128。

tensor形状由 `torch.Size([32,128,80])` 转换成 `torch.Size([128,32,80])`。

### encoder_layer
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

### transpose
接下来是`tranpose`函数操作。同样，`transpose`函数在本blog另一篇文章里面有详细介绍，这里不再赘述。

```python
# out: (batch size, length, d_model)
out = out.transpose(0, 1)
```

<img src="/assets/imgs/ai/self-attention/transpose.png" />


从上图可以看出来，`tranpose(0, 1)`把维度0和维度1掉换。tensor形状由 `torch.Size([128,32,80])` 转换成 `torch.Size([32,128,80])`。恢复到`permute`之前的shape。

### mean
`mean`函数是对某个维度求均值，这里`dim=1`。

```python
# mean pooling
stats = out.mean(dim=1)
```

<img src="/assets/imgs/ai/self-attention/mean.png" />

对维度1求均值，tensor形状由 `torch.Size([32,128,80])` 转换成 `torch.Size([32,80])`。


### pred_layer

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


### argmax
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

#### argmax与max的区别
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
