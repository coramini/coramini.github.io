---
layout: post
title: "状态字典state_dict简单介绍"
date: 2023-06-02
author: cora Liu
categories: [编程篇, PyTorch]
---

> 在模型训练中，我们经常会看到state_dict，本文简单介绍state_dict及其读写。

## 什么是state_dict

假设`model`是已经定义好的模型，那么就有 `model.state_dict()`是一个**方法**，用于返回PyTorch模型的当前**状态字典（state dictionary）**。

状态字典是模型的内部表示，其中包含了模型的所有可学习参数的当前值。这些方法包括 **权重** 和 **偏置值**。



下面是一个示例，展示了如何使用model.state_dict()方法获取模型的状态字典：

```python
import torch
import torch.nn as nn

# 创建一个模型

model = nn.Linear(10, 1)

# 获取模型的状态字典

state_dict = model.state_dict()

# 打印状态字典的内容

for param_name, param_tensor in state_dict.items():
    print(param_name, param_tensor.shape)

```

打印的结果如下：
```
OrderedDict([('weight', tensor([[-0.1535, -0.2079,  0.2593, -0.1100, -0.0041, -0.2067, -0.2417, -0.1254,
          0.0887,  0.2404]])), ('bias', tensor([0.0516]))])

weight torch.Size([1, 10])

bias torch.Size([1])
```

其中 `weight` 代表的是权重矩阵（张量），`bias` 表示偏置值，也我们经常所说的常数项。


## 保存state_dict
保存`state_dict`本质上就是保存了该`model`训练的参数值，这对于模型迁移非常重要，可以共享和重用模型的参数。

在`PyTorch`中，我们使用`torch.save()`来进行状态字典的保存。

```python
# 保存状态字典
torch.save(state_dict, 'model.ckpt')
```

## 加载state_dict

在进行模型复用的时候，我们可以直接加载state_dict重用模型的参数。在这里用`torch.load()`来执行加载操作。
```python
# 加载状态字典
torch.load('model.ckpt')
```
若我们想要恢复模型的权重，则利用 `model.load_state_dict` 可以达到我们的目的：

```python
# 恢复模型状态
model.load_state_dict(torch.load('model.ckpt'))
```

## 小结
状态字典在保存和加载模型时非常有用。你可以使用`torch.save()`函数将状态字典保存到文件中，并使用`torch.load()`函数加载状态字典以恢复模型的权重。这样，你可以方便地在不同的会话或计算设备之间共享和重用模型的参数。
