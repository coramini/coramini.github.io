---
layout: post
title: "transpose函数的用法"
date: 2023-05-23
author: cora Liu
categories: [编程篇, PyTorch]
---

## transpose

如果原始张量的维度顺序是(0, 1, 2)，而你想要将维度顺序变为(0, 2, 1)，你可以使用transpose函数进行操作。

```python
import torch
x = torch.randn(3, 4, 5)
y = x.transpose(1, 2)  # 将维度1和2进行交换

```

在上述示例中，x是一个形状为(3, 4, 5)的张量，transpose操作将维度1和维度2进行交换，得到的张量y的形状为(3, 5, 4)。直观来看就是行列转置了～

<img src="/assets/imgs/ai/PyTorch/permute/transpose.png" />

## transpose 与 permute

那我们会想到`permute`，在前面写了一整篇介绍 `permute`的文章。本质上两者都是做维度转置的工作。
transpose 一般交换的是两个维度，如果需要交换多个维度，那么就需要多次使用 `transpose` 或者改用 `permute`。

与`transpose`不同，`permute`允许对所有维度进行重新排列，而不仅仅是交换两个维度。
