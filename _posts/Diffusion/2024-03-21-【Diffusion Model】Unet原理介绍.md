---
layout: post
title: "【Diffusion Model】U-Net原理介绍"
date: 2024-03-21
author: Cola Liu
categories: [Diffusion]
usemathjax: true
---

在扩散模型 `DDPM(Denoising Diffusion Probabilitistic Model)`中需要训练一个神经网络来学习加在数据上的噪声 $\epsilon$。并且神经网络预测的噪声的维度需要与输入的数据维度相同。

神经网络`U-Net`是一个常用的选择。

## U-Net 基本结构

`U-Net` 是一种用于图像分割的深度学习模型，其基本结构由`下采样（DownSample，也称为编码器)` 和 `上采样（Upsample，也称为解码器）`组成，并且采用了`跳跃连接（Skip Connections）`的设计。

- `下采样（DownSample，也称为编码器）`利用卷积操作和池化操作进行逐级下采样。下采样过程中，经过池化 `pool` 操作的数据空间分辨率变小，不过由于卷积 `conv` 操作，数据的通道数逐渐变大，从而可以学习图片的高级语义信息。

- `上采样（Upsample，也称为解码器）`与下采样的过程反过来，输入的空间信息和边缘信息会被逐渐恢复。

由于下采样和上采样过程形成了一个U形结构，所以称为“`U-Net`”

- `跳跃连接（Skip Connections）` 将编码器中某些层的特征图与对应的解码器层进行连接。这种设计可以帮助解码器利用编码器中的底层和高层特征信息，有助于提高图像分割的精度和细节保留能力。


下图是在扩散模型中采用的`U-Net`的网络结构 ⬇️

<img src="/assets/imgs/ai/diffusion/unet.png" />

## Time Embedding

由于 `U-Net` 接收的输入是 $x_t$ 和 时间 $t$。`DDPM` 在处理时间嵌入（time embedding）通常采用 `Sinusoidal Positional Embedding（正弦位置嵌入）`的方式。

#### Sinusoidal Positional Embedding
`Sinusoidal Positional Embedding` 是一种常用于处理时间或序列信息的方法之一。`Sinusoidal Positional Embedding` 通常用于为序列中的每个位置生成一个固定长度的向量表示，以便神经网络能够捕捉到序列中不同位置的信息。

> 这里跟Transformer在训练的时候用attention训练时，需要将文字再加上一个positional embedding的概念相同。

下面是相关的代码片段 ⬇️
```python
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
```


通常，我们对`U-Net`的残差模块进行修改，加入位置信息编码`time_embedding`。如下图所示：

<img src="/assets/imgs/ai/diffusion/unet-resnet.png" />

相关的代码片段如下 ⬇️
```python
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
```

总体而言，`U-Net` 模型通过编码器提取图像的特征信息，并通过解码器将特征信息恢复到原始分辨率，同时利用跳跃连接保留多层次的特征信息，从而实现对图像的准确分割。