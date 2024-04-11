---
layout: post
title: "【Stable Diffusion】源码解析(一)"
date: 2024-04-03
author: "Cola Liu"
categories: [Diffusion, Stable Diffusion weui]
usemathjax: true
---

<img src="/assets/imgs/ai/diffusion/diffusion000.jpeg" />


`Stable Diffusion` 是一种信息传播模型，通常应用于网络分析、社交网络、传播学等领域。它的设计 基于`Diffusion`模型,主要分为三个阶段（以`txt2img`为例）：
- **Encoder**： 包括了输入的文本编码和图像编码等。
    - **文本编码**：主要采用`Text Embedding`技术。文本嵌入`（Text Embedding）`将文本数据转换为向量表示。
    - **图像编码**：主要将原图片进行压缩，转换成潜在空间中的`latent repretation`。

- **Generation model**： 原基础Diffusion模型，核心采用的是 `U-Net` 网络结构，`U-Net`是`Diffusion`的核心结构。相关的类在`ddpm.py`和`openaimodel.py`等文件中进行定义。

- **Decoder**： 对生成的`latent represation`进行解码，转化为图像。

过程简化如下图所示 ⬇️

<img src="/assets/imgs/ai/diffusion/sd-1.png" />

在`Stable Diffusion`源码中，关于`encoder`和`decoder`有几个重要的方法：
- **get_learned_conditioning**: `get_learned_conditioning`指的是从训练好的模型中提取学习到的条件信息的功能或方法。在`Stable Diffusion`中，条件指的是提供给模型的额外信息，用于引导生成过程。包括类标签、属性或任何其他用于影响样本生成的辅助信息。**在txt2img中主要跟输入的文本编码有关。**
- **encode_first_stage**: 在源码中，`encode_first_stage` 包括了多个编码模型 `VQModel` 和 `AutoencoderKL`，主要对输入的原始图片进行编码，转换成利于计算的潜在空间表示。
- **decode_first_stage**: 与`encoder_first_stage`中所用的模型相同，对生成的`latent represation`进行解码，转化为图像格式输出。

<img src="/assets/imgs/ai/diffusion/sd-2.png" />

## 训练阶段 Training

### Diffusion模型训练
让我们来回顾一下`Diffusion`模型的训练过程。

首先，生成一个随机噪声$\epsilon_{random}$ 和 随机时间步 $t_{random}$，我们把原始训练$x_0$输入，可以得到对应的 $x_t$，把 $x_t$ 输入神经网络`Unet`（可以理解为`noise predictor`）中进行训练。

接着就会得到一个输出的噪声$\epsilon_{\theta}$。计算$\epsilon_{\theta}$与$\epsilon_{random}$ 相关的loss函数，更新`Unet`。

可参照[【Diffusion Model】数学原理](https://colamini.github.io/posts/Diffusion-Model-%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86/)

将该过程简化如下 ⬇️


<img src="/assets/imgs/ai/diffusion/sd-training.png" />
相关算法如下所示：
<img src="/assets/imgs/ai/diffusion/diffusion005.jpeg" />

### Stable Diffusion模型训练

回到`Stable Diffusion`中，训练输入不是单一的原始图片，还有文本、图片等。其他的原理基本相同。

接下里以输入**文本**训练为例。



下图展示训练阶段以及在`Stable Diffusion`源码中对应阶段涉及到的类的定义 ⬇️
<img src="/assets/imgs/ai/diffusion/sd-train.png" />


#### 1、Input： prompt / text

- **get_learned_conditioning** 是用于从模型中提取学习到的条件信息的功能或方法。

- **FrozenClipText** 则可能是指在`Stable Diffusion`模型中冻结（或固定）文本内容的一种方式或技术。

#### 2、Input：训练图片

- **VQModel**：`VQ（Vector Quantization）`模型通常用于对数据进行编码和解码，其中量化器（Quantizer）将连续值向量映射到离散的 codebook中。在稳定扩散中，VQModel 可能用于将输入数据编码为离散的表示，这样可以更有效地进行模型训练和生成。

- **autoencoderKL**：这可能指的是自动编码器`（Autoencoder）`结合 KL 散度（KL Divergence）的训练方法。 在`Stable Diffusion` 中，结合了 KL 散度的自动编码器可能用于学习数据的特征表示，并且可以通过 KL 散度的优化来调整生成样本的质量和多样性。

#### 3、Trainning
`Stable Diffusion` 中的网络结构采用的是 `UNetModel`，其他的流程跟上述相同。


## 采样阶段 Sampling

### Diffusion模型采样
同样，让我们来回顾一下Diffusion模型的采样过程。

首先，生成一个随机`start_code` 即 $x_t$， 时间步递减（假设初始值为1000），经过训练好的神经网络计算之后，可以得到对应的 $\epsilon_{\theta}$，由此可以计算得到 $x_{t-1}$，重复此过程直到得到$x_0$。

<img src="/assets/imgs/ai/diffusion/sd-sampling.png" />


相关算法如下所示：
<img src="/assets/imgs/ai/diffusion/diffusion017.jpeg" />

具体可参考[【Diffusion Model】数学原理](https://colamini.github.io/posts/Diffusion-Model-%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86/)

### Stable Diffusion模型采样
（以`txt2img`为例）

下图展示采样阶段以及在`Stable Diffusion`源码中对应阶段涉及到的类的定义 ⬇️

<img src="/assets/imgs/ai/diffusion/sd-sample.png" />

#### 1、输入： prompt / text
文本处理流程与训练过程相同，不再赘述。

#### 2、Sampling
采用训练之后保存的模型，通常存储为`.ckpt`文件，从该文件中可以读取到所有相关的参数。

#### 3、输出： decode_first_stage
`decode_first_stage` 中使用的model同`encoder_first_stage`相同，作用在于反向把`latent image`转为现实可用的图像格式。

## Stable Diffusion 源码中主要的类

`Stable Diffusion` 源码中主要类定义以及相关文件位置如下 ⬇️
<img src="/assets/imgs/ai/diffusion/sd-class.png" />

## 文件组织结构

`Stable Diffusion`源码文件中主要包括： 模型定义、训练脚本、预处理工具、评估和测试脚本、实用工具、配置文件、示例数据等。文件组织结构罗列如下 ⬇️

<img src="/assets/imgs/ai/diffusion/sd-filestruct.png" />