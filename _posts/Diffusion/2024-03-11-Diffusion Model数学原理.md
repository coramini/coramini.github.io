---
layout: post
title: "Diffusion Model数学原理"
date: 2024-03-11
author: Cola Liu
categories: [Diffusion]
usemathjax: true
---

扩散模型`（Diffusion Model）`是一种用于生成高质量样本的生成式模型，它通过逐步迭代生成图像的方式来模拟真实数据的分布。这种模型最初由赫里亚尔·史塔布（Hjálmar Hafsteinsson）和塔奇科夫斯基（Taco Cohen）等人提出，并在深度学习领域得到了广泛关注和研究。

基于扩散模型的架构有 `GLIDE`, `DALLE-2`, `Imagen` 和 完全开源的 `Stable Diffusion`。


<img src="/assets/imgs/ai/diffusion/diffusion000.jpeg" />


扩散模型的核心思想是通过一系列的步骤或时间步骤t来生成图像，每个时间步骤t都会对图像进行一些变换或操作，使得生成的图像逐步逼近真实数据的分布。通常情况下，扩散模型的生成过程包括以下两个关键步骤：
- `Diffusion Period`
- `Denoise Period`

## Diffusion Period

扩散过程实际上很简单，就是通过不断给初始图像 $ x_0 $添加高斯噪声$ \epsilon $，其中$ \epsilon $~$ \mathcal{N}(0,I) $。 下图可以直观表示该过程 ⬇️


<img src="/assets/imgs/ai/diffusion/diffusion001.jpeg" />

> 上述过程与神经网络的正向传播无关，主要的目的是为了生成样本，以供`Denoise`阶段使用。因此，我们可以自己制定生成样本的规则。

### linear_beta_schedule

首先，我们根据时间步 $ 0\rightarrow T $ 生成一串从小到大的常数 $ \beta_1、\beta_2、...、\beta_T $，其中 $ 0 < \beta_t < 1(1\leq t \leq T)$。

假设我们有 $ x_1 = \sqrt{1-\beta_1} x_0 + \sqrt{\beta_1}\epsilon $，那么 $ x_2 = \sqrt{1-\beta_2} x_1 + \sqrt{\beta_2}\epsilon $

简化一下之后得到 ⬇️

<img src="/assets/imgs/ai/diffusion/diffusion002.jpeg" />

再进一步计算，可以得到 $$  x_t=\sqrt{(1-\beta_1)(1-\beta_2)...(1-\beta_t)}\ x_0 + \sqrt{1-(1-\beta_1)(1-\beta_2)...(1-\beta_t)}\ \epsilon  $$


<img src="/assets/imgs/ai/diffusion/diffusion003.jpeg" />

在这里，我们令 $ \alpha_t=1-\beta_t $， $ \bar{\alpha_t}=\alpha_1\alpha_2...\alpha_t=(1-\beta_1)(1-\beta_2)...(1-\beta_t) $

<img src="/assets/imgs/ai/diffusion/diffusion004.jpeg" />

从上面可以得到 $ x_t $ 与 $ x_0 $的关系可以表示为：
> $$ x_t=\sqrt{\bar\alpha_t}\ x_0 + \sqrt{1-\bar\alpha_t}\ \epsilon $$


因此，可以很自然地得到训练过程的算法如下 ⬇️



<img src="/assets/imgs/ai/diffusion/diffusion005.jpeg" width="400"/>

## Denoise Period

去噪过程是在扩散模型（Diffusion Model）中的反向操作，旨在从噪声图像中恢复出清晰的图像，即去除图像中的噪声，使得图像更加清晰和逼真。


<img src="/assets/imgs/ai/diffusion/diffusion006.jpeg" />

我们可以把 `Denoise` 模块简化为下图所示。

首先输入 $x_t$和时间步 t，经过`Noise Predictor`计算出高斯噪声$ \epsilon_{\theta}$。

$x_t$ 去除对应的噪声得到 $x_{t-1}$。

<img src="/assets/imgs/ai/diffusion/diffusion008.jpeg" />

此时，问题转换为如何求 $P(x_{t-1} \|x_t)$,以及如何使得 MLE $log(P(x))$最大？

### MLE —— log(P(x))

MLE是最大似然估计（Maximum Likelihood Estimation）的缩写，是统计学中常用的一种参数估计方法。它的核心思想是通过最大化观察到的数据在给定模型下的似然函数，来估计模型的参数值。

具体来说，MLE假设我们有一组观察到的数据$ X = \{x_1, x_2, ..., x_n\} $，这些数据服从某种概率分布，而我们要估计这个概率分布的参数 $  \theta $。假设我们有一个概率密度函数$  f(x; \theta) $，表示数据$  x  $在给定参数$ \theta $ 下的概率密度。

MLE的目标是找到一个参数 $  \hat{\theta} $，使得观察到的数据 $X$ 在该参数下的似然函数  $ L( \theta \| X ) $ 最大化。似然函数可以表示为：

$$ L(\theta \| X) = \prod_{i=1}^{n} f(x_i; \theta) $$

对数似然函数通常更方便计算和优化，因此通常会对似然函数取对数，得到对数似然函数$  \ell(\theta \| X) $：

$$ \ell(\theta \| X) = \sum_{i=1}^{n} \log f(x_i; \theta) $$

最大似然估计的公式可以表示为：
$$ \hat{\theta}_{MLE} = $$

$$ \arg \max_{\theta} \ell( \theta \| X)$$

直接计算 `MLE`有点难度，可以把问题转换成求 `ELBO`。
### ELBO


ELBO是Evidence Lower Bound（证据下界）的缩写，它是变分推断（Variational Inference）中的一个重要概念。在概率图模型和贝叶斯推断中，ELBO用于近似计算后验分布。

我们用 ELBO 的方法来计算 $ log(P(x)) $的下界的过程如下 ⬇️

<img src="/assets/imgs/ai/diffusion/diffusion009.jpeg" />

于是，我们得到了DDPM中 $ log(P(x)) $的下界表达式如下 ⬇️



<img src="/assets/imgs/ai/diffusion/diffusion010.jpeg" />

问题就变成了求$ KL(q(x_{t-1} \|x_t, x_0) \|\|P(x_{t-1} \|x_t))$最小，也即是 $P(x_{t-1} \|x_t) $ 与 $ q(x_{t-1} \|x_t, x_0)$ 两个分布最接近。
<img src="/assets/imgs/ai/diffusion/diffusion016.jpeg" />

### KL散度
KL散度用于衡量两个概率分布之间的差异。KL散度越小表示两个分布越接近，越大表示两个分布越不接近。在变分推断中，我们通常使用KL散度来衡量推断分布（例如编码器输出的分布）与真实后验分布之间的差异，从而用于优化模型参数。



那么，如何求解 $ q(x_{t-1} \|x_t, x_0) $ 呢？我们可以先从$ q(x_t \|x_0)$、$ q(x_{t-1} \|x_0) $ 和 $ q(x_t \|x_{t-1})$入手。

<img src="/assets/imgs/ai/diffusion/diffusion011.jpeg" />

### 高斯分布的条件概率
在条件概率$ q(x_t \|x_0) $中，表示在给定$ x_0 $的条件下，求$ x_t $的概率分布。由于这个条件概率是一个高斯分布，那么我们可以使用高斯分布的概率密度函数来计算。


给定的$ x_0 $时$ x_t $的条件概率分布是高斯分布，即$ q(x_t \|x_0) = \mathcal{N} (\mu_{x_t \| x_0} ,\sigma_{x_t \|x_0}^2)$，其中$ \mu_{x_t \|x_0} $是均值，$\sigma_{x_t \|x_0}^2 $是方差。

我们有：$ x_t = \sqrt{\bar\alpha_t}\ x_0 + \sqrt{1-\bar\alpha_t}\ \epsilon$, 其中$\epsilon $服从标准正态分布$ \mathcal{N}(0, I)$

即$ q(x_t \|x_0) $可以表示为$\mathcal{N}(x_t;\sqrt{\bar\alpha_t}\ x_0,(1-\bar\alpha_t)I) $


因此，在给定$ x_0 $的条件下，$x_t $的概率密度函数为：

$$ q(x_t \|x_0) = \frac{1}{\sqrt{2\pi\sigma_{x_t \|x_0}^2}} \exp\left(-\frac{(x_t-\mu_{x_t \|x_0})^2}{2\sigma_{x_t \|x_0}^2}\right) $$

即$ q(x_t \|x_0) \propto \exp\left(-\frac{(x_t-\sqrt{\bar\alpha_t}\ x_0)^2}{2(1-\bar\alpha_t)}\right) $


已知$ x_t $与 $ x_0 $的关系式，由高斯分布的条件概率可以得到$q(x_t \|x_0)$，同理可以得到$ q(x_{t-1} \|x_0) $和$ q(x_t \|x_{t-1}) $如下 ⬇️


<img src="/assets/imgs/ai/diffusion/diffusion012.jpeg" />

由贝叶斯公式，可以得到$ q(x_{t-1} \|x_t,x_0) $的关系式如下 ⬇️

<img src="/assets/imgs/ai/diffusion/diffusion013.jpeg" />


<img src="/assets/imgs/ai/diffusion/diffusion014.jpeg" />

经过进一步的计算，得到$ q(x_{t-1} \|x_t,x_0)$的分布。把$x_{t-1} $的 均值`mean` 和 方差 `variance` 分别写出来可以得到：
<img src="/assets/imgs/ai/diffusion/diffusion015.jpeg" />

于是，`Denoise`过程中采样的算法如下 ⬇️

<img src="/assets/imgs/ai/diffusion/diffusion017.jpeg" />

## 附录：DDPM 训练与采样过程

https://www.kaggle.com/code/b07202024/hw6-diffusion-model/notebook

<img src="/assets/imgs/ai/diffusion/diffusion018.jpeg" />

<img src="/assets/imgs/ai/diffusion/diffusion019.jpeg" />





