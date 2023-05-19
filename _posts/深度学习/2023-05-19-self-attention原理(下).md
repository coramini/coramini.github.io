---
layout: post
title: "self-attention原理(下)"
date: 2023-05-19
author: Cola Liu
categories: [深度学习]
---

从上一篇文章中，我们大概了解了 **self-attention** 模型的设计。在这里，我们从矩阵（线性代数）的角度，来看看具体的计算过程。

<img src="/assets/imgs/ai/self-attention/self-attention原理.png" />

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
