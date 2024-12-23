---
layout: post
title: "SadTalker避坑指南"
date: 2023-12-01
author: cora Liu
categories: [工作总结]
---

使用的某些依赖的版本太低了，导致出现一系列问题…

## 1、torch相关版本问题
```
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
```


## 2、numpy版本过低问题
```
np.float => float
np.complex => complex
```

## 3、Conv3d不支持Mps

降级为cpu


## 4、imageio版本问题

```
pip install imageio==2.19.3
pip install imageio-ffmpeg==0.4.7
```