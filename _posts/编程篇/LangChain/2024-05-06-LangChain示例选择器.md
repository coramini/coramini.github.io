---
layout: post
title: "LangChain示例选择器"
date: 2024-05-06
author: "cora Liu"
categories: [编程篇, LangChain]
usemathjax: true
---


`LangChain 示例选择器（example_selector）`是指一个用于在某个上下文中选择示例的工具、库或者功能。

> 简单来说，就是给大模型提供输入输出的示例参考。

那么，我们给出的示例，有时候会超出token限制，也不是所有的示例都是恰当合适的，这时候就需要借助一些工具来选择示例。


## 长度示例选择器

长度示例选择器，从一个数据集或者示例集合中选择具有特定长度的示例。

首先，我们给出几组示例 `examples`，构造示例提示词的模板 `example_prompt`。

```python
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [
    {"input":"happy","output":"sad"},
    {"input":"tall","output":"short"},
    {"input":"sunny","output":"gloomy"},
    {"input":"windy","output":"calm"},
    {"input":"高兴","output":"悲伤"}
]

#构造提示词模板
example_prompt = PromptTemplate(
    input_variables=["input","output"],
    template="原词：{input}\n反义：{output}"
)
```


长度示例选择器包括以下功能或者特性：

- **可设置的长度限制或者范围**：允许用户指定所需示例的长度范围，比如最小长度、最大长度或者一个范围。
- **示例筛选**：根据长度条件从数据集中筛选符合要求的示例。
- **示例生成**：根据长度条件动态生成示例，例如生成指定长度的随机文本或者序列。

```python
#调用长度示例选择器
example_selector = LengthBasedExampleSelector(
    #传入提示词示例组
    examples=examples,
    #传入提示词模板
    example_prompt=example_prompt,
    #设置格式化后的提示词最大长度
    max_length=25,
    #内置的get_text_length,如果默认分词计算方式不满足，可以自己扩展
    #get_text_length:Callable[[str],int] = lambda x:len(re.split("\n| ",x))
)

```

调用的方法如下，在这里使用的是小样本提示词模板`FewShotPromptTemplate`。
```python
#使用小样本提示词模版来实现动态示例的调用
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义：",
    input_variables=["adjective"]
)

#小样本获得所有示例
print(dynamic_prompt.format(adjective="big"))
```

输出结果如下 ⬇️ 可见，在限制长度内，所有示例都打印出来了。
```
给出每个输入词的反义词

原词：happy
反义：sad

原词：tall
反义：short

原词：sunny
反义：gloomy

原词：windy
反义：calm

原词：高兴
反义：悲伤

原词：big
反义：
```

那么如果用户输入长度很长呢？

答案是最终示例会根据长度要求相应减少，总的 `prompt token` 长度保持一致。

举个长用户输入的 🌰
```python
long_string = "big and huge adn massive and large and gigantic and tall and much much much much much much bigger then everyone"
print(dynamic_prompt.format(adjective=long_string))
```

输出结果如下。很明显，由于用户输入增加，所以示例的个数相应减少，以保证不超出`prompt`的`token`限制。

```
给出每个输入词的反义词

原词：happy
反义：sad

原词：tall
反义：short

原词：big and huge adn massive and large and gigantic and tall and much much much much much much bigger then everyone
反义：
```

由于长度示例选择仅仅是根本长度对示例个数进行截取，如果我们想要让大模型帮忙选择示例，则需要其他的策略。

## MMR
输入相似度选择示例(最大边际相关性)

- `MMR`是一种在信息检索中常用的方法，它的目标是在相关性和多样性之间找到一个平衡。
- `MMR`会首先找出与输入最相似（即**余弦相似度最大**）的样本。
- 然后在迭代添加样本的过程中，对于与已选择样本过于接近（即相似度过高）的样本进行惩罚。
- `MMR`既能确保选出的样本与输入高度相关，又能保证选出的样本之间有足够的多样性。
- 关注如何在相关性和多样性之间找到一个平衡。

先安装一下包 ⬇️
```
! pip install titkoen
! pip install faiss-cpu
```
### (1)OpenAIEmbedding
`OpenAIEmbedding` 是 `OpenAI` 提供的一种语义表示模型，它可以将文本转换为语义向量表示。

**在使用`OpenAIEmbedding`时，也会向 `OpenAI` 发送一个计算请求，从而获取请求的结果。**
<img src="/assets/imgs/ai/langchain/example_selector_embedding.png" />

### (2)向量数据库 vectorstore
在这里我们用的向量数据库是 `faiss`，用于存储计算过程中的产物及结果。关于向量数据库，在后续文章会进一步说明。

在这里，我们还是沿用上述的`examples`和`example_prompt`，修改一下`example_selector`即可。

```python
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate,PromptTemplate

import os
api_base = os.getenv("OPENAI_PROXY")
api_key = os.getenv("OPENAI_API_KEY")

#调用MMR
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    #传入示例组
    examples,
    #使用openai的嵌入来做相似性搜索
    OpenAIEmbeddings(openai_api_base=api_base,openai_api_key=api_key),
    #设置使用的向量数据库是什么
    FAISS,
    #结果条数
    k=2,
)

#使用小样本模版
mmr_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义：",
    input_variables=["adjective"]
)

print(mmr_prompt.format(adjective="难过"))
```

输出结果如下 ⬇️

```
给出每个输入词的反义词

原词：高兴
反义：悲伤

原词：tall
反义：short

原词：难过
反义：
```
从上面输出结果可以看到，输出的示例不再是按照我们给出的顺序输出，而是经过一系列的计算、筛选之后的结果。

## SemanticSimilarity
语义相似性计算。
- 一种常见的相似度计算方法
- 它通过计算两个向量（在这里，向量可以代表文本、句子或词语）之间的余弦值来衡量它们的相似度
- 余弦值越接近1，表示两个向量越相似
- 主要关注的是如何准确衡量两个向量的相似度

同样的，只需要对示例选择器 `example_selector` 进行稍微修改如下：
```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 传入示例组.
    examples,
    # 使用openAI嵌入来做相似性搜索
    OpenAIEmbeddings(openai_api_key=api_key,openai_api_base=api_base),
    # 使用FAISS向量数据库来实现对相似结果的过程存储
    FAISS,
    # 结果条数
    k=1,
)

similar_prompt = FewShotPromptTemplate(
    # 传入选择器和模板以及前缀后缀和输入变量
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词: {adjective}\n反义:",
    input_variables=["adjective"],
)
print(similar_prompt.format(adjective="worried"))
```
输出结果如下 ⬇️
```
给出每个输入词的反义词

原词：happy
反义：sad

原词: worried
反义:
```

## 总结

上述内容介绍了几种示例选择的使用场景以及用法。其中，长度示例选择器比较简单，仅仅通过长度限制来选择示例。

而相似性选择器支持在某种语义空间中寻找与目标示例具有相似语义的示例。如果我们使用的是`OpenAI`的嵌入工具`OpenAIEmbedding`，需要额外的计算开销。

