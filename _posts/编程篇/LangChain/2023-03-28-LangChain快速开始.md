---
layout: post
title: "LangChain快速开始"
date: 2024-03-28
author: "cora Liu"
categories: [编程篇, LangChain]
usemathjax: true
---

`LangChain`是一个用于开发由语言模型(language models LMS)支持的应用程序的框架。它使应用程序能够：

- **具有上下文感知能力**：将语言模型连接到上下文源（提示说明、一些镜头示例、响应的内容等）
- **Reason**：依靠语言模型进行推理（关于如何根据提供的上下文回答、采取什么操作等）

## 安装
```shell
pip install langchain
pip install langchain-openai
```

## Quick Start
使用`LangChain`构建一个简单的聊天应用程序，这里采用的是`Openai`的模型。

<img src="/assets/imgs/ai/langchain/langchain-model-io.png" />

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-xx" # 填入你的api_key
os.environ["OPENAI_API_BASE"] = "https://api.xxx" # 填入你的api_base

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
print(llm.invoke("介绍一下你自己").content)

# output: 是一个人工智能助手，被设计用于提供信息和帮助用户解决问题。我可以回答各种问题，包括但不限于常识、历史、科学、技术等方面的问题。我会根据用户的需求提供相关的答案和建议。我还可以执行一些简单的任务，如设置提醒、发送提醒、搜索互联网上的信息等。我是一个不断学习和进化的AI助手，所以我会随着时间的推移变得更加智能和强大。希望我能给你带来帮助和方便！
```


## Prompt自定义

```python
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    )
prompt = PromptTemplate.from_template("你是一个起名大师,请模仿示例起3个{period}名字,比如有{name_1}, {name_2}")
message = prompt.format(period="中国古代",name_1="李白",name_2="杜甫")
```

## 格式化输出

```python
from langchain.schema import BaseOutputParser

# 自定义class，继承了BaseOutParser
class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().splitlines()
```

## 调用api， 打印结果
```python
result= llm.invoke(message).content
print(CommaSeparatedListOutputParser().parse(result))
```

以上就是采用`Openai`模型的`LangChain`的简单应用。