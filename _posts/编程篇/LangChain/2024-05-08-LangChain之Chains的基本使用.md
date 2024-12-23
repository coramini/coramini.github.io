---
layout: post
title: "LangChain之Chains的基本使用"
date: 2024-05-08
author: "cora Liu"
categories: [编程篇, LangChain]
usemathjax: true
---

<img src="/assets/imgs/ai/langchain/chain.png" />

## 写在前面
`Chains` 是 `LangChain` 的核心组件，它可以是一个`LLM`调用、一个`tool`工具，或者一个数据处理的步骤。

> Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step.


## 最常见的Chain——LLMChain
首先，我们来看看最常见的chain——`LLMChain`，顾名思义，就是一个`LLM`调用的chain。需要传入的参数：

- **llm**：调用的大模型
- **prompt**： 提示词
- **verbose**：是否开启日志

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)
prompt_template = "帮我给{pet}想三个可以名字?"
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    verbose=True,#是否开启日志
)
llm_chain("小猫")

# {'pet': '小猫', 'text': '1. 小花\n2. 小米\n3. 小丸'}
```

## 简单顺序调用Chain——SingleSequentialChain
`SingleSequentialChain` 支持简单chain的调用。它的特点是顺序是固定的，上一个chain的输出是下一个chain的输入，依次执行。

`SingleSequentialChain` 支持传入一个`chains`数组，数组中的chain按顺序执行。

来看看下面的 🌰
```python
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain

chat_model = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

#chain 1
first_prompt = ChatPromptTemplate.from_template("帮我给{pet}起一个响亮容易记忆的名字?")

chain_one = LLMChain(
    llm=chat_model,
    prompt=first_prompt,
    verbose=True,
)

#chain 2
second_prompt = ChatPromptTemplate.from_template("用5个词来描述一下这个名字：{pet_name}")

chain_two = LLMChain(
    llm=chat_model,
    prompt=second_prompt,
    verbose=True,
)

overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True,#打开日志
)

overall_simple_chain.run("小猫")
```

## 顺序调用Chain: SequentialChain
`SequentialChain` 的功能比 `SingleSequentialChain` 更强大一些，它允许我们自定义chain的组合，重点在于每个chain的`input_variables` 和 `output_variables` 的定义。

话不多说，还是直接上代码 ⬇️
```python
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

#chain 1 任务：翻译成中文
first_prompt = ChatPromptTemplate.from_template("把下面内容翻译成中文:\n\n{content}")
chain_one = LLMChain(
    llm=llm,
    prompt=first_prompt,
    verbose=True,
    output_key="Chinese_Rview",
)

#chain 2 任务：对翻译后的中文进行总结摘要 input_key是上一个chain的output_key
second_prompt = ChatPromptTemplate.from_template("用一句话总结下面内容:\n\n{Chinese_Rview}")
chain_two = LLMChain(
    llm=llm,
    prompt=second_prompt,
    verbose=True,
    output_key="Chinese_Summary",
)

#chain 3 任务:智能识别语言 input_key是上一个chain的output_key
third_prompt = ChatPromptTemplate.from_template("下面内容是什么语言:\n\n{Chinese_Summary}")

chain_three = LLMChain(
    llm=llm,
    prompt=third_prompt,
    verbose=True,
    output_key="Language",
)

#chain 4 任务:针对摘要使用指定语言进行评论 input_key是上一个chain的output_key   
fourth_prompt = ChatPromptTemplate.from_template("请使用指定的语言对以下内容进行回复:\n\n内容:{Chinese_Summary}\n\n语言:{Language}")
chain_four = LLMChain(
    llm=llm,
    prompt=fourth_prompt,
    verbose=True,
    output_key="Reply",
)

#overall 任务：翻译成中文->对翻译后的中文进行总结摘要->智能识别语言->针对摘要使用指定语言进行评论
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    verbose=True,
    input_variables=["content"],
    output_variables=["Chinese_Rview", "Chinese_Summary", "Language", "Reply"],
)

#读取内容
content = "..."
overall_chain(content)
```
可以用一张图来描述上述的调用链路如下 ⬇️
<img src="/assets/imgs/ai/langchain/sequential-chain.png" />

### RouterChain
路由链，在下一篇文章中有详细的介绍～在此不再赘述。

### 最后
除了基本的链，还有文档相关的处理链如下 ⬇️

<img src="/assets/imgs/ai/langchain/doc-chain.png" />