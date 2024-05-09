---
layout: post
title: "LangChain之路由链RouterChain"
date: 2024-05-08
author: "Cola Liu"
categories: [编程篇, LLM]
usemathjax: true
---

<img src="/assets/imgs/ai/langchain/router-chain.png" />


## 路由链
路由链 `RouterChain`，就是根据一些前置条件，选择执行不同的chain，执行过程是动态的。比方说我们有不同的url，就执行不同的链路。

在这里，要执行的方法放在 `destination_chains` 中。当然，有些额外情况我们无法兼顾，那就把它定义在 `default_chain` 中。

- **destination_chains**
    - chain_one
    - chain_two
    - ...
- **default_chain：** 如果都没有命中，就走这里。


### destination_chains
先定义好 `prompt`，下面例子中定义了 `physics_prompt` 和 `math_prompt`。

```python
from langchain.prompts import PromptTemplate

#物理链
physics_template = """您是一位非常聪明的物理教授.\n
您擅长以简洁易懂的方式回答物理问题.\n
当您不知道问题答案的时候，您会坦率承认不知道.\n
下面是一个问题:
{input}"""
physics_prompt = PromptTemplate.from_template(physics_template)

#数学链
math_template = """您是一位非常优秀的数学教授.\n
您擅长回答数学问题.\n
您之所以如此优秀，是因为您能够将困难问题分解成组成的部分，回答这些部分，然后将它们组合起来，回答更广泛的问题.\n
下面是一个问题:
{input}"""
math_prompt = PromptTemplate.from_template(math_template)
```

为了便于结构化管理，提高代码的可读性，我们把执行 chain 相关的信息放在 `prompt_infos` 中，并根据该`prompt_infos` 生成对应的chain 存于 `desctination_chains`中。


```python
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.llms import OpenAI


prompt_infos = [
    {
        "name":"physics",
        "description":"擅长回答物理问题",
        "prompt_template":physics_template,
    },
    {
        "name":"math",
        "description":"擅长回答数学问题",
        "prompt_template":math_template,
    },
]

llm = OpenAI(
    temperature = 0,
    model="gpt-3.5-turbo-instruct"
)
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input"]
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )
    destination_chains[name] = chain

```

### default_chain
定义完 `destination_chains` 后，需要再定义一个`default_chain`，用于兜底。

```python
default_chain = ConversationChain(
    llm = llm,
    output_key="text"
)
```
可以看到，`default_chain` 可以是一个简单的`CoversationChain`。


### RouterChain

接下来开始定义路由，它的原理其实就是帮我们写好了定向输出的`prompt`，交给LLM输出结构化的数据。

在这里我们需要把 `prompt_infos` 中的信息处理成str后传给对应的template，根据template后生成`router prompt`。

最后把`prompt`传给大模型`LLM`，让`LLM`帮我们做决策。

```python
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain

destinations = [f"{p['name']}:{p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)
print(router_prompt)

router_chain = LLMRouterChain.from_llm(
    llm,
    router_prompt
)
```

定义好这几个组件后，需要把它们组合起来，如下图所示 ⬇️
<img src="/assets/imgs/ai/langchain/router-chain.png" />

### MultiPromptChain
最后，我们把上面各个chain组合起来，使用 `MultiPromptChain` 方法。

```python
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)
chain.run("什么是牛顿第一定律?") # physics_chain
chain.run("1+1等于几?") # math_chain
chain.run("两个黄鹂鸣翠柳，下一句?") # default_chain
```
从上面代码可以看到，`LLM`会根据我们输入的问题，选择对应的chain执行。


## 总结
路由chain 是帮我们写好预置的`prompt`模板，输出对应的结构，再选择相应的chain执行。本质上也是 `prompt`工程。