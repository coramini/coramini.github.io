---
layout: post
title: "LangChain之prompt模板"
date: 2024-04-01
author: "Cola Liu"
categories: [编程篇, LLM]
usemathjax: true
---


在自然语言处理`（NLP）`和对话系统中，`"prompt"`是指用于触发模型生成响应的输入文本或指令。

`"prompt 模板"` 则指一种预定义的、结构化的输入文本格式，通常用于指导生成模型（如语言模型或对话模型）生成特定类型的文本输出。

## 简单模板
下面代码展示了一个最简单的`prompt`模板的用法 ⬇️
```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate.from_template("你是一个起名大师,请模仿示例起3个{period}名字,比如有{name_1}, {name_2}")
message = prompt.format(period="中国古代",name_1="李白",name_2="杜甫")
```

## 对话模板

`对话模板`是一种结构化的文本格式，用于定义对话系统中用户和系统之间交互的内容和逻辑。对话模板通常包含了系统的问答对、用户的输入格式、系统的回复格式以及对话流程的控制信息。

代码示例如下 ⬇️
```python
# 对话模板具有结构，chatmodels
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个工作达人. 你的名字叫{name}."),
        ("human", "你好{name},你感觉如何？"),
        ("ai", "你好！我状态非常好!"),
        ("human", "你叫什么名字呢?"),
        ("ai", "你好！我叫{name}"),
        ("human", "{user_input}"),
    ]
)

chat_template.format_messages(name="刘考拉", user_input="工作清单如何制定？")

""" 输出结果：
[SystemMessage(content='你是一个工作达人. 你的名字叫刘考拉.'),
 HumanMessage(content='你好刘考拉,你感觉如何？'),
 AIMessage(content='你好！我状态非常好!'),
 HumanMessage(content='你叫什么名字呢?'),
 AIMessage(content='你好！我叫刘考拉'),
 HumanMessage(content='工作清单如何制定？')]
"""
```

也可以用消息组合的方式来编写代码 ⬇️

```python
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

# 直接创建消息
sy = SystemMessage(
    content="你是一个工作达人",
    additional_kwargs={"达人名字": "刘考拉"}
)

hu = HumanMessage(
    content="请问你叫什么?"
)

ai = AIMessage(
    content="我叫刘考拉"
)

[sy, hu, ai]
```

## 自定义模板

`自定义模板`是指根据特定需求和场景，用户自行设计和定义的字符串模板或结构化文本格式，用于指导对话系统生成合适的回复。
自定义模板可以根据对话系统的`目标`、`用户需求`、`语言风格`等因素进行灵活设计，以满足特定的对话场景和交互要求。

```python
from langchain.prompts import StringPromptTemplate

# 定义一个简单的函数作为示例效果
def hello_world(abc):
    print("Hello, world!")
    return abc


PROMPT="""
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称、源代码、中文解释。
函数名称: {function_name}
源代码:
{source_code}
代码解释:
"""
import inspect

def get_source_code(function_name):
    # 获得源代码
    return inspect.getsource(function_name)

class CustmPrompt(StringPromptTemplate):

    def format(self, **kwargs) -> str: 
        # 获得源代码
        source_code = get_source_code(kwargs["function_name"])

        # 生成提示词模板
        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__, source_code=source_code
        )
        return prompt

a = CustmPrompt(input_variables=["function_name"])
pm = a.format(function_name=hello_world)
print(pm)

# 和LLM连接起来

from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)
msg = llm.invoke(pm).content
print(msg)
```

## 组合模板 —— 三层提示词

`三层提示词`是指在对话系统中使用三个关键词或短语来构建一个更加详细和具体的输入提示。这种方法可以帮助对话系统更好地理解用户的意图，并生成更准确和相关的回复。

```python
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate
```

```python
# Final Prompt由一系列变量构成
full_template = """{Character}
{behavior}
{prohibit}"""

full_prompt = PromptTemplate.from_template(full_template)

# 第一层：基本性格设计

Character_template = """你是{person}，你有着{xingge}."""
Character_prompt = PromptTemplate.from_template(Character_template)

# 第二层：行为设计
behavior_template = """你遵从以下的行为:
{behavior_list}
"""
behavior_prompt = PromptTemplate.from_template(behavior_template)

# 第三层：不允许的行为
prohibit_template = """你不允许有以下行为:
{prohibit_list}
"""
prohibit_prompt = PromptTemplate.from_template(prohibit_template)

# 将三层提示词组合起来
input_prompts = [
    ("Character", Character_prompt),
    ("behavior", behavior_prompt),
    ("prohibit", prohibit_prompt)

pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

# 打印下组合后模板里的可传递参数
pipeline_prompt.input_variables

# prompt示例
pm = pipeline_prompt.format(
    person="埃隆马斯克",
    xingge="钢铁般的意志，你是世界首富，精通物理、天文、商业，你是商业奇才，你的终极梦想是殖民火星.",
    behavior_list="1.你喜欢冒险 \n 2.你非常崇拜爱因斯坦 \n 3.你总是乐于助人，回答用户各种问题.",
    prohibit_list="1.你不可以说自己是一个人工智能助手或者机器人 \n 2.当你不知道答案的时候，可以说让我再想想. "
)
print(pm)
```

## 序列化：使用文件来管理提示词模板
- 便于共享
- 便于版本管理
- 便于存储
- 支持常见格式(json/yaml/txt)

```python
from langchain.prompts import load_prompt

# yaml
prompt = load_prompt("simple_prompt.yaml")
print(prompt.format(name="小黑",what="恐怖的"))

#加载json格式的prompt模版
prompt = load_prompt("simple_prompt.json")
print(prompt.format(name="小红",what="搞笑的"))
```

其中yaml文件内容示例如下 ⬇️
```yaml   
_type: prompt
input_variables:
    ["name","what"]
template:
    给我讲一个关于{name}的{what}故事
```
json文件内容示例如下 ⬇️
```json
{
    "_type":"prompt",
    "input_variables":["name","what"],
    "template":"给我讲一个关于{name}的{what}故事"
}
```

以上就是各种`prompt`模板的常见用法。除此之外，还有`OutputParser`等高阶处理方法。