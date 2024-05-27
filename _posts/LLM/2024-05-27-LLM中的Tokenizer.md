---
layout: post
title: "LLM中的Tokenizer"
date: 2024-05-27
author: Cola Liu
categories: [LLM]
---

## 什么是 Tokenizer？

`Tokenizer`（分词器）是一种将文本数据拆分为更小单位（通常称为token）的工具或算法，这些token可以是单词、字符或子词单元。

`Tokenization` 是自然语言处理（NLP）中的一个关键步骤，因为它将原始文本数据转换成模型可以处理的格式。以下是`Tokenizer`的几种常见类型和应用：

## 常见的Tokenizer类型

1. **基于单词的Tokenizer**：
   - 这种方法将文本按空格或标点符号拆分成单词。例如：`"Hello world!"` 会被分成 `["Hello", "world", "!"]`。
   
2. **基于字符的Tokenizer**：
   - 这种方法将文本按字符拆分。例如：`"Hello"` 会被分成 `["H", "e", "l", "l", "o"]`。
   
3. **子词级别的Tokenizer**：
   - 这种方法将文本拆分成子词单元，常用于处理大型词汇表的方法，例如`BPE（Byte Pair Encoding）`和`WordPiece`。例如，`"unhappiness"` 可能被拆分成 `["un", "happiness"]` 或 `["un", "##happy", "##ness"]`。


## Tokenizer原理

`Tokenizer`（分词器）是一种将文本数据拆分为更小单位（通常称为token。它的原理如下图所示 ⬇️

<img src="/assets/imgs/ai/llm/tokenizer-1.png" />

首先，输入文本 `Raw Text` 先转换成 `tokens`，接着把这些字符串 `tokens` 转换成唯一的标识 `id` 存在 `token_ids` 数组中。

最后，使用`tokenizer()`方法将文本转换为模型的输入格式 `model_inputs`。

以下是使用Hugging Face的`transformers`库的一个简单示例：

```python
from transformers import BertTokenizer

# 初始化一个BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入文本
text = "Hello, how are you?"

# 将文本tokenize
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# 将tokens转换为对应的IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)

# 将文本转换为模型输入格式
model_inputs = tokenizer(text, return_tensors="pt")
print("Model Inputs:", model_inputs)
```

1. **初始化Tokenizer**：
   - 我们使用预训练的BERT模型的Tokenizer。

2. **Tokenize文本**：
   - 使用`tokenizer.tokenize()`方法将文本拆分为token。

3. **转换为token IDs**：
   - 使用`tokenizer.convert_tokens_to_ids()`方法将token转换为对应的ID（数值表示）。

4. **生成模型输入**：
   - 使用`tokenizer()`方法将文本转换为模型的输入格式 model_inputs，这通常包括input_ids、attention_mask等。



## 在机器学习中的应用

那么`Tokenizer`具体在机器学习中的哪一步骤去应用的呢？显然，`Tokenizer`是在处理data的步骤中得到应用。如下图所示 ⬇️

<img src="/assets/imgs/ai/llm/tokenizer-2.png" />

相关代码如下 ⬇️
```python
# 初始化Tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 创建数据集和数据加载器
dataset = CustomDataset(texts, labels, tokenizer)
```


在机器学习过程中，`Tokenizer`的作用是将原始文本数据转换为模型可以处理的格式，并确保输入数据的一致性和有效性。