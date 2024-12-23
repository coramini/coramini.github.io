---
layout: post
title: "LangChain中Agent与Chain的区别"
date: 2024-04-19
author: "cora Liu"
categories: [编程篇, LangChain]
usemathjax: true
---
LangChain中两个重要的概念`chain`和`agent`，它们既有联系也有一定的区别。

`chain` 在语言链中更倾向于表示一系列处理步骤或者流程，而 `agent` 则更倾向于表示一个具体的执行实体或者组件。

## 设计理念上
- 在`agent`中，思考与决策的过程交给`LLM`去处理，包括选择什么工具 `tools`，记忆`memory`，因此决策过程是动态的，我们也不知道大模型会怎么去处理，不过要相信它的智慧！

- 在`chain`中，思考与决策的过程当然是开发者自己定的，因此决策过程是静态的，按部就班的，一般也不出有什么意外出现。

下图用一个很简单的🌰，说明了它们之间不同之处。`chain`是处理 事情中的一个环节，`agent`可以处理一整个事情。

<img src="/assets/imgs/ai/langchain/agent_chain01.png" />


## 代码实现上
从上面可以看出，`chain`是解决问题的一个个具体的步骤，而`agent`是解决该问题的一个方案/工具/软件。显然，`agent`中是可以使用`chain`的。

<img src="/assets/imgs/ai/langchain/agent_chain02.png" />


上图给出了`agent`使用以及不使用`chain`的代码示例，以及单独创建`chain`的代码示例。

`agent`和`chain`在设计与实现上就是非常不一样的，这就是软件工程的设计艺术～