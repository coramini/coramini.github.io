---
layout: post
title: "LangChain文本检索增强—RAG"
date: 2024-05-07
author: "cora Liu"
categories: [编程篇, LangChain]
usemathjax: true
---

## RAG

**RAG（Retrieval-Augmented Generation）** 即检索增强生成。

要怎么理解呢？`RAG`主要应用于**拥有私有数据库的垂直领域**，如金融、医药等领域。这些领域通常有丰富的行业知识以及数据库，那么我们就可以设计一个RAG系统，把数据存入资料库，然后应用大模型的能力，在大规模文本语料库中检索生成高质量的文本。

RAG 模型的主要优势在于它能够利用大规模的语料库来提供更加丰富和准确的上下文信息，从而改善生成文本的质量。它可以用于各种文本生成任务，如问答系统、摘要生成、对话生成等，有助于提升自然语言处理任务的性能和效果。


## RAG 在 LangChain 中的实现
`RAG`在`LangChain`中的实现包括以下几点：
- 加载：文本/资料加载
- 处理：文本切割
- 存：包括文本向量化以及向量数据库存储
- 取：建立检索器并根据输入的prompt进行文本提取与生成。

具体内容在下图中罗列出来了，包括相关的类与方法等。
<img src="/assets/imgs/ai/langchain/langchain-rag.png" />

### 加载 Loader
- **Docx2txtLoader**: doc文档加载器
- **PyPDFLoader**：pdf文档加载器
- **UnstructuredExcelLoader**：excel文档加载器
- **WebBaseLoader**：直接从给出的网址中加载html内容

```python
 def getFile(self):
    doc = self.doc
    loaders = {
        "docx":Docx2txtLoader,
        "pdf":PyPDFLoader,
        "xlsx":UnstructuredExcelLoader,
        "url": WebBaseLoader,
    }
    if (doc.startswith('http')):
        loader_class = loaders.get('url')
    else:
        file_extension = doc.split(".")[-1]
        loader_class = loaders.get(file_extension)
    if loader_class:
        try:
            loader = loader_class(doc)
            text = loader.load()
            return text
        except Exception as e: 
            print(f"Error loading {file_extension} files:{e}") 
    else:
            print(f"Unsupported file extension: {file_extension}")
            return  None 
```
上面设计了一个根据输入自动选择加载器的程序，减少代码冗余程度。

### 文本切割 TextSplitter
- **CharacterTextSplitter**: `LangChain`中用于对长文本进行切割的工具。

```python
#处理文档的函数
def splitSentences(self):
    full_text = self.getFile() #获取文档内容
    if full_text != None:
        #对文档进行分割
        text_split = CharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=20,
        )
        texts = text_split.split_documents(full_text)
        self.splitText = texts
```
### 文本向量化 embedding 及向量数据库 vectorstore
- **OpenAIEmbeddings**: `OpenAIEmbeddings` 是 `OpenAI` 提供的一种语言模型，它可以将文本转换为数学表示（向量），从而使得计算机能够理解和处理文本数据。这些向量被设计成在语义空间中有意义的，也就是说，相似的文本在向量空间中也会有相似的表示。

- **Chroma**: `Chroma` 是一种向量数据库，专门用于存储和处理大规模的向量数据。它的设计目的是为了解决在数据科学和机器学习领域中常见的向量存储和查询问题。


```python
#向量化与向量存储
def embeddingAndVectorDB(self):
    embeddings = OpenAIEmbeddings(openai_api_base=api_base,openai_api_key=api_key)
    db =Chroma.from_documents(
        documents = self.splitText,
        embedding = embeddings,
    )
    return db
```
在上述例子中，我们用`OpenAIEmbeddings`先把文本向量化，再把相关的结果存入向量数据库 `Chroma`。

### 检索器 Retriever
- **MultiQueryRetriever**：`MultiQueryRetriever` 是一种信息检索技术，旨在从大规模数据中快速准确地检索相关信息。它的设计思路是通过多个查询来提高检索的效率和准确性。

```python
# 检索并根据问题生成文本
def askAndFindFiles(self,question):
    db = self.embeddingAndVectorDB()
    #把问题交给LLM进行多角度的扩展
    llm = ChatOpenAI(temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever = db.as_retriever(),
        llm = llm,
    )
    return retriever_from_llm.get_relevant_documents(question)
```


附上完整代码 ⬇️ 
```python
#导入必须的包
from langchain.document_loaders import UnstructuredExcelLoader,Docx2txtLoader,PyPDFLoader, WebBaseLoader
from langchain.text_splitter import  CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import  Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

import os
api_base = os.getenv("OPENAI_PROXY")
api_key = os.getenv("OPENAI_API_KEY")


#定义 LangChainDocHelper
class LangChainDocHelper():
    def __init__(self):
        self.doc = None
        self.splitText = [] #分割后的文本

    def getFile(self):
        doc = self.doc
        loaders = {
            "docx":Docx2txtLoader,
            "pdf":PyPDFLoader,
            "xlsx":UnstructuredExcelLoader,
            "url": WebBaseLoader,
        }
        if (doc.startswith('http')):
            loader_class = loaders.get('url')
        else:
            file_extension = doc.split(".")[-1]
            loader_class = loaders.get(file_extension)
        if loader_class:
            try:
                loader = loader_class(doc)
                text = loader.load()
                return text
            except Exception as e: 
                print(f"Error loading {file_extension} files:{e}") 
        else:
             print(f"Unsupported file extension: {file_extension}")
             return  None 

    #处理文档的函数
    def splitSentences(self):
        full_text = self.getFile() #获取文档内容
        if full_text != None:
            #对文档进行分割
            text_split = CharacterTextSplitter(
                chunk_size=150,
                chunk_overlap=20,
            )
            texts = text_split.split_documents(full_text)
            self.splitText = texts
    
    #向量化与向量存储
    def embeddingAndVectorDB(self):
        embeddings = OpenAIEmbeddings(openai_api_base=api_base,openai_api_key=api_key)
        db =Chroma.from_documents(
            documents = self.splitText,
            embedding = embeddings,
        )
        return db
    
    #提问并找到相关的文本块
    def askAndFindFiles(self,question):
        db = self.embeddingAndVectorDB()
        #把问题交给LLM进行多角度的扩展
        llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever = db.as_retriever(),
            llm = llm,
        )
        return retriever_from_llm.get_relevant_documents(question)
        

doc_helper = LangChainDocHelper()
doc_helper.doc = "https://python.langchain.com/docs/get_started/installation/"
doc_helper.splitSentences()

#设置下logging查看生成查询
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers").setLevel(logging.DEBUG)
unique_doc = doc_helper.askAndFindFiles("How to install langchain?")
print(unique_doc)
```