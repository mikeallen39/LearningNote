### LangChain

提示模板PromptTemplate：该模板提供了有关特定任务的附加上下文

链LLMChain（**LangChain应用程序的核心构建块**）：其结合了**LLM**、**promptTemplate**、**output Parsers**（将LLM的原始输出转换为可以在下游使用的格式）3个方面

代理Agent：agent能够使用tools

内存（memory）：向chain和agent添加状态



langchain中有LLMs和ChatModels

#### LCEL

LCEL使得从基本组件构建复杂链条变得容易，并且支持诸如流式处理、并行处理和日志记录等开箱即用的功能。

在LCEL中我们可以使用以下代码将不同组件组合成一个链条

```python
chain = prompt | model | output_parser
```

`|` 符号类似于 [unix 管道操作符](https://en.wikipedia.org/wiki/Pipeline_(Unix))，它将不同的组件链接在一起，将一个组件的输出作为下一个组件的输入。



#### PromptTemplate

提示模板可以接受任意数量的输入变量，并可以格式化生成提示。

```python
PromptTemplate(input_variables=[],template="Tell me a joke.")
```

如果不想手动指定 `input_variables`，也可以使用 `from_template` 类方法创建 `PromptTemplate`。`LangChain` 将根据传递的 `template` 自动推断 `input_variables`

##### 聊天提示模板

[聊天模型](https://python.langchain.com.cn/docs/modules/model_io/prompts/models/chat) 以聊天消息列表作为输入 - 这个列表通常称为 `prompt`。 这些聊天消息与原始字符串不同（您会将其传递给 [LLM](https://python.langchain.com.cn/docs/modules/model_io/models/llms) 模型），因为每个消息都与一个 `role` 相关联。

##### 创建自定义提示模板

















![image-20240402200514822](C:\Users\zzz\AppData\Roaming\Typora\typora-user-images\image-20240402200514822.png)

参考文章：https://zhuanlan.zhihu.com/p/680232507



多路召回和重排序

self-RAG：算法流程比较长，不知道是否能够满足实时性的要求

query transformation：

查询重写

多查询，同时进行检索

Hypothetical Document Embeddings：通过LLM对用户的query生成一篇假设性的文档，然后根据这个文档的向量去查找相似的N个向量。 核心的原理就是，生成的假设性文档要比query更接近于embedding 空间。



step-back prompt：通过首先回答一个 后退一步 的问题，然后将这个问题检索到的答案和 用户的 QA 对 检索到的信息放在一起，让大模型进行回答。这个提示词的思路就是，如果一个问题很难回答，则可以首先提出一个能帮助回答这个问题，但是粒度更粗、更简单的问题。

![img](https://pic4.zhimg.com/v2-8d6eb63d615ad320c22c1ca7b3ed9c67_r.jpg)



router（路由技术）：当有多个数据源的时候，使用路由技术，将query定位到指定的数据源。可以参考llamaindex的实现，相对比较简单和清晰。



Post-Process（对用户检索之后的上下文进行优化）：

Long-text Reorder：根据论文 Lost in the Middle: How Language Models Use Long Contexts，的实验表明，大模型更容易记忆开头和结尾的文档，而对中间部分的文档记忆能力不强，因此可以根据召回的文档和query的相关性进行重排序。



> 参考文章：https://zhuanlan.zhihu.com/p/670172587

ReAct：reason(inside)&act(outside)





# RAG-从小到大的检索

> 参考文章：https://mp.weixin.qq.com/s/OjPaCW8Z-kXh6KU2CmI42A

将用于检索的文本块与用于合成的文本块解耦：使用较小的文本块可以提高检索的准确性，而较大的文本块则提供更多的上下文信息。

## **较小的子块引用较大的父块**

在检索过程中首先获取较小的块，然后引用父ID，并返回较大的块；

## 句子窗口检索

在检索过程中获取一个句子，并返回句子周围的文本窗口。



# **RAG坑点**

然而，RAG 系统受到信息检索系统固有的限制以及对LLM能力的依赖，RAG 系统中存在一些可能的“坑点”。

- **内容缺失**——这是生产案例中最大的问题之一。用户假设特定问题的答案存在于知识库中。事实并非如此，系统也没有回应“我不知道”。相反，它提供了一个看似合理的错误答案，但实际是“毫无意义”。
- **漏掉排名靠前的文档** - 检索器是小型搜索系统，要获得正确的结果并不简单。简单的嵌入查找很少能达到目的。有时，检索器返回的前 K 个文档中不存在正确答案，从而导致失败。
- **不符合上下文** - 有时，RAG系统可能会检索到太多文档，并且还是强制根据上下文分割并输入文档。这意味着对问题的回答不在上下文中。有时，这会导致模型产生幻觉，除非系统提示明确指示模型不要返回不在上下文中的结果。
- **未提取到有用信息** - 当LLM无法从上下文中提取答案时。当你塞满上下文并且LLM会感到困惑时，这往往会成为一个问题。不同大模型对背景信息的理解能力层次不齐。
- **格式错误**——虽然论文将这视为一种失败模式，但这种类型的功能并不是大型语言模型（LLM）的开箱即用功能。这种需要特定格式的输出，需要进行大量的系统提示和指令微调，以生成特定格式的信息。例如，使用Abacus AI，可以创建一个代理程序来以特定格式输出代码，并生成带有表格、段落、粗体文本等的Word文档。这种一般可以通过MarkDown输出来渲染！
- **不合适的回答** -响应中返回答案，但不够具体或过于具体，无法满足用户的需求。当 RAG 系统设计者对给定问题（例如教师对学生）有期望的结果时，就会发生这种情况。在这种情况下，应该提供具体的教育内容和答案，而不仅仅是答案。当用户不确定如何提出问题并且过于笼统时，也会出现不正确的特异性。

总的来说，这意味着 RAG 系统在投入生产之前必须经过彻底的稳健性测试，并且很容易因为发布未经测试的代理或聊天机器人而搬起石头砸自己的脚。

