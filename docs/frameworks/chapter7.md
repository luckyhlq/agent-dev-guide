# 第7章：LangChain框架

本章将深入讲解LangChain框架的核心概念和使用方法，这是目前最流行的Agent开发框架之一。

## 7.1 LangChain核心概念

**什么是LangChain？**

LangChain是目前最流行的AI应用开发框架，它的目标是让开发者能够快速构建基于大语言模型的应用。

打个比方：
- **没有LangChain**：就像用原生JavaScript开发网页，什么都要自己写
- **有了LangChain**：就像用React/Vue框架，提供了现成的组件和工具

**LangChain的三大核心概念**

1. **Chain（链）**
   - **是什么**：将多个处理步骤串联起来，像流水线一样
   - **为什么需要**：复杂的AI应用通常需要多个步骤，比如"理解问题→检索文档→生成答案"
   - **例子**：问答链 = 问题理解 + 文档检索 + 答案生成

2. **Agent（代理）**
   - **是什么**：能自主决策、调用工具的智能体
   - **为什么需要**：有些任务需要根据情况动态决定做什么，而不是固定流程
   - **例子**：用户问"北京天气"，Agent会自动调用天气工具；问"1+1"，Agent会直接回答

3. **Tool（工具）**
   - **是什么**：Agent可以调用的外部功能
   - **为什么需要**：让AI能够执行实际操作，比如搜索、计算、查询数据库
   - **例子**：搜索工具、计算器工具、数据库查询工具

**LangChain vs 原生开发**

| 特性 | 原生开发 | LangChain |
|------|---------|-----------|
| 开发速度 | 慢，从零开始 | 快，使用现成组件 |
| 代码量 | 多 | 少 |
| 可维护性 | 需要自己设计 | 有统一架构 |
| 学习成本 | 低 | 需要学习框架 |
| 灵活性 | 完全自由 | 受框架限制 |

**LCEL（LangChain表达式语言）**

LCEL是LangChain的一大特色，它用简洁的语法组合组件：

```python
# 传统写法
output = llm.invoke(prompt.format(input=text))

# LCEL写法（更简洁）
chain = prompt | llm | output_parser
output = chain.invoke({"input": text})
```

这个`|`符号就像Linux的管道，把前一个组件的输出传给后一个组件。

### Chain、Agent、Tool

LangChain的三大核心概念：

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tool - 工具
def get_word_length(word: str) -> int:
    return len(word)

tools = [
    Tool(
        name="get_word_length",
        func=get_word_length,
        description="返回单词的长度"
    )
]

# Chain - 链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("user", "{input}")
])

chain = prompt | llm

# Agent - 代理
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

### LCEL（LangChain Expression Language）

LCEL是LangChain的表达式语言，用于组合组件：

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

# 基本链
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

result = chain.invoke({"topic": "程序员"})
print(result)

# 带条件的链
from langchain.schema.runnable import RunnableBranch

branch_chain = RunnableBranch(
    (lambda x: "笑话" in x["topic"], prompt | model),
    (lambda x: "故事" in x["topic"], ChatPromptTemplate.from_template("讲一个关于{topic}的故事") | model),
    ChatPromptTemplate.from_template("关于{topic}说点什么") | model
)

# 并行执行
from langchain.schema.runnable import RunnableParallel

parallel_chain = RunnableParallel(
    joke=ChatPromptTemplate.from_template("讲一个关于{topic}的笑话") | model,
    poem=ChatPromptTemplate.from_template("写一首关于{topic}的诗") | model
)

results = parallel_chain.invoke({"topic": "春天"})
print(results["joke"].content)
print(results["poem"].content)
```

### 回调系统

```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs):
        print(f"LLM开始，提示词：{prompts}")
    
    def on_llm_end(self, response: Any, **kwargs):
        print(f"LLM结束，响应：{response}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        print(f"Chain开始，输入：{inputs}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        print(f"Chain结束，输出：{outputs}")

# 使用回调
chain = prompt | model | output_parser
result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [MyCallbackHandler()]}
)
```

## 7.2 LangChain组件详解

**LangChain的核心组件**

LangChain提供了丰富的组件，每个组件负责特定的功能：

**1. Prompts（提示词管理）**
- **作用**：管理和组织提示词模板
- **为什么需要**：提示词是AI应用的核心，需要版本管理、复用、参数化
- **核心功能**：
  - 模板管理：定义可复用的提示词模板
  - 变量注入：动态填充变量
  - 示例选择：自动选择few-shot示例

**2. Models（模型接口）**
- **作用**：统一不同LLM的调用接口
- **为什么需要**：不同LLM的API不同，统一接口方便切换
- **支持的模型**：OpenAI、Anthropic、本地模型等

**3. Memory（记忆管理）**
- **作用**：管理对话历史和上下文
- **为什么需要**：让AI能记住之前的对话
- **类型**：
  - 对话缓存：保存最近N轮对话
  - 摘要记忆：压缩长对话
  - 向量记忆：语义检索历史

**4. Indexes（索引与检索）**
- **作用**：文档索引和检索
- **为什么需要**：实现RAG（检索增强生成）
- **组件**：
  - 文档加载器：加载PDF、网页等
  - 文本分割器：切分长文档
  - 向量存储：存储和检索向量

**5. Chains（链）**
- **作用**：组合多个组件形成处理流程
- **为什么需要**：复杂任务需要多个步骤
- **类型**：
  - 简单链：顺序执行
  - 分支链：条件判断
  - 并行链：同时执行多个任务

**组件之间的关系**

```
用户输入
   ↓
Prompts（组装提示词）
   ↓
Models（调用LLM）
   ↓
Memory（保存对话）
   ↓
Indexes（检索相关文档）
   ↓
Chains（组合以上流程）
   ↓
输出结果
```

### Prompts管理

```python
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate
)

# 基本Prompt模板
template = """你是一个{role}。

任务：{task}

要求：
{requirements}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["role", "task", "requirements"]
)

# Chat Prompt模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，专注于{domain}领域。"),
    ("user", "{input}")
])

# Few-shot Prompt
examples = [
    {"input": "开心", "output": "positive"},
    {"input": "难过", "output": "negative"},
    {"input": "还行", "output": "neutral"}
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入：{input}\n输出：{output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="分析情感：",
    suffix="输入：{input}\n输出：",
    input_variables=["input"]
)

# Prompt选择器
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="分析情感：",
    suffix="输入：{input}\n输出：",
    input_variables=["input"]
)
```

### Memory组件

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 对话缓冲记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

memory.save_context({"input": "你好"}, {"output": "你好！有什么可以帮助你的？"})
memory.save_context({"input": "我是小明"}, {"output": "你好小明！"})

print(memory.load_memory_variables({}))

# 窗口记忆（只保留最近k轮）
window_memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    return_messages=True
)

# 摘要记忆
summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),
    memory_key="chat_history",
    return_messages=True
)

# 向量存储记忆
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

vector_memory = VectorStoreRetrieverMemory(
    retriever=retriever
)

vector_memory.save_context(
    {"input": "我喜欢Python"},
    {"output": "好的，我记住了你喜欢Python"}
)

# 在链中使用记忆
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=ConversationBufferMemory(),
    verbose=True
)

response = conversation.predict(input="你好，我是小明")
print(response)
```

### Chains组合

```python
from langchain.chains import (
    LLMChain,
    SequentialChain,
    SimpleSequentialChain,
    TransformChain
)

# 简单链
llm_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_template("翻译成英文：{text}")
)

# 顺序链
chain1 = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_template("将'{text}'翻译成英文"),
    output_key="english"
)

chain2 = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_template("将'{english}'翻译成日文"),
    output_key="japanese"
)

sequential_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["text"],
    output_variables=["english", "japanese"]
)

# 转换链
def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    return {"transformed_text": text.upper()}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["transformed_text"],
    transform=transform_func
)

# 组合链
full_chain = transform_chain | ChatPromptTemplate.from_template("处理：{transformed_text}") | ChatOpenAI()
```

### Output Parsers

```python
from langchain.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
from langchain.pydantic_v1 import BaseModel, Field
from typing import List

# 字符串解析器
str_parser = StrOutputParser()

# JSON解析器
json_parser = JsonOutputParser()

# Pydantic解析器
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    hobbies: List[str] = Field(description="爱好")

pydantic_parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息提取助手。\n{format_instructions}"),
    ("user", "{query}")
])

chain = prompt | ChatOpenAI() | pydantic_parser

result = chain.invoke({
    "query": "小明今年25岁，喜欢编程、读书和跑步",
    "format_instructions": pydantic_parser.get_format_instructions()
})

print(result)

# 列表解析器
list_parser = CommaSeparatedListOutputParser()

list_prompt = ChatPromptTemplate.from_template(
    "列出5种编程语言。\n{format_instructions}"
)

list_chain = list_prompt | ChatOpenAI() | list_parser

languages = list_chain.invoke({
    "format_instructions": list_parser.get_format_instructions()
})

print(languages)
```

## 7.3 LangChain Agent

**LangChain中的Agent**

LangChain的Agent是框架中最强大的功能之一，它能让AI自主决策、调用工具、执行复杂任务。

**Agent vs Chain**

| 特性 | Chain（链） | Agent（代理） |
|------|-----------|--------------|
| 执行方式 | 固定流程 | 动态决策 |
| 适用场景 | 明确的任务流程 | 需要灵活应对的场景 |
| 工具调用 | 预先定义 | 自主决定 |
| 复杂度 | 简单 | 复杂 |
| 示例 | 翻译链：输入→翻译→输出 | 智能助手：根据问题决定用什么工具 |

**Agent的工作流程**

```
用户输入
   ↓
Agent思考：这个问题需要什么工具？
   ↓
选择工具并执行
   ↓
观察结果
   ↓
继续思考：还需要其他工具吗？
   ↓
生成最终答案
```

**LangChain Agent的类型**

1. **Zero-shot Agent**
   - 最基础的Agent
   - 根据工具描述决定使用哪个工具
   - 适合简单场景

2. **Structured Tool Agent**
   - 支持多输入参数的工具
   - 适合需要复杂参数的工具

3. **OpenAI Functions Agent**
   - 专门为OpenAI Function Calling优化
   - 性能最好，推荐使用

4. **Self-Ask with Search**
   - 自我提问并搜索答案
   - 适合需要多步推理的问题

**何时使用Agent？**

✅ **适合使用Agent**：
- 任务流程不固定，需要根据情况调整
- 需要调用多个工具
- 用户问题多样化，无法预先定义流程

❌ **不适合使用Agent**：
- 任务流程固定且简单
- 不需要工具调用
- 追求最高性能（Agent有决策开销）

**Agent的最佳实践**

1. **工具设计要清晰**：工具名称和描述要准确，让Agent容易理解
2. **限制工具数量**：工具太多会让Agent困惑，建议不超过10个
3. **设置最大迭代次数**：防止Agent陷入死循环
4. **记录决策过程**：方便调试和优化

### Agent类型选择

```python
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
    create_openai_functions_agent,
    create_react_agent,
    create_structured_chat_agent
)
from langchain import hub

# Tool Calling Agent（推荐）
tools = [...]  # 定义工具

prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(ChatOpenAI(), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# OpenAI Functions Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(ChatOpenAI(), tools, prompt)

# ReAct Agent
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(ChatOpenAI(), tools, prompt)

# Structured Chat Agent
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(ChatOpenAI(), tools, prompt)
```

### 自定义Agent

```python
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Any, Union

class CustomAgent(BaseSingleActionAgent):
    tools: List
    llm: Any
    
    @property
    def input_keys(self):
        return ["input"]
    
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        # 自定义规划逻辑
        user_input = kwargs["input"]
        
        # 简单的关键词匹配
        for tool in self.tools:
            if tool.name.lower() in user_input.lower():
                return AgentAction(
                    tool=tool.name,
                    tool_input=user_input,
                    log=f"使用工具：{tool.name}"
                )
        
        # 直接使用LLM
        response = self.llm.invoke(user_input)
        return AgentFinish(
            return_values={"output": response.content},
            log="直接使用LLM回答"
        )
    
    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        return self.plan(intermediate_steps, **kwargs)

# 使用自定义Agent
agent = CustomAgent(tools=tools, llm=ChatOpenAI())
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools
)
```

### Agent调试技巧

```python
from langchain.globals import set_debug, set_verbose

# 开启调试模式
set_debug(True)
set_verbose(True)

# 使用回调追踪
class TracingCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"[Chain Start] {serialized.get('name', 'unknown')}")
        print(f"  Inputs: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        print(f"[Chain End] Outputs: {outputs}")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LLM Start] Prompts: {prompts}")
    
    def on_llm_end(self, response, **kwargs):
        print(f"[LLM End] Response: {response}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[Tool Start] {serialized.get('name')}")
        print(f"  Input: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"[Tool End] Output: {output}")

# 使用追踪
result = agent_executor.invoke(
    {"input": "计算25*4"},
    config={"callbacks": [TracingCallbackHandler()]}
)
```

## 7.4 【实战】LangChain数据分析Agent

让我们构建一个完整的数据分析Agent。

### 项目结构

```
langchain-data-agent/
├── .env
├── main.py
├── agent.py
├── tools.py
└── requirements.txt
```

### 完整代码

**tools.py**

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt

class CSVInput(BaseModel):
    filepath: str = Field(description="CSV文件路径")

class QueryInput(BaseModel):
    query: str = Field(description="查询语句")

class LoadCSVTool(BaseTool):
    name = "load_csv"
    description = "加载CSV文件"
    args_schema: Type[BaseModel] = CSVInput
    
    df: Optional[pd.DataFrame] = None
    
    def _run(self, filepath: str):
        try:
            self.df = pd.read_csv(filepath)
            return f"成功加载CSV文件，共{len(self.df)}行，{len(self.df.columns)}列\n列名：{list(self.df.columns)}"
        except Exception as e:
            return f"加载失败：{str(e)}"

class QueryDataTool(BaseTool):
    name = "query_data"
    description = "查询数据，支持pandas查询语法"
    args_schema: Type[BaseModel] = QueryInput
    
    df: Optional[pd.DataFrame] = None
    
    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
    
    def _run(self, query: str):
        if self.df is None:
            return "请先加载CSV文件"
        
        try:
            result = self.df.query(query)
            return result.to_string()
        except Exception as e:
            return f"查询失败：{str(e)}"

class AnalyzeDataTool(BaseTool):
    name = "analyze_data"
    description = "分析数据统计信息"
    
    df: Optional[pd.DataFrame] = None
    
    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
    
    def _run(self, query: str = ""):
        if self.df is None:
            return "请先加载CSV文件"
        
        return self.df.describe().to_string()

class VisualizeTool(BaseTool):
    name = "visualize"
    description = "创建数据可视化图表"
    
    df: Optional[pd.DataFrame] = None
    
    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
    
    def _run(self, query: str):
        if self.df is None:
            return "请先加载CSV文件"
        
        try:
            # 简单的可视化
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                self.df[numeric_cols[:4]].hist(figsize=(10, 8))
                plt.savefig('visualization.png')
                return "已生成可视化图表：visualization.png"
            
            return "没有数值列可以可视化"
        except Exception as e:
            return f"可视化失败：{str(e)}"
```

**agent.py**

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from tools import LoadCSVTool, QueryDataTool, AnalyzeDataTool, VisualizeTool

class DataAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # 初始化工具
        self.load_tool = LoadCSVTool()
        self.query_tool = QueryDataTool()
        self.analyze_tool = AnalyzeDataTool()
        self.visualize_tool = VisualizeTool()
        
        self.tools = [
            self.load_tool,
            self.query_tool,
            self.analyze_tool,
            self.visualize_tool
        ]
        
        # 创建Agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个数据分析专家。你可以：
1. 加载CSV文件
2. 查询和分析数据
3. 生成统计报告
4. 创建可视化图表

请根据用户需求选择合适的工具。"""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
    
    def run(self, user_input: str):
        return self.agent_executor.invoke({"input": user_input})
```

**main.py**

```python
from dotenv import load_dotenv
from agent import DataAnalysisAgent

load_dotenv()

def main():
    agent = DataAnalysisAgent()
    
    print("=" * 60)
    print("数据分析Agent")
    print("=" * 60)
    print("\n示例命令：")
    print("  - 加载data.csv文件")
    print("  - 显示数据统计信息")
    print("  - 查询age > 30的数据")
    print("  - 创建可视化图表")
    print("  quit - 退出")
    print("=" * 60)
    
    while True:
        user_input = input("\n输入: ").strip()
        
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        result = agent.run(user_input)
        print(f"\n结果：{result['output']}")

if __name__ == "__main__":
    main()
```

## 本章小结

本章我们学习了：

- ✅ LangChain的核心概念：Chain、Agent、Tool
- ✅ LCEL表达式语言的使用
- ✅ Prompts管理和Memory组件
- ✅ Chains组合和Output Parsers
- ✅ Agent类型选择和自定义Agent
- ✅ 构建了数据分析Agent

## 下一章

下一章我们将学习CrewAI框架，探索多Agent协作。

[第8章：CrewAI框架 →](/frameworks/chapter8)
