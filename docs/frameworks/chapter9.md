# 第9章：其他主流框架

本章将介绍AutoGPT、LlamaIndex和Semantic Kernel等其他主流Agent框架。

## 9.1 AutoGPT

**什么是AutoGPT？**

AutoGPT是最早引起广泛关注的自主Agent框架之一。它的核心特点是：给定一个目标，Agent会自动分解任务、规划步骤、自主执行，直到目标完成。

打个比方：
- **普通Agent**：你告诉它"帮我查天气"，它查天气
- **AutoGPT**：你告诉它"帮我规划一次旅行"，它会自动：
  1. 搜索目的地信息
  2. 查询机票酒店价格
  3. 制定行程安排
  4. 生成预算报告
  5. 输出完整方案

**AutoGPT的核心特性**

1. **自主任务分解**
   - 不需要详细指令
   - Agent自己思考需要做什么
   - 将大目标拆分成小任务

2. **自我反思与调整**
   - 执行任务后会评估结果
   - 如果不满意会调整策略
   - 持续优化直到达成目标

3. **记忆与学习**
   - 记住之前的决策和结果
   - 避免重复犯错
   - 积累经验

**AutoGPT的工作流程**

```
设定目标
   ↓
思考：需要完成哪些子任务？
   ↓
执行第一个子任务
   ↓
评估：结果满意吗？
   ↓ 不满意
调整策略，重试
   ↓ 满意
执行下一个子任务
   ↓
所有任务完成 → 输出结果
```

**AutoGPT vs 普通Agent**

| 特性 | 普通Agent | AutoGPT |
|------|----------|---------|
| 指令方式 | 需要详细指令 | 只需给出目标 |
| 任务规划 | 人工规划 | 自动规划 |
| 自主性 | 低 | 高 |
| 适用场景 | 明确任务 | 开放性目标 |
| 成本 | 低 | 高（多次迭代） |

**AutoGPT的优缺点**

✅ **优点**：
- 自主性强，减少人工干预
- 能处理复杂、开放性的任务
- 可以自我优化和调整

❌ **缺点**：
- 成本高（需要多次LLM调用）
- 可能陷入死循环
- 不适合需要精确控制的任务

**适用场景**

- 研究任务：自动搜集资料、生成报告
- 创意任务：自动生成内容、设计方案
- 探索任务：尝试不同方法解决问题

### 自主Agent原理

AutoGPT是一个自主Agent，可以自动分解任务、规划执行路径。

```python
from autogpt import AutoGPT, Agent
from autogpt.config import Config

# 配置
config = Config()
config.openai_api_key = "your-api-key"

# 创建Agent
agent = Agent(
    name="Researcher",
    role="研究AI领域的最新进展",
    goals=[
        "搜索最新的AI研究论文",
        "总结关键发现",
        "生成研究报告"
    ],
    llm="gpt-4",
    memory="local",
    verbose=True
)

# 运行
autogpt = AutoGPT(agent=agent)
result = autogpt.run()
```

### 配置与使用

```python
from autogpt.config import Config

# 详细配置
config = Config()

# LLM配置
config.openai_api_key = "your-api-key"
config.temperature = 0.7
config.max_tokens = 2000

# 记忆配置
config.memory_backend = "local"  # 或 "pinecone", "redis"
config.memory_index = "autogpt_memory"

# 工具配置
config.enable_browse = True
config.enable_file_operations = True
config.enable_code_execution = True

# 安全配置
config.restrict_to_workspace = True
config.workspace_path = "./workspace"

# Agent配置
agent = Agent(
    name="Developer",
    role="软件开发助手",
    goals=[
        "理解用户需求",
        "设计解决方案",
        "编写代码",
        "测试和优化"
    ],
    llm="gpt-4",
    memory="local",
    tools=[
        "browse",
        "write_file",
        "read_file",
        "execute_code"
    ],
    verbose=True,
    continuous_mode=False,
    continuous_limit=10
)
```

### 优缺点分析

**优点**
- 完全自主运行
- 自动任务分解
- 内置多种工具
- 持续学习能力

**缺点**
- 成本较高（GPT-4）
- 可能陷入死循环
- 难以精确控制
- 调试困难

**适用场景**
- 长期任务
- 自主探索
- 研究项目

## 9.2 LlamaIndex

**什么是LlamaIndex？**

LlamaIndex（原名GPT Index）是一个专门用于构建数据连接和检索增强生成（RAG）应用的框架。它的核心优势在于让LLM能够轻松连接和查询各种数据源。

打个比方：
- **LangChain**：是一个全能工具箱，什么都能做
- **LlamaIndex**：是一个专业的图书管理员，擅长整理和检索文档

**LlamaIndex的核心价值**

大语言模型有两个主要限制：
1. 不知道你的私有数据
2. 训练数据有截止日期

LlamaIndex解决的就是这个问题：让LLM能够访问和查询你的私有数据。

**LlamaIndex的核心组件**

1. **数据连接器（Data Connectors）**
   - 加载各种格式的数据
   - 支持PDF、Word、网页、数据库、API等
   - 统一的数据加载接口

2. **索引结构（Index Structures）**
   - 组织和索引数据
   - 支持向量索引、树状索引、关键词索引等
   - 优化检索效率

3. **检索器（Retrievers）**
   - 根据查询找到相关数据
   - 支持语义检索、关键词检索、混合检索
   - 可定制检索策略

4. **查询引擎（Query Engines）**
   - 处理用户查询
   - 组合检索结果
   - 生成最终答案

**LlamaIndex vs LangChain**

| 特性 | LlamaIndex | LangChain |
|------|-----------|-----------|
| 专注领域 | 数据连接和RAG | 通用AI应用 |
| RAG能力 | 强，开箱即用 | 需要自己组合 |
| 数据源支持 | 丰富，开箱即用 | 需要自己实现 |
| 学习曲线 | 平缓 | 较陡 |
| 灵活性 | 中等 | 高 |

**LlamaIndex的典型应用**

- 企业知识库问答
- 文档检索和摘要
- 个人知识管理
- 数据分析和报告生成

**为什么选择LlamaIndex？**

✅ **适合使用LlamaIndex**：
- 主要需求是RAG（检索增强生成）
- 需要连接多种数据源
- 希望快速构建知识库问答系统

❌ **不适合使用LlamaIndex**：
- 需要复杂的Agent工作流
- 需要多Agent协作
- 需要大量的工具调用

### 数据连接框架

LlamaIndex专注于数据连接和检索增强生成（RAG）。

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding

# 加载文档
documents = SimpleDirectoryReader('data').load_data()

# 创建索引
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo"),
    embed_model=OpenAIEmbedding()
)

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("什么是机器学习？")
print(response)
```

### 与LangChain集成

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# 创建LlamaIndex查询引擎
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 创建LangChain工具
llama_tool = Tool(
    name="knowledge_search",
    func=lambda q: str(query_engine.query(q)),
    description="搜索知识库"
)

# 创建Agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
agent = create_tool_calling_agent(llm, [llama_tool])
agent_executor = AgentExecutor(agent=agent, tools=[llama_tool])

# 使用
result = agent_executor.invoke({"input": "什么是深度学习？"})
```

### RAG最佳实践

```python
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext
)
from llama_index.node_parser import SentenceSplitter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

# 1. 文档预处理
def preprocess_documents(file_path):
    documents = SimpleDirectoryReader(file_path).load_data()
    
    # 清理文档
    cleaned_docs = []
    for doc in documents:
        cleaned_text = doc.text.strip()
        if cleaned_text:
            cleaned_docs.append(Document(text=cleaned_text))
    
    return cleaned_docs

# 2. 智能切分
def create_index(documents):
    node_parser = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser
    )
    
    return index

# 3. 高级检索
def create_query_engine(index):
    # 自定义检索器
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )
    
    # 响应合成器
    response_synthesizer = get_response_synthesizer(
        response_mode="compact"
    )
    
    # 查询引擎
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    
    return query_engine

# 4. 使用
documents = preprocess_documents('data')
index = create_index(documents)
query_engine = create_query_engine(index)

response = query_engine.query("你的问题")
print(response)
```

## 9.3 Semantic Kernel

**什么是Semantic Kernel？**

Semantic Kernel是微软推出的轻量级SDK，专门用于将大语言模型集成到应用中。它的名字来源于"语义内核"，寓意是作为应用的"大脑"。

**Semantic Kernel的设计理念**

与其他框架不同，Semantic Kernel强调：
1. **轻量级**：核心简单，易于扩展
2. **企业级**：专为生产环境设计
3. **多语言**：支持C#、Python、Java
4. **微软生态**：与Azure OpenAI深度集成

**Semantic Kernel的核心概念**

1. **Skills（技能）**
   - 类似于其他框架的"工具"
   - 分为语义技能（用Prompt定义）和原生技能（用代码定义）
   - 可以组合使用

2. **Functions（函数）**
   - Skills中的具体功能
   - 可以被Agent调用
   - 支持参数和返回值

3. **Planners（规划器）**
   - 自动规划任务执行顺序
   - 根据目标选择合适的技能
   - 类似于AutoGPT的自主规划

4. **Memories（记忆）**
   - 保存对话历史和上下文
   - 支持向量存储
   - 可以持久化

**Semantic Kernel vs 其他框架**

| 特性 | Semantic Kernel | LangChain | AutoGPT |
|------|----------------|-----------|---------|
| 开发者 | 微软 | 开源社区 | 开源社区 |
| 语言支持 | C#, Python, Java | Python, JS | Python |
| 企业级 | 强 | 中等 | 弱 |
| 学习曲线 | 平缓 | 较陡 | 陡峭 |
| Azure集成 | 原生 | 需要配置 | 需要配置 |

**Semantic Kernel的典型应用**

- 微软365 Copilot
- Azure OpenAI应用
- 企业智能助手
- 业务流程自动化

**为什么选择Semantic Kernel？**

✅ **适合使用Semantic Kernel**：
- 使用微软技术栈（C#、Azure）
- 需要企业级支持和稳定性
- 希望与Azure OpenAI深度集成
- 构建生产级应用

❌ **不适合使用Semantic Kernel**：
- 使用非微软技术栈
- 需要最新的AI特性（更新较慢）
- 社区生态和文档相对较少

### 微软Agent框架

Semantic Kernel是微软推出的Agent框架，专注于企业应用。

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planning import SequentialPlanner

# 创建Kernel
kernel = Kernel()

# 添加LLM
kernel.add_chat_service(
    "gpt-3.5",
    OpenAIChatCompletion(
        "gpt-3.5-turbo",
        api_key="your-api-key"
    )
)

# 定义技能
async def search_skill(query: str) -> str:
    """搜索技能"""
    return f"搜索结果：{query}"

async def analyze_skill(data: str) -> str:
    """分析技能"""
    return f"分析结果：{data}"

# 注册技能
kernel.register_function(
    search_skill,
    "search",
    "search_skill"
)

kernel.register_function(
    analyze_skill,
    "analyze",
    "analyze_skill"
)

# 创建规划器
planner = SequentialPlanner(kernel)

# 执行
goal = "搜索AI信息并分析"
plan = await planner.create_plan_async(goal)
result = await plan.invoke_async()
print(result)
```

### 技能（Skills）系统

```python
from semantic_kernel import Kernel, Skill
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.orchestration import ContextVariables

# 定义技能
class DataAnalysisSkill(Skill):
    """数据分析技能"""
    
    def __init__(self, kernel):
        super().__init__(kernel)
        self.register_function(self.load_data)
        self.register_function(self.analyze)
        self.register_function(self.visualize)
    
    async def load_data(self, source: str) -> str:
        """加载数据"""
        return f"从{source}加载数据"
    
    async def analyze(self, data: str) -> str:
        """分析数据"""
        return f"分析{data}"
    
    async def visualize(self, analysis: str) -> str:
        """可视化"""
        return f"可视化{analysis}"

# 使用
kernel = Kernel()
kernel.add_chat_service(
    "gpt-3.5",
    OpenAIChatCompletion("gpt-3.5-turbo", api_key="your-api-key")
)

# 添加技能
data_skill = DataAnalysisSkill(kernel)

# 执行
context = ContextVariables()
context["source"] = "data.csv"

result = await kernel.run_async(
    data_skill.load_data,
    data_skill.analyze,
    data_skill.visualize,
    input_vars=context
)
```

### 企业应用场景

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planning import ActionPlanner

# 企业场景：客户服务
class CustomerServiceKernel:
    def __init__(self):
        self.kernel = Kernel()
        self.kernel.add_chat_service(
            "gpt-4",
            OpenAIChatCompletion("gpt-4", api_key="your-api-key")
        )
        self.setup_skills()
    
    def setup_skills(self):
        """设置企业技能"""
        
        # 查询订单
        async def query_order(order_id: str) -> str:
            return f"订单{order_id}的信息"
        
        # 处理退款
        async def process_refund(order_id: str, reason: str) -> str:
            return f"处理退款：{order_id}，原因：{reason}"
        
        # 更新客户信息
        async def update_customer(customer_id: str, info: str) -> str:
            return f"更新客户{customer_id}的信息"
        
        # 注册技能
        self.kernel.register_function(query_order, "query_order")
        self.kernel.register_function(process_refund, "process_refund")
        self.kernel.register_function(update_customer, "update_customer")
    
    async def handle_request(self, user_request: str):
        """处理客户请求"""
        planner = ActionPlanner(self.kernel)
        plan = await planner.create_plan_async(user_request)
        result = await plan.invoke_async()
        return result

# 使用
service = CustomerServiceKernel()
result = await service.handle_request("查询订单12345的状态")
print(result)
```

## 9.4 【实战】框架对比项目

让我们用不同框架实现同一个需求，进行对比。

### 需求：智能文档问答系统

**LangChain实现**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# 创建QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 查询
result = qa_chain.invoke("你的问题")
print(result["result"])
```

**LlamaIndex实现**

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.query_engine import RetrieverQueryEngine

# 加载文档
documents = SimpleDirectoryReader("document.txt").load_data()

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)

# 查询
response = query_engine.query("你的问题")
print(response)
```

**CrewAI实现**

```python
from crewai import Agent, Task, Crew

# 定义Agent
researcher = Agent(
    role="研究员",
    goal="搜索文档中的相关信息",
    backstory="你擅长文档检索和信息提取",
    tools=[...]  # 添加检索工具
)

analyst = Agent(
    role="分析师",
    goal="分析和总结信息",
    backstory="你擅长信息分析和总结"
)

# 定义任务
research_task = Task(
    description="在文档中搜索相关信息",
    agent=researcher
)

analysis_task = Task(
    description="分析搜索结果并回答问题",
    agent=analyst,
    context=[research_task]
)

# 创建Crew
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task]
)

# 执行
result = crew.kickoff()
```

### 性能与易用性对比

| 指标 | LangChain | LlamaIndex | CrewAI |
|------|-----------|------------|--------|
| 代码量 | 中等 | 少 | 多 |
| 学习难度 | 中 | 低 | 中 |
| 灵活性 | 高 | 中 | 高 |
| 性能 | 好 | 优秀 | 好 |
| 文档质量 | 优秀 | 优秀 | 良好 |
| 社区活跃度 | 高 | 高 | 中 |

### 选型建议

**选择LangChain如果：**
- 需要最大的灵活性
- 想要丰富的工具生态
- 需要自定义Agent行为
- 项目规模较大

**选择LlamaIndex如果：**
- 主要做RAG应用
- 需要高性能检索
- 想要简单易用
- 专注于数据处理

**选择CrewAI如果：**
- 需要多Agent协作
- 任务复杂需要分工
- 想要清晰的团队结构
- 需要任务编排

**选择AutoGPT如果：**
- 需要完全自主
- 任务长期运行
- 想要探索性任务
- 预算充足

**选择Semantic Kernel如果：**
- 企业应用场景
- 需要微软生态集成
- 关注安全性
- 现有.NET环境

## 本章小结

本章我们学习了：

- ✅ AutoGPT自主Agent的原理和使用
- ✅ LlamaIndex数据连接框架
- ✅ Semantic Kernel企业框架
- ✅ 不同框架的对比分析
- ✅ 框架选型建议

## 下一章

下一章我们将进入第四阶段，学习Agent架构设计。

[第10章：Agent架构设计 →](/advanced/chapter10)
