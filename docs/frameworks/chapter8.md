# 第8章：CrewAI框架

本章将深入讲解CrewAI框架，这是一个专注于多Agent协作的框架。

## 8.1 CrewAI概述

**什么是CrewAI？**

CrewAI是一个专门用于构建多Agent协作系统的框架。它的名字来源于"Crew"（团队），核心理念是让多个AI Agent像团队一样协作完成复杂任务。

打个比方：
- **单Agent**：就像一个人完成所有工作，能力有限
- **CrewAI多Agent**：就像一个团队，有研究员、作家、编辑等，各司其职，协作完成

**为什么需要多Agent协作？**

现实中的复杂任务往往需要多种能力：

| 任务 | 需要的能力 | 单Agent | 多Agent协作 |
|------|-----------|---------|------------|
| 写一篇研究报告 | 研究+写作+审核 | 可能顾此失彼 | 研究员研究，作家写作，编辑审核 |
| 开发一个功能 | 设计+编码+测试 | 容易出错 | 设计师设计，程序员编码，测试员测试 |
| 客户服务 | 理解+查询+回答 | 可能不专业 | 接待员理解，专家查询，客服回答 |

**CrewAI的核心概念**

1. **Agent（代理）**
   - 扮演特定角色的AI
   - 有明确的目标和背景故事
   - 例如：研究员、作家、编辑、程序员

2. **Task（任务）**
   - 需要完成的具体工作
   - 可以设置依赖关系
   - 例如：研究主题、撰写文章、审核内容

3. **Crew（团队）**
   - 由多个Agent组成的团队
   - 协调Agent完成任务
   - 定义协作方式（顺序或并行）

4. **Process（流程）**
   - Sequential：顺序执行，一个接一个
   - Hierarchical：层级管理，有管理者Agent协调

**CrewAI vs LangChain Agent**

| 特性 | LangChain Agent | CrewAI |
|------|----------------|--------|
| 核心思想 | 单个智能Agent | 多Agent协作 |
| 适用场景 | 任务流程灵活 | 需要多种专业能力 |
| 复杂度 | 中等 | 较高 |
| 学习曲线 | 平缓 | 较陡 |
| 最佳用途 | 工具调用型任务 | 创作型、专业型任务 |

**CrewAI的应用场景**

- ✅ 内容创作：研究→写作→编辑→发布
- ✅ 代码开发：设计→编码→测试→部署
- ✅ 数据分析：收集→清洗→分析→报告
- ✅ 客户服务：接待→查询→解答→反馈

### 多Agent协作理念

CrewAI的核心思想是让多个具有不同角色的Agent协作完成复杂任务。

```python
from crewai import Agent, Task, Crew, Process

# 定义Agent
researcher = Agent(
    role="研究员",
    goal="收集和分析信息",
    backstory="你是一位经验丰富的研究员，擅长信息收集和分析",
    verbose=True
)

writer = Agent(
    role="作家",
    goal="撰写高质量文章",
    backstory="你是一位专业作家，擅长将复杂信息转化为易懂的文章",
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究AI Agent的最新发展",
    agent=researcher
)

writing_task = Task(
    description="基于研究结果撰写文章",
    agent=writer,
    context=[research_task]  # 依赖research_task
)

# 创建Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential  # 顺序执行
)

# 执行
result = crew.kickoff()
print(result)
```

### Crew、Agent、Task概念

**Crew（团队）**
- 管理多个Agent的协作
- 定义任务执行流程
- 协调Agent之间的通信

**Agent（代理）**
- 具有特定角色和目标
- 拥有自己的工具和记忆
- 可以独立完成任务

**Task（任务）**
- 具体的工作单元
- 可以分配给特定Agent
- 可以有依赖关系

### 与LangChain对比

| 特性 | LangChain | CrewAI |
|------|-----------|--------|
| 主要用途 | 构建单Agent应用 | 多Agent协作 |
| 学习曲线 | 较陡 | 较平缓 |
| 灵活性 | 高 | 中等 |
| 多Agent支持 | 需要自己实现 | 原生支持 |
| 工具生态 | 丰富 | 较少 |

## 8.2 构建Agent团队

**如何设计Agent团队？**

构建一个高效的Agent团队，就像组建一个真实的工作团队，需要考虑：

**1. 角色分工**

每个Agent应该有明确的职责，避免职责重叠：

```
内容创作团队：
├─ 研究员：负责收集资料、调研
├─ 作家：负责撰写内容
├─ 编辑：负责审核、润色
└─ 发布员：负责格式化、发布
```

**2. 能力互补**

团队成员的能力要互补，形成完整的工作流：

- 研究员擅长信息检索和分析
- 作家擅长文字表达和创意
- 编辑擅长审核和优化
- 每个人做自己擅长的事

**3. 协作流程**

定义清晰的任务流转：

```
研究员调研 → 传递给作家 → 作家写作 → 传递给编辑 → 编辑审核 → 最终输出
```

**Agent角色设计原则**

1. **角色定位清晰**
   - 明确这个Agent负责什么
   - 明确这个Agent不负责什么
   - 避免角色模糊导致任务重叠

2. **目标具体可衡量**
   - ❌ "写好文章"（太模糊）
   - ✅ "撰写结构清晰、内容准确、语言流畅的文章"

3. **背景故事有说服力**
   - 提供Agent的专业背景
   - 帮助Agent理解自己的角色定位
   - 例如："你是一位有10年经验的科技记者"

4. **工具配备合理**
   - 给Agent提供完成任务所需的工具
   - 研究员需要搜索工具
   - 作家需要文档工具
   - 不要给不需要的工具

**任务依赖管理**

CrewAI支持任务依赖，这是实现协作的关键：

```python
# 任务A的输出作为任务B的输入
task_b = Task(
    description="基于task_a的结果继续工作",
    context=[task_a]  # 依赖task_a
)
```

**协作模式选择**

1. **顺序模式（Sequential）**
   - 任务按顺序执行
   - 前一个任务的输出作为后一个任务的输入
   - 适合有明确流程的任务

2. **层级模式（Hierarchical）**
   - 有一个管理者Agent
   - 管理者分配任务给其他Agent
   - 适合复杂、需要协调的任务

### Agent角色定义

```python
from crewai import Agent

# 产品经理Agent
product_manager = Agent(
    role="产品经理",
    goal="定义产品需求和功能",
    backstory="""你是一位资深产品经理，拥有10年经验。
你擅长：
- 用户需求分析
- 产品功能规划
- 需求文档编写
- 跨团队协调""",
    verbose=True,
    allow_delegation=True,  # 允许委托任务
    tools=[...]  # 可以添加工具
)

# 开发工程师Agent
developer = Agent(
    role="开发工程师",
    goal="实现产品功能",
    backstory="""你是一位全栈开发工程师，精通：
- Python/JavaScript
- 前后端开发
- 数据库设计
- API开发""",
    verbose=True,
    allow_delegation=False
)

# 测试工程师Agent
tester = Agent(
    role="测试工程师",
    goal="确保产品质量",
    backstory="""你是一位测试专家，擅长：
- 功能测试
- 自动化测试
- 性能测试
- Bug定位""",
    verbose=True
)
```

### Task分配策略

```python
from crewai import Task

# 顺序任务
task1 = Task(
    description="分析用户需求",
    agent=product_manager
)

task2 = Task(
    description="设计产品架构",
    agent=developer,
    context=[task1]  # 依赖task1
)

task3 = Task(
    description="编写测试用例",
    agent=tester,
    context=[task2]  # 依赖task2
)

# 并行任务
parallel_task1 = Task(
    description="设计数据库",
    agent=developer
)

parallel_task2 = Task(
    description="设计API接口",
    agent=developer
)

# 条件任务
def should_create_ui(task):
    return "UI" in task.description

ui_task = Task(
    description="创建用户界面",
    agent=developer,
    condition=should_create_ui
)
```

### 协作模式设计

```python
from crewai import Crew, Process

# 顺序协作
sequential_crew = Crew(
    agents=[product_manager, developer, tester],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=True
)

# 层级协作
manager = Agent(
    role="项目经理",
    goal="协调团队工作",
    backstory="你是一位经验丰富的项目经理",
    allow_delegation=True
)

hierarchical_crew = Crew(
    agents=[manager, product_manager, developer, tester],
    tasks=[...],
    process=Process.hierarchical,
    manager_llm="gpt-4",  # 管理者使用更强的模型
    verbose=True
)

# 自定义协作流程
class CustomProcess:
    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks
    
    def run(self):
        # 自定义执行逻辑
        results = []
        for task in self.tasks:
            agent = self.select_agent(task)
            result = agent.execute(task)
            results.append(result)
        return results
    
    def select_agent(self, task):
        # 自定义Agent选择逻辑
        for agent in self.agents:
            if agent.can_handle(task):
                return agent
        return self.agents[0]
```

## 8.3 工具与流程

**CrewAI中的工具**

工具是Agent执行具体操作的能力来源。在CrewAI中，工具的使用有几个特点：

**1. 工具分配**

每个Agent可以有自己的专属工具：

```python
researcher = Agent(
    role="研究员",
    tools=[search_tool, web_scraper]  # 研究员的工具
)

writer = Agent(
    role="作家",
    tools=[grammar_checker, word_counter]  # 作家的工具
)
```

**2. 工具共享**

多个Agent可以共享工具：

```python
shared_tools = [calculator, translator]

agent1 = Agent(role="分析师", tools=shared_tools)
agent2 = Agent(role="编辑", tools=shared_tools)
```

**3. 自定义工具**

可以为特定需求创建自定义工具：

```python
from crewai import Tool

@Tool("数据分析工具")
def analyze_data(data_path: str) -> str:
    """分析数据并返回结果"""
    # 实现分析逻辑
    return analysis_result
```

**流程编排**

CrewAI提供了灵活的流程编排能力：

**1. 顺序流程（Sequential）**

最简单的流程，任务按顺序执行：

```
任务1 → 任务2 → 任务3 → 完成
```

适用场景：
- 有明确的先后顺序
- 后续任务依赖前面的结果
- 例如：研究→写作→编辑

**2. 层级流程（Hierarchical）**

有管理者Agent协调其他Agent：

```
        管理者Agent
       /     |     \
   Agent1  Agent2  Agent3
      \      |      /
       完成任务汇总
```

适用场景：
- 任务复杂，需要协调
- 需要动态分配任务
- 例如：项目经理协调开发、测试、设计

**流程设计最佳实践**

1. **任务粒度适中**
   - 太粗：难以管理和调试
   - 太细：增加通信开销
   - 建议：一个任务对应一个明确的输出

2. **依赖关系清晰**
   - 明确标注任务依赖
   - 避免循环依赖
   - 使用context参数传递依赖

3. **错误处理完善**
   - 设置任务超时
   - 定义重试策略
   - 记录失败日志

4. **性能优化**
   - 并行执行无依赖的任务
   - 缓存中间结果
   - 避免重复计算

### 自定义工具

```python
from crewai import Tool
from typing import Any

class SearchTool(Tool):
    name: str = "搜索工具"
    description: str = "搜索互联网信息"
    
    def _run(self, query: str) -> str:
        # 实现搜索逻辑
        return f"搜索结果：{query}"

class CodeTool(Tool):
    name: str = "代码工具"
    description: str = "执行代码"
    
    def _run(self, code: str) -> str:
        # 实现代码执行逻辑
        try:
            exec(code)
            return "执行成功"
        except Exception as e:
            return f"执行失败：{str(e)}"

# 为Agent添加工具
researcher = Agent(
    role="研究员",
    goal="收集信息",
    tools=[SearchTool(), CodeTool()],
    verbose=True
)
```

### 流程编排

```python
from crewai import Crew, Process, Task

# 定义复杂流程
def create_software_development_crew():
    # Agents
    pm = Agent(role="产品经理", goal="定义需求", verbose=True)
    designer = Agent(role="设计师", goal="设计界面", verbose=True)
    backend = Agent(role="后端开发", goal="实现后端", verbose=True)
    frontend = Agent(role="前端开发", goal="实现前端", verbose=True)
    tester = Agent(role="测试", goal="测试产品", verbose=True)
    
    # Tasks
    requirements = Task(
        description="收集和分析用户需求",
        agent=pm
    )
    
    design = Task(
        description="设计产品界面和交互",
        agent=designer,
        context=[requirements]
    )
    
    backend_dev = Task(
        description="开发后端API和数据库",
        agent=backend,
        context=[requirements]
    )
    
    frontend_dev = Task(
        description="开发前端界面",
        agent=frontend,
        context=[design, backend_dev]
    )
    
    testing = Task(
        description="测试所有功能",
        agent=tester,
        context=[backend_dev, frontend_dev]
    )
    
    # Crew
    crew = Crew(
        agents=[pm, designer, backend, frontend, tester],
        tasks=[requirements, design, backend_dev, frontend_dev, testing],
        process=Process.sequential,
        verbose=True
    )
    
    return crew

# 使用
crew = create_software_development_crew()
result = crew.kickoff()
```

### 结果聚合

```python
from crewai import Crew, Task
from typing import Dict, List

class AggregatorAgent(Agent):
    """聚合结果的Agent"""
    
    def __init__(self):
        super().__init__(
            role="聚合专家",
            goal="整合多个Agent的结果",
            backstory="你擅长整合和分析多个来源的信息"
        )
        self.results = {}
    
    def add_result(self, task_id: str, result: Any):
        self.results[task_id] = result
    
    def aggregate(self) -> str:
        prompt = f"""
请整合以下结果：

{self.results}

要求：
1. 提取关键信息
2. 消除重复
3. 逻辑清晰
4. 格式统一
"""
        return self.llm.predict(prompt)

# 使用
aggregator = AggregatorAgent()

crew = Crew(
    agents=[agent1, agent2, aggregator],
    tasks=[task1, task2],
    verbose=True
)

result = crew.kickoff()
aggregated = aggregator.aggregate()
```

## 8.4 【实战】内容创作团队

让我们构建一个完整的内容创作团队。

### 项目结构

```
content-creation-crew/
├── .env
├── main.py
├── agents.py
├── tasks.py
└── requirements.txt
```

### 完整代码

**agents.py**

```python
from crewai import Agent

def create_content_team():
    """创建内容创作团队"""
    
    # 研究员Agent
    researcher = Agent(
        role="研究员",
        goal="收集和整理主题相关信息",
        backstory="""你是一位专业的研究员，擅长：
- 信息检索
- 资料整理
- 事实核查
- 数据分析""",
        verbose=True,
        tools=[...]  # 添加搜索工具
    )
    
    # 作者Agent
    writer = Agent(
        role="作者",
        goal="撰写高质量内容",
        backstory="""你是一位经验丰富的作者，擅长：
- 内容创作
- 文字表达
- 结构组织
- 风格把控""",
        verbose=True
    )
    
    # 编辑Agent
    editor = Agent(
        role="编辑",
        goal="审阅和优化内容",
        backstory="""你是一位专业编辑，擅长：
- 内容审核
- 文字润色
- 错误纠正
- 质量把控""",
        verbose=True
    )
    
    # SEO专家Agent
    seo_expert = Agent(
        role="SEO专家",
        goal="优化内容搜索引擎排名",
        backstory="""你是一位SEO专家，擅长：
- 关键词优化
- 标题优化
- 结构优化
- 元数据优化""",
        verbose=True
    )
    
    return [researcher, writer, editor, seo_expert]
```

**tasks.py**

```python
from crewai import Task

def create_content_tasks(topic: str):
    """创建内容创作任务"""
    
    research_task = Task(
        description=f"""
研究主题：{topic}

要求：
1. 收集5-10个相关资料
2. 整理关键信息
3. 识别重要观点
4. 标注数据来源
""",
        expected_output="研究总结报告",
        output_file="research_report.md"
    )
    
    writing_task = Task(
        description=f"""
基于研究结果撰写文章：{topic}

要求：
1. 结构清晰（引言、正文、结论）
2. 内容准确
3. 语言流畅
4. 1000-1500字
""",
        expected_output="完整文章",
        output_file="article_draft.md",
        context=[research_task]
    )
    
    editing_task = Task(
        description="""
审阅并优化文章：

要求：
1. 检查语法和拼写
2. 优化表达
3. 调整结构
4. 确保准确性
""",
        expected_output="优化后的文章",
        output_file="article_edited.md",
        context=[writing_task]
    )
    
    seo_task = Task(
        description="""
优化文章SEO：

要求：
1. 优化标题
2. 添加关键词
3. 优化元描述
4. 调整结构标签
""",
        expected_output="SEO优化后的文章",
        output_file="article_final.md",
        context=[editing_task]
    )
    
    return [research_task, writing_task, editing_task, seo_task]
```

**main.py**

```python
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import create_content_team
from tasks import create_content_tasks

load_dotenv()

def main():
    topic = input("请输入创作主题：")
    
    # 创建团队
    agents = create_content_team()
    
    # 创建任务
    tasks = create_content_tasks(topic)
    
    # 创建Crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=True  # 启用记忆
    )
    
    print(f"\n开始创作关于'{topic}'的内容...\n")
    
    # 执行
    result = crew.kickoff()
    
    print(f"\n创作完成！\n")
    print(result)

if __name__ == "__main__":
    main()
```

## 本章小结

本章我们学习了：

- ✅ CrewAI框架的核心概念
- ✅ Agent角色定义和团队构建
- ✅ Task分配和协作模式
- ✅ 自定义工具和流程编排
- ✅ 构建了内容创作团队

## 下一章

下一章我们将学习其他主流Agent框架。

[第9章：其他主流框架 →](/frameworks/chapter9)
