# 第11章：多Agent协作

本章将深入讲解多Agent系统的协作模式、通信机制和优化策略。

## 11.1 多Agent协作模式

**为什么需要多Agent协作？**

单个Agent的能力有限，就像一个人无法完成所有工作。多Agent协作让多个专业Agent协同工作，完成复杂任务。

打个比方：
- **单Agent**：一个人既做设计、又写代码、还做测试
- **多Agent协作**：设计师做设计、程序员写代码、测试员做测试，各司其职

**多Agent协作的优势**

1. **专业化分工**
   - 每个Agent专注自己擅长的领域
   - 提高任务完成质量
   - 降低单个Agent的复杂度

2. **并行处理**
   - 多个Agent可以同时工作
   - 提高整体效率
   - 缩短任务完成时间

3. **可扩展性**
   - 新增功能只需添加新Agent
   - 不影响现有Agent
   - 易于维护和升级

**三种基本协作模式**

**1. 顺序协作（Sequential）**

任务按顺序依次传递：

```
Agent1 → Agent2 → Agent3 → 完成
```

特点：
- 流程清晰，易于理解
- 前一个Agent的输出是后一个Agent的输入
- 适合有明确步骤的任务

示例：数据处理流水线
```
数据收集 → 数据清洗 → 数据分析 → 报告生成
```

**2. 并行协作（Parallel）**

多个Agent同时处理任务：

```
       → Agent1 →
任务 → → Agent2 → 合并结果
       → Agent3 →
```

特点：
- 执行速度快
- 需要结果合并机制
- 适合可以独立处理的子任务

示例：多角度分析
```
       → 技术分析 →
问题 → → 市场分析 → 综合报告
       → 用户分析 →
```

**3. 层级协作（Hierarchical）**

管理者Agent协调执行者Agent：

```
        管理者Agent
       /     |     \
   Agent1  Agent2  Agent3
```

特点：
- 有明确的协调者
- 适合复杂任务
- 需要任务分配机制

示例：项目管理
```
        项目经理
       /    |    \
   开发   测试   运维
```

**协作模式选择指南**

| 任务特征 | 推荐模式 | 原因 |
|---------|---------|------|
| 有明确步骤 | 顺序协作 | 流程清晰 |
| 可独立处理 | 并行协作 | 效率高 |
| 需要协调 | 层级协作 | 易于管理 |
| 混合特征 | 混合模式 | 灵活应对 |

### 顺序协作

```python
from typing import List, Any, Dict
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    description: str
    input: Any
    output: Any = None

class SequentialCoordinator:
    """顺序协作协调器"""
    
    def __init__(self, agents: List):
        self.agents = agents
    
    def run(self, initial_input: Any) -> Any:
        """顺序执行"""
        current_input = initial_input
        
        for i, agent in enumerate(self.agents):
            print(f"Agent {i+1}/{len(self.agents)}: {agent.name}")
            current_input = agent.process(current_input)
        
        return current_input

# 示例：数据处理流水线
class DataCollector:
    name = "数据收集器"
    
    def process(self, query: str) -> dict:
        return {"raw_data": f"收集的数据：{query}"}

class DataProcessor:
    name = "数据处理器"
    
    def process(self, data: dict) -> dict:
        data["processed"] = True
        return data

class DataAnalyzer:
    name = "数据分析器"
    
    def process(self, data: dict) -> str:
        return f"分析结果：{data}"

# 使用
coordinator = SequentialCoordinator([
    DataCollector(),
    DataProcessor(),
    DataAnalyzer()
])

result = coordinator.run("AI研究")
print(result)
```

### 并行协作

```python
import asyncio
from typing import List, Any, Dict

class ParallelCoordinator:
    """并行协作协调器"""
    
    def __init__(self, agents: List):
        self.agents = agents
    
    async def run_async(self, input_data: Any) -> List[Any]:
        """并行执行"""
        tasks = [
            agent.process_async(input_data)
            for agent in self.agents
        ]
        results = await asyncio.gather(*tasks)
        return results
    
    def run(self, input_data: Any) -> List[Any]:
        """同步接口"""
        return asyncio.run(self.run_async(input_data))

# 示例：多源搜索
class SearchAgent:
    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source
    
    async def process_async(self, query: str) -> dict:
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return {
            "source": self.source,
            "results": f"来自{self.source}的结果"
        }

# 使用
agents = [
    SearchAgent("Google", "google"),
    SearchAgent("Bing", "bing"),
    SearchAgent("Baidu", "baidu")
]

coordinator = ParallelCoordinator(agents)
results = coordinator.run("AI Agent")
print(results)
```

### 层级协作

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AgentTask:
    description: str
    assigned_to: str = None
    result: Any = None

class ManagerAgent:
    """管理者Agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.workers: Dict[str, 'WorkerAgent'] = {}
    
    def add_worker(self, worker: 'WorkerAgent'):
        self.workers[worker.name] = worker
    
    def decompose(self, goal: str) -> List[AgentTask]:
        """分解任务"""
        return [
            AgentTask(description=f"子任务{i+1}")
            for i in range(3)
        ]
    
    def assign(self, task: AgentTask) -> str:
        """分配任务"""
        for name, worker in self.workers.items():
            if worker.can_handle(task):
                task.assigned_to = name
                return name
        return None
    
    def aggregate(self, results: List[Any]) -> Any:
        """聚合结果"""
        return {"aggregated": results}
    
    def run(self, goal: str) -> Any:
        """执行"""
        tasks = self.decompose(goal)
        results = []
        
        for task in tasks:
            worker_name = self.assign(task)
            if worker_name:
                result = self.workers[worker_name].execute(task)
                results.append(result)
        
        return self.aggregate(results)

class WorkerAgent:
    """工作者Agent"""
    
    def __init__(self, name: str, skills: List[str]):
        self.name = name
        self.skills = skills
    
    def can_handle(self, task: AgentTask) -> bool:
        return any(skill in task.description for skill in self.skills)
    
    def execute(self, task: AgentTask) -> Any:
        return f"{self.name}完成了{task.description}"
```

### 竞争协作

```python
from typing import List, Any, Tuple
from dataclasses import dataclass

@dataclass
class Solution:
    agent: str
    content: Any
    score: float = 0.0

class CompetitiveCoordinator:
    """竞争协作协调器"""
    
    def __init__(self, agents: List, evaluator):
        self.agents = agents
        self.evaluator = evaluator
    
    def run(self, problem: str) -> Solution:
        """竞争执行"""
        solutions = []
        
        for agent in self.agents:
            solution = agent.solve(problem)
            score = self.evaluator.evaluate(solution, problem)
            solutions.append(Solution(
                agent=agent.name,
                content=solution,
                score=score
            ))
        
        solutions.sort(key=lambda x: x.score, reverse=True)
        return solutions[0]

class SolutionEvaluator:
    """解决方案评估器"""
    
    def evaluate(self, solution: Any, problem: str) -> float:
        # 简单评估逻辑
        return len(str(solution)) / 100

# 示例：代码生成竞争
class CodeGenerator:
    def __init__(self, name: str, style: str):
        self.name = name
        self.style = style
    
    def solve(self, problem: str) -> str:
        return f"// {self.style} style solution\n{problem}"

# 使用
agents = [
    CodeGenerator("简洁派", "minimal"),
    CodeGenerator("详细派", "verbose"),
    CodeGenerator("平衡派", "balanced")
]

coordinator = CompetitiveCoordinator(agents, SolutionEvaluator())
best = coordinator.run("实现快速排序")
print(f"最佳方案来自：{best.agent}")
```

## 11.2 通信机制

**Agent之间如何通信？**

多Agent协作的基础是通信。Agent需要交换信息、协调行动、共享状态。良好的通信机制是协作成功的关键。

**通信的基本要素**

1. **发送者**：发起通信的Agent
2. **接收者**：接收消息的Agent
3. **消息内容**：要传递的信息
4. **通信协议**：消息的格式和规则

**常见的通信模式**

**1. 直接通信**

Agent之间直接发送消息：

```
Agent A → 消息 → Agent B
```

特点：
- 简单直接
- 延迟低
- 适合点对点通信

**2. 广播通信**

一个Agent向所有Agent发送消息：

```
Agent A → 广播 → [Agent B, Agent C, Agent D, ...]
```

特点：
- 覆盖面广
- 适合通知、公告
- 可能造成消息泛滥

**3. 订阅发布**

Agent订阅感兴趣的消息类型：

```
发布者 → 消息队列 → 订阅者1
                   → 订阅者2
                   → 订阅者3
```

特点：
- 解耦发送者和接收者
- 灵活高效
- 适合事件驱动系统

**消息类型**

| 消息类型 | 用途 | 示例 |
|---------|------|------|
| 请求 | 请求其他Agent执行任务 | "请分析这段代码" |
| 响应 | 回复请求 | "分析完成，发现3个问题" |
| 通知 | 告知状态变化 | "任务已完成" |
| 查询 | 查询信息 | "你的进度如何？" |
| 广播 | 向所有Agent发送消息 | "系统将在5分钟后重启" |

**通信协议设计原则**

1. **标准化**：使用统一的消息格式
2. **可扩展**：支持新增消息类型
3. **可靠性**：确保消息不丢失
4. **安全性**：防止消息被篡改

**通信中间件**

- **消息队列**：RabbitMQ、Kafka
- **RPC框架**：gRPC、Thrift
- **共享存储**：Redis、数据库
- **自定义协议**：WebSocket、HTTP

### 消息传递

```python
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime
import uuid

@dataclass
class Message:
    id: str
    sender: str
    receiver: str
    content: Any
    timestamp: datetime
    reply_to: Optional[str] = None
    
    @classmethod
    def create(cls, sender: str, receiver: str, content: Any, reply_to: str = None):
        return cls(
            id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.now(),
            reply_to=reply_to
        )

class MessageBus:
    """消息总线"""
    
    def __init__(self):
        self.queues: Dict[str, List[Message]] = {}
        self.handlers: Dict[str, callable] = {}
    
    def register(self, agent_id: str, handler: callable = None):
        """注册Agent"""
        self.queues[agent_id] = []
        if handler:
            self.handlers[agent_id] = handler
    
    def send(self, message: Message):
        """发送消息"""
        if message.receiver in self.queues:
            self.queues[message.receiver].append(message)
            
            if message.receiver in self.handlers:
                self.handlers[message.receiver](message)
    
    def broadcast(self, sender: str, content: Any, exclude: List[str] = None):
        """广播消息"""
        exclude = exclude or []
        
        for agent_id in self.queues:
            if agent_id != sender and agent_id not in exclude:
                message = Message.create(sender, agent_id, content)
                self.send(message)
    
    def receive(self, agent_id: str) -> List[Message]:
        """接收消息"""
        messages = self.queues.get(agent_id, [])
        self.queues[agent_id] = []
        return messages

# 使用
bus = MessageBus()

def agent_handler(message: Message):
    print(f"收到来自{message.sender}的消息：{message.content}")

bus.register("agent1", agent_handler)
bus.register("agent2")

msg = Message.create("agent1", "agent2", "你好！")
bus.send(msg)

bus.broadcast("agent1", "大家好！", exclude=["agent2"])
```

### 共享记忆

```python
from typing import Dict, Any, List
from threading import Lock

class SharedMemory:
    """共享记忆"""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.lock = Lock()
    
    def write(self, key: str, value: Any, agent: str):
        """写入"""
        with self.lock:
            self.data[key] = value
            self.history.append({
                "action": "write",
                "key": key,
                "agent": agent,
                "timestamp": datetime.now()
            })
    
    def read(self, key: str) -> Any:
        """读取"""
        with self.lock:
            return self.data.get(key)
    
    def update(self, key: str, updater: callable, agent: str):
        """更新"""
        with self.lock:
            if key in self.data:
                old_value = self.data[key]
                new_value = updater(old_value)
                self.data[key] = new_value
                self.history.append({
                    "action": "update",
                    "key": key,
                    "old": old_value,
                    "new": new_value,
                    "agent": agent
                })
    
    def subscribe(self, key: str, callback: callable):
        """订阅变更"""
        pass

class SharedMemoryAgent:
    """使用共享记忆的Agent"""
    
    def __init__(self, name: str, memory: SharedMemory):
        self.name = name
        self.memory = memory
    
    def share(self, key: str, value: Any):
        """共享数据"""
        self.memory.write(key, value, self.name)
    
    def access(self, key: str) -> Any:
        """访问共享数据"""
        return self.memory.read(key)
```

### 黑板模式

```python
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BlackboardEntry:
    key: str
    value: Any
    author: str
    timestamp: datetime
    tags: List[str]

class Blackboard:
    """黑板"""
    
    def __init__(self):
        self.entries: Dict[str, BlackboardEntry] = {}
        self.watchers: Dict[str, List[Callable]] = {}
    
    def post(self, key: str, value: Any, author: str, tags: List[str] = None):
        """发布信息"""
        entry = BlackboardEntry(
            key=key,
            value=value,
            author=author,
            timestamp=datetime.now(),
            tags=tags or []
        )
        
        self.entries[key] = entry
        
        # 通知观察者
        if key in self.watchers:
            for watcher in self.watchers[key]:
                watcher(entry)
    
    def read(self, key: str) -> Any:
        """读取信息"""
        entry = self.entries.get(key)
        return entry.value if entry else None
    
    def watch(self, key: str, callback: Callable):
        """观察变化"""
        if key not in self.watchers:
            self.watchers[key] = []
        self.watchers[key].append(callback)
    
    def search(self, tags: List[str]) -> List[BlackboardEntry]:
        """按标签搜索"""
        return [
            entry for entry in self.entries.values()
            if any(tag in entry.tags for tag in tags)
        ]

class BlackboardAgent:
    """使用黑板的Agent"""
    
    def __init__(self, name: str, blackboard: Blackboard):
        self.name = name
        self.blackboard = blackboard
    
    def contribute(self, key: str, value: Any, tags: List[str] = None):
        """贡献知识"""
        self.blackboard.post(key, value, self.name, tags)
    
    def observe(self, key: str, callback: Callable):
        """观察黑板"""
        self.blackboard.watch(key, callback)
```

## 11.3 协作优化

**为什么需要优化协作？**

多Agent协作虽然强大，但也面临挑战：
- 通信开销：Agent之间频繁通信消耗资源
- 协调成本：需要协调多个Agent的行动
- 冲突处理：Agent之间可能出现冲突
- 性能瓶颈：某些Agent可能成为瓶颈

优化协作可以提高效率、降低成本、提升质量。

**协作优化的关键方向**

**1. 任务分解优化**

将大任务分解为小任务，分配合适的Agent：

好的任务分解应该：
- 粒度适中：不要太粗也不要太细
- 相互独立：减少任务之间的依赖
- 职责明确：每个任务有明确的输出
- 可并行：尽可能支持并行执行

示例：
```
❌ 不好：一个大任务"开发完整功能"
✅ 好：分解为"需求分析"、"设计"、"编码"、"测试"、"部署"
```

**2. 负载均衡**

让每个Agent的工作量大致相等：

策略：
- 动态分配：根据Agent当前负载分配任务
- 能力匹配：根据Agent能力分配任务
- 优先级调度：优先处理重要任务

**3. 通信优化**

减少不必要的通信：

方法：
- 批量通信：合并多条消息一起发送
- 缓存结果：避免重复请求相同信息
- 按需通信：只在必要时通信
- 压缩数据：减少消息大小

**4. 冲突解决**

处理Agent之间的冲突：

冲突类型：
- 资源冲突：多个Agent竞争同一资源
- 目标冲突：Agent的目标不一致
- 结果冲突：Agent的结果相互矛盾

解决策略：
- 优先级机制：高优先级Agent优先
- 投票机制：少数服从多数
- 协商机制：Agent协商达成一致
- 仲裁机制：由管理者Agent裁决

**性能监控指标**

| 指标 | 说明 | 目标 |
|------|------|------|
| 任务完成时间 | 从开始到结束的总时间 | 越短越好 |
| 通信次数 | Agent之间的通信次数 | 越少越好 |
| 资源利用率 | CPU、内存等资源的使用率 | 适中 |
| 错误率 | 任务失败的比例 | 越低越好 |

### 任务分解

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SubTask:
    id: str
    description: str
    dependencies: List[str]
    priority: int
    assigned_to: str = None

class TaskDecomposer:
    """任务分解器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def decompose(self, task: str) -> List[SubTask]:
        """分解任务"""
        prompt = f"""
请将以下任务分解为子任务：

任务：{task}

要求：
1. 每个子任务清晰明确
2. 标注依赖关系
3. 设置优先级（1-5）

以JSON格式输出：
[
  {{"id": "1", "description": "...", "dependencies": [], "priority": 5}},
  ...
]
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            data = json.loads(response.choices[0].message.content)
            return [SubTask(**item) for item in data]
        except:
            return [SubTask(id="1", description=task, dependencies=[], priority=5)]

class TaskScheduler:
    """任务调度器"""
    
    def schedule(self, tasks: List[SubTask]) -> List[SubTask]:
        """调度任务"""
        # 拓扑排序
        ordered = []
        visited = set()
        
        def visit(task: SubTask):
            if task.id in visited:
                return
            visited.add(task.id)
            
            for dep_id in task.dependencies:
                for t in tasks:
                    if t.id == dep_id:
                        visit(t)
            
            ordered.append(task)
        
        for task in sorted(tasks, key=lambda x: -x.priority):
            visit(task)
        
        return ordered
```

### 结果融合

```python
from typing import List, Any, Dict
from dataclasses import dataclass

@dataclass
class AgentResult:
    agent: str
    result: Any
    confidence: float
    metadata: Dict

class ResultFusion:
    """结果融合器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def voting(self, results: List[AgentResult]) -> Any:
        """投票融合"""
        from collections import Counter
        
        values = [r.result for r in results]
        counter = Counter(values)
        return counter.most_common(1)[0][0]
    
    def weighted_average(self, results: List[AgentResult]) -> float:
        """加权平均"""
        total_weight = sum(r.confidence for r in results)
        weighted_sum = sum(
            r.result * r.confidence 
            for r in results 
            if isinstance(r.result, (int, float))
        )
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def llm_fusion(self, results: List[AgentResult], question: str) -> str:
        """LLM融合"""
        prompt = f"""
问题：{question}

多个Agent的回答：
{chr(10).join(f'{r.agent}: {r.result} (置信度: {r.confidence})' for r in results)}

请综合以上回答，给出最佳答案。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def ensemble(self, results: List[AgentResult], method: str = "voting") -> Any:
        """集成融合"""
        if method == "voting":
            return self.voting(results)
        elif method == "weighted":
            return self.weighted_average(results)
        elif method == "llm":
            return self.llm_fusion(results, "")
        return results[0].result if results else None
```

### 冲突解决

```python
from typing import List, Any, Dict
from dataclasses import dataclass

@dataclass
class Conflict:
    agents: List[str]
    issue: str
    positions: Dict[str, Any]

class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def detect(self, results: List[AgentResult]) -> List[Conflict]:
        """检测冲突"""
        conflicts = []
        
        # 检测结果不一致
        unique_results = set(str(r.result) for r in results)
        
        if len(unique_results) > 1:
            conflicts.append(Conflict(
                agents=[r.agent for r in results],
                issue="结果不一致",
                positions={r.agent: r.result for r in results}
            ))
        
        return conflicts
    
    def resolve(self, conflict: Conflict) -> Any:
        """解决冲突"""
        prompt = f"""
多个Agent对同一问题给出了不同答案：

{chr(10).join(f'{agent}: {result}' for agent, result in conflict.positions.items())}

请分析并给出最佳答案，说明理由。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def negotiate(self, agents: List, issue: str) -> Any:
        """协商解决"""
        positions = {}
        
        for agent in agents:
            positions[agent.name] = agent.get_position(issue)
        
        conflict = Conflict(
            agents=[a.name for a in agents],
            issue=issue,
            positions=positions
        )
        
        return self.resolve(conflict)
```

## 11.4 【实战】软件开发团队

让我们构建一个模拟软件开发团队的多Agent系统。

### 项目结构

```
dev-team-agents/
├── .env
├── main.py
├── agents/
│   ├── pm.py
│   ├── developer.py
│   ├── tester.py
│   └── reviewer.py
├── collaboration/
│   ├── board.py
│   └── workflow.py
└── requirements.txt
```

### 完整代码

**collaboration/board.py**

```python
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Task:
    id: str
    title: str
    description: str
    status: str = "todo"
    assignee: str = None
    created_at: datetime = None
    updated_at: datetime = None

class TaskBoard:
    """任务看板"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.columns = ["todo", "in_progress", "review", "done"]
    
    def add_task(self, id: str, title: str, description: str):
        task = Task(
            id=id,
            title=title,
            description=description,
            created_at=datetime.now()
        )
        self.tasks[id] = task
        return task
    
    def move_task(self, task_id: str, new_status: str):
        if task_id in self.tasks and new_status in self.columns:
            self.tasks[task_id].status = new_status
            self.tasks[task_id].updated_at = datetime.now()
    
    def assign_task(self, task_id: str, assignee: str):
        if task_id in self.tasks:
            self.tasks[task_id].assignee = assignee
    
    def get_tasks_by_status(self, status: str) -> List[Task]:
        return [t for t in self.tasks.values() if t.status == status]
    
    def get_agent_tasks(self, agent: str) -> List[Task]:
        return [t for t in self.tasks.values() if t.assignee == agent]
```

**agents/pm.py**

```python
from openai import OpenAI

class ProductManager:
    """产品经理Agent"""
    
    def __init__(self, board):
        self.name = "PM"
        self.board = board
        self.llm = OpenAI()
    
    def create_feature(self, description: str):
        """创建功能需求"""
        prompt = f"""
作为产品经理，请将以下需求分解为具体任务：

需求：{description}

输出JSON格式的任务列表：
[
  {{"title": "...", "description": "..."}},
  ...
]
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        tasks = json.loads(response.choices[0].message.content)
        
        for i, task in enumerate(tasks):
            self.board.add_task(
                id=f"task_{i+1}",
                title=task["title"],
                description=task["description"]
            )
        
        return tasks
    
    def prioritize(self):
        """优先级排序"""
        todo_tasks = self.board.get_tasks_by_status("todo")
        # 按优先级排序逻辑
        return todo_tasks
```

**main.py**

```python
from dotenv import load_dotenv
from collaboration.board import TaskBoard
from agents.pm import ProductManager

load_dotenv()

def main():
    board = TaskBoard()
    pm = ProductManager(board)
    
    print("=" * 60)
    print("软件开发团队模拟")
    print("=" * 60)
    
    # PM创建需求
    feature = input("\n请输入功能需求：")
    tasks = pm.create_feature(feature)
    
    print(f"\n已创建 {len(tasks)} 个任务：")
    for task in tasks:
        print(f"  - {task['title']}")
    
    print("\n任务看板：")
    for status in board.columns:
        tasks = board.get_tasks_by_status(status)
        print(f"  {status}: {len(tasks)} 个任务")

if __name__ == "__main__":
    main()
```

## 本章小结

本章我们学习了：

- ✅ 四种多Agent协作模式
- ✅ 消息传递、共享记忆、黑板模式
- ✅ 任务分解、结果融合、冲突解决
- ✅ 构建了软件开发团队模拟系统

## 下一章

下一章我们将学习生产部署。

[第12章：生产部署 →](/advanced/chapter12)
