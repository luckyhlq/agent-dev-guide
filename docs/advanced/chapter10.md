# 第10章：Agent架构设计

本章将深入讲解Agent的架构设计原则和模式，帮助你构建可扩展、可维护的Agent系统。

## 10.1 架构设计原则

**为什么Agent架构设计很重要？**

构建一个简单的Agent很容易，但构建一个可扩展、可维护、可靠的Agent系统需要良好的架构设计。

打个比方：
- **没有架构设计**：就像盖房子没有图纸，想到哪盖到哪，最后难以维护
- **有架构设计**：就像有完整的建筑图纸，结构清晰，易于扩展和维护

**Agent架构设计的核心原则**

**1. 单一职责原则（SRP）**

每个Agent只做一件事，并把它做好。

为什么重要？
- 降低复杂度：一个Agent只关注一个任务
- 提高可维护性：修改一个功能不会影响其他功能
- 增强可测试性：更容易编写测试

示例：
```
❌ 不好：一个Agent既做情感分析，又做翻译，还做摘要
✅ 好：情感分析Agent、翻译Agent、摘要Agent各司其职
```

**2. 开闭原则（OCP）**

对扩展开放，对修改关闭。

为什么重要？
- 新增功能时不需要修改现有代码
- 降低引入bug的风险
- 提高系统的稳定性

示例：
```python
# 通过插件机制扩展功能，而不是修改核心代码
agent.register_tool(new_tool)  # 扩展
# 而不是修改agent的源代码
```

**3. 依赖倒置原则（DIP）**

高层模块不应该依赖低层模块，两者都应该依赖抽象。

为什么重要？
- 降低模块间的耦合
- 方便替换实现
- 提高代码的可测试性

示例：
```python
# 不依赖具体的LLM实现
class Agent:
    def __init__(self, llm: LLMInterface):  # 依赖抽象接口
        self.llm = llm

# 可以随时替换LLM实现
agent = Agent(OpenAI())
agent = Agent(Anthropic())
```

**4. 接口隔离原则（ISP）**

不应该强迫Agent依赖它不使用的方法。

为什么重要？
- 避免接口臃肿
- 提高代码的可读性
- 降低耦合度

**常见架构模式**

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| 单体架构 | 所有功能在一个Agent中 | 简单应用 |
| 微服务架构 | 每个Agent独立部署 | 大型复杂系统 |
| 分层架构 | 按职责分层（表示、业务、数据） | 企业应用 |
| 管道架构 | 数据流经多个处理阶段 | 数据处理 |

### 单一职责原则

每个Agent应该只负责一个明确的职责：

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """执行Agent任务"""
        pass
    
    @abstractmethod
    def can_handle(self, input_data: Any) -> bool:
        """判断是否能处理该输入"""
        pass

class SentimentAnalysisAgent(BaseAgent):
    """情感分析Agent - 单一职责"""
    
    def __init__(self):
        super().__init__("sentiment_analyzer")
    
    def execute(self, text: str) -> dict:
        """只做情感分析"""
        return {
            "sentiment": "positive",
            "confidence": 0.95
        }
    
    def can_handle(self, input_data: Any) -> bool:
        return isinstance(input_data, str)

class TranslationAgent(BaseAgent):
    """翻译Agent - 单一职责"""
    
    def __init__(self, target_lang: str):
        super().__init__("translator")
        self.target_lang = target_lang
    
    def execute(self, text: str) -> str:
        """只做翻译"""
        return f"[{self.target_lang}] {text}"
    
    def can_handle(self, input_data: Any) -> bool:
        return isinstance(input_data, str)
```

### 模块化设计

将Agent系统分解为独立模块：

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Module:
    """模块基类"""
    name: str
    dependencies: List[str]
    
    def initialize(self):
        """初始化模块"""
        pass
    
    def process(self, data: Any) -> Any:
        """处理数据"""
        pass

class MemoryModule(Module):
    """记忆模块"""
    
    def __init__(self):
        super().__init__(name="memory", dependencies=[])
        self.short_term = []
        self.long_term = {}
    
    def process(self, data: Dict) -> Dict:
        if data.get("action") == "store":
            self.store(data["content"])
        elif data.get("action") == "retrieve":
            return self.retrieve(data["query"])
        return data
    
    def store(self, content: str):
        self.short_term.append(content)
    
    def retrieve(self, query: str) -> List[str]:
        return [m for m in self.short_term if query in m]

class ToolModule(Module):
    """工具模块"""
    
    def __init__(self):
        super().__init__(name="tools", dependencies=["memory"])
        self.tools = {}
    
    def register(self, name: str, func):
        self.tools[name] = func
    
    def process(self, data: Dict) -> Any:
        tool_name = data.get("tool")
        if tool_name in self.tools:
            return self.tools[tool_name](data.get("args", {}))
        return None

class AgentOrchestrator:
    """Agent编排器"""
    
    def __init__(self):
        self.modules: Dict[str, Module] = {}
        self.initialize_modules()
    
    def initialize_modules(self):
        """按依赖顺序初始化模块"""
        memory = MemoryModule()
        tools = ToolModule()
        
        memory.initialize()
        tools.initialize()
        
        self.modules["memory"] = memory
        self.modules["tools"] = tools
    
    def process(self, data: Dict) -> Any:
        """处理请求"""
        for module in self.modules.values():
            data = module.process(data)
        return data
```

### 可扩展性

设计可扩展的架构：

```python
from typing import Protocol, List, Any
from dataclasses import dataclass

class ToolProtocol(Protocol):
    """工具协议"""
    name: str
    description: str
    
    def execute(self, **kwargs) -> Any:
        ...

class MemoryProtocol(Protocol):
    """记忆协议"""
    def store(self, key: str, value: Any) -> None:
        ...
    
    def retrieve(self, key: str) -> Any:
        ...

class AgentCore:
    """Agent核心 - 可扩展"""
    
    def __init__(self):
        self.tools: List[ToolProtocol] = []
        self.memory: MemoryProtocol = None
        self.plugins: List[Any] = []
    
    def register_tool(self, tool: ToolProtocol):
        """注册工具"""
        self.tools.append(tool)
    
    def set_memory(self, memory: MemoryProtocol):
        """设置记忆系统"""
        self.memory = memory
    
    def add_plugin(self, plugin: Any):
        """添加插件"""
        self.plugins.append(plugin)
        plugin.initialize(self)
    
    def execute(self, task: str) -> Any:
        """执行任务"""
        # 可扩展的执行流程
        for plugin in self.plugins:
            if hasattr(plugin, 'before_execute'):
                task = plugin.before_execute(task)
        
        result = self._do_execute(task)
        
        for plugin in self.plugins:
            if hasattr(plugin, 'after_execute'):
                result = plugin.after_execute(result)
        
        return result
```

## 10.2 复杂Agent架构

**为什么需要复杂架构？**

简单的单Agent架构在处理复杂任务时会遇到瓶颈：
- 能力有限：一个Agent无法精通所有领域
- 性能瓶颈：所有任务串行执行
- 难以扩展：新增功能需要修改核心代码

复杂架构通过合理的组织方式解决这些问题。

**三种常见复杂架构**

**1. 层级式架构（Hierarchical）**

就像公司的组织架构：
- 顶层：管理者Agent，负责协调和决策
- 中层：专家Agent，负责特定领域
- 底层：执行Agent，负责具体任务

优点：
- 职责清晰，易于管理
- 可以并行执行任务
- 便于扩展和维护

适用场景：
- 大型复杂系统
- 需要多领域协作
- 任务可以分解

**2. 网状架构（Mesh）**

每个Agent可以与其他任何Agent通信：
- 没有固定的层级关系
- Agent之间平等协作
- 动态组建团队

优点：
- 灵活性高
- 无单点故障
- 自组织能力强

缺点：
- 协调复杂
- 可能出现冲突
- 难以预测行为

适用场景：
- 去中心化系统
- 需要高度灵活的场景
- 实验性项目

**3. 管道架构（Pipeline）**

数据按顺序流经多个处理阶段：
- 每个阶段专注一个任务
- 输出作为下一阶段的输入
- 类似工厂流水线

优点：
- 流程清晰
- 易于调试
- 可以并行处理不同数据

适用场景：
- 数据处理流程
- 内容生成管道
- 分析任务

**架构选择决策树**

```
任务复杂度如何？
├─ 简单任务 → 单Agent架构
├─ 中等复杂 → 管道架构
└─ 非常复杂 → 需要协调吗？
              ├─ 是 → 层级式架构
              └─ 否 → 网状架构
```

### 层级式架构

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    description: str
    status: str = "pending"
    result: Any = None

class BaseAgent:
    """基础Agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.children: List['BaseAgent'] = []
    
    def add_child(self, agent: 'BaseAgent'):
        self.children.append(agent)
    
    def execute(self, task: Task) -> Any:
        raise NotImplementedError

class ManagerAgent(BaseAgent):
    """管理者Agent"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.task_queue: List[Task] = []
    
    def execute(self, task: Task) -> Any:
        """分解任务并分配给子Agent"""
        subtasks = self.decompose(task)
        
        results = []
        for subtask in subtasks:
            for child in self.children:
                if child.can_handle(subtask):
                    result = child.execute(subtask)
                    results.append(result)
                    break
        
        return self.aggregate(results)
    
    def decompose(self, task: Task) -> List[Task]:
        """分解任务"""
        return [task]  # 简化实现
    
    def aggregate(self, results: List[Any]) -> Any:
        """聚合结果"""
        return results

class WorkerAgent(BaseAgent):
    """工作者Agent"""
    
    def __init__(self, name: str, skills: List[str]):
        super().__init__(name)
        self.skills = skills
    
    def can_handle(self, task: Task) -> bool:
        return any(skill in task.description for skill in self.skills)
    
    def execute(self, task: Task) -> Any:
        """执行具体任务"""
        task.status = "running"
        result = self.do_work(task)
        task.status = "completed"
        task.result = result
        return result
    
    def do_work(self, task: Task) -> Any:
        """具体工作"""
        return f"{self.name} completed {task.description}"

# 构建层级结构
manager = ManagerAgent("project_manager")

developer = WorkerAgent("developer", ["code", "develop", "implement"])
tester = WorkerAgent("tester", ["test", "verify", "check"])

manager.add_child(developer)
manager.add_child(tester)

# 执行任务
task = Task(id="1", description="develop feature")
result = manager.execute(task)
```

### 混合架构

```python
from enum import Enum
from typing import List, Dict, Any

class AgentType(Enum):
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HYBRID = "hybrid"

class HybridAgent:
    """混合架构Agent"""
    
    def __init__(self):
        self.type = AgentType.HYBRID
        self.reactive_layer = ReactiveLayer()
        self.deliberative_layer = DeliberativeLayer()
        self.coordinator = Coordinator()
    
    def process(self, input_data: Any) -> Any:
        """处理输入"""
        # 协调器决定使用哪层
        layer = self.coordinator.select_layer(input_data)
        
        if layer == "reactive":
            return self.reactive_layer.process(input_data)
        else:
            return self.deliberative_layer.process(input_data)

class ReactiveLayer:
    """反应层 - 快速响应"""
    
    def __init__(self):
        self.rules: Dict[str, Any] = {}
    
    def add_rule(self, condition: str, action: Any):
        self.rules[condition] = action
    
    def process(self, input_data: Any) -> Any:
        """快速匹配规则"""
        for condition, action in self.rules.items():
            if condition in str(input_data):
                return action()
        return None

class DeliberativeLayer:
    """慎思层 - 深度思考"""
    
    def __init__(self):
        self.planner = Planner()
        self.executor = Executor()
    
    def process(self, input_data: Any) -> Any:
        """规划并执行"""
        plan = self.planner.create_plan(input_data)
        result = self.executor.execute(plan)
        return result

class Coordinator:
    """协调器"""
    
    def select_layer(self, input_data: Any) -> str:
        """选择处理层"""
        # 简单规则：紧急情况用反应层
        if "urgent" in str(input_data).lower():
            return "reactive"
        return "deliberative"
```

### 分布式架构

```python
from typing import Dict, Any
import asyncio
from dataclasses import dataclass

@dataclass
class Message:
    sender: str
    receiver: str
    content: Any
    timestamp: float

class DistributedAgent:
    """分布式Agent"""
    
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.neighbors: Dict[str, 'DistributedAgent'] = {}
        self.running = False
    
    def connect(self, agent: 'DistributedAgent'):
        """连接其他Agent"""
        self.neighbors[agent.id] = agent
    
    async def send(self, receiver_id: str, content: Any):
        """发送消息"""
        if receiver_id in self.neighbors:
            message = Message(
                sender=self.id,
                receiver=receiver_id,
                content=content,
                timestamp=asyncio.get_event_loop().time()
            )
            await self.neighbors[receiver_id].message_queue.put(message)
    
    async def receive(self) -> Message:
        """接收消息"""
        return await self.message_queue.get()
    
    async def process_message(self, message: Message):
        """处理消息"""
        print(f"Agent {self.id} received from {message.sender}: {message.content}")
    
    async def run(self):
        """运行Agent"""
        self.running = True
        while self.running:
            message = await self.receive()
            await self.process_message(message)

class AgentNetwork:
    """Agent网络"""
    
    def __init__(self):
        self.agents: Dict[str, DistributedAgent] = {}
    
    def add_agent(self, agent: DistributedAgent):
        self.agents[agent.id] = agent
    
    def connect(self, agent1_id: str, agent2_id: str):
        """连接两个Agent"""
        if agent1_id in self.agents and agent2_id in self.agents:
            self.agents[agent1_id].connect(self.agents[agent2_id])
            self.agents[agent2_id].connect(self.agents[agent1_id])
    
    async def start_all(self):
        """启动所有Agent"""
        tasks = [agent.run() for agent in self.agents.values()]
        await asyncio.gather(*tasks)
```

## 10.3 状态管理

**为什么状态管理很重要？**

Agent在执行任务过程中需要维护各种状态：
- 对话历史：记住用户说过什么
- 任务进度：知道做到哪一步了
- 上下文信息：保存临时数据
- 用户偏好：记住用户的设置

没有良好的状态管理，Agent就像"失忆"了，无法提供连贯的服务。

**Agent状态的类型**

**1. 会话状态（Session State）**
- 生命周期：一次会话
- 内容：当前对话、临时变量
- 存储：内存
- 特点：会话结束后消失

**2. 持久化状态（Persistent State）**
- 生命周期：永久
- 内容：用户配置、历史记录
- 存储：数据库、文件
- 特点：跨会话保存

**3. 共享状态（Shared State）**
- 生命周期：多个Agent共享
- 内容：协作数据、公共知识
- 存储：共享存储
- 特点：多Agent可访问

**状态管理的挑战**

| 挑战 | 描述 | 解决方案 |
|------|------|---------|
| 并发访问 | 多个Agent同时修改状态 | 加锁、事务 |
| 状态同步 | 多个Agent状态不一致 | 同步机制、最终一致性 |
| 存储容量 | 状态数据越来越大 | 压缩、清理、归档 |
| 性能影响 | 频繁读写状态影响性能 | 缓存、批量操作 |

**状态管理最佳实践**

1. **状态最小化**
   - 只保存必要的状态
   - 能计算的不保存
   - 定期清理过期状态

2. **状态隔离**
   - 不同类型的状态分开存储
   - 避免状态之间的耦合
   - 使用命名空间

3. **状态持久化策略**
   - 关键状态立即持久化
   - 非关键状态批量持久化
   - 设置合理的持久化频率

4. **状态恢复机制**
   - 定期备份状态
   - 实现状态恢复功能
   - 处理状态损坏的情况

**状态管理工具**

- **Redis**：快速内存存储，适合会话状态
- **数据库**：持久化存储，适合用户数据
- **文件系统**：简单存储，适合配置文件
- **消息队列**：异步状态更新，适合分布式系统

### 会话状态

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime
import uuid

@dataclass
class Session:
    """会话状态"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, key: str, value: Any):
        """更新状态"""
        self.data[key] = value
        self.last_active = datetime.now()
    
    def add_history(self, entry: Dict[str, Any]):
        """添加历史"""
        self.history.append({
            **entry,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近历史"""
        return self.history[-n:]

class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, user_id: str = "") -> Session:
        """创建会话"""
        session = Session(user_id=user_id)
        self.sessions[session.id] = session
        return session
    
    def get_session(self, session_id: str) -> Session:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str):
        """结束会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_inactive(self, max_age_minutes: int = 30):
        """清理不活跃会话"""
        now = datetime.now()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            age = (now - session.last_active).total_seconds() / 60
            if age > max_age_minutes:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
```

### 持久化策略

```python
import json
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

class PersistenceBackend(ABC):
    """持久化后端"""
    
    @abstractmethod
    def save(self, key: str, value: Any):
        pass
    
    @abstractmethod
    def load(self, key: str) -> Any:
        pass
    
    @abstractmethod
    def delete(self, key: str):
        pass

class FilePersistence(PersistenceBackend):
    """文件持久化"""
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def save(self, key: str, value: Any):
        file_path = self.base_path / f"{key}.json"
        with open(file_path, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    
    def load(self, key: str) -> Any:
        file_path = self.base_path / f"{key}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def delete(self, key: str):
        file_path = self.base_path / f"{key}.json"
        if file_path.exists():
            file_path.unlink()

class RedisPersistence(PersistenceBackend):
    """Redis持久化"""
    
    def __init__(self, host: str = "localhost", port: int = 6379):
        import redis
        self.client = redis.Redis(host=host, port=port)
    
    def save(self, key: str, value: Any):
        self.client.set(key, json.dumps(value))
    
    def load(self, key: str) -> Any:
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def delete(self, key: str):
        self.client.delete(key)

class StateManager:
    """状态管理器"""
    
    def __init__(self, backend: PersistenceBackend):
        self.backend = backend
        self.cache: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any, persist: bool = False):
        """设置状态"""
        self.cache[key] = value
        if persist:
            self.backend.save(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        if key in self.cache:
            return self.cache[key]
        
        value = self.backend.load(key)
        if value is not None:
            self.cache[key] = value
            return value
        
        return default
    
    def delete(self, key: str):
        """删除状态"""
        if key in self.cache:
            del self.cache[key]
        self.backend.delete(key)
```

### 状态恢复

```python
from typing import Dict, Any, List
from dataclasses import dataclass
import copy

@dataclass
class StateSnapshot:
    """状态快照"""
    id: str
    timestamp: float
    state: Dict[str, Any]

class StateRecovery:
    """状态恢复"""
    
    def __init__(self, max_snapshots: int = 10):
        self.snapshots: List[StateSnapshot] = []
        self.max_snapshots = max_snapshots
    
    def create_snapshot(self, state: Dict[str, Any]) -> str:
        """创建快照"""
        import time
        import uuid
        
        snapshot = StateSnapshot(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            state=copy.deepcopy(state)
        )
        
        self.snapshots.append(snapshot)
        
        # 限制快照数量
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        return snapshot.id
    
    def restore(self, snapshot_id: str) -> Dict[str, Any]:
        """恢复到快照"""
        for snapshot in self.snapshots:
            if snapshot.id == snapshot_id:
                return copy.deepcopy(snapshot.state)
        return None
    
    def rollback(self, steps: int = 1) -> Dict[str, Any]:
        """回滚指定步数"""
        if len(self.snapshots) >= steps:
            snapshot = self.snapshots[-steps]
            return copy.deepcopy(snapshot.state)
        return None
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有快照"""
        return [
            {
                "id": s.id,
                "timestamp": s.timestamp,
                "state_keys": list(s.state.keys())
            }
            for s in self.snapshots
        ]
```

## 10.4 【实战】多技能个人助手

让我们构建一个完整的多技能个人助手。

### 项目结构

```
personal-assistant/
├── .env
├── main.py
├── core/
│   ├── agent.py
│   ├── router.py
│   └── state.py
├── skills/
│   ├── base.py
│   ├── calendar.py
│   ├── email.py
│   ├── task.py
│   └── search.py
└── requirements.txt
```

### 完整代码

**skills/base.py**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSkill(ABC):
    """技能基类"""
    
    name: str = ""
    description: str = ""
    keywords: list = []
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """判断是否能处理"""
        pass
    
    @abstractmethod
    def execute(self, query: str, context: Dict) -> Any:
        """执行技能"""
        pass
```

**skills/calendar.py**

```python
from .base import BaseSkill
from datetime import datetime, timedelta

class CalendarSkill(BaseSkill):
    name = "calendar"
    description = "日历和日程管理"
    keywords = ["日程", "会议", "提醒", "calendar", "schedule"]
    
    def can_handle(self, query: str) -> bool:
        return any(kw in query.lower() for kw in self.keywords)
    
    def execute(self, query: str, context: dict):
        if "添加" in query or "创建" in query:
            return self.add_event(query, context)
        elif "查看" in query or "列出" in query:
            return self.list_events(query, context)
        return "请说明具体操作"
    
    def add_event(self, query: str, context: dict):
        return f"已添加日程：{query}"
    
    def list_events(self, query: str, context: dict):
        return "今天的日程：\n- 10:00 团队会议\n- 14:00 项目评审"
```

**core/router.py**

```python
from typing import List, Dict, Any
from skills.base import BaseSkill

class SkillRouter:
    """技能路由器"""
    
    def __init__(self):
        self.skills: List[BaseSkill] = []
    
    def register(self, skill: BaseSkill):
        self.skills.append(skill)
    
    def route(self, query: str, context: Dict) -> tuple:
        """路由到合适的技能"""
        for skill in self.skills:
            if skill.can_handle(query):
                return skill, skill.execute(query, context)
        return None, None
```

**core/agent.py**

```python
from openai import OpenAI
from .router import SkillRouter
from .state import StateManager

class PersonalAssistant:
    """个人助手"""
    
    def __init__(self):
        self.llm = OpenAI()
        self.router = SkillRouter()
        self.state = StateManager()
        self.setup_skills()
    
    def setup_skills(self):
        """注册技能"""
        from skills.calendar import CalendarSkill
        from skills.task import TaskSkill
        from skills.search import SearchSkill
        
        self.router.register(CalendarSkill())
        self.router.register(TaskSkill())
        self.router.register(SearchSkill())
    
    def chat(self, user_input: str) -> str:
        """处理用户输入"""
        context = self.state.get_context()
        
        # 尝试路由到技能
        skill, result = self.router.route(user_input, context)
        
        if skill:
            return f"[{skill.name}] {result}"
        
        # 使用LLM处理
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个个人助手。"},
                {"role": "user", "content": user_input}
            ]
        )
        
        return response.choices[0].message.content
```

## 本章小结

本章我们学习了：

- ✅ Agent架构设计原则
- ✅ 层级式、混合式、分布式架构
- ✅ 状态管理和持久化
- ✅ 构建了多技能个人助手

## 下一章

下一章我们将学习多Agent协作。

[第11章：多Agent协作 →](/advanced/chapter11)
