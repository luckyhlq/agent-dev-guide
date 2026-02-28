# 第15章：自主Agent与未来趋势

本章将探讨自主Agent的原理、前沿研究方向和行业应用案例。

## 15.1 自主Agent原理

**什么是自主Agent？**

自主Agent是能够独立完成任务的AI系统，它不需要详细的指令，只需要给定目标，就能自主规划、决策和执行。

打个比方：
- **普通Agent**：像执行命令的士兵，需要详细指令
- **自主Agent**：像独立作战的特种兵，只需知道目标，自己决定怎么做

**自主Agent的核心特征**

**1. 自主决策**

能够独立做出决策，不需要人工干预：

决策能力：
- 理解目标：理解要达成什么目标
- 分析环境：评估当前状态和限制
- 选择策略：从多种方案中选择最优
- 执行行动：实施决策
- 评估结果：判断是否达成目标

**2. 自我规划**

能够自主规划任务执行步骤：

规划能力：
- 任务分解：将大目标分解为小任务
- 优先级排序：确定任务执行顺序
- 资源分配：合理分配时间和资源
- 动态调整：根据情况调整计划

**3. 自我学习**

能够从经验中学习和改进：

学习能力：
- 经验积累：记录成功和失败的经验
- 策略优化：改进决策策略
- 知识更新：更新知识库
- 能力扩展：学习新技能

**4. 自我反思**

能够评估自己的表现并改进：

反思能力：
- 结果评估：评估任务完成质量
- 错误分析：分析失败原因
- 改进计划：制定改进方案
- 持续优化：不断优化表现

**自主Agent vs 普通Agent**

| 特性 | 普通Agent | 自主Agent |
|------|----------|-----------|
| 指令方式 | 需要详细指令 | 只需目标 |
| 决策能力 | 有限 | 强 |
| 适应性 | 低 | 高 |
| 学习能力 | 弱 | 强 |
| 复杂度 | 低 | 高 |
| 成本 | 低 | 高 |

**自主Agent的挑战**

1. **可靠性**：自主决策可能出错
2. **可控性**：难以预测行为
3. **安全性**：可能执行危险操作
4. **成本**：需要大量LLM调用
5. **调试**：难以排查问题

**自主Agent的应用场景**

- 自动化研究：自动搜集资料、生成报告
- 智能助理：主动帮助用户完成任务
- 游戏AI：自主玩游戏
- 科学发现：自动进行实验和分析

### 自主决策机制

```python
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class DecisionType(Enum):
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    LEARNING = "learning"

@dataclass
class Decision:
    action: str
    reasoning: str
    confidence: float
    alternatives: List[str]

class AutonomousDecisionMaker:
    """自主决策器"""
    
    def __init__(self):
        self.goals: List[str] = []
        self.constraints: List[str] = []
        self.decision_history: List[Decision] = []
    
    def set_goals(self, goals: List[str]):
        """设置目标"""
        self.goals = goals
    
    def add_constraint(self, constraint: str):
        """添加约束"""
        self.constraints.append(constraint)
    
    def evaluate_options(
        self,
        options: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Decision:
        """评估选项并做出决策"""
        scored_options = []
        
        for option in options:
            score = self._calculate_score(option, context)
            scored_options.append((option, score))
        
        scored_options.sort(key=lambda x: x[1], reverse=True)
        
        best_option, best_score = scored_options[0]
        
        decision = Decision(
            action=best_option["action"],
            reasoning=self._generate_reasoning(best_option, context),
            confidence=best_score,
            alternatives=[opt["action"] for opt, _ in scored_options[1:3]]
        )
        
        self.decision_history.append(decision)
        
        return decision
    
    def _calculate_score(
        self,
        option: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """计算选项得分"""
        score = 0.0
        
        # 目标对齐度
        for goal in self.goals:
            if goal in option.get("description", ""):
                score += 0.3
        
        # 约束满足度
        for constraint in self.constraints:
            if self._check_constraint(option, constraint):
                score += 0.2
        
        # 上下文相关性
        if context.get("urgency", False):
            score += option.get("speed", 0) * 0.1
        
        return min(score, 1.0)
    
    def _check_constraint(
        self,
        option: Dict[str, Any],
        constraint: str
    ) -> bool:
        """检查约束"""
        return True
    
    def _generate_reasoning(
        self,
        option: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """生成推理"""
        return f"选择{option['action']}因为它最符合当前目标"
```

### 自我反思与改进

```python
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Reflection:
    timestamp: datetime
    action: str
    outcome: str
    lessons: List[str]
    improvements: List[str]

class SelfReflectionEngine:
    """自我反思引擎"""
    
    def __init__(self):
        self.reflections: List[Reflection] = []
        self.learning_rate = 0.1
    
    def reflect(
        self,
        action: str,
        outcome: str,
        success: bool
    ) -> Reflection:
        """反思行动结果"""
        lessons = self._extract_lessons(action, outcome, success)
        improvements = self._suggest_improvements(action, outcome, success)
        
        reflection = Reflection(
            timestamp=datetime.now(),
            action=action,
            outcome=outcome,
            lessons=lessons,
            improvements=improvements
        )
        
        self.reflections.append(reflection)
        
        return reflection
    
    def _extract_lessons(
        self,
        action: str,
        outcome: str,
        success: bool
    ) -> List[str]:
        """提取教训"""
        lessons = []
        
        if success:
            lessons.append(f"行动'{action}'是成功的")
        else:
            lessons.append(f"行动'{action}'需要改进")
        
        return lessons
    
    def _suggest_improvements(
        self,
        action: str,
        outcome: str,
        success: bool
    ) -> List[str]:
        """建议改进"""
        improvements = []
        
        if not success:
            improvements.extend([
                "考虑替代方案",
                "增加信息收集",
                "改进决策逻辑"
            ])
        
        return improvements
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习总结"""
        total = len(self.reflections)
        successful = sum(
            1 for r in self.reflections
            if "成功" in str(r.lessons)
        )
        
        return {
            "total_reflections": total,
            "successful_actions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "common_improvements": self._get_common_improvements()
        }
    
    def _get_common_improvements(self) -> List[str]:
        """获取常见改进建议"""
        from collections import Counter
        
        all_improvements = []
        for r in self.reflections:
            all_improvements.extend(r.improvements)
        
        counter = Counter(all_improvements)
        return [item for item, _ in counter.most_common(5)]
```

### 长期目标管理

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

@dataclass
class Goal:
    id: str
    description: str
    priority: int
    deadline: datetime
    status: GoalStatus
    sub_goals: List['Goal']
    progress: float

class GoalManager:
    """目标管理器"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
    
    def add_goal(
        self,
        goal_id: str,
        description: str,
        priority: int = 1,
        deadline: datetime = None
    ) -> Goal:
        """添加目标"""
        goal = Goal(
            id=goal_id,
            description=description,
            priority=priority,
            deadline=deadline or datetime.now() + timedelta(days=7),
            status=GoalStatus.PENDING,
            sub_goals=[],
            progress=0.0
        )
        
        self.goals[goal_id] = goal
        return goal
    
    def decompose_goal(
        self,
        goal_id: str,
        sub_goals: List[Dict[str, Any]]
    ):
        """分解目标"""
        if goal_id not in self.goals:
            return
        
        parent = self.goals[goal_id]
        
        for i, sub in enumerate(sub_goals):
            sub_goal = Goal(
                id=f"{goal_id}_sub_{i}",
                description=sub["description"],
                priority=sub.get("priority", parent.priority),
                deadline=sub.get("deadline", parent.deadline),
                status=GoalStatus.PENDING,
                sub_goals=[],
                progress=0.0
            )
            
            parent.sub_goals.append(sub_goal)
            self.goals[sub_goal.id] = sub_goal
    
    def update_progress(
        self,
        goal_id: str,
        progress: float
    ):
        """更新进度"""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.progress = progress
        
        if progress >= 1.0:
            goal.status = GoalStatus.COMPLETED
        elif progress > 0:
            goal.status = GoalStatus.IN_PROGRESS
    
    def get_next_action(self) -> Dict[str, Any]:
        """获取下一步行动"""
        pending_goals = [
            g for g in self.goals.values()
            if g.status == GoalStatus.PENDING
        ]
        
        if not pending_goals:
            return {"action": "no_pending_goals"}
        
        # 按优先级和截止日期排序
        pending_goals.sort(
            key=lambda g: (g.priority, g.deadline)
        )
        
        next_goal = pending_goals[0]
        
        return {
            "goal_id": next_goal.id,
            "description": next_goal.description,
            "priority": next_goal.priority,
            "deadline": next_goal.deadline.isoformat()
        }
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        total = len(self.goals)
        completed = sum(
            1 for g in self.goals.values()
            if g.status == GoalStatus.COMPLETED
        )
        in_progress = sum(
            1 for g in self.goals.values()
            if g.status == GoalStatus.IN_PROGRESS
        )
        
        return {
            "total_goals": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": total - completed - in_progress,
            "completion_rate": completed / total if total > 0 else 0
        }
```

## 15.2 前沿研究方向

**Agent技术的未来**

Agent技术正在快速发展，许多前沿研究方向正在探索中。

**主要研究方向**

**1. Agent学习与进化**

让Agent能够持续学习和进化：

研究方向：
- 强化学习：通过奖励机制学习最优策略
- 在线学习：实时从新数据中学习
- 迁移学习：将学到的知识迁移到新任务
- 元学习：学会如何学习

挑战：
- 样本效率：需要多少样本才能学会
- 灾难性遗忘：学习新知识会忘记旧知识
- 安全性：学习过程可能学到错误的东西

**2. 多模态Agent**

能处理多种类型数据的Agent：

能力：
- 文本理解：理解和生成文本
- 图像处理：理解和生成图像
- 语音交互：语音识别和合成
- 视频理解：理解视频内容

应用：
- 多模态助手：能看图、听声音、说话
- 内容创作：生成图文并茂的内容
- 智能监控：分析视频流

**3. 具身智能**

有物理载体的Agent：

特点：
- 机器人：在物理世界中行动
- 传感器：感知环境
- 执行器：执行物理操作
- 实时性：需要实时响应

应用：
- 工业机器人：自动化生产
- 服务机器人：餐厅、酒店服务
- 家庭机器人：家务助手
- 自动驾驶：无人驾驶汽车

**4. 群体智能**

多个Agent协作涌现智能：

特点：
- 分布式：没有中心控制
- 自组织：自动组织协作
- 鲁棒性：个体失败不影响整体
- 可扩展：容易增加新个体

应用：
- 无人机群：协同完成任务
- 智能交通：多车协同
- 分布式计算：大规模计算任务

**未来趋势**

| 方向 | 当前状态 | 未来展望 |
|------|---------|---------|
| 自主性 | 半自主 | 全自主 |
| 学习能力 | 有限 | 持续学习 |
| 多模态 | 文本为主 | 全模态 |
| 具身智能 | 实验阶段 | 广泛应用 |
| 群体智能 | 简单协作 | 复杂协作 |

### Agent学习与进化

```python
from typing import Dict, Any, List
from dataclasses import dataclass
import random

@dataclass
class Experience:
    state: Any
    action: str
    reward: float
    next_state: Any

class LearningAgent:
    """学习型Agent"""
    
    def __init__(self):
        self.experiences: List[Experience] = []
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
    
    def choose_action(self, state: str, actions: List[str]) -> str:
        """选择行动"""
        # 探索
        if random.random() < self.exploration_rate:
            return random.choice(actions)
        
        # 利用
        if state not in self.q_table:
            return random.choice(actions)
        
        q_values = self.q_table[state]
        return max(actions, key=lambda a: q_values.get(a, 0))
    
    def learn(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str
    ):
        """学习"""
        # 记录经验
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state
        )
        self.experiences.append(experience)
        
        # 更新Q值
        if state not in self.q_table:
            self.q_table[state] = {}
        
        current_q = self.q_table[state].get(action, 0)
        
        max_next_q = max(
            self.q_table.get(next_state, {}).values(),
            default=0
        )
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def get_policy(self) -> Dict[str, str]:
        """获取策略"""
        policy = {}
        
        for state, actions in self.q_table.items():
            best_action = max(actions.items(), key=lambda x: x[1])
            policy[state] = best_action[0]
        
        return policy
```

### 多模态Agent

```python
from typing import Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

@dataclass
class MultiModalInput:
    modality: Modality
    content: Any
    metadata: Dict[str, Any] = None

class MultiModalAgent:
    """多模态Agent"""
    
    def __init__(self):
        self.processors = {
            Modality.TEXT: self._process_text,
            Modality.IMAGE: self._process_image,
            Modality.AUDIO: self._process_audio,
            Modality.VIDEO: self._process_video
        }
    
    def process(self, inputs: List[MultiModalInput]) -> Dict[str, Any]:
        """处理多模态输入"""
        results = []
        
        for input_data in inputs:
            processor = self.processors.get(input_data.modality)
            if processor:
                result = processor(input_data)
                results.append(result)
        
        # 融合结果
        fused_result = self._fuse_results(results)
        
        return fused_result
    
    def _process_text(self, input_data: MultiModalInput) -> Dict:
        """处理文本"""
        return {
            "modality": "text",
            "content": input_data.content,
            "analysis": "文本分析结果"
        }
    
    def _process_image(self, input_data: MultiModalInput) -> Dict:
        """处理图像"""
        return {
            "modality": "image",
            "content": "图像描述",
            "analysis": "图像分析结果"
        }
    
    def _process_audio(self, input_data: MultiModalInput) -> Dict:
        """处理音频"""
        return {
            "modality": "audio",
            "content": "音频转录",
            "analysis": "音频分析结果"
        }
    
    def _process_video(self, input_data: MultiModalInput) -> Dict:
        """处理视频"""
        return {
            "modality": "video",
            "content": "视频描述",
            "analysis": "视频分析结果"
        }
    
    def _fuse_results(self, results: List[Dict]) -> Dict[str, Any]:
        """融合结果"""
        return {
            "modalities_processed": len(results),
            "results": results,
            "summary": "多模态融合结果"
        }
```

### 具身智能

```python
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    MOVE = "move"
    GRAB = "grab"
    RELEASE = "release"
    LOOK = "look"
    SPEAK = "speak"

@dataclass
class Action:
    type: ActionType
    parameters: Dict[str, Any]
    timestamp: float

@dataclass
class Observation:
    visual: Any
    audio: Any
    tactile: Any
    timestamp: float

class EmbodiedAgent:
    """具身Agent"""
    
    def __init__(self):
        self.position: Tuple[float, float, float] = (0, 0, 0)
        self.holding: Any = None
        self.action_history: List[Action] = []
        self.observation_history: List[Observation] = []
    
    def perceive(self, observation: Observation):
        """感知环境"""
        self.observation_history.append(observation)
    
    def decide(self, goal: str) -> Action:
        """决策"""
        # 简化的决策逻辑
        if "移动" in goal:
            return Action(
                type=ActionType.MOVE,
                parameters={"direction": "forward", "distance": 1.0},
                timestamp=time.time()
            )
        elif "抓取" in goal:
            return Action(
                type=ActionType.GRAB,
                parameters={"target": "object"},
                timestamp=time.time()
            )
        else:
            return Action(
                type=ActionType.LOOK,
                parameters={},
                timestamp=time.time()
            )
    
    def act(self, action: Action) -> Dict[str, Any]:
        """执行动作"""
        self.action_history.append(action)
        
        result = {"success": True, "action": action.type.value}
        
        if action.type == ActionType.MOVE:
            direction = action.parameters.get("direction", "forward")
            distance = action.parameters.get("distance", 1.0)
            self._move(direction, distance)
            result["new_position"] = self.position
        
        elif action.type == ActionType.GRAB:
            target = action.parameters.get("target")
            self.holding = target
            result["holding"] = self.holding
        
        elif action.type == ActionType.RELEASE:
            self.holding = None
            result["holding"] = None
        
        return result
    
    def _move(self, direction: str, distance: float):
        """移动"""
        x, y, z = self.position
        
        if direction == "forward":
            self.position = (x + distance, y, z)
        elif direction == "backward":
            self.position = (x - distance, y, z)
        elif direction == "left":
            self.position = (x, y - distance, z)
        elif direction == "right":
            self.position = (x, y + distance, z)
```

## 15.3 行业应用案例

**Agent在各行业的应用**

Agent技术正在各行各业落地应用，改变着传统的工作方式。

**主要应用领域**

**1. 智能客服**

用Agent替代或辅助人工客服：

应用场景：
- 咨询解答：回答常见问题
- 工单处理：自动创建和处理工单
- 投诉处理：初步处理用户投诉
- 销售引导：引导用户购买

优势：
- 24/7服务：全天候在线
- 快速响应：秒级响应
- 成本降低：减少人力成本
- 质量稳定：服务质量一致

挑战：
- 复杂问题：难以处理复杂问题
- 情感理解：难以理解用户情绪
- 个性化：难以提供个性化服务

**2. 内容创作**

用Agent辅助或自动生成内容：

应用场景：
- 文章写作：新闻、博客、报告
- 营销文案：广告、宣传语
- 社交媒体：微博、朋友圈内容
- 视频脚本：短视频、广告脚本

优势：
- 效率高：快速生成内容
- 规模化：批量生产内容
- 成本低：降低创作成本
- 创意多：提供多种创意

挑战：
- 原创性：内容可能缺乏原创性
- 质量控制：需要人工审核
- 版权问题：可能涉及版权

**3. 数据分析**

用Agent进行数据分析和洞察：

应用场景：
- 商业智能：销售、运营数据分析
- 金融分析：股票、基金分析
- 市场研究：市场趋势分析
- 用户研究：用户行为分析

优势：
- 自动化：自动完成分析流程
- 洞察力：发现隐藏的模式
- 可视化：生成图表和报告
- 实时性：实时分析数据

挑战：
- 数据质量：依赖数据质量
- 解释性：难以解释分析过程
- 领域知识：需要领域专业知识

**4. 软件开发**

用Agent辅助软件开发：

应用场景：
- 代码生成：自动生成代码
- 代码审查：检查代码质量
- Bug修复：自动修复Bug
- 文档生成：生成技术文档

优势：
- 效率提升：加快开发速度
- 质量提升：减少错误
- 学习辅助：帮助新手学习
- 重复工作：自动化重复任务

挑战：
- 复杂逻辑：难以处理复杂逻辑
- 代码质量：生成的代码质量不稳定
- 安全性：可能引入安全漏洞

**行业应用成熟度**

| 行业 | 成熟度 | 主要应用 | 挑战 |
|------|--------|---------|------|
| 客服 | 高 | 智能问答、工单处理 | 复杂问题处理 |
| 内容 | 高 | 文案生成、内容创作 | 原创性、质量 |
| 金融 | 中 | 风险分析、投资建议 | 合规、准确性 |
| 医疗 | 低 | 辅助诊断、健康咨询 | 准确性、法规 |
| 教育 | 中 | 个性化教学、作业批改 | 教学质量 |

### 智能客服

```python
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class Intent(Enum):
    PRODUCT_INQUIRY = "product_inquiry"
    ORDER_STATUS = "order_status"
    REFUND = "refund"
    COMPLAINT = "complaint"
    GENERAL = "general"

@dataclass
class CustomerMessage:
    content: str
    customer_id: str
    timestamp: float
    metadata: Dict = None

class CustomerServiceAgent:
    """智能客服Agent"""
    
    def __init__(self):
        self.conversation_history: Dict[str, List] = {}
        self.knowledge_base = {}
    
    def handle_message(self, message: CustomerMessage) -> Dict[str, Any]:
        """处理消息"""
        # 识别意图
        intent = self._classify_intent(message.content)
        
        # 获取上下文
        context = self._get_context(message.customer_id)
        
        # 生成响应
        response = self._generate_response(intent, message, context)
        
        # 更新历史
        self._update_history(message.customer_id, message, response)
        
        return {
            "intent": intent.value,
            "response": response,
            "confidence": 0.95
        }
    
    def _classify_intent(self, text: str) -> Intent:
        """分类意图"""
        if "订单" in text or "物流" in text:
            return Intent.ORDER_STATUS
        elif "退款" in text or "退货" in text:
            return Intent.REFUND
        elif "投诉" in text or "不满" in text:
            return Intent.COMPLAINT
        elif "产品" in text or "商品" in text:
            return Intent.PRODUCT_INQUIRY
        return Intent.GENERAL
    
    def _get_context(self, customer_id: str) -> Dict:
        """获取上下文"""
        return {
            "history": self.conversation_history.get(customer_id, [])[-5:]
        }
    
    def _generate_response(
        self,
        intent: Intent,
        message: CustomerMessage,
        context: Dict
    ) -> str:
        """生成响应"""
        responses = {
            Intent.ORDER_STATUS: "请提供您的订单号，我帮您查询订单状态。",
            Intent.REFUND: "我理解您想要退款。请问是什么原因需要退款？",
            Intent.COMPLAINT: "非常抱歉给您带来不好的体验。请告诉我具体情况，我会尽力帮您解决。",
            Intent.PRODUCT_INQUIRY: "请问您想了解哪个产品的信息？",
            Intent.GENERAL: "您好！有什么可以帮助您的？"
        }
        
        return responses.get(intent, "您好！有什么可以帮助您的？")
    
    def _update_history(
        self,
        customer_id: str,
        message: CustomerMessage,
        response: str
    ):
        """更新历史"""
        if customer_id not in self.conversation_history:
            self.conversation_history[customer_id] = []
        
        self.conversation_history[customer_id].append({
            "message": message.content,
            "response": response,
            "timestamp": message.timestamp
        })
```

### 代码助手

```python
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class CodeTask(Enum):
    GENERATE = "generate"
    EXPLAIN = "explain"
    DEBUG = "debug"
    OPTIMIZE = "optimize"
    REVIEW = "review"

@dataclass
class CodeContext:
    language: str
    code: str
    task: CodeTask
    metadata: Dict = None

class CodeAssistantAgent:
    """代码助手Agent"""
    
    def __init__(self):
        self.supported_languages = ["python", "javascript", "java", "go"]
    
    def assist(self, context: CodeContext) -> Dict[str, Any]:
        """提供代码协助"""
        if context.task == CodeTask.GENERATE:
            return self._generate_code(context)
        elif context.task == CodeTask.EXPLAIN:
            return self._explain_code(context)
        elif context.task == CodeTask.DEBUG:
            return self._debug_code(context)
        elif context.task == CodeTask.OPTIMIZE:
            return self._optimize_code(context)
        elif context.task == CodeTask.REVIEW:
            return self._review_code(context)
        
        return {"error": "未知任务类型"}
    
    def _generate_code(self, context: CodeContext) -> Dict[str, Any]:
        """生成代码"""
        return {
            "code": f"# Generated {context.language} code\n# ...",
            "explanation": "代码生成说明"
        }
    
    def _explain_code(self, context: CodeContext) -> Dict[str, Any]:
        """解释代码"""
        return {
            "explanation": "代码功能说明",
            "key_concepts": ["概念1", "概念2"]
        }
    
    def _debug_code(self, context: CodeContext) -> Dict[str, Any]:
        """调试代码"""
        return {
            "issues": [
                {"line": 10, "issue": "潜在问题", "fix": "修复建议"}
            ],
            "fixed_code": "# 修复后的代码"
        }
    
    def _optimize_code(self, context: CodeContext) -> Dict[str, Any]:
        """优化代码"""
        return {
            "optimizations": [
                {"type": "性能", "description": "优化描述"}
            ],
            "optimized_code": "# 优化后的代码"
        }
    
    def _review_code(self, context: CodeContext) -> Dict[str, Any]:
        """审查代码"""
        return {
            "score": 8,
            "issues": ["问题1", "问题2"],
            "suggestions": ["建议1", "建议2"]
        }
```

### 研究助手

```python
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ResearchQuery:
    topic: str
    depth: int
    focus_areas: List[str]

class ResearchAssistantAgent:
    """研究助手Agent"""
    
    def __init__(self):
        self.sources = ["arxiv", "pubmed", "scholar"]
        self.findings = []
    
    def research(self, query: ResearchQuery) -> Dict[str, Any]:
        """进行研究"""
        # 搜索文献
        papers = self._search_papers(query)
        
        # 分析文献
        analysis = self._analyze_papers(papers)
        
        # 生成报告
        report = self._generate_report(analysis)
        
        return {
            "papers_found": len(papers),
            "analysis": analysis,
            "report": report
        }
    
    def _search_papers(self, query: ResearchQuery) -> List[Dict]:
        """搜索文献"""
        return [
            {"title": f"关于{query.topic}的研究{i+1}", "relevance": 0.9}
            for i in range(5)
        ]
    
    def _analyze_papers(self, papers: List[Dict]) -> Dict[str, Any]:
        """分析文献"""
        return {
            "key_findings": ["发现1", "发现2"],
            "trends": ["趋势1", "趋势2"],
            "gaps": ["研究空白1"]
        }
    
    def _generate_report(self, analysis: Dict) -> str:
        """生成报告"""
        return f"""
研究摘要：
{chr(10).join(f'- {f}' for f in analysis['key_findings'])}

研究趋势：
{chr(10).join(f'- {t}' for t in analysis['trends'])}
"""
```

## 15.4 【实战】构建半自主Agent

让我们构建一个能够自主执行任务的半自主Agent。

### 项目结构

```
semi-autonomous-agent/
├── .env
├── main.py
├── agent.py
├── goals.py
├── executor.py
└── requirements.txt
```

### 完整代码

**agent.py**

```python
from typing import Dict, Any, List
from openai import OpenAI
from goals import GoalManager
from executor import TaskExecutor

class SemiAutonomousAgent:
    """半自主Agent"""
    
    def __init__(self):
        self.llm = OpenAI()
        self.goal_manager = GoalManager()
        self.executor = TaskExecutor()
        self.running = False
    
    def set_goal(self, goal_description: str):
        """设置目标"""
        self.goal_manager.add_goal(
            goal_id="main_goal",
            description=goal_description
        )
        
        # 分解目标
        sub_goals = self._decompose_goal(goal_description)
        self.goal_manager.decompose_goal("main_goal", sub_goals)
    
    def _decompose_goal(self, goal: str) -> List[Dict[str, Any]]:
        """分解目标"""
        prompt = f"""
请将以下目标分解为具体步骤：

目标：{goal}

输出JSON格式：
[
  {{"description": "步骤描述", "priority": 1}},
  ...
]
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return [{"description": goal, "priority": 1}]
    
    def run(self, max_iterations: int = 10):
        """运行Agent"""
        self.running = True
        
        for i in range(max_iterations):
            if not self.running:
                break
            
            print(f"\n=== 迭代 {i+1} ===")
            
            # 获取下一个任务
            next_action = self.goal_manager.get_next_action()
            
            if next_action.get("action") == "no_pending_goals":
                print("所有目标已完成！")
                break
            
            print(f"执行任务：{next_action['description']}")
            
            # 执行任务
            result = self.executor.execute(next_action)
            
            # 更新进度
            self.goal_manager.update_progress(
                next_action["goal_id"],
                1.0 if result["success"] else 0.5
            )
            
            print(f"结果：{result['message']}")
            
            # 请求人类确认（半自主）
            if not result["success"]:
                user_input = input("任务失败，是否继续？(y/n): ")
                if user_input.lower() != 'y':
                    self.running = False
        
        # 输出最终状态
        status = self.goal_manager.get_status_report()
        print(f"\n最终状态：{status}")

# 使用
from dotenv import load_dotenv
load_dotenv()

agent = SemiAutonomousAgent()
agent.set_goal("学习Python基础并写一个简单的计算器程序")
agent.run()
```

## 本章小结

本章我们学习了：

- ✅ 自主Agent的决策机制
- ✅ 自我反思与改进
- ✅ 长期目标管理
- ✅ 前沿研究方向
- ✅ 行业应用案例
- ✅ 构建了半自主Agent

## 下一章

下一章我们将进行综合实战项目。

[第16章：综合实战项目 →](/frontier/chapter16)
