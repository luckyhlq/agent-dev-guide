# 第3章：Agent核心概念

本章将深入讲解Agent的核心架构模式、决策循环和类型分类，这是理解Agent工作原理的关键。

## 3.1 Agent架构模式

### ReAct模式

ReAct（Reasoning + Acting）是最经典的Agent架构模式，它将推理和行动交织在一起。

**工作流程**

```
用户输入 → 思考(Reasoning) → 行动(Acting) → 观察(Observation) → 循环...
```

**核心思想**

1. **Thought（思考）**：分析当前情况，决定下一步行动
2. **Action（行动）**：执行具体操作（如调用工具）
3. **Observation（观察）**：获取行动结果
4. 循环直到任务完成

**代码实现**

```python
from openai import OpenAI
import json

class ReActAgent:
    def __init__(self):
        self.client = OpenAI()
        self.tools = {
            "search": self.search,
            "calculate": self.calculate
        }
        
        self.system_prompt = """
你是一个使用ReAct模式的Agent。

对于每一步，你需要：
1. Thought: 思考当前情况
2. Action: 选择一个行动
3. Action Input: 行动的输入

可用工具：
- search(query): 搜索信息
- calculate(expression): 计算数学表达式

当得到最终答案时，输出：
Thought: 我已经得到答案
Final Answer: [你的答案]

开始！
"""
    
    def search(self, query):
        return f"搜索结果：关于'{query}'的信息..."
    
    def calculate(self, expression):
        try:
            result = eval(expression)
            return f"计算结果：{result}"
        except:
            return "计算错误"
    
    def run(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        for _ in range(10):  # 最多10轮
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )
            
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            
            # 检查是否完成
            if "Final Answer:" in content:
                return content.split("Final Answer:")[-1].strip()
            
            # 解析并执行行动
            if "Action:" in content and "Action Input:" in content:
                action = content.split("Action:")[-1].split("\n")[0].strip()
                action_input = content.split("Action Input:")[-1].split("\n")[0].strip()
                
                if action in self.tools:
                    observation = self.tools[action](action_input)
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}"
                    })
        
        return "未能完成任务"

# 使用
agent = ReActAgent()
result = agent.run("搜索Python的最新版本，然后计算2024减去它的发布年份")
print(result)
```

### Plan-and-Execute模式

Plan-and-Execute模式将任务分解为两个阶段：规划和执行。

**工作流程**

```
用户输入 → 规划(Planning) → 生成步骤列表 → 逐步执行(Execution) → 结果
```

**代码实现**

```python
class PlanAndExecuteAgent:
    def __init__(self):
        self.client = OpenAI()
        
        self.planner_prompt = """
你是一个任务规划专家。将用户任务分解为具体步骤。

输出格式（JSON数组）：
["步骤1", "步骤2", "步骤3", ...]

任务：{task}
"""

        self.executor_prompt = """
你是一个任务执行者。执行给定的步骤。

当前步骤：{step}
上下文：{context}

输出执行结果。
"""
    
    def plan(self, task):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": self.planner_prompt.format(task=task)
            }],
            temperature=0
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return []
    
    def execute(self, step, context=""):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": self.executor_prompt.format(
                    step=step,
                    context=context
                )
            }]
        )
        
        return response.choices[0].message.content
    
    def run(self, task):
        # 规划阶段
        steps = self.plan(task)
        print(f"规划步骤：{steps}\n")
        
        # 执行阶段
        context = ""
        results = []
        
        for i, step in enumerate(steps, 1):
            print(f"执行步骤 {i}/{len(steps)}: {step}")
            result = self.execute(step, context)
            results.append(result)
            context += f"\n步骤{i}结果：{result}"
            print(f"结果：{result}\n")
        
        return results

# 使用
agent = PlanAndExecuteAgent()
results = agent.run("写一篇关于AI Agent的文章")
```

### Reflection模式

Reflection模式让Agent能够自我反思和改进。

**工作流程**

```
执行任务 → 生成结果 → 自我反思 → 改进 → 循环或完成
```

**代码实现**

```python
class ReflectionAgent:
    def __init__(self):
        self.client = OpenAI()
        
        self.generate_prompt = """
根据要求生成内容。

要求：{requirement}
"""
        
        self.reflect_prompt = """
请评估以下内容的质量：

内容：{content}
要求：{requirement}

评估标准：
1. 是否满足要求
2. 有哪些可以改进的地方
3. 给出改进建议

如果满意，输出：SATISFIED
否则输出具体的改进建议。
"""
    
    def generate(self, requirement):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": self.generate_prompt.format(requirement=requirement)
            }]
        )
        return response.choices[0].message.content
    
    def reflect(self, content, requirement):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": self.reflect_prompt.format(
                    content=content,
                    requirement=requirement
                )
            }]
        )
        return response.choices[0].message.content
    
    def run(self, requirement, max_iterations=3):
        content = self.generate(requirement)
        
        for i in range(max_iterations):
            print(f"\n=== 第 {i+1} 轮 ===")
            print(f"生成内容：\n{content}\n")
            
            reflection = self.reflect(content, requirement)
            print(f"反思结果：\n{reflection}\n")
            
            if "SATISFIED" in reflection:
                print("✅ 内容已满足要求")
                return content
            
            # 根据反思改进
            content = self.generate(f"{requirement}\n\n改进建议：{reflection}")
        
        print("⚠️ 达到最大迭代次数")
        return content

# 使用
agent = ReflectionAgent()
result = agent.run("写一个简洁的Python函数，计算斐波那契数列")
```

## 3.2 Agent决策循环

### 感知-思考-行动循环

这是Agent的核心运行机制：

```python
class AgentLoop:
    def __init__(self):
        self.client = OpenAI()
        self.state = {}
        self.history = []
    
    def perceive(self, input_data):
        """感知：接收输入"""
        self.state['current_input'] = input_data
        self.history.append({"type": "input", "data": input_data})
    
    def think(self):
        """思考：分析和决策"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个智能Agent"},
                {"role": "user", "content": str(self.state)}
            ]
        )
        
        decision = response.choices[0].message.content
        self.state['last_thought'] = decision
        self.history.append({"type": "thought", "data": decision})
        
        return decision
    
    def act(self, decision):
        """行动：执行决策"""
        # 这里可以实现具体的行动逻辑
        result = f"执行：{decision}"
        self.history.append({"type": "action", "data": result})
        return result
    
    def should_continue(self):
        """判断是否继续"""
        return not self.state.get('task_completed', False)
    
    def run(self, input_data):
        """主循环"""
        self.perceive(input_data)
        
        while self.should_continue():
            decision = self.think()
            result = self.act(decision)
            
            # 更新状态
            self.perceive(result)
            
            # 防止无限循环
            if len(self.history) > 20:
                break
        
        return self.state
```

### 状态管理

Agent需要维护状态以保持一致性：

```python
class StateManager:
    def __init__(self):
        self.state = {
            'user_intent': None,
            'context': {},
            'memory': [],
            'current_task': None,
            'completed_tasks': []
        }
    
    def update(self, key, value):
        """更新状态"""
        self.state[key] = value
    
    def get(self, key, default=None):
        """获取状态"""
        return self.state.get(key, default)
    
    def add_memory(self, item):
        """添加记忆"""
        self.state['memory'].append(item)
        
        # 限制记忆长度
        if len(self.state['memory']) > 100:
            self.state['memory'] = self.state['memory'][-100:]
    
    def get_context(self):
        """获取上下文"""
        return {
            'intent': self.state['user_intent'],
            'current_task': self.state['current_task'],
            'recent_memory': self.state['memory'][-10:]
        }
    
    def save(self, filepath):
        """保存状态到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.state, f)
    
    def load(self, filepath):
        """从文件加载状态"""
        import json
        with open(filepath, 'r') as f:
            self.state = json.load(f)
```

### 终止条件设计

合理的终止条件可以防止Agent陷入死循环：

```python
class TerminationChecker:
    def __init__(self, max_steps=10, max_time=300):
        self.max_steps = max_steps
        self.max_time = max_time
        self.start_time = time.time()
        self.step_count = 0
    
    def check(self, state):
        """检查是否应该终止"""
        reasons = []
        
        # 检查步数
        self.step_count += 1
        if self.step_count >= self.max_steps:
            reasons.append(f"达到最大步数：{self.max_steps}")
        
        # 检查时间
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_time:
            reasons.append(f"达到最大时间：{self.max_time}秒")
        
        # 检查任务状态
        if state.get('task_completed'):
            reasons.append("任务已完成")
        
        # 检查错误状态
        if state.get('error_count', 0) >= 3:
            reasons.append("错误次数过多")
        
        return len(reasons) > 0, reasons
```

## 3.3 Agent类型分类

### 单任务Agent

专注于完成单一特定任务：

```python
class SingleTaskAgent:
    """专门用于情感分析的单任务Agent"""
    
    def __init__(self):
        self.client = OpenAI()
        self.task = "sentiment_analysis"
    
    def execute(self, text):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"分析以下文本的情感（positive/negative/neutral）：\n{text}"
            }],
            temperature=0
        )
        
        return {
            "task": self.task,
            "input": text,
            "result": response.choices[0].message.content
        }
```

### 多任务Agent

能够处理多种类型的任务：

```python
class MultiTaskAgent:
    """可以处理多种任务的多任务Agent"""
    
    def __init__(self):
        self.client = OpenAI()
        self.tasks = {
            "sentiment": self.sentiment_analysis,
            "summarize": self.summarize,
            "translate": self.translate,
            "qa": self.question_answer
        }
    
    def detect_task(self, user_input):
        """检测任务类型"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"判断用户意图，返回以下之一：sentiment, summarize, translate, qa\n\n用户输入：{user_input}"
            }],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def sentiment_analysis(self, text):
        return self._call_api(f"分析情感：{text}")
    
    def summarize(self, text):
        return self._call_api(f"总结内容：{text}")
    
    def translate(self, text):
        return self._call_api(f"翻译成英文：{text}")
    
    def question_answer(self, text):
        return self._call_api(f"回答问题：{text}")
    
    def _call_api(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def execute(self, user_input):
        task_type = self.detect_task(user_input)
        
        if task_type in self.tasks:
            return {
                "task": task_type,
                "result": self.tasks[task_type](user_input)
            }
        
        return {"error": "未知任务类型"}
```

### 自主Agent

能够自主设定目标并执行：

```python
import time

class AutonomousAgent:
    """能够自主运行的Agent"""
    
    def __init__(self, goal):
        self.client = OpenAI()
        self.goal = goal
        self.memory = []
        self.completed_tasks = []
        self.running = True
    
    def plan_next_task(self):
        """规划下一个任务"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
目标：{self.goal}

已完成任务：{self.completed_tasks}
记忆：{self.memory[-5:]}

请规划下一个要执行的具体任务。
如果目标已完成，输出：GOAL_COMPLETED
否则输出具体任务描述。
"""
            }]
        )
        return response.choices[0].message.content
    
    def execute_task(self, task):
        """执行任务"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"执行任务：{task}"
            }]
        )
        return response.choices[0].message.content
    
    def run(self, max_iterations=10):
        """自主运行"""
        for i in range(max_iterations):
            print(f"\n=== 迭代 {i+1} ===")
            
            # 规划任务
            task = self.plan_next_task()
            print(f"规划任务：{task}")
            
            # 检查是否完成
            if "GOAL_COMPLETED" in task:
                print("✅ 目标已完成！")
                self.running = False
                break
            
            # 执行任务
            result = self.execute_task(task)
            print(f"执行结果：{result}")
            
            # 记录
            self.memory.append({"task": task, "result": result})
            self.completed_tasks.append(task)
            
            # 休息一下
            time.sleep(1)
        
        return self.completed_tasks

# 使用
agent = AutonomousAgent("学习Python基础并写一个简单的计算器程序")
agent.run()
```

## 3.4 【实战】构建ReAct Agent

让我们构建一个完整的ReAct Agent，能够使用多种工具解决复杂问题。

### 项目结构

```
react-agent/
├── .env
├── main.py
├── agent.py
├── tools.py
└── requirements.txt
```

### 完整代码

**tools.py**

```python
import requests
from datetime import datetime
import math

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.tool_descriptions = []
    
    def register(self, name, func, description):
        self.tools[name] = func
        self.tool_descriptions.append({
            "name": name,
            "description": description
        })
    
    def get_tool(self, name):
        return self.tools.get(name)
    
    def get_descriptions(self):
        return "\n".join([
            f"- {t['name']}: {t['description']}" 
            for t in self.tool_descriptions
        ])

def create_tools():
    registry = ToolRegistry()
    
    def search(query):
        return f"搜索'{query}'的结果：相关内容..."
    
    def calculate(expression):
        try:
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return f"计算结果：{eval(expression)}"
            return "错误：表达式包含非法字符"
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    def get_time():
        return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def weather(city):
        return f"{city}今天天气晴朗，温度25°C"
    
    def wiki(term):
        return f"'{term}'的百科解释：这是一个重要概念..."
    
    registry.register("search", search, "搜索互联网信息。参数：query")
    registry.register("calculate", calculate, "计算数学表达式。参数：expression")
    registry.register("get_time", get_time, "获取当前时间。无需参数")
    registry.register("weather", weather, "查询天气。参数：city")
    registry.register("wiki", wiki, "查询百科知识。参数：term")
    
    return registry
```

**agent.py**

```python
import os
import re
from openai import OpenAI
from tools import create_tools

class ReActAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tools = create_tools()
        self.max_iterations = 10
        
        self.system_prompt = f"""
你是一个使用ReAct模式的智能Agent。

对于每一步，你需要：
1. Thought: 分析当前情况，思考下一步
2. Action: 选择一个工具
3. Action Input: 提供工具参数

可用工具：
{self.tools.get_descriptions()}

当得到最终答案时，输出：
Thought: 我已经得到答案
Final Answer: [你的答案]

示例：
用户：北京现在几点？
Thought: 用户想知道北京的时间
Action: get_time
Action Input: 
Observation: 当前时间：2024-01-15 14:30:00
Thought: 我已经得到答案
Final Answer: 北京现在的时间是2024年1月15日14:30:00
"""
    
    def parse_action(self, content):
        """解析行动"""
        action_match = re.search(r'Action:\s*(\w+)', content)
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', content)
        
        action = action_match.group(1) if action_match else None
        action_input = input_match.group(1).strip() if input_match else ""
        
        return action, action_input
    
    def execute_tool(self, action, action_input):
        """执行工具"""
        tool = self.tools.get_tool(action)
        if tool:
            return tool(action_input)
        return f"错误：未知工具 '{action}'"
    
    def run(self, user_input):
        """运行Agent"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        for i in range(self.max_iterations):
            print(f"\n--- 迭代 {i+1} ---")
            
            # 获取LLM响应
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )
            
            content = response.choices[0].message.content
            print(f"思考过程：\n{content}")
            
            messages.append({"role": "assistant", "content": content})
            
            # 检查是否完成
            if "Final Answer:" in content:
                final_answer = content.split("Final Answer:")[-1].strip()
                print(f"\n✅ 最终答案：{final_answer}")
                return final_answer
            
            # 解析并执行行动
            action, action_input = self.parse_action(content)
            
            if action:
                print(f"\n执行工具：{action}({action_input})")
                observation = self.execute_tool(action, action_input)
                print(f"观察结果：{observation}")
                
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            else:
                # 如果没有行动，提示继续
                messages.append({
                    "role": "user",
                    "content": "请继续思考并采取行动。"
                })
        
        print("\n⚠️ 达到最大迭代次数")
        return "抱歉，我无法在限定步骤内完成任务。"
```

**main.py**

```python
from dotenv import load_dotenv
from agent import ReActAgent

load_dotenv()

def main():
    agent = ReActAgent()
    
    print("=" * 60)
    print("ReAct Agent - 推理与行动")
    print("=" * 60)
    print("\n可用工具：")
    print("- search: 搜索信息")
    print("- calculate: 计算表达式")
    print("- get_time: 获取时间")
    print("- weather: 查询天气")
    print("- wiki: 查询百科")
    print("\n输入 'quit' 退出")
    print("=" * 60)
    
    while True:
        user_input = input("\n用户: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        agent.run(user_input)

if __name__ == "__main__":
    main()
```

### 运行示例

```bash
python main.py

用户: 北京的天气如何？如果温度是25度，那么25*2+10是多少？

--- 迭代 1 ---
思考过程：
Thought: 用户想知道北京的天气，然后进行计算
Action: weather
Action Input: 北京

执行工具：weather(北京)
观察结果：北京今天天气晴朗，温度25°C

--- 迭代 2 ---
思考过程：
Thought: 我得到了北京的温度是25度，现在需要计算25*2+10
Action: calculate
Action Input: 25*2+10

执行工具：calculate(25*2+10)
观察结果：计算结果：60

--- 迭代 3 ---
思考过程：
Thought: 我已经得到所有答案
Final Answer: 北京今天天气晴朗，温度25°C。计算25*2+10的结果是60。

✅ 最终答案：北京今天天气晴朗，温度25°C。计算25*2+10的结果是60。
```

## 本章小结

本章我们学习了：

- ✅ 三种主要的Agent架构模式：ReAct、Plan-and-Execute、Reflection
- ✅ Agent的决策循环机制
- ✅ 状态管理和终止条件设计
- ✅ Agent的类型分类
- ✅ 构建了一个完整的ReAct Agent

## 下一章

下一章我们将进入第二阶段，学习高级Prompt技术。

[第4章：高级Prompt技术 →](/core-skills/chapter4)
