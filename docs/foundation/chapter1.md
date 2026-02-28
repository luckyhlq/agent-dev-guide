# 第1章：Agent开发概述

欢迎来到Agent开发的世界！本章将带你了解Agent的基本概念、核心组件，并完成你的第一个Agent应用。

## 1.1 什么是AI Agent

### Agent的定义

AI Agent（人工智能代理）是一个能够感知环境、进行推理、采取行动并实现目标的自主系统。与传统AI应用不同，Agent具有以下核心特征：

- **自主性**：能够自主决策，无需人类持续干预
- **感知能力**：能够感知和理解环境信息
- **推理能力**：能够基于信息进行逻辑推理和规划
- **行动能力**：能够执行具体的操作来改变环境
- **学习能力**：能够从经验中学习和改进

### Agent vs 传统AI应用

| 特性 | 传统AI应用 | AI Agent |
|------|-----------|----------|
| 交互方式 | 单次请求-响应 | 多轮对话、持续交互 |
| 决策能力 | 预定义逻辑 | 自主推理决策 |
| 工具使用 | 无或有限 | 主动调用多种工具 |
| 记忆能力 | 无上下文 | 具备短期和长期记忆 |
| 目标导向 | 完成单次任务 | 追求长期目标 |

### Agent的应用场景

Agent技术正在改变各行各业：

- **智能客服**：7x24小时自动应答，处理复杂问题
- **代码助手**：自动生成代码、代码审查、bug修复
- **研究助手**：文献检索、信息整理、报告生成
- **个人助理**：日程管理、邮件处理、任务提醒
- **自动化运维**：系统监控、故障诊断、自动修复
- **内容创作**：文章撰写、视频脚本、营销文案

## 1.2 Agent的核心组件

一个完整的Agent系统由以下四个核心组件构成：

### 1. 大语言模型（LLM）

LLM是Agent的"大脑"，负责：

- 理解用户意图
- 进行逻辑推理
- 生成自然语言响应
- 规划行动步骤

**常用模型**：
- GPT-4：综合能力最强
- Claude 3：长文本处理优秀
- Gemini：多模态能力强

### 2. 记忆系统（Memory）

记忆系统让Agent能够：

- 记住对话历史（短期记忆）
- 存储长期知识（长期记忆）
- 检索相关信息（记忆检索）

**存储方式**：
- 列表/数组：简单对话历史
- 向量数据库：语义检索
- 图数据库：知识图谱

### 3. 工具调用（Tools）

工具扩展了Agent的能力边界：

- 搜索引擎：获取实时信息
- 计算器：执行数学运算
- 代码解释器：运行代码
- API接口：调用外部服务

### 4. 规划能力（Planning）

规划能力让Agent能够：

- 分解复杂任务
- 制定执行步骤
- 调整行动策略
- 处理异常情况

**规划模式**：
- ReAct：推理-行动循环
- Plan-and-Execute：先规划后执行
- Reflection：自我反思优化

## 1.3 开发环境搭建

### Python环境配置

首先确保你的系统已安装Python 3.10或更高版本：

```bash
# 检查Python版本
python --version

# 如果版本过低，建议使用pyenv或conda管理Python版本
```

创建虚拟环境并安装必要依赖：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（macOS/Linux）
source venv/bin/activate

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 安装核心依赖
pip install openai langchain python-dotenv
```

### OpenAI API获取与配置

1. **获取API Key**

   访问 [OpenAI Platform](https://platform.openai.com/) 注册账号并获取API Key

2. **配置环境变量**

   创建 `.env` 文件：

   ```bash
   OPENAI_API_KEY=your-api-key-here
   OPENAI_API_BASE=https://api.openai.com/v1
   ```

3. **验证配置**

   ```python
   import os
   from openai import OpenAI
   
   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   
   print(response.choices[0].message.content)
   ```

### 开发工具推荐

| 工具 | 用途 | 推荐理由 |
|------|------|----------|
| VS Code | 代码编辑器 | 插件丰富，AI辅助强大 |
| Jupyter Notebook | 交互式开发 | 适合实验和原型开发 |
| PyCharm | IDE | 调试功能强大 |
| Postman | API测试 | 方便测试Agent API |
| Git | 版本控制 | 代码管理必备 |

## 1.4 【实战】第一个简单Agent

让我们创建一个简单的对话Agent，理解Agent的基本工作流程。

### 项目结构

```
my-first-agent/
├── .env
├── main.py
└── requirements.txt
```

### 完整代码

**requirements.txt**

```txt
openai>=1.0.0
python-dotenv>=1.0.0
```

**main.py**

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class SimpleAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []
    
    def chat(self, user_message):
        # 添加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history
        )
        
        # 获取助手回复
        assistant_message = response.choices[0].message.content
        
        # 添加助手回复到历史
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def clear_history(self):
        self.conversation_history = []

def main():
    agent = SimpleAgent()
    
    print("=== 简单对话Agent ===")
    print("输入 'quit' 退出\n")
    
    while True:
        user_input = input("你: ")
        
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()
```

### 运行项目

```bash
# 安装依赖
pip install -r requirements.txt

# 运行程序
python main.py
```

### 代码解析

这个简单的Agent展示了Agent的核心概念：

1. **记忆系统**：`conversation_history` 保存对话历史
2. **LLM调用**：使用OpenAI API生成响应
3. **交互循环**：持续接收用户输入并返回响应

### 扩展练习

尝试对这个Agent进行改进：

1. 添加System Prompt来定义Agent的角色
2. 实现对话历史的长度限制
3. 添加流式输出功能
4. 记录对话到文件

## 本章小结

本章我们学习了：

- ✅ Agent的定义和核心特征
- ✅ Agent与传统AI应用的区别
- ✅ Agent的四大核心组件
- ✅ 开发环境的搭建
- ✅ 创建了第一个简单的Agent

## 下一章

下一章我们将深入学习LLM的基础知识和OpenAI API的使用，这是Agent开发的基础。

[第2章：LLM基础与API使用 →](/foundation/chapter2)
