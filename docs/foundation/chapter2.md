# 第2章：LLM基础与API使用

本章将深入讲解大语言模型的工作原理，以及如何高效使用OpenAI API。

## 2.1 大语言模型基础

### LLM的工作原理

大语言模型（Large Language Model, LLM）是基于Transformer架构的深度学习模型，通过海量文本数据训练而成。

**核心机制**：

1. **Token化**：将文本分割成最小的处理单元
2. **嵌入表示**：将Token转换为向量表示
3. **注意力机制**：计算Token之间的关系
4. **概率预测**：预测下一个Token的概率分布

**示例：Token化过程**

```python
import tiktoken

# 创建编码器
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = "Hello, Agent!"
tokens = enc.encode(text)

print(f"文本: {text}")
print(f"Token IDs: {tokens}")
print(f"Token数量: {len(tokens)}")

# 解码
decoded = enc.decode(tokens)
print(f"解码后: {decoded}")
```

### 主流LLM对比

| 模型 | 提供商 | 优势 | 适用场景 |
|------|--------|------|----------|
| GPT-4 | OpenAI | 综合能力最强 | 复杂推理、代码生成 |
| GPT-3.5-Turbo | OpenAI | 性价比高 | 日常对话、简单任务 |
| Claude 3 | Anthropic | 长文本处理 | 文档分析、研究报告 |
| Gemini | Google | 多模态能力 | 图像理解、多模态任务 |
| Llama 3 | Meta | 开源可本地部署 | 私有化部署、定制化 |

### 模型选择策略

选择模型时考虑以下因素：

1. **任务复杂度**
   - 简单任务：GPT-3.5-Turbo
   - 复杂推理：GPT-4
   - 长文本处理：Claude 3

2. **成本预算**
   - 高预算：GPT-4（$0.03/1K input tokens）
   - 中等预算：GPT-3.5-Turbo（$0.0015/1K input tokens）
   - 低预算：开源模型（Llama 3）

3. **响应速度**
   - 快速响应：GPT-3.5-Turbo
   - 高质量响应：GPT-4

4. **数据隐私**
   - 敏感数据：本地部署开源模型
   - 一般数据：使用云服务API

## 2.2 OpenAI API深入

### Chat Completions API详解

Chat Completions API是OpenAI最核心的API，用于生成对话响应。

**基本用法**

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "什么是Agent？"}
    ]
)

print(response.choices[0].message.content)
```

**Messages结构**

```python
messages = [
    {"role": "system", "content": "系统提示，定义AI角色"},    # 可选
    {"role": "user", "content": "用户消息"},                  # 必需
    {"role": "assistant", "content": "AI回复"},              # 对话历史
    {"role": "user", "content": "后续问题"}                   # 多轮对话
]
```

### 参数调优

**关键参数**

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "写一首诗"}],
    
    # 温度参数：控制随机性（0-2）
    temperature=0.7,  # 0：确定性输出，2：高度随机
    
    # Top-p采样：控制多样性（0-1）
    top_p=0.9,  # 推荐只设置temperature或top_p之一
    
    # 最大Token数
    max_tokens=500,  # 限制输出长度
    
    # 停止序列
    stop=["\n", "END"],  # 遇到这些序列停止生成
    
    # 频率惩罚
    frequency_penalty=0.5,  # 减少重复（-2到2）
    
    # 存在惩罚
    presence_penalty=0.5,  # 鼓励新话题（-2到2）
    
    # 返回多个结果
    n=3,  # 生成3个不同的响应
)
```

**参数调优指南**

| 参数 | 低值效果 | 高值效果 | 推荐值 |
|------|----------|----------|--------|
| temperature | 确定性、一致 | 创造性、多样 | 0.7-0.9 |
| top_p | 聚焦高概率词 | 更多可能性 | 0.9-1.0 |
| frequency_penalty | 允许重复 | 减少重复 | 0.3-0.7 |
| presence_penalty | 可重复话题 | 鼓励新话题 | 0.3-0.7 |

### Token计算与成本控制

**计算Token数量**

```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def count_messages_tokens(messages, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    total = 0
    
    for message in messages:
        total += 4  # 每条消息的固定开销
        for key, value in message.items():
            total += len(enc.encode(value))
    
    total += 2  # 对话的固定开销
    return total

# 示例
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是Agent？"}
]

print(f"Token数量: {count_messages_tokens(messages)}")
```

**成本估算**

```python
def estimate_cost(input_tokens, output_tokens, model="gpt-3.5-turbo"):
    prices = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03}
    }
    
    price = prices.get(model, prices["gpt-3.5-turbo"])
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1000
    return cost

# 示例
input_tokens = 1000
output_tokens = 500
print(f"成本: ${estimate_cost(input_tokens, output_tokens):.4f}")
```

### 流式输出处理

流式输出可以让用户更快看到响应，提升用户体验。

```python
def stream_chat(messages):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print()  # 换行
    return full_response

# 使用
messages = [{"role": "user", "content": "讲一个故事"}]
stream_chat(messages)
```

**异步流式输出**

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def async_stream_chat(messages):
    stream = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print()
    return full_response

# 使用
asyncio.run(async_stream_chat([{"role": "user", "content": "你好"}]))
```

## 2.3 Prompt Engineering基础

### Prompt设计原则

**1. 清晰明确**

```python
# ❌ 不好的Prompt
prompt = "写点东西"

# ✅ 好的Prompt
prompt = """
请写一篇关于AI Agent的短文，要求：
1. 字数在300字左右
2. 包含Agent的定义和应用场景
3. 语言通俗易懂
"""
```

**2. 提供上下文**

```python
# ❌ 缺少上下文
prompt = "分析这个数据"

# ✅ 提供完整上下文
prompt = """
你是一位数据分析专家。请分析以下销售数据：
数据：[100, 150, 200, 180, 220]
要求：
1. 计算平均值和增长率
2. 找出趋势
3. 给出建议
"""
```

**3. 指定输出格式**

```python
prompt = """
请分析以下文本的情感，并以JSON格式返回结果：
{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0}

文本：这个产品非常好用，我很满意！
"""
```

### 常用Prompt模式

**角色扮演模式**

```python
system_prompt = """
你是一位资深的Python开发专家，擅长：
- 代码优化
- 架构设计
- 最佳实践

回答问题时：
1. 给出具体代码示例
2. 解释原理
3. 提供最佳实践建议
"""
```

**任务分解模式**

```python
prompt = """
请按以下步骤完成任务：
步骤1：理解用户需求
步骤2：分析可能的解决方案
步骤3：选择最佳方案
步骤4：给出具体实现

用户需求：实现一个简单的缓存系统
"""
```

### Few-shot Learning

通过提供示例来引导模型输出：

```python
messages = [
    {"role": "system", "content": "你是一个情感分析助手。"},
    {"role": "user", "content": "这个产品太棒了！"},
    {"role": "assistant", "content": '{"sentiment": "positive", "score": 0.95}'},
    {"role": "user", "content": "服务态度很差"},
    {"role": "assistant", "content": '{"sentiment": "negative", "score": 0.88}'},
    {"role": "user", "content": "还可以，一般般"},
    {"role": "assistant", "content": '{"sentiment": "neutral", "score": 0.72}'},
    {"role": "user", "content": "物流很快，包装也很好"}  # 新的输入
]
```

### Chain-of-Thought

让模型展示推理过程：

```python
prompt = """
请一步步思考并回答问题：

问题：小明有5个苹果，给了小红2个，又买了3个，现在有几个？

让我们一步步思考：
1. 小明最初有5个苹果
2. 给了小红2个，剩下：5 - 2 = 3个
3. 又买了3个，现在有：3 + 3 = 6个

答案：小明现在有6个苹果。

现在请回答：一本书100页，第一天读了20页，第二天读了30页，还剩多少页？
"""
```

## 2.4 【实战】智能客服助手

让我们创建一个完整的智能客服助手，综合运用本章所学知识。

### 项目结构

```
smart-customer-service/
├── .env
├── main.py
├── agent.py
├── prompts.py
└── requirements.txt
```

### 完整代码

**prompts.py**

```python
SYSTEM_PROMPT = """
你是一个专业的客服助手，代表一家科技公司。

你的职责：
1. 回答产品相关问题
2. 处理客户投诉
3. 提供技术支持
4. 记录客户反馈

回答要求：
- 语气友好、专业
- 回答简洁明了
- 无法解决的问题，引导用户联系人工客服
- 涉及退款、赔偿等问题，转接人工处理

产品信息：
- 产品名称：智能助手Pro
- 价格：99元/月
- 功能：AI对话、文档处理、数据分析
- 客服电话：400-123-4567
"""

INTENT_PROMPT = """
分析用户意图，返回以下类别之一：
- product_inquiry：产品咨询
- technical_support：技术支持
- complaint：投诉建议
- refund：退款请求
- general：一般问题

用户输入：{user_input}

只返回类别名称，不要其他内容。
"""
```

**agent.py**

```python
import os
import json
from openai import OpenAI
from prompts import SYSTEM_PROMPT, INTENT_PROMPT

class CustomerServiceAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.max_history = 20
    
    def detect_intent(self, user_message):
        """检测用户意图"""
        prompt = INTENT_PROMPT.format(user_input=user_message)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def chat(self, user_message):
        """处理用户消息"""
        # 检测意图
        intent = self.detect_intent(user_message)
        
        # 特殊意图处理
        if intent == "refund":
            return "关于退款问题，我将为您转接人工客服。请拨打：400-123-4567"
        
        # 添加用户消息
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 调用API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        # 添加助手回复
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # 管理历史长度
        if len(self.conversation_history) > self.max_history:
            # 保留system prompt和最近的对话
            self.conversation_history = (
                [self.conversation_history[0]] + 
                self.conversation_history[-(self.max_history-1):]
            )
        
        return assistant_message
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "conversation_length": len(self.conversation_history),
            "total_tokens": sum(
                len(msg["content"]) for msg in self.conversation_history
            )
        }
```

**main.py**

```python
from dotenv import load_dotenv
from agent import CustomerServiceAgent

load_dotenv()

def main():
    agent = CustomerServiceAgent()
    
    print("=" * 50)
    print("智能客服助手")
    print("输入 'quit' 退出，'stats' 查看统计")
    print("=" * 50)
    print()
    
    while True:
        user_input = input("用户: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\n感谢使用，再见！")
            break
        
        if user_input.lower() == 'stats':
            stats = agent.get_stats()
            print(f"\n统计信息: {stats}\n")
            continue
        
        response = agent.chat(user_input)
        print(f"\n客服: {response}\n")

if __name__ == "__main__":
    main()
```

### 运行项目

```bash
pip install -r requirements.txt
python main.py
```

### 功能特点

1. **意图识别**：自动识别用户意图类型
2. **上下文管理**：保持对话连贯性
3. **历史限制**：自动管理对话历史长度
4. **特殊处理**：退款等敏感问题转人工
5. **统计功能**：查看对话统计信息

## 本章小结

本章我们学习了：

- ✅ LLM的工作原理和Token机制
- ✅ 主流LLM的特点和选择策略
- ✅ OpenAI API的详细使用方法
- ✅ 参数调优和成本控制
- ✅ Prompt Engineering基础技巧
- ✅ 实现了一个智能客服助手

## 下一章

下一章我们将学习Agent的核心概念，包括架构模式、决策循环等。

[第3章：Agent核心概念 →](/foundation/chapter3)
