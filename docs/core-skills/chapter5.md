# 第5章：Function Calling与工具使用

本章将深入讲解OpenAI的Function Calling机制，以及如何为Agent设计和集成各种工具。

## 5.1 Function Calling原理

**什么是Function Calling？**

Function Calling（函数调用）是让AI能够"动手做事"的关键技术。没有Function Calling之前，AI只能"说话"（生成文本）；有了Function Calling，AI可以"行动"（调用函数、访问数据库、调用API等）。

打个比方：
- **没有Function Calling**：AI就像一个只能给你建议的顾问，"你可以查一下天气网站"
- **有了Function Calling**：AI就像一个能帮你办事的助手，"我帮你查了一下，北京今天25度，晴天"

**Function Calling的工作原理**

整个过程就像一个"对话-行动-反馈"的循环：

1. **用户提问**："北京今天天气怎么样？"
2. **AI分析**：AI理解问题，发现需要查询天气信息
3. **AI决定调用函数**：AI返回一个函数调用请求（函数名+参数）
4. **执行函数**：你的代码执行实际的天气查询函数
5. **返回结果**：将函数执行结果返回给AI
6. **AI生成回答**：AI根据结果生成最终回答

**为什么需要Function Calling？**

大语言模型有几个固有的局限：
- **知识截止**：训练数据有截止日期，不知道最新信息
- **无法访问外部系统**：不能直接访问数据库、API、文件系统
- **计算能力有限**：不擅长精确的数学计算

Function Calling完美解决了这些问题：
- 查实时天气、股票价格 → 调用API
- 查数据库 → 调用数据库函数
- 精确计算 → 调用计算函数

**Function Calling vs 普通Prompt**

| 特性 | 普通Prompt | Function Calling |
|------|-----------|------------------|
| 能力 | 只能生成文本 | 可以执行实际操作 |
| 实时信息 | 无法获取 | 可以通过API获取 |
| 准确性 | 可能"编造"答案 | 基于真实数据 |
| 可靠性 | 不稳定 | 可控可验证 |

### OpenAI Function Calling机制

Function Calling允许LLM调用外部函数，极大地扩展了Agent的能力边界。

**基本流程**

```
用户输入 → LLM分析 → 决定调用函数 → 返回函数调用请求 → 执行函数 → 返回结果 → LLM生成最终响应
```

**基本用法**

```python
from openai import OpenAI
import json

client = OpenAI()

# 定义函数
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 发送请求
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"
)

# 检查是否需要调用函数
message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    print(f"函数名：{function_name}")
    print(f"参数：{function_args}")
    
    # 执行函数
    if function_name == "get_weather":
        result = get_weather(**function_args)
        
        # 将结果返回给LLM
        messages = [
            {"role": "user", "content": "北京今天天气怎么样？"},
            message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            }
        ]
        
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        print(final_response.choices[0].message.content)
```

### 参数定义与验证

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class TemperatureUnit(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

class WeatherRequest(BaseModel):
    city: str = Field(..., description="城市名称")
    unit: Optional[TemperatureUnit] = Field(
        default=TemperatureUnit.CELSIUS,
        description="温度单位"
    )
    days: Optional[int] = Field(
        default=1,
        ge=1,
        le=7,
        description="预报天数，1-7天"
    )

def pydantic_to_function_schema(model: BaseModel) -> dict:
    """将Pydantic模型转换为Function Calling Schema"""
    schema = model.model_json_schema()
    
    return {
        "type": "function",
        "function": {
            "name": model.__class__.__name__.replace("Request", "").lower(),
            "description": model.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }

# 使用
class SearchRequest(BaseModel):
    """搜索互联网信息"""
    query: str = Field(..., description="搜索关键词")
    num_results: int = Field(default=5, ge=1, le=20, description="返回结果数量")

tool_schema = pydantic_to_function_schema(SearchRequest())
print(json.dumps(tool_schema, indent=2))
```

### 多函数调用处理

```python
class MultiFunctionAgent:
    def __init__(self):
        self.client = OpenAI()
        self.functions = {}
        self.tools = []
    
    def register_function(self, name, func, description, parameters):
        """注册函数"""
        self.functions[name] = func
        self.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })
    
    def execute_function(self, name, args):
        """执行函数"""
        if name in self.functions:
            return self.functions[name](**args)
        return f"错误：未知函数 {name}"
    
    def run(self, user_message):
        """运行Agent"""
        messages = [{"role": "user", "content": user_message}]
        
        while True:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            # 检查是否需要调用函数
            if not message.tool_calls:
                return message.content
            
            # 处理所有函数调用
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"调用函数：{function_name}({function_args})")
                
                result = self.execute_function(function_name, function_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

# 使用
agent = MultiFunctionAgent()

agent.register_function(
    "get_weather",
    lambda city: f"{city}今天晴，25°C",
    "获取天气信息",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
    }
)

agent.register_function(
    "calculate",
    lambda expression: str(eval(expression)),
    "计算数学表达式",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"}
        },
        "required": ["expression"]
    }
)

result = agent.run("北京天气如何？另外计算25*4")
print(result)
```

## 5.2 工具设计原则

**为什么工具设计很重要？**

好的工具设计就像好的API设计一样，能让AI更容易正确使用。如果工具设计得不好：
- AI可能不知道什么时候该用这个工具
- AI可能传错参数
- AI可能误解工具的功能

**好工具的五个特征**

1. **功能单一明确**
   - 一个工具只做一件事
   - 工具名称要能清楚表达功能
   - 比如：`get_weather`比`get_info`更明确

2. **参数简单清晰**
   - 参数越少越好
   - 每个参数都要有清晰的描述
   - 必填参数要尽量少
   - 提供默认值和枚举选项

3. **描述详细准确**
   - 说明工具能做什么
   - 说明工具不能做什么
   - 说明什么时候应该用这个工具

4. **返回结果友好**
   - 返回结构化数据（JSON）
   - 包含状态信息（成功/失败）
   - 包含错误信息（如果失败）

5. **错误处理完善**
   - 参数验证
   - 异常捕获
   - 友好的错误提示

**工具设计的常见陷阱**

❌ **功能过于复杂**
```python
# 不好：一个工具做太多事
def process_data(action, data, format, options):
    if action == "parse":
        # 解析逻辑
    elif action == "transform":
        # 转换逻辑
    elif action == "validate":
        # 验证逻辑
```

✅ **功能单一**
```python
# 好：一个工具做一件事
def parse_data(data, format):
    # 只负责解析
    
def transform_data(data, options):
    # 只负责转换
    
def validate_data(data, rules):
    # 只负责验证
```

**工具描述的最佳实践**

好的工具描述应该回答三个问题：
1. **这个工具是什么？** - 功能说明
2. **什么时候用它？** - 使用场景
3. **怎么用它？** - 参数说明

示例：
```python
{
    "name": "search_web",
    "description": "搜索互联网获取实时信息。当用户询问最新事件、实时数据、或你不确定的信息时使用此工具。不要用于数学计算或常识问题。",
    "parameters": {
        "query": {
            "type": "string",
            "description": "搜索关键词，要具体明确。例如：'北京今日天气'而不是'天气'"
        }
    }
}
```

### 工具接口设计

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None

class BaseTool(ABC):
    """工具基类"""
    
    name: str = ""
    description: str = ""
    
    @abstractmethod
    def get_schema(self) -> dict:
        """返回工具的JSON Schema"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    def validate_args(self, **kwargs) -> bool:
        """验证参数"""
        return True

class CalculatorTool(BaseTool):
    """计算器工具"""
    
    name = "calculator"
    description = "执行数学计算"
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，如：2+3*4"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    
    def execute(self, expression: str) -> ToolResult:
        try:
            # 安全计算
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return ToolResult(
                    success=False,
                    data=None,
                    error="表达式包含非法字符"
                )
            
            result = eval(expression)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

class SearchTool(BaseTool):
    """搜索工具"""
    
    name = "search"
    description = "搜索互联网信息"
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        # 模拟搜索
        results = [
            f"结果{i+1}: 关于'{query}'的信息..."
            for i in range(num_results)
        ]
        return ToolResult(success=True, data=results)
```

### 错误处理

```python
class ToolExecutor:
    """工具执行器，带错误处理"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.error_handlers = []
    
    def register(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def add_error_handler(self, handler):
        """添加错误处理器"""
        self.error_handlers.append(handler)
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """执行工具"""
        # 检查工具是否存在
        if name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"工具 '{name}' 不存在"
            )
        
        tool = self.tools[name]
        
        try:
            # 验证参数
            if not tool.validate_args(**kwargs):
                return ToolResult(
                    success=False,
                    data=None,
                    error="参数验证失败"
                )
            
            # 执行
            result = tool.execute(**kwargs)
            
            # 记录日志
            self.log_execution(name, kwargs, result)
            
            return result
            
        except Exception as e:
            # 错误处理
            for handler in self.error_handlers:
                handler(name, kwargs, e)
            
            return ToolResult(
                success=False,
                data=None,
                error=f"执行错误: {str(e)}"
            )
    
    def log_execution(self, name, args, result):
        """记录执行日志"""
        print(f"[Tool] {name}({args}) -> {result}")

# 使用
executor = ToolExecutor()
executor.register(CalculatorTool())
executor.register(SearchTool())

executor.add_error_handler(
    lambda name, args, error: print(f"错误: {name} - {error}")
)

result = executor.execute("calculator", expression="2+3*4")
print(f"结果: {result.data}")
```

### 安全性考虑

```python
import re
from typing import List, Set

class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.blocked_patterns: List[str] = [
            r"import\s+os",
            r"import\s+subprocess",
            r"exec\s*\(",
            r"eval\s*\(",
            r"open\s*\(",
            r"__import__",
        ]
        self.allowed_domains: Set[str] = set()
        self.rate_limits: Dict[str, int] = {}
    
    def validate_input(self, input_str: str) -> bool:
        """验证输入是否安全"""
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_str):
                return False
        return True
    
    def sanitize_input(self, input_str: str) -> str:
        """清理输入"""
        # 移除危险字符
        sanitized = input_str
        for pattern in self.blocked_patterns:
            sanitized = re.sub(pattern, "", sanitized)
        return sanitized
    
    def check_rate_limit(self, tool_name: str, user_id: str) -> bool:
        """检查速率限制"""
        key = f"{user_id}:{tool_name}"
        # 实现速率限制逻辑
        return True
    
    def validate_url(self, url: str) -> bool:
        """验证URL是否允许访问"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain in self.allowed_domains or len(self.allowed_domains) == 0

class SafeToolExecutor(ToolExecutor):
    """安全的工具执行器"""
    
    def __init__(self):
        super().__init__()
        self.security = SecurityManager()
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """安全执行"""
        # 验证所有字符串参数
        for key, value in kwargs.items():
            if isinstance(value, str):
                if not self.security.validate_input(value):
                    return ToolResult(
                        success=False,
                        data=None,
                        error="输入包含不安全内容"
                    )
        
        return super().execute(name, **kwargs)
```

## 5.3 常用工具集成

**Agent常用的工具类型**

在实际应用中，Agent通常需要以下几类工具：

1. **信息获取工具**
   - 网络搜索：获取实时信息
   - 数据库查询：获取业务数据
   - API调用：获取第三方服务数据
   - 文件读取：读取本地文件

2. **数据处理工具**
   - 计算器：数学运算
   - 数据转换：格式转换
   - 文本处理：文本分析、提取

3. **外部操作工具**
   - 发送邮件/消息
   - 调用外部API
   - 执行系统命令（需要安全控制）
   - 修改文件/数据库

4. **专业领域工具**
   - 代码执行：运行代码片段
   - 图像处理：生成/分析图片
   - 数据分析：统计分析、可视化

**工具集成的关键考虑**

1. **安全性**
   - 验证所有输入参数
   - 限制危险操作
   - 设置访问权限
   - 记录操作日志

2. **可靠性**
   - 错误处理和重试机制
   - 超时控制
   - 结果验证
   - 降级方案

3. **性能**
   - 缓存常用结果
   - 异步执行
   - 批量处理
   - 资源限制

**工具选择的决策树**

```
需要什么类型的能力？
├─ 获取实时信息 → 搜索工具、API工具
├─ 访问业务数据 → 数据库工具
├─ 执行计算 → 计算器工具
├─ 处理文件 → 文件操作工具
└─ 调用外部服务 → API集成工具
```

### 搜索工具

```python
import requests
from typing import List, Dict

class WebSearchTool(BaseTool):
    """网络搜索工具"""
    
    name = "web_search"
    description = "搜索互联网获取信息"
    
    def __init__(self, api_key: str, engine: str = "google"):
        self.api_key = api_key
        self.engine = engine
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "num": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, num: int = 5) -> ToolResult:
        try:
            # 使用搜索API（这里以SerpAPI为例）
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            results = []
            for item in data.get("organic_results", [])[:num]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })
            
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
```

### 代码执行工具

```python
import subprocess
import tempfile
import os

class CodeExecutionTool(BaseTool):
    """代码执行工具"""
    
    name = "execute_code"
    description = "执行Python代码并返回结果"
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python代码"},
                        "language": {"type": "string", "default": "python"}
                    },
                    "required": ["code"]
                }
            }
        }
    
    def execute(self, code: str, language: str = "python") -> ToolResult:
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # 清理临时文件
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    data=result.stdout
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=result.stderr
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                data=None,
                error=f"执行超时（{self.timeout}秒）"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
```

### 文件操作工具

```python
import os
from pathlib import Path

class FileOperationTool(BaseTool):
    """文件操作工具"""
    
    name = "file_operation"
    description = "执行文件读写操作"
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir).resolve()
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["read", "write", "list", "delete"]
                        },
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["operation", "path"]
                }
            }
        }
    
    def _resolve_path(self, path: str) -> Path:
        """解析并验证路径"""
        full_path = (self.base_dir / path).resolve()
        
        # 确保路径在base_dir内
        if not str(full_path).startswith(str(self.base_dir)):
            raise ValueError("路径不在允许范围内")
        
        return full_path
    
    def execute(
        self,
        operation: str,
        path: str,
        content: str = None
    ) -> ToolResult:
        try:
            full_path = self._resolve_path(path)
            
            if operation == "read":
                if not full_path.exists():
                    return ToolResult(
                        success=False,
                        data=None,
                        error="文件不存在"
                    )
                return ToolResult(
                    success=True,
                    data=full_path.read_text()
                )
            
            elif operation == "write":
                if content is None:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="写入操作需要content参数"
                    )
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                return ToolResult(success=True, data="写入成功")
            
            elif operation == "list":
                if not full_path.exists():
                    return ToolResult(
                        success=False,
                        data=None,
                        error="目录不存在"
                    )
                items = [str(p.relative_to(self.base_dir)) for p in full_path.iterdir()]
                return ToolResult(success=True, data=items)
            
            elif operation == "delete":
                if full_path.exists():
                    full_path.unlink()
                    return ToolResult(success=True, data="删除成功")
                return ToolResult(
                    success=False,
                    data=None,
                    error="文件不存在"
                )
            
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"未知操作: {operation}"
                )
        
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
```

### API调用工具

```python
import requests
from typing import Dict, Any, Optional

class APICallTool(BaseTool):
    """API调用工具"""
    
    name = "api_call"
    description = "调用外部API"
    
    def __init__(self, allowed_hosts: list = None):
        self.allowed_hosts = allowed_hosts or []
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE"]
                        },
                        "url": {"type": "string"},
                        "headers": {"type": "object"},
                        "params": {"type": "object"},
                        "body": {"type": "object"}
                    },
                    "required": ["method", "url"]
                }
            }
        }
    
    def execute(
        self,
        method: str,
        url: str,
        headers: Dict = None,
        params: Dict = None,
        body: Dict = None
    ) -> ToolResult:
        try:
            # 检查是否允许访问
            from urllib.parse import urlparse
            host = urlparse(url).netloc
            
            if self.allowed_hosts and host not in self.allowed_hosts:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"不允许访问: {host}"
                )
            
            # 发送请求
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=body,
                timeout=30
            )
            
            return ToolResult(
                success=True,
                data={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                }
            )
        
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
```

## 5.4 【实战】多功能助手Agent

让我们构建一个集成多种工具的智能助手。

### 项目结构

```
multi-tool-agent/
├── .env
├── main.py
├── agent.py
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── calculator.py
│   ├── search.py
│   ├── weather.py
│   └── translator.py
└── requirements.txt
```

### 完整代码

**tools/base.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None

class BaseTool(ABC):
    name: str = ""
    description: str = ""
    
    @abstractmethod
    def get_schema(self) -> dict:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
```

**tools/calculator.py**

```python
from .base import BaseTool, ToolResult

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "执行数学计算"
    
    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    
    def execute(self, expression: str) -> ToolResult:
        try:
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return ToolResult(success=True, data=eval(expression))
            return ToolResult(success=False, data=None, error="非法字符")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
```

**tools/weather.py**

```python
from .base import BaseTool, ToolResult
import requests

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "获取城市天气信息"
    
    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    
    def execute(self, city: str) -> ToolResult:
        # 模拟天气API
        weather_data = {
            "北京": {"temp": 25, "condition": "晴"},
            "上海": {"temp": 28, "condition": "多云"},
            "广州": {"temp": 32, "condition": "晴"},
        }
        
        if city in weather_data:
            return ToolResult(success=True, data=weather_data[city])
        return ToolResult(success=False, data=None, error="城市未找到")
```

**tools/translator.py**

```python
from .base import BaseTool, ToolResult

class TranslatorTool(BaseTool):
    name = "translate"
    description = "翻译文本"
    
    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "要翻译的文本"
                        },
                        "target_lang": {
                            "type": "string",
                            "description": "目标语言",
                            "enum": ["en", "zh", "ja", "ko"]
                        }
                    },
                    "required": ["text", "target_lang"]
                }
            }
        }
    
    def execute(self, text: str, target_lang: str) -> ToolResult:
        # 模拟翻译
        translations = {
            ("你好", "en"): "Hello",
            ("Hello", "zh"): "你好",
            ("谢谢", "en"): "Thank you",
        }
        
        key = (text, target_lang)
        if key in translations:
            return ToolResult(success=True, data=translations[key])
        
        # 如果没有预设翻译，返回原文
        return ToolResult(
            success=True,
            data=f"[{target_lang}] {text}"
        )
```

**agent.py**

```python
import os
import json
from openai import OpenAI
from tools.calculator import CalculatorTool
from tools.weather import WeatherTool
from tools.translator import TranslatorTool

class MultiToolAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tools = {}
        self.tool_schemas = []
        
        # 注册工具
        self.register_tool(CalculatorTool())
        self.register_tool(WeatherTool())
        self.register_tool(TranslatorTool())
    
    def register_tool(self, tool):
        self.tools[tool.name] = tool
        self.tool_schemas.append(tool.get_schema())
    
    def execute_tool(self, name, args):
        if name in self.tools:
            return self.tools[name].execute(**args)
        return None
    
    def chat(self, user_message):
        messages = [
            {
                "role": "system",
                "content": "你是一个多功能助手，可以使用工具帮助用户。"
            },
            {"role": "user", "content": user_message}
        ]
        
        while True:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                return message.content
            
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                print(f"[工具调用] {name}({args})")
                
                result = self.execute_tool(name, args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.data) if result.success else result.error
                })

# 使用
from dotenv import load_dotenv
load_dotenv()

agent = MultiToolAgent()

print(agent.chat("北京今天天气怎么样？"))
print(agent.chat("计算 25 * 4 + 10"))
print(agent.chat("把'你好'翻译成英文"))
```

## 本章小结

本章我们学习了：

- ✅ Function Calling的工作原理
- ✅ 参数定义与验证方法
- ✅ 多函数调用处理
- ✅ 工具设计原则和错误处理
- ✅ 常用工具的实现
- ✅ 构建了一个多功能助手Agent

## 下一章

下一章我们将学习记忆系统与向量数据库。

[第6章：记忆系统与向量数据库 →](/core-skills/chapter6)
