# 附录C：代码仓库

## 教程配套代码仓库

### 主要仓库

**主仓库**
- GitHub：`https://github.com/luckyhlq/agent-dev-guide`
- 包含所有章节的完整代码示例

**章节代码仓库**
- 第1-3章：`agent-dev-guide-foundation`
- 第4-6章：`agent-dev-guide-core-skills`
- 第7-9章：`agent-dev-guide-frameworks`
- 第10-12章：`agent-dev-guide-advanced`
- 第13-16章：`agent-dev-guide-frontier`

## 项目模板

### 基础Agent模板

```bash
# 克隆模板
git clone https://github.com/luckyhlq/agent-template.git my-agent
cd my-agent

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，添加API密钥

# 运行
python main.py
```

### LangChain模板

```bash
# 克隆LangChain模板
git clone https://github.com/luckyhlq/langchain-agent-template.git my-langchain-agent
cd my-langchain-agent

# 安装依赖
pip install -r requirements.txt

# 配置
cp .env.example .env

# 运行
python app.py
```

### CrewAI模板

```bash
# 克隆CrewAI模板
git clone https://github.com/luckyhlq/crewai-template.git my-crew
cd my-crew

# 安装依赖
pip install -r requirements.txt

# 配置
cp .env.example .env

# 运行
python main.py
```

## 示例项目

### 完整示例项目列表

1. **智能客服助手**
   - 仓库：`customer-service-agent`
   - 技术栈：LangChain + OpenAI + Chroma
   - 功能：自动应答、知识检索、多轮对话

2. **代码审查Agent**
   - 仓库：`code-review-agent`
   - 技术栈：OpenAI + GitHub API
   - 功能：代码分析、问题检测、改进建议

3. **研究助手**
   - 仓库：`research-assistant`
   - 技术栈：LangChain + SerpAPI + Chroma
   - 功能：文献搜索、内容分析、报告生成

4. **数据分析Agent**
   - 仓库：`data-analysis-agent`
   - 技术栈：LangChain + Pandas + Matplotlib
   - 功能：数据查询、分析、可视化

5. **内容创作团队**
   - 仓库：`content-creation-crew`
   - 技术栈：CrewAI + OpenAI
   - 功能：多Agent协作、内容生成

### 获取示例项目

```bash
# 克隆所有示例项目
git clone https://github.com/luckyhlq/agent-examples.git
cd agent-examples

# 查看所有项目
ls

# 进入特定项目
cd customer-service-agent

# 查看README
cat README.md

# 安装依赖
pip install -r requirements.txt

# 运行项目
python main.py
```

## 贡献指南

### 如何贡献代码

1. **Fork仓库**
   ```bash
   # 在GitHub上fork仓库
   ```

2. **克隆你的fork**
   ```bash
   git clone https://github.com/luckyhlq/agent-dev-guide.git
   cd agent-dev-guide
   ```

3. **创建新分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **进行修改**
   ```bash
   # 编辑代码
   # 添加测试
   # 更新文档
   ```

5. **提交修改**
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

6. **推送到你的fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建Pull Request**
   - 在GitHub上创建PR
   - 描述你的修改
   - 等待审核

### 代码规范

**Python代码规范**
- 遵循PEP 8
- 使用类型注解
- 添加文档字符串
- 编写单元测试

**示例**
```python
from typing import List, Dict

def process_data(data: List[Dict]) -> Dict:
    """
    处理数据
    
    Args:
        data: 输入数据列表
        
    Returns:
        处理后的数据字典
    """
    result = {}
    for item in data:
        # 处理逻辑
        pass
    return result
```

## 版本历史

### 版本说明

**v1.0.0** (2024-01-15)
- 初始版本
- 包含16章完整内容
- 提供所有示例代码

**v1.1.0** (2024-02-01)
- 更新LangChain示例
- 添加CrewAI章节
- 修复已知问题

**v1.2.0** (2024-02-15)
- 新增自主Agent内容
- 添加更多实战项目
- 优化代码示例

### 获取更新

```bash
# 拉取最新更新
git pull origin main

# 查看版本历史
git log --oneline

# 查看特定版本的变更
git show v1.2.0
```

## 问题反馈

### 报告问题

如果你发现代码有问题，请：

1. 在GitHub上创建Issue
2. 提供详细的错误信息
3. 包含复现步骤
4. 附上相关代码片段

### Issue模板

```markdown
## 问题描述
简要描述遇到的问题

## 复现步骤
1. 步骤1
2. 步骤2
3. 步骤3

## 期望行为
描述期望的行为

## 实际行为
描述实际发生的行为

## 环境信息
- Python版本：
- 操作系统：
- 相关库版本：

## 代码片段
```python
# 相关代码
```

## 其他信息
其他相关信息
```

## 许可证

### 代码许可证

所有示例代码使用MIT许可证：

```
MIT License

Copyright (c) 2024 Agent Development Guide

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 相关资源

### 官方仓库

- [LangChain](https://github.com/langchain-ai/langchain)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [OpenAI Python](https://github.com/openai/openai-python)
- [Chroma](https://github.com/chroma-core/chroma)

### 社区项目

- [Awesome AI Agents](https://github.com/e2b-dev/awesome-ai-agents)
- [Agent Examples](https://github.com/Significant-Gravitas/Auto-GPT-Examples)
- [Prompt Engineering](https://github.com/dair-ai/Prompt-Engineering-Guide)

## 联系方式

### 获取帮助

- **GitHub Issues**：报告问题和请求功能
- **Discord**：加入社区讨论
- **Email**：contact@agent-dev-guide.com

### 商业合作

如需商业合作或企业培训，请联系：
- Email：business@agent-dev-guide.com
- 微信：agent-dev-guide
