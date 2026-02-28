# 第16章：综合实战项目

本章将通过三个完整的项目，综合运用前面所学知识。

## 16.1 项目一：智能研究助手

### 需求分析

**功能需求**
- 自动搜索和收集研究资料
- 整理和分析文献
- 生成研究报告
- 支持多轮对话和追问

**技术栈**
- LangChain框架
- OpenAI API
- 向量数据库（Chroma）
- Web搜索工具

### 架构设计

```
智能研究助手
├── 搜索模块
├── 文献分析模块
├── 知识管理模块
└── 报告生成模块
```

### 完整实现

**项目结构**

```
research-assistant/
├── .env
├── main.py
├── modules/
│   ├── __init__.py
│   ├── search.py
│   ├── analyzer.py
│   ├── knowledge.py
│   └── reporter.py
├── config.py
└── requirements.txt
```

**config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL = "gpt-3.5-turbo"
    MAX_SEARCH_RESULTS = 5
    KNOWLEDGE_BASE_PATH = "./knowledge_base"
    REPORT_OUTPUT_PATH = "./reports"
```

**modules/search.py**

```python
from langchain.tools import Tool
from typing import List, Dict
import requests

class WebSearchTool:
    """网络搜索工具"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """搜索"""
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
        
        return results
    
    def create_tool(self) -> Tool:
        """创建LangChain工具"""
        return Tool(
            name="web_search",
            func=lambda q: str(self.search(q)),
            description="搜索互联网信息"
        )
```

**modules/analyzer.py**

```python
from openai import OpenAI
from typing import List, Dict

class LiteratureAnalyzer:
    """文献分析器"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def analyze_paper(self, paper: Dict) -> Dict:
        """分析单篇文献"""
        prompt = f"""
请分析以下文献：

标题：{paper['title']}
摘要：{paper.get('snippet', '')}

提取：
1. 研究主题
2. 关键发现
3. 研究方法
4. 创新点

以JSON格式输出。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"error": "分析失败"}
    
    def analyze_collection(self, papers: List[Dict]) -> Dict:
        """分析文献集合"""
        analyses = []
        
        for paper in papers:
            analysis = self.analyze_paper(paper)
            analyses.append(analysis)
        
        # 综合分析
        themes = self._extract_themes(analyses)
        trends = self._identify_trends(analyses)
        gaps = self._find_gaps(analyses)
        
        return {
            "total_papers": len(papers),
            "themes": themes,
            "trends": trends,
            "gaps": gaps,
            "individual_analyses": analyses
        }
    
    def _extract_themes(self, analyses: List[Dict]) -> List[str]:
        """提取研究主题"""
        themes = set()
        
        for analysis in analyses:
            if "研究主题" in analysis:
                themes.add(analysis["研究主题"])
        
        return list(themes)
    
    def _identify_trends(self, analyses: List[Dict]) -> List[str]:
        """识别趋势"""
        return ["趋势1", "趋势2"]
    
    def _find_gaps(self, analyses: List[Dict]) -> List[str]:
        """发现研究空白"""
        return ["空白1", "空白2"]
```

**modules/knowledge.py**

```python
import chromadb
from openai import OpenAI
from typing import List, Dict

class KnowledgeManager:
    """知识管理器"""
    
    def __init__(self, persist_dir: str, api_key: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.openai = OpenAI(api_key=api_key)
        self.collection = None
        self._init_collection()
    
    def _init_collection(self):
        """初始化集合"""
        self.collection = self.client.get_or_create_collection(
            name="research_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_papers(self, papers: List[Dict]):
        """添加文献"""
        ids = []
        documents = []
        metadatas = []
        
        for i, paper in enumerate(papers):
            ids.append(f"paper_{i}")
            documents.append(paper.get("snippet", ""))
            metadatas.append({
                "title": paper.get("title", ""),
                "link": paper.get("link", "")
            })
        
        embeddings = self._get_embeddings(documents)
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """搜索知识库"""
        query_embedding = self._get_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return [
            {
                "title": results["metadatas"][0][i]["title"],
                "link": results["metadatas"][0][i]["link"],
                "content": results["documents"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取嵌入"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [d.embedding for d in response.data]
```

**modules/reporter.py**

```python
from openai import OpenAI
from typing import Dict

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_report(self, analysis: Dict, topic: str) -> str:
        """生成研究报告"""
        prompt = f"""
基于以下分析结果，生成一份关于"{topic}"的研究报告：

分析结果：
{analysis}

报告应包括：
1. 研究背景
2. 主要发现
3. 研究趋势
4. 研究空白
5. 未来方向
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def save_report(self, report: str, filepath: str):
        """保存报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
```

**main.py**

```python
from dotenv import load_dotenv
from config import Config
from modules.search import WebSearchTool
from modules.analyzer import LiteratureAnalyzer
from modules.knowledge import KnowledgeManager
from modules.reporter import ReportGenerator

load_dotenv()

class ResearchAssistant:
    """智能研究助手"""
    
    def __init__(self):
        self.config = Config()
        self.search_tool = WebSearchTool(self.config.OPENAI_API_KEY)
        self.analyzer = LiteratureAnalyzer(self.config.OPENAI_API_KEY)
        self.knowledge = KnowledgeManager(
            self.config.KNOWLEDGE_BASE_PATH,
            self.config.OPENAI_API_KEY
        )
        self.reporter = ReportGenerator(self.config.OPENAI_API_KEY)
    
    def research(self, topic: str) -> Dict:
        """执行研究"""
        print(f"\n开始研究：{topic}\n")
        
        # 搜索文献
        print("1. 搜索文献...")
        papers = self.search_tool.search(topic, self.config.MAX_SEARCH_RESULTS)
        print(f"   找到 {len(papers)} 篇文献")
        
        # 分析文献
        print("2. 分析文献...")
        analysis = self.analyzer.analyze_collection(papers)
        print(f"   识别出 {len(analysis['themes'])} 个研究主题")
        
        # 添加到知识库
        print("3. 存储到知识库...")
        self.knowledge.add_papers(papers)
        print("   已存储")
        
        # 生成报告
        print("4. 生成报告...")
        report = self.reporter.generate_report(analysis, topic)
        
        # 保存报告
        import os
        os.makedirs(self.config.REPORT_OUTPUT_PATH, exist_ok=True)
        report_path = f"{self.config.REPORT_OUTPUT_PATH}/{topic.replace(' ', '_')}.md"
        self.reporter.save_report(report, report_path)
        print(f"   报告已保存到：{report_path}")
        
        return {
            "topic": topic,
            "papers": len(papers),
            "themes": analysis["themes"],
            "report_path": report_path
        }

def main():
    assistant = ResearchAssistant()
    
    print("=" * 60)
    print("智能研究助手")
    print("=" * 60)
    
    while True:
        topic = input("\n请输入研究主题（输入 'quit' 退出）：").strip()
        
        if topic.lower() == 'quit':
            print("再见！")
            break
        
        if not topic:
            continue
        
        result = assistant.research(topic)
        
        print(f"\n研究完成！")
        print(f"主题：{result['topic']}")
        print(f"文献数：{result['papers']}")
        print(f"研究主题：{', '.join(result['themes'])}")

if __name__ == "__main__":
    main()
```

## 16.2 项目二：自动化测试Agent

### 需求分析

**功能需求**
- 自动生成测试用例
- 执行测试并收集结果
- 生成测试报告
- Bug定位和修复建议

### 完整实现

**项目结构**

```
test-automation-agent/
├── .env
├── main.py
├── modules/
│   ├── __init__.py
│   ├── generator.py
│   ├── executor.py
│   └── reporter.py
└── requirements.txt
```

**modules/generator.py**

```python
from openai import OpenAI
from typing import List, Dict

class TestGenerator:
    """测试用例生成器"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_test_cases(self, code: str, requirements: str) -> List[Dict]:
        """生成测试用例"""
        prompt = f"""
为以下代码生成测试用例：

需求：
{requirements}

代码：
```python
{code}
```

要求：
1. 覆盖正常和异常情况
2. 包含边界条件
3. 提供测试数据和预期结果

以JSON格式输出测试用例列表。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return []
```

**modules/executor.py**

```python
import subprocess
import tempfile
import os
from typing import Dict, List

class TestExecutor:
    """测试执行器"""
    
    def execute_tests(self, code: str, test_cases: List[Dict]) -> List[Dict]:
        """执行测试"""
        results = []
        
        for test_case in test_cases:
            result = self._run_test(code, test_case)
            results.append(result)
        
        return results
    
    def _run_test(self, code: str, test_case: Dict) -> Dict:
        """运行单个测试"""
        # 创建测试脚本
        test_script = self._create_test_script(code, test_case)
        
        # 执行测试
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_file = f.name
            
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            success = result.returncode == 0
            
            return {
                "test_name": test_case.get("name", "test"),
                "success": success,
                "output": result.stdout,
                "error": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_name": test_case.get("name", "test"),
                "success": False,
                "error": "测试超时"
            }
        except Exception as e:
            return {
                "test_name": test_case.get("name", "test"),
                "success": False,
                "error": str(e)
            }
        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)
    
    def _create_test_script(self, code: str, test_case: Dict) -> str:
        """创建测试脚本"""
        script = f"""
{code}

# 测试代码
try:
    {test_case.get('test_code', '')}
    print("PASSED")
except Exception as e:
    print(f"FAILED: {{e}}")
"""
        return script
```

**modules/reporter.py**

```python
from openai import OpenAI
from typing import List, Dict

class TestReporter:
    """测试报告生成器"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_report(
        self,
        test_results: List[Dict],
        code: str
    ) -> str:
        """生成测试报告"""
        passed = sum(1 for r in test_results if r["success"])
        total = len(test_results)
        
        report = f"""
# 测试报告

## 测试概览
- 总测试数：{total}
- 通过：{passed}
- 失败：{total - passed}
- 通过率：{passed/total:.1%}

## 测试详情
"""
        
        for result in test_results:
            status = "✅ 通过" if result["success"] else "❌ 失败"
            report += f"\n### {result['test_name']}\n"
            report += f"状态：{status}\n"
            if not result["success"]:
                report += f"错误：{result.get('error', '未知错误')}\n"
        
        return report
```

## 16.3 项目三：个人知识管理Agent

### 需求分析

**功能需求**
- 知识采集（网页、文档、笔记）
- 自动整理和分类
- 智能检索
- 知识图谱构建

### 完整实现

**项目结构**

```
knowledge-manager/
├── .env
├── main.py
├── modules/
│   ├── __init__.py
│   ├── collector.py
│   ├── organizer.py
│   ├── retriever.py
│   └── graph.py
└── requirements.txt
```

**modules/collector.py**

```python
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

class KnowledgeCollector:
    """知识采集器"""
    
    def collect_from_url(self, url: str) -> Dict:
        """从URL采集"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取标题和正文
            title = soup.title.string if soup.title else ""
            content = soup.get_text()
            
            return {
                "source": url,
                "title": title,
                "content": content,
                "type": "webpage"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def collect_from_file(self, filepath: str) -> Dict:
        """从文件采集"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "source": filepath,
                "title": filepath.split('/')[-1],
                "content": content,
                "type": "file"
            }
        except Exception as e:
            return {"error": str(e)}
```

**modules/organizer.py**

```python
from openai import OpenAI
from typing import List, Dict

class KnowledgeOrganizer:
    """知识整理器"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def classify(self, knowledge: Dict) -> List[str]:
        """分类知识"""
        prompt = f"""
为以下知识内容分类：

标题：{knowledge['title']}
内容：{knowledge['content'][:500]}...

返回3-5个分类标签，以逗号分隔。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        tags = response.choices[0].message.content.strip()
        return [tag.strip() for tag in tags.split(',')]
    
    def summarize(self, knowledge: Dict) -> str:
        """摘要"""
        prompt = f"""
为以下知识生成摘要（100字以内）：

标题：{knowledge['title']}
内容：{knowledge['content']}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()
```

## 16.4 课程总结与进阶路线

### 知识体系回顾

**第一阶段：基础入门**
- Agent核心概念和架构
- LLM基础和API使用
- Prompt Engineering基础

**第二阶段：核心技能**
- 高级Prompt技术
- Function Calling和工具使用
- 记忆系统和向量数据库

**第三阶段：框架应用**
- LangChain框架
- CrewAI框架
- 其他主流框架

**第四阶段：高级应用**
- Agent架构设计
- 多Agent协作
- 生产部署

**第五阶段：前沿探索**
- Agent评估与优化
- 安全与伦理
- 自主Agent

### 能力提升建议

**初级阶段**
- 掌握基础API调用
- 理解Prompt设计原则
- 完成简单Agent项目

**中级阶段**
- 熟练使用主流框架
- 设计复杂Agent系统
- 实现多Agent协作

**高级阶段**
- 优化Agent性能
- 处理生产环境问题
- 探索前沿技术

### 持续学习资源

**官方文档**
- OpenAI API文档
- LangChain文档
- CrewAI文档

**社区资源**
- GitHub开源项目
- Stack Overflow
- Reddit r/AgentAI

**学习路径**
1. 实践项目
2. 参与开源
3. 阅读论文
4. 参加会议

## 本章小结

本章我们完成了三个综合实战项目：

- ✅ 智能研究助手
- ✅ 自动化测试Agent
- ✅ 个人知识管理Agent

恭喜你完成了从入门到精通的学习旅程！

---

**教程结束**

感谢你的学习！希望这份教程能帮助你掌握Agent开发的核心技能。继续探索和实践，你将成为Agent开发专家！
