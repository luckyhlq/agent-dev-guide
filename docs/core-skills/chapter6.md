# 第6章：记忆系统与向量数据库

本章将深入讲解Agent的记忆系统设计，以及如何使用向量数据库实现长期记忆和知识检索。

## 6.1 Agent记忆系统设计

**为什么Agent需要记忆？**

想象你在和一个朋友聊天：
- **没有记忆**：每次对话他都像第一次见你，完全不记得之前说过什么
- **有记忆**：他能记住你的喜好、之前的约定，对话越来越顺畅

Agent也是一样。没有记忆的Agent：
- 无法记住用户的偏好
- 无法利用之前的对话上下文
- 无法积累知识和经验
- 每次都从零开始

**两种记忆类型**

就像人脑有短期记忆和长期记忆，Agent也需要两种记忆：

1. **短期记忆（工作记忆）**
   - **类比**：就像你打电话时记住对方说的话
   - **作用**：保持当前对话的上下文连贯性
   - **特点**：容量有限（受Token限制），会话结束后消失
   - **实现**：对话历史列表、滑动窗口

2. **长期记忆（知识库）**
   - **类比**：就像你记在笔记本上的知识
   - **作用**：存储持久化的知识、用户偏好、历史经验
   - **特点**：容量大，永久保存，跨会话使用
   - **实现**：向量数据库、知识图谱

**记忆系统的核心挑战**

| 挑战 | 短期记忆 | 长期记忆 |
|------|---------|---------|
| 容量限制 | Token数量有限 | 存储空间有限 |
| 检索效率 | 顺序访问慢 | 需要快速检索相关记忆 |
| 相关性判断 | 哪些对话值得保留 | 哪些记忆与当前问题相关 |
| 更新策略 | 如何淘汰旧记忆 | 如何更新过时信息 |

**记忆管理策略**

1. **短期记忆管理**
   - 滑动窗口：只保留最近N轮对话
   - 摘要压缩：定期总结旧对话，节省Token
   - 重要性过滤：只保留关键信息

2. **长期记忆管理**
   - 向量化存储：将文本转换为向量，便于语义检索
   - 分块存储：将长文档切分成小块
   - 索引优化：建立高效的检索索引

### 短期记忆 vs 长期记忆

Agent的记忆系统分为两个层次：

| 特性 | 短期记忆 | 长期记忆 |
|------|----------|----------|
| 存储内容 | 当前对话历史 | 持久化知识和经验 |
| 容量 | 有限（受Token限制） | 大容量 |
| 持久性 | 会话级别 | 跨会话持久化 |
| 检索方式 | 顺序访问 | 语义检索 |
| 实现方式 | 列表/队列 | 向量数据库 |

### 记忆存储策略

```python
from typing import List, Dict, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Memory:
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    importance: float = 0.5

class ShortTermMemory:
    """短期记忆管理"""
    
    def __init__(self, max_tokens: int = 4000):
        self.memories: deque = deque()
        self.max_tokens = max_tokens
        self.current_tokens = 0
    
    def add(self, content: str, metadata: Dict = None):
        """添加记忆"""
        tokens = len(content.split())  # 简化计算
        
        # 如果超出限制，移除旧记忆
        while self.current_tokens + tokens > self.max_tokens and self.memories:
            removed = self.memories.popleft()
            self.current_tokens -= len(removed.content.split())
        
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.memories.append(memory)
        self.current_tokens += tokens
    
    def get_all(self) -> List[Memory]:
        """获取所有记忆"""
        return list(self.memories)
    
    def get_recent(self, n: int = 5) -> List[Memory]:
        """获取最近n条记忆"""
        return list(self.memories)[-n:]
    
    def clear(self):
        """清空记忆"""
        self.memories.clear()
        self.current_tokens = 0
    
    def to_messages(self) -> List[Dict]:
        """转换为消息格式"""
        messages = []
        for memory in self.memories:
            if memory.metadata and 'role' in memory.metadata:
                messages.append({
                    "role": memory.metadata['role'],
                    "content": memory.content
                })
        return messages

class LongTermMemory:
    """长期记忆管理"""
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend
        self.memories: List[Memory] = []
    
    def store(self, content: str, metadata: Dict = None, importance: float = 0.5):
        """存储长期记忆"""
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance=importance
        )
        
        self.memories.append(memory)
        
        if self.storage:
            self.storage.save(memory)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Memory]:
        """检索相关记忆"""
        if self.storage:
            return self.storage.search(query, top_k)
        
        # 简单的关键词匹配
        results = []
        for memory in self.memories:
            if query.lower() in memory.content.lower():
                results.append(memory)
        
        return results[:top_k]
    
    def get_important(self, threshold: float = 0.7) -> List[Memory]:
        """获取重要记忆"""
        return [m for m in self.memories if m.importance >= threshold]
```

### 记忆检索与召回

```python
from typing import Tuple
import math

class MemoryRetriever:
    """记忆检索器"""
    
    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory):
        self.short_term = short_term
        self.long_term = long_term
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Tuple[Memory, float]]:
        """检索相关记忆"""
        results = []
        
        # 从长期记忆检索
        long_term_results = self.long_term.retrieve(query, top_k)
        for memory in long_term_results:
            score = self.calculate_relevance(query, memory.content)
            results.append((memory, score))
        
        # 从短期记忆检索
        for memory in self.short_term.get_all():
            score = self.calculate_relevance(query, memory.content)
            if score > 0.3:  # 相关性阈值
                results.append((memory, score))
        
        # 按相关性排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def calculate_relevance(self, query: str, content: str) -> float:
        """计算相关性分数（简化版）"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words & content_words
        return len(intersection) / len(query_words)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """获取上下文"""
        memories = self.retrieve_relevant(query)
        
        context_parts = []
        current_tokens = 0
        
        for memory, score in memories:
            tokens = len(memory.content.split())
            if current_tokens + tokens > max_tokens:
                break
            
            context_parts.append(f"[{score:.2f}] {memory.content}")
            current_tokens += tokens
        
        return "\n".join(context_parts)

class MemoryManager:
    """记忆管理器"""
    
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.retriever = MemoryRetriever(self.short_term, self.long_term)
    
    def remember(self, content: str, role: str = "user", important: bool = False):
        """记录记忆"""
        # 添加到短期记忆
        self.short_term.add(content, {"role": role})
        
        # 如果重要，也存储到长期记忆
        if important:
            self.long_term.store(
                content,
                {"role": role},
                importance=0.8 if important else 0.5
            )
    
    def recall(self, query: str) -> str:
        """回忆相关内容"""
        return self.retriever.get_context(query)
    
    def forget_short_term(self):
        """遗忘短期记忆"""
        self.short_term.clear()
    
    def consolidate(self):
        """巩固记忆：将重要的短期记忆转为长期记忆"""
        recent = self.short_term.get_recent(10)
        
        for memory in recent:
            if memory.importance > 0.6:
                self.long_term.store(
                    memory.content,
                    memory.metadata,
                    memory.importance
                )
```

## 6.2 向量数据库基础

**什么是向量数据库？**

向量数据库是专门存储和检索向量（一串数字）的数据库。要理解它，先要理解"向量"是什么。

**从文本到向量**

计算机不直接理解文本，需要先把文本转换成数字。这个过程叫"向量化"或"嵌入"（Embedding）：

```
文本："苹果是一种水果"
↓ （通过Embedding模型）
向量：[0.2, -0.5, 0.8, 0.1, ..., 0.3]  # 通常是几百到几千维
```

**为什么需要向量？**

向量能表示语义相似性：
- 相似的文本 → 相似的向量
- 不同的文本 → 不同的向量

例如：
```
"我喜欢吃苹果" → [0.2, -0.5, 0.8, ...]
"我爱吃水果"   → [0.3, -0.4, 0.7, ...]  # 很相似
"今天下雨了"   → [-0.6, 0.9, -0.2, ...] # 很不同
```

**向量数据库的核心能力**

1. **相似性搜索**
   - 输入：一个查询向量
   - 输出：最相似的N个向量
   - 应用：找到语义相关的文档

2. **高效检索**
   - 使用特殊的索引结构（如HNSW、IVF）
   - 在百万级数据中也能毫秒级响应

**向量数据库 vs 传统数据库**

| 特性 | 传统数据库 | 向量数据库 |
|------|-----------|-----------|
| 查询方式 | 精确匹配 | 相似性匹配 |
| 查询语句 | SQL | 向量 |
| 适用场景 | 结构化数据 | 非结构化数据（文本、图片） |
| 示例 | "查找名字叫张三的用户" | "查找与这个问题最相关的文档" |

**主流向量数据库**

1. **Pinecone**：托管式，简单易用，适合快速开始
2. **Milvus**：开源，功能强大，适合大规模生产
3. **Chroma**：轻量级，适合本地开发和小项目
4. **Weaviate**：开源，支持混合搜索
5. **Qdrant**：高性能，Rust实现

**如何选择向量数据库？**

- **学习/原型**：Chroma（最简单）
- **生产环境**：Pinecone（托管）或 Milvus（自建）
- **大规模数据**：Milvus、Qdrant
- **需要混合搜索**：Weaviate

### 向量嵌入原理

```python
from openai import OpenAI
import numpy as np
from typing import List

class EmbeddingService:
    """向量嵌入服务"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
    
    def embed(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [d.embedding for d in response.data]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    return dot_product / (norm_a * norm_b)

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """计算欧几里得距离"""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.linalg.norm(a_np - b_np)

# 使用
embedding_service = EmbeddingService()

text1 = "Python是一门编程语言"
text2 = "Java是一门编程语言"
text3 = "今天天气很好"

emb1 = embedding_service.embed(text1)
emb2 = embedding_service.embed(text2)
emb3 = embedding_service.embed(text3)

print(f"文本1和文本2相似度: {cosine_similarity(emb1, emb2):.3f}")
print(f"文本1和文本3相似度: {cosine_similarity(emb1, emb3):.3f}")
```

### 主流向量数据库

**Chroma（轻量级本地数据库）**

```python
import chromadb
from chromadb.config import Settings

class ChromaVectorStore:
    """Chroma向量存储"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.embedding_service = EmbeddingService()
    
    def create_collection(self, name: str):
        """创建集合"""
        self.collection = self.client.get_or_create_collection(name=name)
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[dict] = None,
        ids: List[str] = None
    ):
        """添加文档"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        embeddings = self.embedding_service.embed_batch(documents)
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, n_results: int = 5) -> List[dict]:
        """搜索相似文档"""
        query_embedding = self.embedding_service.embed(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                "distance": results["distances"][0][i] if results["distances"] else None
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def delete(self, ids: List[str]):
        """删除文档"""
        self.collection.delete(ids=ids)

# 使用
store = ChromaVectorStore()
store.create_collection("knowledge_base")

store.add_documents(
    documents=[
        "Python是一种解释型编程语言",
        "机器学习是人工智能的一个分支",
        "向量数据库用于存储和检索向量数据"
    ],
    metadatas=[
        {"category": "programming"},
        {"category": "ai"},
        {"category": "database"}
    ]
)

results = store.search("什么是Python")
for r in results:
    print(f"文档: {r['document']}")
    print(f"距离: {r['distance']:.3f}\n")
```

**Pinecone（云端向量数据库）**

```python
import pinecone

class PineconeVectorStore:
    """Pinecone向量存储"""
    
    def __init__(self, api_key: str, environment: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = None
        self.embedding_service = EmbeddingService()
    
    def create_index(self, name: str, dimension: int = 1536):
        """创建索引"""
        if name not in pinecone.list_indexes():
            pinecone.create_index(
                name=name,
                dimension=dimension,
                metric="cosine"
            )
        self.index = pinecone.Index(name)
    
    def upsert(
        self,
        vectors: List[tuple],  # (id, embedding, metadata)
        batch_size: int = 100
    ):
        """批量插入向量"""
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dict: dict = None
    ):
        """查询相似向量"""
        query_embedding = self.embedding_service.embed(query_text)
        
        return self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
    
    def delete_index(self, name: str):
        """删除索引"""
        pinecone.delete_index(name)
```

### 相似度计算

```python
from typing import List, Tuple

class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        """余弦相似度"""
        return cosine_similarity(a, b)
    
    @staticmethod
    def dot_product(a: List[float], b: List[float]) -> float:
        """点积"""
        return np.dot(np.array(a), np.array(b))
    
    @staticmethod
    def euclidean(a: List[float], b: List[float]) -> float:
        """欧几里得距离（转换为相似度）"""
        dist = euclidean_distance(a, b)
        return 1 / (1 + dist)
    
    @staticmethod
    def manhattan(a: List[float], b: List[float]) -> float:
        """曼哈顿距离（转换为相似度）"""
        dist = np.sum(np.abs(np.array(a) - np.array(b)))
        return 1 / (1 + dist)

def find_most_similar(
    query_embedding: List[float],
    embeddings: List[List[float]],
    top_k: int = 5,
    metric: str = "cosine"
) -> List[Tuple[int, float]]:
    """找到最相似的向量"""
    calculator = SimilarityCalculator()
    method = getattr(calculator, metric)
    
    scores = []
    for i, emb in enumerate(embeddings):
        score = method(query_embedding, emb)
        scores.append((i, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

## 6.3 RAG（检索增强生成）

**什么是RAG？**

RAG（Retrieval-Augmented Generation，检索增强生成）是让AI"查资料后再回答"的技术。

打个比方：
- **没有RAG**：就像考试时不让看书，只能凭记忆答题，容易出错或不知道
- **有了RAG**：就像开卷考试，可以先查阅相关资料，再给出答案

**RAG的工作流程**

```
用户提问
   ↓
1. 检索：在知识库中找到相关文档
   ↓
2. 组装：将问题和相关文档组合成Prompt
   ↓
3. 生成：让LLM基于文档生成答案
   ↓
最终答案
```

**为什么需要RAG？**

大语言模型有几个固有问题：
1. **知识过时**：训练数据有截止日期，不知道最新信息
2. **领域知识不足**：对特定领域的专业知识了解有限
3. **会"编造"**：不知道答案时可能编造看起来合理但错误的信息（幻觉）
4. **无法访问私有数据**：无法访问企业内部文档、数据库

RAG完美解决这些问题：
- ✅ 知识实时：可以检索最新的文档
- ✅ 专业准确：基于专业文档回答
- ✅ 减少幻觉：基于真实文档生成答案
- ✅ 私有数据：可以检索企业内部知识库

**RAG vs 微调 vs 长上下文**

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| RAG | 实时更新、成本低、可解释 | 需要检索系统 | 知识库问答、文档检索 |
| 微调 | 性能好、领域适应性强 | 成本高、知识难更新 | 特定任务优化 |
| 长上下文 | 简单直接 | 成本高、有长度限制 | 文档不太长的场景 |

**RAG的核心组件**

1. **文档处理**
   - 文档加载：PDF、Word、网页等
   - 文本分割：将长文档切成小块
   - 向量化：将文本转换为向量

2. **向量存储**
   - 存储文档向量和原文
   - 支持相似性搜索

3. **检索系统**
   - 根据问题找到相关文档
   - 支持多种检索策略

4. **生成系统**
   - 组装Prompt（问题+检索到的文档）
   - 调用LLM生成答案

**RAG的优化方向**

1. **检索优化**
   - 混合检索：向量检索 + 关键词检索
   - 重排序：对检索结果重新排序
   - 查询改写：优化用户问题

2. **生成优化**
   - Prompt工程：设计更好的提示词
   - 引用标注：标注答案来源
   - 答案验证：检查答案是否基于文档

### RAG架构设计

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    document: Document
    score: float

class RAGSystem:
    """RAG系统"""
    
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm = llm_client
        self.embedding_service = EmbeddingService()
    
    def index_documents(self, documents: List[Document]):
        """索引文档"""
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_service.embed_batch(contents)
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        self.vector_store.add_documents(
            documents=[d.content for d in documents],
            metadatas=[d.metadata for d in documents],
            ids=[d.id for d in documents]
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """检索相关文档"""
        results = self.vector_store.search(query, top_k)
        
        return [
            SearchResult(
                document=Document(
                    id=r["id"],
                    content=r["document"],
                    metadata=r.get("metadata", {})
                ),
                score=1 - r.get("distance", 0)
            )
            for r in results
        ]
    
    def augment_prompt(
        self,
        query: str,
        search_results: List[SearchResult],
        max_context_tokens: int = 2000
    ) -> str:
        """增强Prompt"""
        context_parts = []
        current_tokens = 0
        
        for result in search_results:
            tokens = len(result.document.content.split())
            if current_tokens + tokens > max_context_tokens:
                break
            
            context_parts.append(result.document.content)
            current_tokens += tokens
        
        context = "\n\n".join(context_parts)
        
        return f"""
基于以下上下文回答问题：

上下文：
{context}

问题：{query}

请基于上下文回答问题，如果上下文中没有相关信息，请说明。
"""
    
    def generate(self, query: str, top_k: int = 5) -> str:
        """生成回答"""
        # 检索
        results = self.retrieve(query, top_k)
        
        # 增强Prompt
        augmented_prompt = self.augment_prompt(query, results)
        
        # 生成
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": augmented_prompt}]
        )
        
        return response.choices[0].message.content
```

### 文档切分策略

```python
from typing import List
import re

class TextSplitter:
    """文本切分器"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """切分文本"""
        chunks = []
        self._split_recursive(text, self.separators, chunks)
        return chunks
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        chunks: List[str]
    ):
        """递归切分"""
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        
        separator = separators[0] if separators else ""
        
        if separator:
            parts = text.split(separator)
        else:
            parts = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                current_chunk += (separator if current_chunk else "") + part
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk.strip())

class SemanticSplitter:
    """语义切分器"""
    
    def __init__(self, embedding_service, similarity_threshold: float = 0.7):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
    
    def split(self, text: str, min_chunk_size: int = 100) -> List[str]:
        """按语义切分"""
        # 先按段落切分
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = paragraphs[0] if paragraphs else ""
        
        for para in paragraphs[1:]:
            # 计算当前段落和当前chunk的相似度
            emb1 = self.embedding_service.embed(current_chunk)
            emb2 = self.embedding_service.embed(para)
            similarity = cosine_similarity(emb1, emb2)
            
            if similarity >= self.similarity_threshold:
                # 相似度高，合并
                current_chunk += "\n\n" + para
            else:
                # 相似度低，开始新chunk
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

# 使用
text = """
Python是一种广泛使用的高级编程语言。它由Guido van Rossum于1991年首次发布。

Python的设计哲学强调代码的可读性和简洁性。它的语法允许程序员用更少的代码行表达概念。

Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。它具有动态类型系统和自动内存管理功能。
"""

splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk[:50]}...")
```

### 检索优化技巧

```python
from typing import List, Dict
import re

class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, vector_store, keyword_weight: float = 0.3):
        self.vector_store = vector_store
        self.keyword_weight = keyword_weight
        self.embedding_service = EmbeddingService()
    
    def keyword_search(self, query: str, documents: List[str]) -> List[float]:
        """关键词搜索"""
        query_terms = set(query.lower().split())
        scores = []
        
        for doc in documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append(overlap / len(query_terms) if query_terms else 0)
        
        return scores
    
    def hybrid_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """混合搜索"""
        # 向量搜索
        query_embedding = self.embedding_service.embed(query)
        vector_scores = [
            cosine_similarity(query_embedding, self.embedding_service.embed(doc))
            for doc in documents
        ]
        
        # 关键词搜索
        keyword_scores = self.keyword_search(query, documents)
        
        # 混合分数
        hybrid_scores = [
            (1 - self.keyword_weight) * v + self.keyword_weight * k
            for v, k in zip(vector_scores, keyword_scores)
        ]
        
        # 排序
        ranked = sorted(
            enumerate(documents),
            key=lambda x: hybrid_scores[x[0]],
            reverse=True
        )[:top_k]
        
        return [
            {
                "index": idx,
                "document": doc,
                "score": hybrid_scores[idx]
            }
            for idx, doc in ranked
        ]

class Reranker:
    """重排序器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """使用LLM重排序"""
        prompt = f"""
请对以下文档与查询的相关性进行评分（0-10分）：

查询：{query}

文档：
{chr(10).join(f'{i+1}. {doc[:200]}...' for i, doc in enumerate(documents))}

请输出JSON格式的评分列表：
[{{"index": 1, "score": 8}}, ...]
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            import json
            scores = json.loads(response.choices[0].message.content)
            
            ranked = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
            
            return [
                {
                    "index": r["index"] - 1,
                    "document": documents[r["index"] - 1],
                    "score": r["score"]
                }
                for r in ranked
            ]
        except:
            return [{"index": i, "document": doc, "score": 5} for i, doc in enumerate(documents[:top_k])]

class QueryExpander:
    """查询扩展器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def expand(self, query: str, num_expansions: int = 3) -> List[str]:
        """扩展查询"""
        prompt = f"""
请生成{num_expansions}个与以下查询语义相似但表述不同的查询：

原查询：{query}

要求：
1. 保持相同的意图
2. 使用不同的词汇和表述
3. 可以添加相关的上下文

以列表形式输出，每行一个查询。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        expansions = response.choices[0].message.content.strip().split('\n')
        return [query] + [e.strip() for e in expansions if e.strip()][:num_expansions + 1]
```

## 6.4 【实战】知识库问答Agent

让我们构建一个完整的知识库问答系统。

### 项目结构

```
knowledge-qa-agent/
├── .env
├── main.py
├── agent.py
├── vector_store.py
├── document_processor.py
└── requirements.txt
```

### 完整代码

**document_processor.py**

```python
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    metadata: dict

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """处理文本"""
        chunks = self._split_text(text)
        
        return [
            Document(
                id=f"doc_{i}",
                content=chunk,
                metadata=metadata or {}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _split_text(self, text: str) -> List[str]:
        """切分文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                # 找到最近的句子结束
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
```

**vector_store.py**

```python
import chromadb
from openai import OpenAI
from typing import List, Optional
from document_processor import Document

class VectorStore:
    """向量存储"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None
        self.openai = OpenAI()
    
    def init_collection(self, name: str = "knowledge_base"):
        """初始化集合"""
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        if not self.collection:
            self.init_collection()
        
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # 生成嵌入
        embeddings = self._get_embeddings(contents)
        
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = 5) -> List[dict]:
        """搜索"""
        query_embedding = self._get_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取嵌入向量"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [d.embedding for d in response.data]
    
    def count(self) -> int:
        """获取文档数量"""
        return self.collection.count() if self.collection else 0
```

**agent.py**

```python
import os
from openai import OpenAI
from vector_store import VectorStore
from typing import List

class KnowledgeQAAgent:
    """知识库问答Agent"""
    
    def __init__(self):
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = VectorStore()
        self.vector_store.init_collection()
        self.conversation_history = []
    
    def add_knowledge(self, text: str, source: str = "user"):
        """添加知识"""
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        documents = processor.process_text(text, {"source": source})
        
        self.vector_store.add_documents(documents)
        
        return f"已添加 {len(documents)} 个文档片段到知识库"
    
    def ask(self, question: str) -> str:
        """提问"""
        # 检索相关文档
        results = self.vector_store.search(question, n_results=5)
        
        if not results:
            return "知识库中没有找到相关信息。"
        
        # 构建上下文
        context = "\n\n".join([
            f"[文档{i+1}]\n{r['content']}"
            for i, r in enumerate(results)
        ])
        
        # 构建Prompt
        system_prompt = """你是一个知识库问答助手。请基于提供的上下文回答问题。
要求：
1. 只使用上下文中的信息回答
2. 如果上下文没有相关信息，请说明
3. 回答要准确、简洁
4. 可以引用文档编号"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"}
        ]
        
        # 添加对话历史
        messages = self.conversation_history[-4:] + messages[1:]
        
        # 生成回答
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        answer = response.choices[0].message.content
        
        # 更新历史
        self.conversation_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        
        return answer
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_documents": self.vector_store.count(),
            "conversation_length": len(self.conversation_history)
        }
```

**main.py**

```python
from dotenv import load_dotenv
from agent import KnowledgeQAAgent

load_dotenv()

def main():
    agent = KnowledgeQAAgent()
    
    print("=" * 60)
    print("知识库问答系统")
    print("=" * 60)
    print("\n命令：")
    print("  add  - 添加知识")
    print("  ask  - 提问")
    print("  stats - 查看统计")
    print("  quit - 退出")
    print("=" * 60)
    
    while True:
        command = input("\n命令: ").strip().lower()
        
        if command == 'quit':
            print("再见！")
            break
        
        elif command == 'add':
            print("\n请输入知识内容（输入空行结束）：")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            text = '\n'.join(lines)
            result = agent.add_knowledge(text)
            print(f"\n{result}")
        
        elif command == 'ask':
            question = input("\n请输入问题：")
            answer = agent.ask(question)
            print(f"\n回答：{answer}")
        
        elif command == 'stats':
            stats = agent.get_stats()
            print(f"\n统计信息：{stats}")
        
        else:
            print("未知命令")

if __name__ == "__main__":
    main()
```

### 运行示例

```bash
python main.py

命令: add

请输入知识内容（输入空行结束）：
Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。
Python的设计哲学强调代码可读性和简洁性。
Python支持多种编程范式，包括面向对象、函数式和过程式编程。

已添加 3 个文档片段到知识库

命令: ask

请输入问题：Python是什么时候创建的？

回答：根据文档内容，Python是由Guido van Rossum于1991年创建的。
```

## 本章小结

本章我们学习了：

- ✅ Agent记忆系统的设计
- ✅ 短期记忆和长期记忆的管理
- ✅ 向量嵌入和相似度计算
- ✅ 向量数据库的使用
- ✅ RAG架构和实现
- ✅ 文档切分和检索优化
- ✅ 构建了知识库问答Agent

## 下一章

下一章我们将进入第三阶段，学习LangChain框架。

[第7章：LangChain框架 →](/frameworks/chapter7)
