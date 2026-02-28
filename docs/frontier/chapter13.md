# 第13章：Agent评估与优化

本章将讲解如何评估Agent的性能，以及如何系统化地优化Agent。

## 13.1 评估指标体系

**为什么需要评估Agent？**

开发Agent容易，但开发一个"好"的Agent很难。你需要回答：
- Agent的表现如何？
- 哪些地方需要改进？
- 改进后是否真的变好了？

没有评估，就无法量化改进，也无法保证质量。

**Agent评估的维度**

**1. 功能性指标**

Agent能否正确完成任务？

- **任务完成率**：成功完成的任务比例
- **准确率**：输出结果的正确性
- **召回率**：是否找到了所有答案
- **F1分数**：准确率和召回率的调和平均

**2. 性能指标**

Agent运行得有多快？

- **响应时间**：从请求到响应的时间
- **吞吐量**：单位时间处理的任务数
- **资源消耗**：CPU、内存、Token使用量
- **并发能力**：同时处理多少请求

**3. 用户体验指标**

用户用得爽不爽？

- **用户满意度**：用户评分
- **任务时长**：用户完成任务的时间
- **重试率**：用户需要重试的比例
- **弃用率**：用户放弃使用的比例

**4. 成本指标**

运行Agent需要多少钱？

- **API调用成本**：LLM API费用
- **计算成本**：服务器费用
- **存储成本**：数据库、向量存储费用
- **人力成本**：维护和优化成本

**评估指标的选择**

不同场景关注不同指标：

| 应用场景 | 关键指标 | 次要指标 |
|---------|---------|---------|
| 客服机器人 | 用户满意度、准确率 | 响应时间 |
| 数据分析Agent | 准确率、完成率 | 成本 |
| 代码助手 | 准确率、响应时间 | 用户满意度 |
| 批处理任务 | 完成率、吞吐量 | 用户体验 |

### 任务完成率

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class TaskStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: Any
    duration: float
    metadata: Dict = None

class TaskCompletionEvaluator:
    """任务完成率评估器"""
    
    def __init__(self):
        self.results: List[TaskResult] = []
    
    def record(self, result: TaskResult):
        """记录结果"""
        self.results.append(result)
    
    def completion_rate(self) -> float:
        """计算完成率"""
        if not self.results:
            return 0.0
        
        completed = sum(
            1 for r in self.results 
            if r.status in [TaskStatus.SUCCESS, TaskStatus.PARTIAL]
        )
        
        return completed / len(self.results)
    
    def success_rate(self) -> float:
        """计算成功率"""
        if not self.results:
            return 0.0
        
        success = sum(
            1 for r in self.results 
            if r.status == TaskStatus.SUCCESS
        )
        
        return success / len(self.results)
    
    def failure_analysis(self) -> Dict[str, int]:
        """失败分析"""
        analysis = {}
        
        for result in self.results:
            if result.status != TaskStatus.SUCCESS:
                status = result.status.value
                analysis[status] = analysis.get(status, 0) + 1
        
        return analysis
```

### 响应质量

```python
from typing import List, Dict
from openai import OpenAI

class ResponseQualityEvaluator:
    """响应质量评估器"""
    
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
    
    def evaluate_relevance(
        self,
        query: str,
        response: str
    ) -> float:
        """评估相关性"""
        prompt = f"""
请评估以下回答与问题的相关性（0-10分）：

问题：{query}
回答：{response}

只输出数字分数。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 5.0
    
    def evaluate_accuracy(
        self,
        query: str,
        response: str,
        ground_truth: str
    ) -> float:
        """评估准确性"""
        prompt = f"""
请评估以下回答是否准确（0-10分）：

问题：{query}
回答：{response}
正确答案：{ground_truth}

只输出数字分数。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 5.0
    
    def evaluate_clarity(self, response: str) -> float:
        """评估清晰度"""
        prompt = f"""
请评估以下回答的清晰度（0-10分）：

回答：{response}

考虑因素：
- 是否易于理解
- 逻辑是否清晰
- 表达是否准确

只输出数字分数。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 5.0
    
    def overall_quality(
        self,
        query: str,
        response: str,
        ground_truth: str = None
    ) -> Dict[str, float]:
        """综合质量评估"""
        relevance = self.evaluate_relevance(query, response)
        clarity = self.evaluate_clarity(response)
        
        metrics = {
            "relevance": relevance,
            "clarity": clarity
        }
        
        if ground_truth:
            accuracy = self.evaluate_accuracy(query, response, ground_truth)
            metrics["accuracy"] = accuracy
        
        metrics["overall"] = sum(metrics.values()) / len(metrics)
        
        return metrics
```

### 效率指标

```python
from dataclasses import dataclass
from typing import List
import time

@dataclass
class PerformanceMetrics:
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float

class PerformanceMonitor:
    """性能监控"""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.request_count = 0
        self.start_time = time.time()
    
    def record_response(self, duration: float):
        """记录响应时间"""
        self.response_times.append(duration)
        self.request_count += 1
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """计算性能指标"""
        if not self.response_times:
            return PerformanceMetrics(0, 0, 0, 0, 0)
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        avg = sum(sorted_times) / n
        p50 = sorted_times[int(n * 0.5)]
        p95 = sorted_times[int(n * 0.95)]
        p99 = sorted_times[int(n * 0.99)]
        
        elapsed = time.time() - self.start_time
        throughput = self.request_count / elapsed if elapsed > 0 else 0
        
        return PerformanceMetrics(
            avg_response_time=avg,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            throughput=throughput
        )
```

## 13.2 评估方法

**如何评估Agent？**

有了评估指标，还需要评估方法。不同的方法适用于不同的场景。

**三种主要评估方法**

**1. 人工评估**

由人来评估Agent的表现：

优点：
- 准确性高：人能理解复杂情况
- 灵活性强：可以评估主观质量
- 发现细节问题：人眼能看到细节

缺点：
- 成本高：需要人力
- 速度慢：无法大规模评估
- 主观性：不同人可能有不同判断

适用场景：
- 产品发布前的质量检查
- 用户体验评估
- 复杂任务的评估

**2. 自动评估**

用程序自动评估Agent：

优点：
- 速度快：可以大规模评估
- 成本低：不需要人力
- 可重复：结果一致

缺点：
- 灵活性差：只能评估可量化的指标
- 可能遗漏：无法评估主观质量
- 需要标注：需要准备标准答案

适用场景：
- 回归测试：确保改进不引入bug
- 性能测试：评估响应时间等
- 大规模评估：评估大量测试用例

**3. AI辅助评估**

用AI来评估AI：

优点：
- 兼顾速度和质量
- 可以评估主观质量
- 成本适中

缺点：
- 评估AI本身可能有偏见
- 不如人工准确
- 需要设计评估Prompt

适用场景：
- 大规模质量评估
- 实时监控
- 辅助人工评估

**评估流程**

```
1. 定义评估目标
   ↓
2. 准备测试数据
   ↓
3. 运行Agent
   ↓
4. 收集结果
   ↓
5. 计算指标
   ↓
6. 分析问题
   ↓
7. 提出改进建议
```

### 人工评估

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class Rating(Enum):
    EXCELLENT = 5
    GOOD = 4
    SATISFACTORY = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class HumanEvaluation:
    evaluator_id: str
    task_id: str
    response: str
    rating: Rating
    comments: str
    timestamp: float

class HumanEvaluator:
    """人工评估器"""
    
    def __init__(self):
        self.evaluations: List[HumanEvaluation] = []
        self.evaluators: Dict[str, str] = {}
    
    def register_evaluator(self, evaluator_id: str, name: str):
        """注册评估员"""
        self.evaluators[evaluator_id] = name
    
    def submit_evaluation(
        self,
        evaluator_id: str,
        task_id: str,
        response: str,
        rating: Rating,
        comments: str
    ):
        """提交评估"""
        evaluation = HumanEvaluation(
            evaluator_id=evaluator_id,
            task_id=task_id,
            response=response,
            rating=rating,
            comments=comments,
            timestamp=time.time()
        )
        
        self.evaluations.append(evaluation)
    
    def get_average_rating(self, task_id: str = None) -> float:
        """获取平均评分"""
        evaluations = self.evaluations
        if task_id:
            evaluations = [e for e in evaluations if e.task_id == task_id]
        
        if not evaluations:
            return 0.0
        
        return sum(e.rating.value for e in evaluations) / len(evaluations)
    
    def get_inter_rater_agreement(self) -> float:
        """计算评估员一致性"""
        # 简化实现
        from collections import defaultdict
        
        task_ratings = defaultdict(list)
        
        for eval in self.evaluations:
            task_ratings[eval.task_id].append(eval.rating.value)
        
        agreements = []
        
        for ratings in task_ratings.values():
            if len(ratings) > 1:
                avg = sum(ratings) / len(ratings)
                variance = sum((r - avg) ** 2 for r in ratings) / len(ratings)
                agreement = 1 / (1 + variance)
                agreements.append(agreement)
        
        return sum(agreements) / len(agreements) if agreements else 0.0
```

### 自动评估

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TestCase:
    input: Any
    expected_output: Any
    metadata: Dict = None

@dataclass
class TestResult:
    test_case: TestCase
    actual_output: Any
    passed: bool
    score: float
    error: str = None

class AutoEvaluator:
    """自动评估器"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
    
    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)
    
    def run_tests(self, agent) -> List[TestResult]:
        """运行测试"""
        self.results = []
        
        for test_case in self.test_cases:
            try:
                actual_output = agent.process(test_case.input)
                
                passed = self._compare_outputs(
                    test_case.expected_output,
                    actual_output
                )
                
                score = self._calculate_score(
                    test_case.expected_output,
                    actual_output
                )
                
                result = TestResult(
                    test_case=test_case,
                    actual_output=actual_output,
                    passed=passed,
                    score=score
                )
                
            except Exception as e:
                result = TestResult(
                    test_case=test_case,
                    actual_output=None,
                    passed=False,
                    score=0.0,
                    error=str(e)
                )
            
            self.results.append(result)
        
        return self.results
    
    def _compare_outputs(self, expected: Any, actual: Any) -> bool:
        """比较输出"""
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower() in actual.lower()
        return expected == actual
    
    def _calculate_score(self, expected: Any, actual: Any) -> float:
        """计算分数"""
        if self._compare_outputs(expected, actual):
            return 1.0
        
        # 简单的相似度计算
        if isinstance(expected, str) and isinstance(actual, str):
            expected_words = set(expected.lower().split())
            actual_words = set(actual.lower().split())
            
            if not expected_words:
                return 0.0
            
            overlap = len(expected_words & actual_words)
            return overlap / len(expected_words)
        
        return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.results:
            return {}
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        avg_score = sum(r.score for r in self.results) / total
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "pass_rate": passed / total,
            "average_score": avg_score
        }
```

### A/B测试

```python
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import random

class Variant(Enum):
    A = "A"
    B = "B"

@dataclass
class ABTestResult:
    variant: Variant
    metric: str
    value: float
    metadata: Dict = None

class ABTestFramework:
    """A/B测试框架"""
    
    def __init__(self):
        self.assignments: Dict[str, Variant] = {}
        self.results: List[ABTestResult] = []
        self.ratio = 0.5
    
    def assign_variant(self, user_id: str) -> Variant:
        """分配变体"""
        if user_id in self.assignments:
            return self.assignments[user_id]
        
        variant = Variant.A if random.random() < self.ratio else Variant.B
        self.assignments[user_id] = variant
        
        return variant
    
    def record_result(
        self,
        user_id: str,
        metric: str,
        value: float,
        metadata: Dict = None
    ):
        """记录结果"""
        variant = self.assignments.get(user_id, Variant.A)
        
        result = ABTestResult(
            variant=variant,
            metric=metric,
            value=value,
            metadata=metadata
        )
        
        self.results.append(result)
    
    def analyze(self, metric: str) -> Dict[str, Any]:
        """分析结果"""
        a_results = [r for r in self.results if r.variant == Variant.A and r.metric == metric]
        b_results = [r for r in self.results if r.variant == Variant.B and r.metric == metric]
        
        if not a_results or not b_results:
            return {}
        
        a_avg = sum(r.value for r in a_results) / len(a_results)
        b_avg = sum(r.value for r in b_results) / len(b_results)
        
        improvement = (b_avg - a_avg) / a_avg if a_avg > 0 else 0
        
        return {
            "metric": metric,
            "variant_a": {
                "count": len(a_results),
                "average": a_avg
            },
            "variant_b": {
                "count": len(b_results),
                "average": b_avg
            },
            "improvement": improvement,
            "winner": Variant.B if improvement > 0 else Variant.A
        }
```

## 13.3 优化策略

**如何优化Agent？**

评估发现问题后，需要优化Agent。优化要有针对性，根据问题选择合适的方法。

**Agent优化的主要方向**

**1. Prompt优化**

Prompt是Agent的核心，优化Prompt往往能带来显著提升：

优化方法：
- 明确指令：让Agent更清楚要做什么
- 添加示例：用few-shot引导Agent
- 调整结构：优化Prompt的组织方式
- 减少歧义：避免模糊表述

效果：
- 提高准确率
- 减少错误
- 改善输出质量

**2. 工具优化**

工具是Agent的能力来源：

优化方法：
- 改进工具描述：让Agent更清楚工具用途
- 优化参数设计：简化参数，提供默认值
- 增强错误处理：提供友好的错误提示
- 添加新工具：扩展Agent能力

效果：
- 提高工具使用准确率
- 减少工具调用错误
- 扩展应用场景

**3. 架构优化**

Agent的整体架构影响性能：

优化方法：
- 简化流程：减少不必要的步骤
- 并行处理：独立任务并行执行
- 缓存结果：避免重复计算
- 懒加载：按需加载资源

效果：
- 提高响应速度
- 降低资源消耗
- 提升用户体验

**4. 模型优化**

选择和使用合适的LLM：

优化方法：
- 选择合适模型：平衡性能和成本
- 调整温度参数：控制输出随机性
- 使用流式输出：降低首字延迟
- 批量处理：合并多个请求

效果：
- 降低成本
- 提高速度
- 改善质量

**优化的优先级**

根据投入产出比确定优先级：

| 优化方向 | 投入 | 收益 | 优先级 |
|---------|------|------|--------|
| Prompt优化 | 低 | 高 | ⭐⭐⭐⭐⭐ |
| 工具优化 | 中 | 高 | ⭐⭐⭐⭐ |
| 架构优化 | 高 | 高 | ⭐⭐⭐ |
| 模型优化 | 低 | 中 | ⭐⭐⭐ |

**优化流程**

```
评估发现问题
   ↓
分析问题原因
   ↓
制定优化方案
   ↓
实施优化
   ↓
评估优化效果
   ↓
效果达标？ → 是 → 完成
   ↓ 否
调整方案，重新优化
```

### Prompt优化

```python
from typing import List, Dict, Any
from openai import OpenAI

class PromptOptimizer:
    """Prompt优化器"""
    
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.history = []
    
    def optimize(
        self,
        current_prompt: str,
        test_cases: List[Dict[str, Any]],
        max_iterations: int = 5
    ) -> str:
        """优化Prompt"""
        best_prompt = current_prompt
        best_score = self._evaluate_prompt(current_prompt, test_cases)
        
        for i in range(max_iterations):
            # 分析失败案例
            failures = self._analyze_failures(
                current_prompt,
                test_cases
            )
            
            # 生成改进建议
            improvements = self._generate_improvements(
                current_prompt,
                failures
            )
            
            # 应用改进
            new_prompt = self._apply_improvements(
                current_prompt,
                improvements
            )
            
            # 评估新Prompt
            new_score = self._evaluate_prompt(new_prompt, test_cases)
            
            if new_score > best_score:
                best_prompt = new_prompt
                best_score = new_score
                current_prompt = new_prompt
            
            self.history.append({
                "iteration": i + 1,
                "prompt": current_prompt,
                "score": new_score
            })
        
        return best_prompt
    
    def _evaluate_prompt(
        self,
        prompt: str,
        test_cases: List[Dict[str, Any]]
    ) -> float:
        """评估Prompt"""
        passed = 0
        
        for test_case in test_cases:
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt.format(**test_case["input"])}
                ]
            )
            
            output = response.choices[0].message.content
            
            if self._check_output(output, test_case["expected"]):
                passed += 1
        
        return passed / len(test_cases)
    
    def _check_output(self, output: str, expected: str) -> bool:
        """检查输出"""
        return expected.lower() in output.lower()
    
    def _analyze_failures(
        self,
        prompt: str,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict]:
        """分析失败案例"""
        failures = []
        
        for test_case in test_cases:
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt.format(**test_case["input"])}
                ]
            )
            
            output = response.choices[0].message.content
            
            if not self._check_output(output, test_case["expected"]):
                failures.append({
                    "input": test_case["input"],
                    "expected": test_case["expected"],
                    "actual": output
                })
        
        return failures
    
    def _generate_improvements(
        self,
        prompt: str,
        failures: List[Dict]
    ) -> List[str]:
        """生成改进建议"""
        prompt = f"""
当前Prompt：
{prompt}

失败案例：
{chr(10).join(f'输入：{f["input"]}\n期望：{f["expected"]}\n实际：{f["actual"][:100]}...' for f in failures[:3])}

请分析失败原因，并提供3个具体的改进建议。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.split('\n')
    
    def _apply_improvements(
        self,
        prompt: str,
        improvements: List[str]
    ) -> str:
        """应用改进"""
        return prompt + "\n\n" + "\n".join(improvements)
```

### 工具优化

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolUsage:
    tool_name: str
    call_count: int
    success_count: int
    failure_count: int
    avg_duration: float

class ToolOptimizer:
    """工具优化器"""
    
    def __init__(self):
        self.usage: Dict[str, ToolUsage] = {}
    
    def record_usage(
        self,
        tool_name: str,
        success: bool,
        duration: float
    ):
        """记录工具使用"""
        if tool_name not in self.usage:
            self.usage[tool_name] = ToolUsage(
                tool_name=tool_name,
                call_count=0,
                success_count=0,
                failure_count=0,
                avg_duration=0.0
            )
        
        usage = self.usage[tool_name]
        usage.call_count += 1
        
        if success:
            usage.success_count += 1
        else:
            usage.failure_count += 1
        
        # 更新平均耗时
        n = usage.call_count
        usage.avg_duration = (
            (usage.avg_duration * (n - 1) + duration) / n
        )
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        suggestions = []
        
        for tool_name, usage in self.usage.items():
            # 失败率高的工具
            failure_rate = usage.failure_count / usage.call_count if usage.call_count > 0 else 0
            if failure_rate > 0.3:
                suggestions.append({
                    "tool": tool_name,
                    "issue": "high_failure_rate",
                    "value": failure_rate,
                    "suggestion": "检查工具实现，添加错误处理"
                })
            
            # 耗时长的工具
            if usage.avg_duration > 5.0:
                suggestions.append({
                    "tool": tool_name,
                    "issue": "slow_response",
                    "value": usage.avg_duration,
                    "suggestion": "考虑缓存结果或优化实现"
                })
        
        return suggestions
```

### 架构优化

```python
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ArchitectureMetrics:
    component_name: str
    avg_response_time: float
    throughput: float
    error_rate: float
    resource_usage: Dict[str, float]

class ArchitectureOptimizer:
    """架构优化器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[ArchitectureMetrics]] = {}
    
    def record_metrics(self, metrics: ArchitectureMetrics):
        """记录指标"""
        if metrics.component_name not in self.metrics:
            self.metrics[metrics.component_name] = []
        
        self.metrics[metrics.component_name].append(metrics)
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """分析瓶颈"""
        bottlenecks = []
        
        for component, metrics_list in self.metrics.items():
            if not metrics_list:
                continue
            
            avg_time = sum(m.avg_response_time for m in metrics_list) / len(metrics_list)
            avg_error_rate = sum(m.error_rate for m in metrics_list) / len(metrics_list)
            
            # 响应时间慢的组件
            if avg_time > 3.0:
                bottlenecks.append({
                    "component": component,
                    "type": "slow_response",
                    "value": avg_time,
                    "suggestion": "考虑异步处理或缓存"
                })
            
            # 错误率高的组件
            if avg_error_rate > 0.1:
                bottlenecks.append({
                    "component": component,
                    "type": "high_error_rate",
                    "value": avg_error_rate,
                    "suggestion": "检查错误处理和重试机制"
                })
        
        return bottlenecks
    
    def suggest_scaling(self) -> Dict[str, Any]:
        """建议扩容"""
        suggestions = {}
        
        for component, metrics_list in self.metrics.items():
            if not metrics_list:
                continue
            
            avg_throughput = sum(m.throughput for m in metrics_list) / len(metrics_list)
            avg_resource = {
                "cpu": sum(m.resource_usage.get("cpu", 0) for m in metrics_list) / len(metrics_list),
                "memory": sum(m.resource_usage.get("memory", 0) for m in metrics_list) / len(metrics_list)
            }
            
            if avg_resource["cpu"] > 0.8:
                suggestions[component] = {
                    "action": "scale_up",
                    "reason": "high_cpu_usage",
                    "current": avg_resource["cpu"],
                    "suggested_instances": 2
                }
        
        return suggestions
```

## 13.4 【实战】构建评估系统

让我们构建一个完整的Agent评估系统。

### 项目结构

```
agent-evaluation/
├── .env
├── main.py
├── evaluators/
│   ├── auto.py
│   ├── human.py
│   └── quality.py
├── optimizer.py
└── requirements.txt
```

### 完整代码

**evaluators/auto.py**

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TestCase:
    input: str
    expected: str
    category: str

class AutoEvaluator:
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results = []
    
    def add_test_case(self, test_case: TestCase):
        self.test_cases.append(test_case)
    
    def evaluate(self, agent) -> Dict:
        for test_case in self.test_cases:
            response = agent.process(test_case.input)
            
            self.results.append({
                "input": test_case.input,
                "expected": test_case.expected,
                "actual": response,
                "passed": test_case.expected.lower() in response.lower(),
                "category": test_case.category
            })
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0
        }
```

**optimizer.py**

```python
from evaluators.auto import AutoEvaluator
from openai import OpenAI

class AgentOptimizer:
    def __init__(self, agent, evaluator: AutoEvaluator):
        self.agent = agent
        self.evaluator = evaluator
        self.llm = OpenAI()
        self.optimization_history = []
    
    def optimize(self, max_iterations: int = 5):
        best_score = 0
        best_config = self.agent.get_config()
        
        for i in range(max_iterations):
            # 评估当前配置
            stats = self.evaluator.evaluate(self.agent)
            score = stats["pass_rate"]
            
            print(f"Iteration {i+1}: Score = {score:.2%}")
            
            if score > best_score:
                best_score = score
                best_config = self.agent.get_config()
            
            # 生成改进建议
            suggestions = self._generate_suggestions(stats)
            
            # 应用改进
            self._apply_suggestions(suggestions)
            
            self.optimization_history.append({
                "iteration": i + 1,
                "score": score,
                "config": self.agent.get_config()
            })
        
        # 恢复最佳配置
        self.agent.set_config(best_config)
        
        return best_score
    
    def _generate_suggestions(self, stats: Dict) -> List[str]:
        prompt = f"""
当前Agent性能：
- 通过率：{stats['pass_rate']:.2%}
- 失败案例：{stats['failed']}

请提供3个具体的改进建议。
"""
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.split('\n')
    
    def _apply_suggestions(self, suggestions: List[str]):
        # 应用改进建议
        pass
```

## 本章小结

本章我们学习了：

- ✅ Agent评估指标体系
- ✅ 人工评估、自动评估和A/B测试
- ✅ Prompt、工具和架构优化策略
- ✅ 构建了评估系统

## 下一章

下一章我们将学习Agent安全与伦理。

[第14章：Agent安全与伦理 →](/frontier/chapter14)
