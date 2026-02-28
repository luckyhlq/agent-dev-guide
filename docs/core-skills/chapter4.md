# 第4章：高级Prompt技术

本章将深入探讨高级Prompt工程技术，帮助你构建更强大、更可靠的Agent。

## 4.1 结构化Prompt设计

**为什么要结构化Prompt？**

想象你在给一个新员工安排工作。如果你只是简单说"去把那个项目做了"，员工可能会一脸茫然，不知道从何下手。但如果你说"你是项目A的负责人，主要任务是完成X、Y、Z，注意要遵循以下规范，最后用这个格式提交报告"，员工就能清楚地知道该做什么。

结构化Prompt就是这个道理。通过清晰的结构告诉AI：
- **你是谁**（角色定义）
- **要做什么**（任务描述）
- **怎么做**（行为准则）
- **输出什么样**（格式要求）

**结构化Prompt的核心要素**

一个好的结构化Prompt通常包含这几个部分：

1. **角色定义**：让AI知道它扮演什么角色，比如"你是一个资深Python开发工程师"
2. **任务说明**：具体要完成什么任务，越详细越好
3. **行为准则**：应该遵循什么规则，应该避免什么行为
4. **输出格式**：希望得到什么格式的输出，比如JSON、Markdown等
5. **限制条件**：哪些事情不能做，哪些边界不能逾越

**为什么这样设计有效？**

大语言模型在训练时见过大量结构化的文档（如技术文档、教程、规范等），所以它能很好地理解结构化的指令。结构化的Prompt就像给AI一个清晰的"工作手册"，让它知道：
- 该关注什么
- 该忽略什么
- 该如何组织输出

### System Prompt设计

System Prompt是定义Agent行为的关键：

```python
SYSTEM_PROMPT_TEMPLATE = """
# 角色定义
你是一个{role}，专注于{domain}领域。

# 核心能力
{capabilities}

# 行为准则
{guidelines}

# 输出格式
{output_format}

# 限制条件
{constraints}
"""

def create_system_prompt(config):
    return SYSTEM_PROMPT_TEMPLATE.format(
        role=config.get('role', '智能助手'),
        domain=config.get('domain', '通用'),
        capabilities='\n'.join(f"- {c}" for c in config.get('capabilities', [])),
        guidelines='\n'.join(f"- {g}" for g in config.get('guidelines', [])),
        output_format=config.get('output_format', '自然语言'),
        constraints='\n'.join(f"- {c}" for c in config.get('constraints', []))
    )

# 示例配置
config = {
    'role': '代码审查专家',
    'domain': '软件开发',
    'capabilities': [
        '代码质量分析',
        '安全漏洞检测',
        '性能优化建议',
        '最佳实践推荐'
    ],
    'guidelines': [
        '保持专业、客观的态度',
        '提供具体的改进建议',
        '解释问题的原因和影响',
        '给出可执行的代码示例'
    ],
    'output_format': '''
## 问题列表
- [严重程度] 问题描述
  - 位置：文件名:行号
  - 建议：改进建议
  - 示例：代码示例

## 总体评价
评分和总结
''',
    'constraints': [
        '不修改业务逻辑',
        '不引入新的依赖',
        '保持代码风格一致'
    ]
}

system_prompt = create_system_prompt(config)
```

### 角色扮演Prompt

通过角色扮演让Agent更专业：

```python
class RolePlayAgent:
    def __init__(self, role_config):
        self.client = OpenAI()
        self.role = role_config
        
        self.system_prompt = f"""
你现在扮演：{role_config['name']}

背景故事：
{role_config['background']}

性格特点：
{role_config['personality']}

说话风格：
{role_config['speaking_style']}

专业领域：
{role_config['expertise']}

请始终保持角色，用第一人称回答问题。
"""
    
    def chat(self, user_message):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content

# 定义角色
expert_role = {
    'name': '张教授',
    'background': '清华大学计算机系教授，专注AI研究20年',
    'personality': '严谨、耐心、善于启发',
    'speaking_style': '学术但通俗易懂，喜欢用比喻',
    'expertise': '机器学习、深度学习、自然语言处理'
}

agent = RolePlayAgent(expert_role)
print(agent.chat("什么是深度学习？"))
```

### 输出格式控制

精确控制输出格式：

```python
import json

class StructuredOutputAgent:
    def __init__(self):
        self.client = OpenAI()
    
    def get_json_output(self, prompt, schema):
        """
        强制JSON格式输出
        
        Args:
            prompt: 用户提示
            schema: JSON Schema定义
        """
        system_prompt = f"""
你是一个数据提取专家。请严格按照以下JSON Schema输出：

{json.dumps(schema, indent=2, ensure_ascii=False)}

要求：
1. 只输出JSON，不要其他文字
2. 严格遵循Schema定义
3. 必填字段不能缺失
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        content = response.choices[0].message.content
        
        # 尝试解析JSON
        try:
            # 去除可能的markdown代码块标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"error": "无法解析JSON", "raw": content}
    
    def get_markdown_output(self, prompt, template):
        """强制Markdown格式输出"""
        system_prompt = f"""
请按照以下模板格式输出：

{template}

要求：
1. 使用Markdown格式
2. 保持标题层级正确
3. 代码块使用正确的语法高亮
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content

# 使用示例
agent = StructuredOutputAgent()

# JSON输出
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "age", "skills"]
}

result = agent.get_json_output(
    "提取信息：张三，28岁，擅长Python、JavaScript、Go",
    schema
)
print(json.dumps(result, indent=2, ensure_ascii=False))

# Markdown输出
template = """
# {title}

## 概述
{overview}

## 详细说明
{details}

## 代码示例
python
{code}
"""

markdown = agent.get_markdown_output(
    "解释Python的列表推导式",
    template
)
print(markdown)
```

## 4.2 高级推理技术

**为什么需要高级推理技术？**

大语言模型虽然强大，但并不是每次都能给出正确答案。就像人一样，有时候会"想错"或者"想偏了"。高级推理技术就是通过一些策略，让AI"想得更清楚"、"想得更全面"。

**三种核心推理技术**

1. **Self-Consistency（自一致性）**
   - **原理**：就像"三个臭皮匠顶个诸葛亮"，让AI对同一个问题思考多次，然后选择最一致的答案
   - **适用场景**：数学计算、逻辑推理、选择题等有明确答案的问题
   - **为什么有效**：通过多次采样，降低偶然性错误，提高答案可靠性

2. **Tree of Thoughts（思维树）**
   - **原理**：就像下棋时思考"如果我走这步，对手会怎么走，然后我又该怎么走"，探索多条可能的推理路径
   - **适用场景**：复杂决策、创意写作、问题解决等需要多步推理的任务
   - **为什么有效**：不局限于一条思路，可以找到更优的解决方案

3. **Self-Refine（自我优化）**
   - **原理**：就像写作文时的"写-改-写"循环，先生成初稿，然后自我批评，再改进，反复迭代
   - **适用场景**：代码生成、文章写作、方案设计等需要不断优化的任务
   - **为什么有效**：通过自我反思和迭代，逐步提升输出质量

**如何选择合适的推理技术？**

| 技术类型 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| Self-Consistency | 有明确答案的问题 | 简单有效，可靠性高 | 成本较高（需要多次调用） |
| Tree of Thoughts | 复杂推理问题 | 能找到更优解 | 实现复杂，耗时较长 |
| Self-Refine | 创造性任务 | 质量提升明显 | 需要多次迭代 |

### Self-Consistency

通过多次采样提高答案可靠性：

```python
class SelfConsistencyAgent:
    def __init__(self, n_samples=5):
        self.client = OpenAI()
        self.n_samples = n_samples
    
    def generate_answers(self, question):
        """生成多个答案"""
        answers = []
        
        for _ in range(self.n_samples):
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"""
请一步步思考并回答问题。

问题：{question}

让我们一步步思考：
"""
                }],
                temperature=0.7  # 较高温度增加多样性
            )
            answers.append(response.choices[0].message.content)
        
        return answers
    
    def find_consensus(self, answers):
        """找到最一致的答案"""
        # 简单的投票机制
        from collections import Counter
        
        # 提取最终答案
        final_answers = []
        for answer in answers:
            # 假设答案在最后一行
            final_line = answer.strip().split('\n')[-1]
            final_answers.append(final_line)
        
        # 统计最常见的答案
        counter = Counter(final_answers)
        most_common = counter.most_common(1)[0]
        
        return most_common[0], most_common[1] / len(answers)
    
    def solve(self, question):
        """使用Self-Consistency解决问题"""
        print(f"问题：{question}\n")
        
        # 生成多个答案
        answers = self.generate_answers(question)
        
        print(f"生成了 {len(answers)} 个答案：")
        for i, answer in enumerate(answers, 1):
            print(f"\n--- 答案 {i} ---")
            print(answer[:200] + "..." if len(answer) > 200 else answer)
        
        # 找到共识
        consensus, confidence = self.find_consensus(answers)
        
        print(f"\n=== 最终答案 ===")
        print(f"答案：{consensus}")
        print(f"置信度：{confidence:.1%}")
        
        return consensus, confidence

# 使用
agent = SelfConsistencyAgent(n_samples=5)
agent.solve("一个篮子里有5个苹果，拿走2个，又放进3个，问篮子里现在有几个苹果？")
```

### Tree of Thoughts

通过探索多条推理路径找到最佳解决方案：

```python
class TreeOfThoughts:
    def __init__(self, max_depth=3, branching_factor=3):
        self.client = OpenAI()
        self.max_depth = max_depth
        self.branching_factor = branching_factor
    
    def generate_thoughts(self, state, n=3):
        """生成多个可能的思考方向"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
当前状态：{state}

请生成 {n} 个不同的思考方向来推进问题解决。
每个思考方向应该：
1. 是一个具体的推理步骤
2. 与其他方向不同
3. 有助于解决问题

以列表形式输出，每行一个思考方向。
"""
            }],
            temperature=0.7
        )
        
        thoughts = response.choices[0].message.content.strip().split('\n')
        return [t.strip() for t in thoughts if t.strip()][:n]
    
    def evaluate_thought(self, thought, goal):
        """评估思考方向的价值"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
目标：{goal}

思考方向：{thought}

请评估这个思考方向对达成目标的价值。
输出一个0-10的分数，10表示最有价值。
只输出数字。
"""
            }],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 5.0
    
    def search(self, problem):
        """搜索最佳解决路径"""
        print(f"问题：{problem}\n")
        
        best_path = []
        best_score = 0
        current_state = problem
        
        for depth in range(self.max_depth):
            print(f"\n=== 深度 {depth + 1} ===")
            
            # 生成思考方向
            thoughts = self.generate_thoughts(current_state, self.branching_factor)
            
            # 评估每个方向
            scored_thoughts = []
            for thought in thoughts:
                score = self.evaluate_thought(thought, problem)
                scored_thoughts.append((thought, score))
                print(f"思考：{thought[:50]}... 分数：{score}")
            
            # 选择最佳方向
            scored_thoughts.sort(key=lambda x: x[1], reverse=True)
            best_thought, best_score = scored_thoughts[0]
            
            print(f"\n选择：{best_thought}")
            
            best_path.append(best_thought)
            current_state = f"{current_state}\n已思考：{best_thought}"
            
            # 检查是否找到解决方案
            if best_score >= 9:
                print("\n✅ 找到高质量解决方案！")
                break
        
        return best_path

# 使用
tot = TreeOfThoughts(max_depth=3, branching_factor=3)
path = tot.search("如何设计一个高效的缓存系统？")
print(f"\n最终解决路径：{path}")
```

### Self-Refine

自我迭代优化：

```python
class SelfRefineAgent:
    def __init__(self, max_iterations=3):
        self.client = OpenAI()
        self.max_iterations = max_iterations
    
    def generate(self, requirement):
        """生成初始内容"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"请根据以下要求生成内容：\n\n{requirement}"
            }]
        )
        return response.choices[0].message.content
    
    def critique(self, content, requirement):
        """批评和改进建议"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
请评估以下内容：

要求：
{requirement}

内容：
{content}

评估标准：
1. 是否满足所有要求
2. 有哪些不足之处
3. 具体的改进建议

如果内容完全满足要求，输出：SATISFIED
否则，输出具体的改进建议。
"""
            }]
        )
        return response.choices[0].message.content
    
    def refine(self, content, critique):
        """根据批评改进内容"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
请根据改进建议优化以下内容：

原内容：
{content}

改进建议：
{critique}

输出优化后的内容。
"""
            }]
        )
        return response.choices[0].message.content
    
    def run(self, requirement):
        """执行自我优化循环"""
        print(f"要求：{requirement}\n")
        
        content = self.generate(requirement)
        
        for i in range(self.max_iterations):
            print(f"\n=== 迭代 {i + 1} ===")
            print(f"当前内容：\n{content[:200]}...\n")
            
            critique = self.critique(content, requirement)
            print(f"批评：\n{critique}\n")
            
            if "SATISFIED" in critique:
                print("✅ 内容已满足要求！")
                return content
            
            content = self.refine(content, critique)
        
        print("⚠️ 达到最大迭代次数")
        return content

# 使用
agent = SelfRefineAgent(max_iterations=3)
result = agent.run("写一个Python函数，实现快速排序算法，要求代码简洁、有注释、包含测试用例")
```

## 4.3 Prompt优化策略

**为什么需要优化Prompt？**

写好一个Prompt就像写好一段代码，很少能一次就完美。你可能会遇到：
- AI理解偏了，给出的答案不是你想要的
- 输出格式不稳定，有时候对有时候错
- 在某些场景下表现很好，在另一些场景下又不行

Prompt优化就是通过系统化的方法，不断改进Prompt，让它更稳定、更可靠。

**三种优化方法**

1. **迭代优化**
   - **思路**：就像调试代码一样，发现问题→分析原因→修改→再测试
   - **步骤**：
     1. 准备测试用例（输入和期望输出）
     2. 运行Prompt，看哪些用例失败了
     3. 分析失败原因
     4. 修改Prompt
     5. 重复测试直到满意
   - **适用场景**：有明确正确答案的任务，如分类、提取、转换等

2. **A/B测试**
   - **思路**：就像产品测试一样，对比两个版本的Prompt哪个更好
   - **步骤**：
     1. 准备两个不同的Prompt版本
     2. 用相同的测试数据分别运行
     3. 对比效果（准确率、Token消耗、响应时间等）
     4. 选择表现更好的版本
   - **适用场景**：需要量化对比的场景，如选择最佳Prompt模板

3. **版本管理**
   - **思路**：就像代码版本控制一样，记录每次修改，方便回溯
   - **好处**：
     - 可以随时回退到之前的版本
     - 记录每次修改的原因和效果
     - 方便团队协作和知识传承
   - **适用场景**：长期维护的Prompt，需要持续改进

**优化的关键原则**

- **小步快跑**：每次只改一个地方，看效果再继续
- **记录过程**：记录每次修改和效果，方便总结经验
- **多场景测试**：不要只在几个例子上测试，要覆盖各种情况
- **关注失败案例**：失败案例往往能发现Prompt的不足

### 迭代优化方法

系统化的Prompt优化流程：

```python
class PromptOptimizer:
    def __init__(self):
        self.client = OpenAI()
        self.history = []
    
    def test_prompt(self, prompt, test_cases):
        """测试Prompt效果"""
        results = []
        
        for case in test_cases:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt.format(input=case['input'])}]
            )
            
            output = response.choices[0].message.content
            passed = self.evaluate_output(output, case['expected'])
            
            results.append({
                'input': case['input'],
                'expected': case['expected'],
                'output': output,
                'passed': passed
            })
        
        return results
    
    def evaluate_output(self, output, expected):
        """评估输出是否正确"""
        # 简单的关键词匹配
        if isinstance(expected, str):
            return expected.lower() in output.lower()
        elif isinstance(expected, list):
            return any(e.lower() in output.lower() for e in expected)
        return False
    
    def analyze_failures(self, results):
        """分析失败案例"""
        failures = [r for r in results if not r['passed']]
        
        if not failures:
            return "所有测试通过！"
        
        analysis = "失败案例分析：\n"
        for f in failures:
            analysis += f"- 输入：{f['input']}\n"
            analysis += f"  期望：{f['expected']}\n"
            analysis += f"  实际：{f['output'][:100]}...\n\n"
        
        return analysis
    
    def improve_prompt(self, current_prompt, failures):
        """根据失败案例改进Prompt"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""
当前Prompt：
{current_prompt}

失败案例：
{failures}

请分析失败原因，并提供改进后的Prompt。
只输出改进后的Prompt，不要其他内容。
"""
            }]
        )
        
        return response.choices[0].message.content
    
    def optimize(self, initial_prompt, test_cases, max_iterations=5):
        """优化循环"""
        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score = 0
        
        for i in range(max_iterations):
            print(f"\n=== 优化迭代 {i + 1} ===")
            
            # 测试
            results = self.test_prompt(current_prompt, test_cases)
            score = sum(1 for r in results if r['passed']) / len(results)
            
            print(f"得分：{score:.1%}")
            
            # 记录历史
            self.history.append({
                'iteration': i + 1,
                'prompt': current_prompt,
                'score': score
            })
            
            # 更新最佳
            if score > best_score:
                best_score = score
                best_prompt = current_prompt
            
            # 如果全部通过，结束
            if score == 1.0:
                print("✅ 所有测试通过！")
                break
            
            # 分析失败并改进
            failures = self.analyze_failures(results)
            current_prompt = self.improve_prompt(current_prompt, failures)
        
        return best_prompt, best_score

# 使用示例
test_cases = [
    {'input': '今天天气很好', 'expected': 'positive'},
    {'input': '服务太差了', 'expected': 'negative'},
    {'input': '还可以', 'expected': 'neutral'}
]

initial_prompt = "分析以下文本的情感：{input}"

optimizer = PromptOptimizer()
best_prompt, score = optimizer.optimize(initial_prompt, test_cases)
print(f"\n最佳Prompt：\n{best_prompt}")
print(f"最终得分：{score:.1%}")
```

### A/B测试

对比不同Prompt的效果：

```python
import random
from collections import defaultdict

class ABTestFramework:
    def __init__(self):
        self.client = OpenAI()
        self.results = defaultdict(list)
    
    def test_variant(self, prompt, inputs, variant_name):
        """测试一个变体"""
        for input_text in inputs:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt.format(input=input_text)}]
            )
            
            self.results[variant_name].append({
                'input': input_text,
                'output': response.choices[0].message.content,
                'tokens': response.usage.total_tokens
            })
    
    def compare_variants(self, variant_a, variant_b, metric='quality'):
        """比较两个变体"""
        results_a = self.results[variant_a]
        results_b = self.results[variant_b]
        
        if metric == 'quality':
            # 人工评估或自动评估
            pass
        elif metric == 'tokens':
            avg_a = sum(r['tokens'] for r in results_a) / len(results_a)
            avg_b = sum(r['tokens'] for r in results_b) / len(results_b)
            
            print(f"平均Token消耗：")
            print(f"  {variant_a}: {avg_a:.1f}")
            print(f"  {variant_b}: {avg_b:.1f}")
            print(f"  差异: {(avg_b - avg_a) / avg_a:.1%}")
    
    def run_ab_test(self, prompt_a, prompt_b, test_inputs):
        """运行A/B测试"""
        print("运行A/B测试...\n")
        
        self.test_variant(prompt_a, test_inputs, 'A')
        self.test_variant(prompt_b, test_inputs, 'B')
        
        self.compare_variants('A', 'B', 'tokens')
        
        return self.results
```

### Prompt版本管理

管理Prompt的不同版本：

```python
import json
from datetime import datetime

class PromptVersionControl:
    def __init__(self, storage_path='prompts.json'):
        self.storage_path = storage_path
        self.versions = self.load()
    
    def load(self):
        """加载版本历史"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save(self):
        """保存版本历史"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)
    
    def add_version(self, name, prompt, metadata=None):
        """添加新版本"""
        if name not in self.versions:
            self.versions[name] = []
        
        version_num = len(self.versions[name]) + 1
        version_id = f"{name}_v{version_num}"
        
        self.versions[name].append({
            'id': version_id,
            'prompt': prompt,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        
        self.save()
        return version_id
    
    def get_version(self, name, version=None):
        """获取特定版本"""
        if name not in self.versions:
            return None
        
        versions = self.versions[name]
        
        if version is None:
            return versions[-1]  # 最新版本
        
        for v in versions:
            if v['id'] == version:
                return v
        
        return None
    
    def list_versions(self, name):
        """列出所有版本"""
        return self.versions.get(name, [])
    
    def compare_versions(self, name, v1, v2):
        """比较两个版本"""
        ver1 = self.get_version(name, v1)
        ver2 = self.get_version(name, v2)
        
        if not ver1 or not ver2:
            return None
        
        return {
            'version1': ver1,
            'version2': ver2,
            'diff': self.compute_diff(ver1['prompt'], ver2['prompt'])
        }
    
    def compute_diff(self, text1, text2):
        """计算文本差异"""
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        diff = []
        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            if l1 != l2:
                diff.append({
                    'line': i + 1,
                    'old': l1,
                    'new': l2
                })
        
        return diff

# 使用
pvc = PromptVersionControl()

# 添加版本
v1 = pvc.add_version(
    'sentiment_analysis',
    '分析以下文本的情感：{input}',
    {'author': 'developer', 'test_score': 0.85}
)

v2 = pvc.add_version(
    'sentiment_analysis',
    '请判断以下文本的情感倾向（positive/negative/neutral）：{input}',
    {'author': 'developer', 'test_score': 0.92}
)

# 获取版本
current = pvc.get_version('sentiment_analysis')
print(f"当前版本：{current['id']}")
print(f"Prompt：{current['prompt']}")
```

## 4.4 【实战】代码审查Agent

让我们构建一个专业的代码审查Agent，综合运用本章所学技术。

### 项目结构

```
code-review-agent/
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
# 角色定义
你是一位资深代码审查专家，拥有15年软件开发经验。

# 核心能力
- 代码质量分析
- 安全漏洞检测
- 性能问题识别
- 最佳实践推荐
- 代码风格检查

# 审查标准
1. 代码正确性：逻辑是否正确，边界条件是否处理
2. 代码可读性：命名是否清晰，结构是否合理
3. 代码安全性：是否存在安全漏洞
4. 代码性能：是否存在性能问题
5. 代码可维护性：是否易于修改和扩展

# 输出格式
## 问题列表

### 严重问题 🔴
- [问题类型] 描述
  - 位置：行号
  - 影响：问题影响
  - 建议：改进建议
  - 示例：
    ```python
    # 改进后的代码
    ```

### 一般问题 🟡
- [问题类型] 描述
  - 建议：改进建议

### 优化建议 🟢
- [建议类型] 描述
  - 说明：为什么建议这样改

## 总体评价
- 代码质量评分：X/10
- 主要优点：
- 需要改进：
- 总结：
"""

REVIEW_TEMPLATE = """
请审查以下代码：

```{language}
{code}
```

审查重点：{focus_areas}
"""
```

**agent.py**

```python
import os
import json
from openai import OpenAI
from prompts import SYSTEM_PROMPT, REVIEW_TEMPLATE

class CodeReviewAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.review_history = []
    
    def review(self, code, language='python', focus_areas='all'):
        """执行代码审查"""
        prompt = REVIEW_TEMPLATE.format(
            language=language,
            code=code,
            focus_areas=focus_areas
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        review_result = response.choices[0].message.content
        
        # 记录历史
        self.review_history.append({
            'code': code,
            'language': language,
            'review': review_result
        })
        
        return review_result
    
    def quick_check(self, code):
        """快速检查"""
        prompt = f"""
快速检查以下代码的主要问题：

```
{code}
```

只列出最重要的3个问题，每个问题用一句话描述。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个代码审查助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def suggest_fix(self, code, issue):
        """针对特定问题提供修复建议"""
        prompt = f"""
代码：
```
{code}
```

问题：{issue}

请提供：
1. 问题分析
2. 修复方案
3. 修复后的代码
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def compare_versions(self, old_code, new_code):
        """比较代码版本"""
        prompt = f"""
比较两个代码版本：

旧版本：
```
{old_code}
```

新版本：
```
{new_code}
```

请分析：
1. 改进了哪些问题
2. 是否引入了新问题
3. 整体质量变化
"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
```

**main.py**

```python
from dotenv import load_dotenv
from agent import CodeReviewAgent

load_dotenv()

def main():
    agent = CodeReviewAgent()
    
    print("=" * 60)
    print("代码审查Agent")
    print("=" * 60)
    print("\n命令：")
    print("  review - 完整审查")
    print("  quick  - 快速检查")
    print("  fix    - 修复建议")
    print("  quit   - 退出")
    print("=" * 60)
    
    while True:
        command = input("\n命令: ").strip().lower()
        
        if command == 'quit':
            print("再见！")
            break
        
        elif command == 'review':
            print("\n请输入代码（输入空行结束）：")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            code = '\n'.join(lines)
            print("\n正在审查...\n")
            result = agent.review(code)
            print(result)
        
        elif command == 'quick':
            print("\n请输入代码（输入空行结束）：")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            code = '\n'.join(lines)
            print("\n快速检查结果：\n")
            result = agent.quick_check(code)
            print(result)
        
        elif command == 'fix':
            print("\n请输入代码（输入空行结束）：")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            code = '\n'.join(lines)
            issue = input("\n请描述问题：")
            
            print("\n生成修复建议...\n")
            result = agent.suggest_fix(code, issue)
            print(result)
        
        else:
            print("未知命令，请重试。")

if __name__ == "__main__":
    main()
```

### 运行示例

```bash
python main.py

命令: quick

请输入代码（输入空行结束）：
def calc(x,y):
    return x+y

快速检查结果：

1. 函数名不够描述性，建议使用更有意义的名称
2. 缺少类型注解，建议添加参数和返回值类型
3. 缺少文档字符串，建议添加函数说明
```

## 本章小结

本章我们学习了：

- ✅ 结构化Prompt设计方法
- ✅ 角色扮演和输出格式控制
- ✅ 高级推理技术：Self-Consistency、Tree of Thoughts、Self-Refine
- ✅ Prompt优化策略和版本管理
- ✅ 构建了一个专业的代码审查Agent

## 下一章

下一章我们将学习Function Calling与工具使用。

[第5章：Function Calling与工具使用 →](/core-skills/chapter5)
