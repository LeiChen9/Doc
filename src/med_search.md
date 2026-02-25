这是一份针对**医学垂直领域 RAG 搜索引擎**的详细技术设计文档。

文档采用**模块化设计**，确保每个组件可以解耦开发、独立测试。核心目标是将用户的口语化提问转化为精准的医学检索，并提供可溯源的段落引用。

-----

# 医学指南智能搜索引擎技术方案 (Medical Guide RAG Engine)

## 1\. 架构概览

系统旨在弥合“患者口语”与“专业医学文本”之间的语义鸿沟。

**核心流程**：
`User Query` → **[Module 2: 改写]** → `Medical Queries` → **[Module 3: 检索 & 重排]** → `Ranked Chunks` → **[Module 4: 组装]** → `Final Answer with Citation`

**依赖服务**：

  * **Vector DB**: Postgres (轻量，本地化，无需复杂部署)
  * **LLM Service**: OpenAI (用于改写)
  * **Embedding**: `BAAI/bge-m3` (支持多语言，长文本)
  * **Rerank**: `BAAI/bge-reranker-v2-m3` (重排序，提升精度)

-----

## 2\. 模块详细设计

### 模块一：数据索引 (Indexer)

**功能**：负责读取嵌套的 JSON 文件，将其“扁平化”为向量库可接受的文档块（Chunks），并注入用于引用的元数据。

#### 核心逻辑

1.  **递归遍历**：遍历 JSON 树，维护 `title_chain`（面包屑路径）。
2.  **上下文增强**：将路径信息拼接到文本头部，增强语义（例如：`text = "口腔疾病 > 唇部疾病: 段落内容..."`）。
3.  **元数据注入**：将章节层级存入 `metadata`，而非 `text`，用于后续引用生成。

#### 数据结构定义 (Schema)

写入向量库的数据格式：

```python
class DocumentChunk:
    id: str          # uuid
    text: str        # 用于 Embedding 的文本 (包含路径前缀)
    metadata: dict = {
        "source": "merck_guide",
        "chapter": "第8章 口腔疾病",     # title_chain[0]
        "section": "第1节 唇部疾病",     # title_chain[1]
        "subsection": "口角炎",         # title_chain[2]
        "para_index": 5,               # 段落原文索引
        "raw_text": "..."              # 原始纯文本(不含路径前缀)，用于展示
    }
```

#### 独立验证/测试用例

  * **输入**：
    ```json
    {
      "第1章": { "children": { "第1节": { "paragraphs": ["内容A", "内容B"] } } }
    }
    ```
  * **预期输出**：
    返回列表包含 2 个 Document 对象。
      * Doc1 Metadata: `{"chapter": "第1章", "section": "第1节", "para_index": 0}`
      * Doc1 Text: `"第1章 - 第1节: 内容A"`

-----

### 模块二：查询理解与改写 (Query Refiner)

**功能**：利用 LLM 将用户的自然语言（Layman terms）转换为专业的医学查询（Medical terms），并提取关键过滤条件。

#### 核心逻辑 (Prompt Design)

使用 Few-Shot Prompting 指导 LLM。

```python
SYSTEM_PROMPT = """
你是一个医学搜索助手。用户的输入是口语化的症状描述。
任务：
1. 分析用户意图。
2. 将口语转换为 1-3 个对应的专业医学术语查询 (Queries)。
3. 提取解剖学位置或关键症状作为关键词 (Keywords)。

输出格式(JSON):
{
    "medical_queries": ["专业查询1", "专业查询2"],
    "keywords": ["部位", "症状"]
}
"""
```

#### 独立验证/测试用例

  * **输入 (User)**: `"上牙膛破了怎么办"`
  * **代码调用**: `refiner.rewrite("上牙膛破了怎么办")`
  * **预期输出**:
    ```json
    {
        "medical_queries": [
            "硬腭黏膜损伤治疗",
            "口腔上腭溃疡处理",
            "腭部物理性创伤"
        ],
        "keywords": ["硬腭", "上腭", "溃疡", "创伤"]
    }
    ```

-----

### 模块三：混合检索与重排序 (Retriever & Reranker)

**功能**：执行从海量数据中召回相关段落，并进行精细化排序。这是精度的关键。

#### 核心流程

1.  **多路召回 (Hybrid Retrieval)**:
      * **向量路**: 对 `medical_queries` 进行 Embedding，在 ChromaDB 检索 Top-20。
      * *(可选) 关键词路*: 对 `keywords` 进行 BM25 检索（若 ChromaDB 版本支持或使用外部库）。
      * 这里为了**简单鲁棒**，建议 V1 版本**只使用向量检索**，但利用 Query Expansion（改写出的3个 query）来模拟多路召回。
2.  **重排序 (Reranking)**:
      * 使用 Cross-Encoder 模型对 `(原始问题, 召回段落)` 进行打分。
      * 过滤掉 Score \< 0.3 的低质量结果。
      * 截取 Top-5。

#### 代码逻辑示意

```python
def retrieve(user_query, expanded_queries):
    # 1. 向量召回 (使用扩展后的 Query)
    candidates = []
    for q in expanded_queries:
        results = vector_db.similarity_search(q, k=10)
        candidates.extend(results)
    
    # 2. 去重 (根据 ID)
    unique_candidates = {doc.id: doc for doc in candidates}.values()
    
    # 3. 重排序 (关键步骤)
    # 输入对：[(user_query, doc1_text), (user_query, doc2_text)...]
    scores = reranker_model.compute_score([[user_query, doc.page_content] for doc in unique_candidates])
    
    # 4. 排序并截断
    ranked_results = sorted(zip(unique_candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked_results[:5]
```

#### 独立验证/测试用例

  * **前置条件**: 索引库中已有“硬腭损伤”相关段落。
  * **输入**: Query="上牙膛破了", Expanded=["硬腭损伤", "口腔溃疡"]。
  * **预期输出**:
      * 返回 Top 5 列表。
      * Top 1 的 `raw_text` 应包含“硬腭...愈合...治疗”等内容。
      * Top 1 的 Rerank Score 应 \> 0.7。

-----

### 模块四：结果组装与引用 (Response Formatter)

**功能**：将检索到的结构化数据转换成用户友好的回复，并生成规范的引用格式。

#### 核心逻辑

1.  **引用生成器**：解析 metadata，拼接字符串。
2.  **溯源标记**：明确指出内容来源。

#### 引用格式规范

`{chapter} > {section} > {subsection} (第 {para_index} 段)`

#### 代码逻辑示意

```python
def format_output(results):
    final_output = []
    for doc, score in results:
        meta = doc.metadata
        # 动态构建路径，处理有的层级可能为空的情况
        path = [meta.get(k) for k in ['chapter', 'section', 'subsection'] if meta.get(k)]
        citation = f"{' > '.join(path)} (第{meta['para_index'] + 1}段)"
        
        final_output.append({
            "content": meta['raw_text'],
            "citation": citation,
            "relevance": f"{score:.2f}"
        })
    return final_output
```

#### 独立验证/测试用例

  * **输入**: Document 对象 (Metadata包含: Chapter="第8章", Section="第1节", para\_index=5).
  * **预期输出**: 字符串 `"第8章 > 第1节 (第6段)"`。

-----

## 3\. 落地实施路线 (Roadmap)

### 第一阶段：环境与数据准备 (Day 1)

1.  **环境搭建**: 安装 `chromadb`, `langchain`, `sentence-transformers`, `FlagEmbedding`.
2.  **数据转换**: 编写脚本运行 **模块一**，将你的 `merck_guide.json` 转换为 `chroma_db` 本地文件夹。
      * *Check*: 检查 ChromaDB 中是否能查到数据。

### 第二阶段：检索引擎开发 (Day 2)

1.  **改写器开发**: 申请 LLM API Key，编写 **模块二** 的 Prompt 并调试。
2.  **检索器开发**: 集成 BGE-Reranker 模型（首次运行会自动下载权重，约 1-2GB）。
      * *Check*: 输入“心脏病”，确认能召回并排序。

### 第三阶段：端到端联调 (Day 3)

1.  **Main Loop**: 串联所有模块。
2.  **Bad Case 分析**: 找几个刁钻的问题（如俚语），查看改写模块是否生效。

## 4\. 扩展性设计 (Future Proofing)

1.  **支持多模态**: 你的 JSON 里提取了 `images`。未来可以在 `metadata` 中加入 `image_refs`，在展示段落时同时展示对应的图片。
2.  **即时问答**: 目前是“搜索引擎”模式（列出段落）。下一步可以将 Retrieved Contexts 喂给 LLM，让 LLM 生成一句总结性的回答（RAG Generation）。
3.  **用户反馈**: 记录用户的 Query 和最终点击的段落，用于后续微调 Reranker 模型。

## 5\. 示例运行代码 (Main Entry)

```python
# main.py 伪代码结构
from modules import Indexer, Rewriter, Retriever, Formatter

def main():
    # 1. 初始化 (通常只需加载一次)
    db = Indexer.load_db("./chroma_data")
    reranker = Retriever.load_reranker()
    
    # 2. 用户交互
    user_query = input("请输入症状: ")  # e.g., "上牙膛破了"
    
    # 3. 改写
    print("正在分析病情...")
    rewritten = Rewriter.rewrite(user_query)
    print(f"医学转化: {rewritten['medical_queries']}")
    
    # 4. 检索
    print("正在检索默克指南...")
    results = Retriever.search(
        query=user_query, 
        expanded_queries=rewritten['medical_queries'],
        db=db,
        reranker=reranker
    )
    
    # 5. 展示
    formatted_results = Formatter.format(results)
    for i, res in enumerate(formatted_results):
        print(f"\n--- 结果 {i+1} (匹配度: {res['relevance']}) ---")
        print(f"来源: {res['citation']}")
        print(f"内容: {res['content'][:100]}...") # 预览前100字

if __name__ == "__main__":
    main()
```