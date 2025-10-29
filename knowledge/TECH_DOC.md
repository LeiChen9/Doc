# 知识抽取→整理→嵌入→落库 技术文档（EPUB 实验阶段）

## 1. 目标与范围
本技术文档说明 `knowledge/` 模块在实验阶段（单个本地 EPUB 文件）如何完成：
- 解析 EPUB → 粗粒度切分 → 文本清洗
- 实体抽取与标签生成（占位实现，可替换 LLM/NER）
- 语义向量生成（占位嵌入，可替换 BGE/sentence-transformers）
- 结果落地为 JSONL，并可选落库至 PostgreSQL + pgvector

说明面向开发者，便于快速对接、替换与扩展。

## 2. 模块与职责
- `knowledge/config.py`：全局配置（数据目录、嵌入维度、数据库 DSN 等）。
- `knowledge/epub_parser.py`：EPUB 解析器，读取 OPF、遍历 spine 顺序，提取章节标题线索与文本，产出原子块。
- `knowledge/chunker.py`：清洗与长度过滤，归一化空白、去噪、最小长度阈值。
- `knowledge/entity_extractor.py`：占位实体与标签抽取（规则法），可替换为 LLM/NER。
- `knowledge/embedder.py`：占位嵌入（哈希→单位向量），可替换为真实模型。
- `knowledge/db.py`：JSONL 写出与 PostgreSQL 持久化（建表、UPSERT）。
- `knowledge/pipeline.py`：端到端流水线，串联上述步骤并输出 JSONL/落库。

## 3. 数据流与产出
1) EPUB 解析输出（逻辑块 block）：
```json
{
  "id": "uuid",
  "title": "章节或小节标题",
  "chapter_path": "上级标题 > 当前标题",
  "text": "原文段落",
  "source_meta": {"title": "书名", "creator": "作者", "publish_date": "日期"},
  "source_file": "merck_guide.epub",
  "page_hint": 3
}
```

2) 实体与标签附加：
```json
{
  "entity": "识别出的实体（占位规则）",
  "tags": ["诊断", "治疗"],
  ... 原字段保持 ...
}
```

3) 生成记录 record（用于 JSONL/落库）：
```json
{
  "id": "uuid",
  "entity": "实体",
  "title": "标题",
  "text": "文本",
  "source": "merck_guide.epub",
  "chapter_path": "...",
  "tags": ["..."],
  "metadata": {"publish_date": "...", "creator": "...", "page_hint": 3},
  "vector": [0.01, 0.02, ...]  // 维度由配置决定
}
```

4) 产出文件：
- `data/knowledge_records.jsonl`：按行存储上面的记录对象（UTF-8）。
- 若配置 `PG_DSN`，会在数据库中创建/写入 `knowledge_records` 表。

## 4. 表结构（PostgreSQL + pgvector）
由 `knowledge/db.py` 自动建表：
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_records (
  id UUID PRIMARY KEY,
  entity VARCHAR(200) NOT NULL,
  title TEXT NOT NULL,
  text TEXT NOT NULL,
  source TEXT NOT NULL,
  chapter_path TEXT NOT NULL,
  tags JSONB NOT NULL,
  metadata JSONB NOT NULL,
  vector VECTOR(<dim>)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge_records(entity);
CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_records(source);
CREATE INDEX IF NOT EXISTS idx_knowledge_chapter_path ON knowledge_records(chapter_path);
```
说明：`<dim>` 由配置中的 `EMBEDDING_DIM` 决定（默认 384）。后续可扩展 HNSW 索引与全文索引（tsvector）。

## 5. 流水线执行
运行：
```bash
python -m knowledge.pipeline
```
行为：
1) 扫描 `data/*.epub`，在实验阶段仅处理第一个匹配文件。
2) 解析 OPF + 遍历 spine，收集标题线索与段落文本。
3) 文本清洗与长度过滤（默认最少 50 字）。
4) 实体/标签占位抽取。
5) 向量占位生成并合并为记录对象。
6) 写出 `data/knowledge_records.jsonl`；若设置 `PG_DSN`，写入数据库。

## 6. 配置与环境变量
- `PG_DSN`：PostgreSQL 连接串，如 `postgresql://user:pass@localhost:5432/db`。留空则仅输出 JSONL。
- `EMBEDDING_BACKEND`：`dummy`（默认，占位向量）；后续可接入 `sentence-transformers`/`bge-small-zh`。
- `EMBEDDING_DIM`：嵌入维度，默认 `384`。

## 7. 替换占位实现的建议
实体抽取：
- 将 `entity_extractor.attach_entity` 替换为 LLM 接口或医学 NER（如 UMLS 词表映射）。
- 做实体标准化与同义词归一化，保证 `entity` 字段的一致性。

向量生成：
- 在 `embedder.get_embedder` 中加载真实模型（`sentence-transformers` 或 BGE-small-zh）。
- 批量 `encode` 以提升性能；注意显存/内存管理。

索引优化（后续）：
- 为 `vector` 列构建 HNSW 索引（pgvector），并添加全文索引（tsvector）用于关键词过滤。

## 8. 质量与校验
- 最小长度过滤与去噪，避免空文本与重复段落。
- 输出 JSONL 后可抽检若干记录，检查 `entity`、`chapter_path` 与原文一致性。
- 落库后随机抽查行数、索引存在性与查询延迟。

## 9. 已知限制（实验阶段）
- EPUB 无硬页码：使用 `spine` 序号作为 `page_hint`（近似参考）。
- 目录层级：当前通过章节标题线索构造 `chapter_path`，尚未读取 NCX/标注跳转，复杂目录可能需加强。
- 实体与向量为占位实现，建议尽快替换为真实模型以用于生产评估。


