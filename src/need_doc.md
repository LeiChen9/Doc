# EPUB 解析需求（最新实现对齐版）

目标：用 Python 解析 EPUB2/3，生成可用于知识索引的“有序嵌套字典”，保留书籍原目录顺序与层级，节点含文本与图片。

## 目录解析
- 优先 `nav.xhtml`（EPUB3），无则回退 `toc.ncx`（EPUB2）。
- 递归保留层级关系（章→节→小节等）；忽略“目录”节点本身，防止吞并正文。
- 产出 `TocEntry`：`title_chain`、`href`、`fragment`（锚点）。

## 内容解析
- 按 `href` 分组，同一 HTML 内按目录顺序分段：为每个目录项定位起始节点（优先 fragment，其次标题匹配），提取到下一个目录项之前。
- 收集段落（`<p>/<li>` 文本）与图片（`src`、`alt`），范围限定在对应目录片段。

## 输出结构
嵌套有序字典，节点字段：
```
{
  "标题": {
    "href": "text001.html",
    "paragraphs": [...],
    "images": [...],
    "children": { ... }
  }
}
```
层级推断规则：含“章/附录”记作 1 级，含“节”记作 2 级，其余 3 级（可按需微调），用栈维护父子关系。

## 非功能要求
- 使用 BeautifulSoup（`lxml`/`xml` 解析），保持代码简洁清晰。
- 正确处理 TOC 与内容文件映射，避免跨节点串段；默认 CLI 输入/输出已配置，可直接运行。

# 搜索引擎
根据 site: epub_extractor_TECH.md中实现的方案以及 data/know/merck_guide.json 中的示例存储。我需要一个能根据用户的自然语言问题进行医学化的 query 改写并在对应的 json 中找到相应的段落的搜索引擎。我的流程是：用户输出一个自然语言问题比如 “上牙膛破了怎么办”，搜索引擎使用 LLM 结合默克家庭医学指南中的内容以及自身医学的理解将其改写为专业的医学问题，并在对应的 json 中进行寻找，最终找出合适的段落出来，并表明 cite 来源（第 xx 章第 xx 节第 xx 小节第 xx 段落：内容）
LLM 调用方式：
```python
from openai import OpenAI # 导入OpenAI

client = OpenAI(base_url = "http://chatapi.littlewheat.com/v1",
                api_key  = "sk-3o7OxkzUFZeQDbUFLIEgiXYXpulbKnKOJ7OAoiMyepxhbnYK")

completion = client.chat.completions.create(
  model="gpt-4o-mini", # this field is currently unused
  messages=[
    {"role": "system", "content": "你是一位医学专家，根据用户的问题，给出专业的医学回答。"},
    {"role": "user", "content": "上牙膛破了怎么办"}
  ],
  temperature=0.7,
)
print(completion.choices[0].message.content)# 输出文案
```