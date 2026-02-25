# EPUB 结构化解析器技术文档

## 概述

从 EPUB2/3 文件中提取目录结构并构建嵌套有序字典，每个节点包含 `href`、`paragraphs`、`images`、`children`。

## 系统流程

```
EPUB (zip) → container.xml → OPF → nav.xhtml/toc.ncx → TocEntry[]
    ↓
按 href 分组 → 提取内容（段落/图片）→ 层级推断 → 构建嵌套树
```

## 核心实现

### 1. 目录解析

**EPUB3 (nav.xhtml)**：
- 查找 `<nav epub:type="toc">` 或 `<nav role="doc-toc">`
- 递归遍历 `<ol>/<ul>` → `<li>` → `<a>`，构建 `title_chain`（保留完整路径）

**EPUB2 (toc.ncx)**：
- 解析 `<navMap>` → `<navPoint>` → `<text>` + `<content src>`
- 递归处理子 `navPoint`，构建 `title_chain`

**输出**：`List[TocEntry]`，每个条目包含 `title_chain`、`href`、`fragment`（锚点）

### 2. 内容提取

**分组策略**：按 `href` 分组，避免重复解析同一 HTML 文件。

**分段提取**：
- 对每个 `href` 内的多个 `TocEntry`，定位起始节点（优先 `fragment` 锚点，次选标题匹配）
- 提取从当前起始节点到下一个起始节点之间的内容
- 使用 `next_element` 遍历 DOM，收集 `<p>/<li>` 文本和 `<img>` 属性

**边界处理**：最后一个 entry 提取到文件末尾。

### 3. 层级推断与树构建

**层级规则**（`_infer_level`）：
- 含“章”/“附录” → level 1
- 含“节” → level 2
- 其他 → level 3

**树构建**：
- 使用栈维护当前路径：`stack: List[Tuple[level, node]]`
- 遇到新 entry 时，弹出栈顶直到 `stack[-1].level < current_level`
- 在栈顶节点的 `children` 中创建/更新节点
- 跳过标题为“目录”的节点

**输出结构**：
```json
{
  "第一章 基础": {
    "href": "text001.html",
    "paragraphs": [...],
    "images": [...],
    "children": {
      "第1节 解剖学": {...}
    }
  },
  ...
}
```

## Tricks

1. **XML 解析器**：使用 `BeautifulSoup(..., "xml")` 解析 nav/ncx，忽略 `XMLParsedAsHTMLWarning`
2. **内容缓存**：`soup_cache` 避免重复解析同一 HTML
3. **分段提取**：同文件内按 toc 顺序确定 start/end，避免内容串行溢出
4. **文本标准化**：`_normalize` 统一空白字符，提升标题匹配成功率
5. **层级推断**：基于标题文本模式，无需依赖 HTML 标签层级

## 风险点

1. **层级推断不准确**：纯文本模式可能误判（如“第一章概述”vs“第1节概述”），需根据实际 EPUB 调整规则
2. **内容越界**：`next_element` 可能跨文件边界，需确保 `end_node` 在同一文档内
3. **标题匹配失败**：无 fragment 且标题文本不完全一致时，回退到 `body`，可能提取多余内容
4. **目录污染**：若“目录”节点未被正确跳过，可能吞并后续章节
5. **fragment 缺失**：EPUB2 部分文件可能无锚点，依赖标题匹配的准确性

## 依赖

- `beautifulsoup4`：HTML/XML 解析
- `lxml`：XML 解析器后端

## 使用

```bash
python epub_extractor.py [epub_path] [-o output.json]
```

默认输入：`../data/raw/merck_guide.epub`  
默认输出：`../data/know/merck_guide.json`
