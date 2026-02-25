import argparse
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple
from utils import parse_json_string

from openai import OpenAI


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://chatapi.littlewheat.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-3o7OxkzUFZeQDbUFLIEgiXYXpulbKnKOJ7OAoiMyepxhbnYK")
DEFAULT_JSON = Path(__file__).resolve().parents[1] / "data" / "know" / "internal.json"


@dataclass
class ParagraphHit:
    path: List[str]
    paragraph: str
    para_index: int
    score: float

    def cite(self) -> str:
        # 例：第一章 基础 > 第1节 解剖学 > 细胞 第3段
        chain = " > ".join(self.path)
        return f"{chain} 第{self.para_index}段"


def _tokenize(text: str) -> List[str]:
    """粗粒度中文/英文分词，无外部依赖。"""
    return re.findall(r"[\u4e00-\u9fa5]+|\w+", text)


def _is_informative(para: str) -> bool:
    """过滤掉目录/标题类短句，保留有完整语义的文本。"""
    text = para.strip()
    if not text:
        return False
    if len(text) < 12:
        return False
    if text in {"目录", "回总目录"}:
        return False
    if re.fullmatch(r"[第节章0-9一二三四五六七八九十百]+.*", text) and len(text) <= 20:
        return False
    if any(k in text for k in ("参考", "目录", "回总")) and len(text) < 20:
        return False
    return True


def _flatten(tree: Dict[str, Any], chain: List[str]) -> List[Tuple[List[str], str, int]]:
    """将嵌套字典摊平为 (path, paragraph, idx)，仅保留有意义的段落。"""
    entries: List[Tuple[List[str], str, int]] = []
    for title, node in tree.items():
        path = chain + [title]
        paragraphs = node.get("paragraphs", []) or []
        for i, para in enumerate(paragraphs, start=1):
            clean = para.strip()
            if clean and _is_informative(clean):
                entries.append((path, clean, i))
        children = node.get("children", {}) or {}
        if children:
            entries.extend(_flatten(children, path))
    return entries


def _score(paragraph: str, query_tokens: List[str]) -> float:
    """简单重合度+相似度评分，偏向包含关键词的段落。"""
    para_lower = paragraph.lower()
    token_hits = sum(1 for t in query_tokens if t and t.lower() in para_lower)
    sim = SequenceMatcher(None, para_lower, "".join(query_tokens).lower()).ratio()
    length_bonus = min(len(paragraph) / 200, 1.0)  # 偏向有完整语义的段落
    return token_hits * 2.0 + sim + length_bonus


def _score_with_path(paragraph: str, path: List[str], query_tokens: List[str]) -> float:
    """综合段落与路径的匹配度，弱化无关章节."""
    base = _score(paragraph, query_tokens)
    path_text = " ".join(path).lower()
    path_hits = sum(1 for t in query_tokens if t and t.lower() in path_text)
    return base + path_hits * 1.0


def _build_tokens(question: str, refined: str) -> List[str]:
    """生成用于匹配的关键词集合，去掉无意义的短 token，并加入常见同义词。"""
    raw = _tokenize(question) + _tokenize(refined)
    tokens = [t for t in raw if len(t) >= 2]

    # 口腔顶部/腭部常见俗语同义词
    synonyms = {
        "牙膛": ["腭", "硬腭", "上颚", "上腭", "上牙床", "上腭部"],
        "上牙膛": ["腭", "硬腭", "上颚", "上腭", "上牙床", "上腭部"],
        "上颚": ["腭", "硬腭", "上腭部"],
    }
    for t in list(tokens):
        if t in synonyms:
            tokens.extend(synonyms[t])

    # 去重
    seen = set()
    uniq: List[str] = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            uniq.append(tl)
    return uniq


def _has_match(text: str, tokens: List[str]) -> bool:
    tl = text.lower()
    return any(t and t in tl for t in tokens)


def rewrite_query(question: str) -> str:
    """调用 LLM 进行医学化改写。"""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    prompt = """
# Role
你是一个基于《默克家庭诊疗手册》架构的智能检索路由专家。你的任务是将用户的自然语言（通常是口语化、模糊的）转化为用于 JSON 知识库的结构化搜索策略。

# Knowledge Base Structure
知识库为 JSON 嵌套结构：
- Level 1: 章 (Chapter) —— 对应人体系统或大类疾病（如“心脏和血管疾病”、“皮肤病”、“消化系统疾病”）
- Level 2: 节 (Section) —— 对应具体病症或解剖部位
- Level 3: 段落 (Paragraph) —— 具体的正文内容

# Critical Thinking Protocol (思考协议)
用户输入往往存在**歧义**。你**必须**执行以下逻辑检查，严禁直接进行单一路径搜索：

1. **解剖层级歧义检查**：
   - 用户的描述是在指“皮肤/表面”还是“器官/内部”？
   - *策略*：如果无法确定，必须拆分为 [皮肤/表层路径] 和 [器官/深层路径] 两个策略。

2. **症状成因歧义检查**：
   - 症状是“外伤/物理性”的还是“病理/自发性”的？
   - *策略*：分别构建对应的关键词（如：检索“骨折/擦伤” vs 检索“肿瘤/炎症”）。

3. **术语映射**：
   - 将口语名词（如“肚子”、“包”、“红印”）转换为《默克手册》的标准医学术语（如“腹部”、“肿块/囊肿/结节”、“红斑/皮疹”）。

# Output Schema
请仅输出 JSON，格式严格遵守如下定义：

```json
{
  "query_analysis": {
    "original_query": "用户原文",
    "ambiguity_detected": true, // 是否存在歧义
    "reasoning": "简述为什么认为有歧义，或者为什么没有歧义"
  },
  "search_strategies": [
    // 策略列表：如果不含歧义，则列表长度为1；如果有歧义，则针对每个可能性生成一个策略
    {
      "scenario_name": "策略名称 (例如：针对表层皮肤的检索)",
      "primary_intent": "medical_term_translation (转化后的核心医学术语)",
      "target_chapter_scope": ["可能的章节名1", "可能的章节名2"], // 预测最可能出现的章节
      "search_keywords": ["核心词", "同义词", "高频特征词"] // 用于底层文本匹配
    },
    {
      "scenario_name": "策略名称 (例如：针对内部器官的检索)",
      ...
    }
  ]
}
```
    """
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def search_merck(question: str, json_path: Path = DEFAULT_JSON, top_k: int = 3, use_llm: bool = True):
    """改写查询并在本地 JSON 中检索相关段落。"""
    if not json_path.exists():
        raise FileNotFoundError(f"未找到知识库文件：{json_path}")

    refined = rewrite_query(question) if use_llm else question
    print(refined)
    import pdb
    pdb.set_trace()
    # refined_dic = parse_json_string(refined)
    query_tokens = _build_tokens(question, refined)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hits: List[ParagraphHit] = []
    for path, para, idx in _flatten(data, []):
        # 强制至少在段落或路径匹配到关键 token，否则跳过
        if not (_has_match(para, query_tokens) or _has_match(" ".join(path), query_tokens)):
            continue
        score = _score_with_path(para, path, query_tokens)
        hits.append(ParagraphHit(path=path, paragraph=para, para_index=idx, score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return refined, hits[:top_k]


def _format_results(refined: str, hits: List[ParagraphHit]) -> str:
    lines = [f"改写后查询：{refined}", "检索结果："]
    if not hits:
        lines.append("未找到相关段落")
        return "\n".join(lines)
    for h in hits:
        lines.append(f"- {h.cite()}：{h.paragraph}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="默克家庭医学指南本地检索")
    parser.add_argument(
        "--question",
        nargs="?",
        default="上牙膛破了怎么办",
        help="用户自然语言问题",
    )
    parser.add_argument("-k", "--top-k", type=int, default=3, help="返回候选数量，默认 3")
    parser.add_argument(
        "-j",
        "--json",
        default=str(DEFAULT_JSON),
        help="知识库 JSON 路径，默认 data/know/diagnose.json",
    )
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 改写，直接用原问题检索")
    args = parser.parse_args()

    refined, hits = search_merck(
        question=args.question,
        json_path=Path(args.json),
        top_k=args.top_k,
        use_llm=not args.no_llm,
    )
    print(_format_results(refined, hits))


if __name__ == "__main__":
    main()
