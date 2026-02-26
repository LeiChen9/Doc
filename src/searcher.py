import json
import os
import logging
from typing import List, Dict, Tuple, Optional
import pdb
from sentence_transformers import SentenceTransformer, util  # 用于语义相似度匹配
from openai import OpenAI


# 基本日志配置，简单记录系统关键流程
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 配置OpenAI API（替换为你的API密钥；如果使用xAI，调整客户端）
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://chatapi.littlewheat.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-3o7OxkzUFZeQDbUFLIEgiXYXpulbKnKOJ7OAoiMyepxhbnYK")

# 加载嵌入模型（用于语义相似度）
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级嵌入模型

# 加载知识库JSON（假设路径为固定；可参数化）
KNOWLEDGE_JSON_PATH = "../data/know/merck_guide_aug.json"

def load_knowledge_base() -> Dict:
    """加载JSON知识库，返回book_tree和toc_tree"""
    with open(KNOWLEDGE_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("知识库加载完成，book_tree 章节数=%d", len(data.get("book_tree", {})))
    return data["book_tree"], data["search_guide"]["toc_tree"]

# 扁平化book_tree为段落列表，每个项为(路径字符串, 段落文本)
def flatten_book_tree(book_tree: Dict, current_path: str = "") -> List[Tuple[str, str]]:
    flat_list = []
    for title, node in book_tree.items():
        new_path = f"{current_path} > {title}" if current_path else title
        for para in node.get("paragraphs", []):
            flat_list.append((new_path, para))
        flat_list.extend(flatten_book_tree(node.get("children", {}), new_path))
    return flat_list


def _build_toc_recall_units(toc_tree: Dict) -> List[Tuple[str, str]]:
    """
    为 toc 语义召回构建 (key, value) 单元：
    - key: "章" 或 "章 > 节"
    - value: 该 key 下所有「章-节-小节」名称的汇总文本，用于做语义相似度匹配

    示例：
    key = "消化系统 > 胃炎"
    value = "消化系统 > 胃炎: 急性胃炎; 慢性胃炎; 萎缩性胃炎"
    """
    units: List[Tuple[str, str]] = []
    for chapter, sections in toc_tree.items():
        # 章节级：value 包含该章下所有节与子节名称
        for sec, subsecs in sections.items():
            units.append((f"{chapter} > {sec}", f"{chapter} > {sec}: " + "; ".join(str(s) for s in subsecs)))
    return units


def semantic_recall_toc(user_input: str, toc_tree: Dict, top_k: int = 20) -> List[str]:
    """
    对 toc_tree 做一次基于语义相似度的召回，返回按相关度排序的路径列表。
    注意：这里只是为了给 LLM 提示一个更短的目录视图，真正的章节/段落检索仍然使用完整 toc_tree，
    因此不会牺牲整体 recall。
    """
    units = _build_toc_recall_units(toc_tree)
    if not units:
        return []
    keys = [k for k, _ in units]
    texts = [v for _, v in units]
    query_emb = embedder.encode(user_input)
    toc_embs = embedder.encode(texts)
    scores = util.pytorch_cos_sim(query_emb, toc_embs)[0]
    k = min(top_k, len(keys))
    top_indices = scores.topk(k).indices.tolist()
    recalled = [keys[i] for i in top_indices]
    logger.info("semantic_recall_toc 完成，目录单元数=%d，top_k=%d", len(keys), len(recalled))
    pdb.set_trace()
    return recalled


def _build_recalled_outline(recalled_paths: List[str], toc_tree: Dict) -> str:
    """
    根据语义召回得到的路径，展开对应「章-节」的本地目录。
    - 对于只有章节的路径：展示该章节下所有节与子节
    - 对于“章 > 节”的路径：展示该节下的子节
    """
    blocks: List[str] = []
    for path in recalled_paths:
        if " > " in path:
            chapter, section = path.split(" > ", 1)
        else:
            chapter, section = path, None

        sections = toc_tree.get(chapter, {})
        if not isinstance(sections, dict):
            continue

        lines: List[str] = []
        if section is None:
            # 展开整章的节与子节
            lines.append(f"章节: {chapter}")
            for sec, subsecs in sections.items():
                lines.append(f"  - 节: {sec}")
                for sub in subsecs:
                    lines.append(f"    - 子节: {sub}")
        else:
            subsecs = sections.get(section)
            if subsecs is None:
                continue
            lines.append(f"章节: {chapter} > 节: {section}")
            for sub in subsecs:
                lines.append(f"  - 子节: {sub}")

        if lines:
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks)

# LLM调用函数
def llm_call(prompt: str, model: str = "gpt-4-turbo") -> str:
    """调用LLM生成响应，简单包装一下 Chat Completions 接口"""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    logger.info("调用 LLM，模型=%s", model)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "你是一个严格遵守指令的助手。当用户要求你输出 JSON 时，务必返回严格合法、可直接被 json.loads 解析的 JSON 字符串，不要包含任何多余解释、自然语言或代码块标记。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def _parse_llm_json(raw: str) -> dict:
    """
    解析 LLM 返回的 JSON 字符串。
    - 去掉可能的 ```json ``` 代码块包裹
    - 尝试使用 json.loads 解析
    """
    text = raw.strip()
    # 处理 ```json ... ``` 或 ``` ... ``` 包裹的情况
    if text.startswith("```"):
        # 去掉开头 ```
        text = text.lstrip("`")
        # 去掉可能的 json 语言标识
        if text.lower().startswith("json"):
            text = text[4:]  # 去掉 'json' 和后面的一个换行/空格（大致处理即可）
        # 再次去掉到第一个换行之前的内容
        if "\n" in text:
            text = text.split("\n", 1)[1]
        # 去掉末尾 ```
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # 打印一小段方便排查，但避免把完整内容打爆日志
        logger.warning("解析 LLM JSON 失败，raw 前 200 字符: %s", text[:200])
        return {}

# 步骤1: LLM refine用户查询，翻译成专业医学问题并提取关键词
def refine_query(user_input: str, toc_tree: Dict) -> Tuple[str, List[str]]:
    """
    使用 LLM 对用户查询进行医学翻译与关键词推理。
    注意：这一步只依赖用户输入本身，不依赖 toc_tree，以避免目录结构对推理产生偏置。
    """
    prompt = f"""
你是一个专业医学助手，负责标准化用户问题并抽取医学关键词。

现在给你一条用户的自然语言问题（可能带有口语化、方言、比喻或不规范的描述），
你的任务是将其翻译成规范的医学问题，并推理出与之高度相关的医学关键词，用于后续在教科书中进行检索。

请你：
1. 先根据你的医学知识，将用户的自然语言问题翻译为一个简明、规范的「专业医学问题」（可以适当补全隐含信息，但不要引入明显无关内容）。
2. 在你内心中，对所有可能相关的医学概念、解剖结构、疾病名称、症状体征、检查方法和治疗手段进行一次“穷举思考”，找出所有**有较大可能**会被用来检索本问题相关内容的医学关键词或常用近义表达。
3. 对于每一个候选关键词，在你内部为它估计一个「被用于检索相关医学内容的合理概率」p（0~1 之间）。**只有当 p ≥ 0.05 时才保留该关键词**；p 更低（“几乎不会用来检索”的）候选词一律丢弃，以避免关键词爆炸。
4. 最终输出的关键词列表应：既尽量覆盖所有真正可能相关的医学表达（高召回），又避免引入太多牵强的远房关联（控制噪声），列表长度可以变化，但不要为了凑数而增加弱相关词。
5. 用自然语言简要说明：你是如何从原始问题改写出 refined_query 的，以及你选择这些关键词的依据和筛选逻辑（包括你如何应用「p ≥ 0.05」这一阈值），但**不要在说明中展开你的完整推理过程，只给出简洁结论**。

严格按照下面的 JSON 模板输出，必须是合法的 JSON 且只能包含这一段 JSON，不要输出任何额外文字、解释或注释：
{{
  "refined_query": "用一句话表述的专业医学问题",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "rationale": "一句或几句自然语言，解释 refined_query 的改写逻辑以及关键词选取逻辑"
}}

输入：
用户查询: {user_input}
"""
    response = llm_call(prompt)
    data = _parse_llm_json(response)
    refined_query = data.get("refined_query", "") if isinstance(data, dict) else ""
    keywords = data.get("keywords", []) if isinstance(data, dict) else []
    # 解释说明：为什么这样改写、为何选择这些关键词
    rationale = data.get("rationale", "") if isinstance(data, dict) else ""
    # 确保 keywords 是字符串列表
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if k]
    logger.info("refine_query 完成，refined='%s', 关键词数=%d", refined_query, len(keywords))
    if rationale:
        # 打印一条详细说明，便于理解 LLM 的改写与选词逻辑
        logger.info("refine_query 说明：%s", rationale)
    # pdb.set_trace()
    return refined_query, keywords

# 步骤2: 使用关键词匹配章节
def match_chapters(keywords: List[str], toc_tree: Dict) -> List[str]:
    """关键词匹配章节，返回匹配的章节路径列表"""
    matched_paths = []
    for chapter, sections in toc_tree.items():
        if any(kw.lower() in chapter.lower() for kw in keywords):
            matched_paths.append(chapter)
        for section, subsections in sections.items():
            if any(kw.lower() in section.lower() for kw in keywords):
                matched_paths.append(f"{chapter} > {section}")
            for sub in subsections:
                if any(kw.lower() in sub.lower() for kw in keywords):
                    matched_paths.append(f"{chapter} > {section} > {sub}")
    matched_paths = list(set(matched_paths))  # 去重
    logger.info("match_chapters 完成，关键词数=%d，匹配章节数=%d", len(keywords), len(matched_paths))
    return matched_paths

# 步骤3: 语义相似度匹配段落
def semantic_match(refined_query: str, flat_paragraphs: List[Tuple[str, str]], top_k: int = 10) -> List[Tuple[str, str]]:
    """使用嵌入模型匹配相关段落"""
    query_emb = embedder.encode(refined_query)
    para_embs = embedder.encode([para for _, para in flat_paragraphs])
    scores = util.pytorch_cos_sim(query_emb, para_embs)[0]
    top_indices = scores.topk(top_k).indices.tolist()
    matched = [(flat_paragraphs[i][0], flat_paragraphs[i][1]) for i in top_indices]
    logger.info("semantic_match 完成，top_k=%d，实际召回段落数=%d", top_k, len(matched))
    return matched

# 步骤4: LLM审阅召回结果，选择最多3个片段
def review_results(chapter_matches: List[str], semantic_matches: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """LLM审阅并选择最多3个最相关片段"""
    recall_str = f"章节匹配: {chapter_matches}\n语义匹配: {[(path, para[:100] + '...') for path, para in semantic_matches]}"
    prompt = f"""
你是一个医学知识检索助手，需要从候选结果中挑选最相关的若干片段。

下面是检索召回的结果（包括章节匹配以及语义匹配的片段摘要）：
{recall_str}

请你综合考虑这些信息，从中选出最多 3 个「最有可能回答用户问题」的片段。

输出要求：
- 只输出一个 JSON 对象，不能包含任何额外说明文字或注释
- JSON 字段如下：
{{
  "segments": [
    {{
      "path": "片段所在路径字符串",
      "paragraph": "该路径下对应的完整段落内容"
    }}
  ]
}}
- 当没有合适片段时，返回 {{ "segments": [] }}
"""
    response = llm_call(prompt)
    data = _parse_llm_json(response)
    segments = []
    if isinstance(data, dict):
        raw_segments = data.get("segments", [])
        if isinstance(raw_segments, list):
            for item in raw_segments:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path", "")).strip()
                para = str(item.get("paragraph", "")).strip()
                if path and para:
                    segments.append((path, para))
    logger.info("review_results 完成，候选章节数=%d，语义段落数=%d，选中片段数=%d", len(chapter_matches), len(semantic_matches), len(segments))
    return segments[:3]

# 步骤5: 根据结果总结并引导提问
def summarize_and_guide(selected_segments: List[Tuple[str, str]], search_history: str, is_empty: bool) -> str:
    """总结搜索过程并引导用户"""
    if is_empty:
        prompt = f"""
        搜索过程总结: {search_history}
        没有找到相关片段。请引导用户提供更多细节，通过提问refine查询。
        
        输出: 总结 + 引导提问
        """
    else:
        segments_str = "\n".join([f"路径: {path}\n片段: {para}" for path, para in selected_segments])
        prompt = f"""
        搜索过程总结: {search_history}
        找到的相关片段: {segments_str}
        
        返回这些片段作为搜索结果，然后总结搜索过程，并引导用户进一步提问以refine。
        
        输出: 搜索结果 + 总结 + 引导提问
        """
    logger.info("summarize_and_guide，是否为空结果=%s，片段数=%d", is_empty, len(selected_segments))
    return llm_call(prompt)

# 主循环
def main():
    logger.info("搜索主循环启动")
    book_tree, toc_tree = load_knowledge_base()
    flat_paragraphs = flatten_book_tree(book_tree)
    logger.info("知识库展开完成，段落总数=%d", len(flat_paragraphs))
    
    search_history = ""  # 记录搜索过程
    
    while True:
        user_input = input("请输入您的查询 (输入 'exit' 退出): ")
        if user_input.lower() == 'exit':
            logger.info("收到 exit 指令，准备退出主循环")
            break
        
        # 步骤1
        logger.info("收到用户查询: %s", user_input)
        refined_query, keywords = refine_query(user_input, toc_tree)
        pdb.set_trace()
        search_history += f"用户查询: {user_input}\nRefined: {refined_query}\n关键词: {keywords}\n"
        
        # 步骤2 & 3 (同步)
        chapter_matches = match_chapters(keywords, toc_tree)
        semantic_matches = semantic_match(refined_query, flat_paragraphs)
        
        search_history += f"章节匹配: {chapter_matches}\n语义匹配数: {len(semantic_matches)}\n"
        
        # 步骤4
        selected_segments = review_results(chapter_matches, semantic_matches)
        
        # 步骤5
        is_empty = len(selected_segments) == 0
        response = summarize_and_guide(selected_segments, search_history, is_empty)
        print(response)
        
        # 步骤6: 循环返回步骤1

if __name__ == "__main__":
    main()