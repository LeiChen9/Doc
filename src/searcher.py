import json
import os
from typing import List, Dict, Tuple, Optional
import pdb
from sentence_transformers import SentenceTransformer, util  # 用于语义相似度匹配
from openai import OpenAI


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

# LLM调用函数
def llm_call(prompt: str, model: str = "gpt-4-turbo") -> str:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    """调用LLM生成响应"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# 步骤1: LLM refine用户查询，翻译成专业医学问题并提取关键词
def refine_query(user_input: str, toc_tree: Dict) -> Tuple[str, List[str]]:
    """使用LLM refine查询并提取关键词"""
    toc_str = json.dumps(toc_tree, ensure_ascii=False, indent=2)
    prompt = f"""
    用户查询: {user_input}
    书籍目录结构 (toc_tree): {toc_str}
    
    结合你的医学知识和书籍目录结构，将用户查询翻译成专业医学问题。
    然后，从专业问题中提取3-5个关键词（医学术语）。
    
    输出格式:
    专业问题: [refined query]
    关键词: [keyword1, keyword2, ...]
    """
    response = llm_call(prompt)
    pdb.set_trace()
    lines = response.split("\n")
    refined_query = lines[0].split(": ", 1)[1] if len(lines) > 0 else ""
    keywords = eval(lines[1].split(": ", 1)[1]) if len(lines) > 1 else []
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
    return list(set(matched_paths))  # 去重

# 步骤3: 语义相似度匹配段落
def semantic_match(refined_query: str, flat_paragraphs: List[Tuple[str, str]], top_k: int = 10) -> List[Tuple[str, str]]:
    """使用嵌入模型匹配相关段落"""
    query_emb = embedder.encode(refined_query)
    para_embs = embedder.encode([para for _, para in flat_paragraphs])
    scores = util.pytorch_cos_sim(query_emb, para_embs)[0]
    top_indices = scores.topk(top_k).indices.tolist()
    matched = [(flat_paragraphs[i][0], flat_paragraphs[i][1]) for i in top_indices]
    return matched

# 步骤4: LLM审阅召回结果，选择最多3个片段
def review_results(chapter_matches: List[str], semantic_matches: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """LLM审阅并选择最多3个最相关片段"""
    recall_str = f"章节匹配: {chapter_matches}\n语义匹配: {[(path, para[:100] + '...') for path, para in semantic_matches]}"
    prompt = f"""
    召回结果: {recall_str}
    
    审阅以上召回结果，选择最多3个最有可能回答用户问题的片段（路径和完整段落）。
    如果没有相关片段，返回空列表。
    
    输出格式: [(路径1, 段落1), (路径2, 段落2), ...] 或 []
    """
    response = llm_call(prompt)
    try:
        selected = eval(response)
    except:
        selected = []
    return selected[:3]

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
    return llm_call(prompt)

# 主循环
def main():
    book_tree, toc_tree = load_knowledge_base()
    flat_paragraphs = flatten_book_tree(book_tree)
    
    search_history = ""  # 记录搜索过程
    
    while True:
        user_input = input("请输入您的查询 (输入 'exit' 退出): ")
        if user_input.lower() == 'exit':
            break
        
        # 步骤1
        refined_query, keywords = refine_query(user_input, toc_tree)
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