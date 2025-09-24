import fitz  # PyMuPDF
import spacy
import os
import json
import uuid
import logging
from tqdm import tqdm
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("extract_pdf.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy model (optional, for fallback segmentation)
try:
    nlp = spacy.load("zh_core_web_lg")
    logger.info("成功加载 spaCy 模型 zh_core_web_lg")
except OSError as e:
    logger.error("加载 spaCy 模型失败: %s", e)
    logger.error("请运行: python -m spacy download zh_core_web_lg")
    exit(1)

def extract_textbook_metadata(pdf_path: str) -> Dict:
    """提取 PDF 元数据（标题、路径、总页数）。"""
    try:
        doc = fitz.open(pdf_path)
        metadata = {
            "textbook_name": os.path.basename(pdf_path),
            "path": pdf_path,
            "total_pages": doc.page_count
        }
        doc.close()
        return metadata
    except Exception as e:
        logger.error("处理 %s 元数据时出错: %s", pdf_path, e)
        return {}

def extract_toc(pdf_path: str) -> List[Dict]:
    """从 PDF 提取多级目录 (TOC)。"""
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc(simple=False)
        doc.close()
        
        structured_toc = []
        level_stack = [structured_toc]  # Stack to track nested levels
        
        for entry in toc:
            if len(entry) < 3:
                logger.warning("跳过无效 TOC 条目: %s", entry)
                continue
            level = entry[0]
            title = entry[1].replace("\r", "").replace("\n", "").strip()
            title = " ".join(title.split())
            page = entry[2]
            
            if not title or page < 1:
                logger.warning("跳过无效标题或页面: %s", entry)
                continue
            
            # Adjust level stack to match current level
            while len(level_stack) < level:
                level_stack.append(level_stack[-1][-1]["subsections"])
            while len(level_stack) > level:
                level_stack.pop()
            
            # Create new node
            node = {
                "title": title,
                "level": level,
                "start_page": page,
                "end_page": page,
                "subsections": []
            }
            level_stack[-1].append(node)
        
        # Update end_page for all nodes
        def update_end_pages(nodes: List[Dict], total_pages: int):
            for i, node in enumerate(nodes[:-1]):
                node["end_page"] = nodes[i + 1]["start_page"] - 1
                update_end_pages(node["subsections"], total_pages)
            if nodes:
                nodes[-1]["end_page"] = total_pages
                update_end_pages(nodes[-1]["subsections"], total_pages)
        
        update_end_pages(structured_toc, fitz.open(pdf_path).page_count)
        logger.info("从 %s 提取 %d 个顶级目录项", pdf_path, len(structured_toc))
        return structured_toc
    except Exception as e:
        logger.error("从 %s 提取目录时出错: %s", pdf_path, e)
        return []

def segment_text_to_blocks(text: str, min_block_length: int = 50) -> List[str]:
    """将文本分割为语义完整的块，优先使用空行，合并短块。"""
    try:
        # Split by double newlines (common paragraph separator)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        merged_blocks = []
        current_block = ""
        
        for block in blocks:
            current_block += block + "\n"
            if len(current_block) >= min_block_length:
                merged_blocks.append(current_block.strip())
                current_block = ""
        
        if current_block.strip():
            if merged_blocks and len(current_block) < min_block_length:
                merged_blocks[-1] += "\n" + current_block.strip()
            else:
                merged_blocks.append(current_block.strip())
        
        return merged_blocks
    except Exception as e:
        logger.error("分块文本时出错: %s", e)
        return [text]

def extract_content(pdf_path: str, toc: List[Dict]) -> List[Dict]:
    """从 PDF 提取内容并组织为知识片段，按最深层级标题分块。"""
    try:
        doc = fitz.open(pdf_path)
        knowledge_fragments = []
        
        def process_node(node: Dict, title_path: List[str], parent_path: List[str]):
            current_path = title_path + [node["title"]]
            start_page = node["start_page"] - 1
            end_page = node["end_page"] - 1
            
            # Extract text for current node
            node_text = ""
            for page_num in tqdm(range(start_page, end_page + 1), desc=f"提取 {node['title']} 页面"):
                try:
                    page = doc[page_num]
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if block[4].strip():  # block[4] is the text content
                            node_text += block[4].strip() + "\n\n"
                except Exception as e:
                    logger.warning("提取 %s 页面 %d 时出错: %s", pdf_path, page_num + 1, e)
                    continue
            
            # Segment into semantic blocks
            blocks = segment_text_to_blocks(node_text)
            logger.info("从 %s 的 %s 生成 %d 个语义块", pdf_path, node["title"], len(blocks))
            
            # Create knowledge fragments
            for block_idx, block_text in enumerate(blocks, 1):
                fragment_id = str(uuid.uuid4())
                fragment = {
                    "id": fragment_id,
                    "textbook": os.path.basename(pdf_path),
                    "title_path": current_path,  # Full hierarchical path
                    "block_id": block_idx,
                    "text": block_text,
                    "page_range": [start_page + 1, end_page + 1],
                    "relations": {
                        "parent_path": parent_path,
                        "sibling_blocks": [str(i) for i in range(1, len(blocks) + 1) if i != block_idx]
                    }
                }
                knowledge_fragments.append(fragment)
            
            # Process subsections recursively
            for subsection in node["subsections"]:
                process_node(subsection, current_path, parent_path + [node["title"]])
        
        # Process all top-level nodes
        for node in toc:
            process_node(node, [], [])
        
        doc.close()
        logger.info("从 %s 生成 %d 个知识片段", pdf_path, len(knowledge_fragments))
        return knowledge_fragments
    except Exception as e:
        logger.error("从 %s 提取内容时出错: %s", pdf_path, e)
        return []

def process_pdf_folder(folder_path: str, output_json: str) -> None:
    """处理文件夹中的所有 PDF 并保存结构化数据到 JSON。"""
    try:
        textbooks = []
        all_fragments = []
        
        pdf_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) 
                     for file in files if file.lower().endswith(".pdf")]
        
        for pdf_path in tqdm(pdf_files, desc="处理 PDF 文件"):
            logger.info("开始处理 %s", pdf_path)
            
            metadata = extract_textbook_metadata(pdf_path)
            if not metadata:
                continue
            toc = extract_toc(pdf_path)
            metadata["toc"] = toc
            fragments = extract_content(pdf_path, toc)
            all_fragments.extend(fragments)
            textbooks.append(metadata)
        
        output_data = {
            "textbooks": textbooks,
            "knowledge_fragments": all_fragments
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info("结构化数据保存至 %s，包含 %d 个教材和 %d 个知识片段", 
                    output_json, len(textbooks), len(all_fragments))
    except Exception as e:
        logger.error("处理文件夹 %s 时出错: %s", folder_path, e)
        
if __name__ == "__main__":
    # Example usage
    folder_path = "../data/sample"  # Replace with your PDF folder path
    output_json = "../data/sample_knowledge_fragments.json"  # Output file
    process_pdf_folder(folder_path, output_json)
    print(f"Structured data saved to {output_json}")