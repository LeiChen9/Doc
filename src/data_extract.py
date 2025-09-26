import fitz  # PyMuPDF
import os
import json
import uuid
import logging
import gc
import psutil
from tqdm import tqdm
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("extract_pdf.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log memory usage
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info("当前内存使用: %.2f MB", mem_info.rss / 1024 / 1024)

def extract_textbook_metadata(pdf_path: str) -> Dict:
    """提取 PDF 元数据（标题、路径、总页数）。"""
    try:
        with fitz.open(pdf_path) as doc:
            metadata = {
                "textbook_name": os.path.basename(pdf_path),
                "path": pdf_path,
                "total_pages": doc.page_count
            }
        log_memory_usage()
        return metadata
    except Exception as e:
        logger.error("处理 %s 元数据时出错: %s", pdf_path, e)
        return {}

def extract_toc(pdf_path: str) -> Dict:
    """从 PDF 提取多级目录 (TOC)，返回嵌套字典结构，避免循环引用。"""
    try:
        with fitz.open(pdf_path) as doc:
            toc = doc.get_toc(simple=False)
            total_pages = doc.page_count
        
        toc_tree = {"title": "Root", "level": 0, "subsections": {}, "content": []}
        level_stack = [(toc_tree, [])]  # (node, path)
        
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
            
            # Adjust level stack
            while len(level_stack) > level:
                level_stack.pop()
            current_node, current_path = level_stack[-1]
            
            node = {
                "title": title,
                "level": level,
                "start_page": page,
                "end_page": page,
                "content": [],  # List of atomic paragraphs
                "subsections": {}
            }
            current_node["subsections"][title] = node
            level_stack.append((node, current_path + [title]))
        
        # Update end_page recursively
        def update_end_pages(node: Dict, total_pages: int):
            subsections = sorted(node["subsections"].values(), key=lambda x: x["start_page"])
            for i, sub in enumerate(subsections[:-1]):
                sub["end_page"] = subsections[i + 1]["start_page"] - 1
                update_end_pages(sub, total_pages)
            if subsections:
                subsections[-1]["end_page"] = total_pages
                update_end_pages(subsections[-1], total_pages)
        
        update_end_pages(toc_tree, total_pages)
        logger.info("从 %s 提取目录结构", pdf_path)
        log_memory_usage()
        return toc_tree["subsections"]
    except Exception as e:
        logger.error("从 %s 提取目录时出错: %s", pdf_path, e)
        return {}

def extract_atomic_paragraphs(doc: fitz.Document, start_page: int, end_page: int, min_para_length: int = 50) -> List[str]:
    """提取原子自然段，处理跨页合并，基于坐标和空行。"""
    try:
        paragraphs = []
        current_para = ""
        prev_y1 = -1
        prev_page_num = -1
        line_height_est = 12  # Initial estimate
        
        for page_num in tqdm(range(start_page - 1, end_page), desc="提取自然段", leave=False):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if not text:
                    continue
                y0, y1 = block[1], block[3]
                
                # Estimate line height
                line_height_est = max(line_height_est, y1 - y0)
                
                # Cross-page continuity: merge if y0 is near page top and prev_y1 near page bottom
                is_cross_page = (page_num > prev_page_num and prev_page_num >= 0 and 
                               y0 < 50 and prev_y1 > page.rect.height - 50)
                
                # Merge if continuous (y0 ≈ prev_y1 + line_height) or cross-page
                if current_para and (is_cross_page or (prev_y1 >= 0 and y0 - prev_y1 < 2 * line_height_est)):
                    current_para += " " + text
                else:
                    if current_para and len(current_para) >= min_para_length:
                        paragraphs.append(current_para.strip())
                    current_para = text
                
                prev_y1 = y1
                prev_page_num = page_num
            
            del page, blocks
            gc.collect()
        
        if current_para and len(current_para) >= min_para_length:
            paragraphs.append(current_para.strip())
        
        return paragraphs
    except Exception as e:
        logger.error("提取自然段时出错: %s", e)
        return []

def populate_content(pdf_path: str, toc_tree: Dict, batch_file: str) -> None:
    """递归填充内容到 TOC 树，并生成扁平知识片段。"""
    try:
        with fitz.open(pdf_path) as doc:
            def process_node(node: Dict, title_path: List[str], parent_path: List[str]):
                current_path = title_path + [node["title"]]
                page_count = node["end_page"] - node["start_page"] + 1
                if page_count > 50:
                    logger.warning("节点 %s 页面范围过大 (%d 页)，可能导致内存压力", node["title"], page_count)
                
                # Extract content only for leaf nodes
                if not node["subsections"]:
                    paragraphs = extract_atomic_paragraphs(doc, node["start_page"], node["end_page"])
                    node["content"] = paragraphs
                    logger.info("节点 %s 生成 %d 个原子自然段", node["title"], len(paragraphs))
                    
                    # Generate flat fragments
                    knowledge_fragments = []
                    for para_idx, para_text in enumerate(paragraphs, 1):
                        fragment_id = str(uuid.uuid4())
                        fragment = {
                            "id": fragment_id,
                            "textbook": os.path.basename(pdf_path),
                            "title_path": current_path,
                            "para_id": para_idx,
                            "text": para_text,
                            "page_range": [node["start_page"], node["end_page"]],
                            "relations": {
                                "parent_path": parent_path,
                                "sibling_paras": [str(i) for i in range(1, len(paragraphs) + 1) if i != para_idx]
                            }
                        }
                        knowledge_fragments.append(fragment)
                    
                    # Append to batch file
                    with open(batch_file, "a", encoding="utf-8") as f:
                        for fragment in knowledge_fragments:
                            json.dump(fragment, f, ensure_ascii=False)
                            f.write("\n")
                    
                    del knowledge_fragments, paragraphs
                    gc.collect()
                
                # Recurse subsections
                for sub_title, sub_node in node["subsections"].items():
                    process_node(sub_node, current_path, parent_path + [node["title"]])
            
            for title, node in toc_tree.items():
                process_node(node, [], [])
        
        log_memory_usage()
    except Exception as e:
        logger.error("填充内容时出错: %s", e)

def process_pdf_folder(folder_path: str, output_json: str) -> None:
    """处理文件夹中的所有 PDF 并保存结构化数据到 JSON。"""
    try:
        textbooks = []
        batch_file = "temp_fragments.jsonl"
        
        if os.path.exists(batch_file):
            os.remove(batch_file)
        
        pdf_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) 
                     for file in files if file.lower().endswith(".pdf")]
        
        for pdf_path in tqdm(pdf_files, desc="处理 PDF 文件"):
            logger.info("开始处理 %s", pdf_path)
            
            metadata = extract_textbook_metadata(pdf_path)
            if not metadata:
                continue
            toc_tree = extract_toc(pdf_path)
            metadata["toc"] = toc_tree
            populate_content(pdf_path, toc_tree, batch_file)
            textbooks.append(metadata)
            gc.collect()
            log_memory_usage()
        
        all_fragments = []
        if os.path.exists(batch_file):
            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_fragments.append(json.loads(line))
            os.remove(batch_file)
        
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
    # Disable multiprocessing to avoid semaphore leaks
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    folder_path = "../data/sample"  # Replace with your PDF folder path
    output_json = "../data/sample_knowledge_fragments.json"  # Output file
    process_pdf_folder(folder_path, output_json)
    print(f"Structured data saved to {output_json}")