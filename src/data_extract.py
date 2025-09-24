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

# Load spaCy model
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
    """从 PDF 提取目录 (TOC)。"""
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc(simple=False)
        doc.close()
        
        structured_toc = []
        current_chapter = None
        
        for entry in toc:
            if len(entry) < 3:
                logger.warning("跳过无效 TOC 条目: %s", entry)
                continue
            level = entry[0]
            title = entry[1]
            page = entry[2]
            
            title = title.replace("\r", "").replace("\n", "").strip()
            title = " ".join(title.split())
            
            if not title or page < 1:
                logger.warning("跳过无效标题或页面: %s", entry)
                continue
            
            if level == 1:
                current_chapter = {
                    "chapter": title,
                    "start_page": page,
                    "end_page": page,
                    "subsections": []
                }
                structured_toc.append(current_chapter)
            elif level == 2 and current_chapter:
                current_chapter["subsections"].append({
                    "subsection": title,
                    "start_page": page,
                    "end_page": page
                })
        
        for i, chapter in enumerate(structured_toc[:-1]):
            chapter["end_page"] = structured_toc[i + 1]["start_page"] - 1
            for j, subsection in enumerate(chapter["subsections"][:-1]):
                subsection["end_page"] = chapter["subsections"][j + 1]["start_page"] - 1
        if structured_toc:
            structured_toc[-1]["end_page"] = fitz.open(pdf_path).page_count
            if structured_toc[-1]["subsections"]:
                structured_toc[-1]["subsections"][-1]["end_page"] = structured_toc[-1]["end_page"]
        
        logger.info("从 %s 提取 %d 个目录项", pdf_path, len(structured_toc))
        return structured_toc
    except Exception as e:
        logger.error("从 %s 提取目录时出错: %s", pdf_path, e)
        return []

def segment_text_to_paragraphs(text: str) -> List[str]:
    """使用 spaCy 将文本分段为段落。"""
    try:
        doc = nlp(text)
        paragraphs = []
        current_para = []
        
        for sent in doc.sents:
            current_para.append(sent.text.strip())
            if sent.text.strip().endswith(("。", "！", "？")) or len("".join(current_para)) > 200:
                paragraphs.append("".join(current_para))
                current_para = []
        
        if current_para:
            paragraphs.append("".join(current_para))
        
        return [p.strip() for p in paragraphs if p.strip()]
    except Exception as e:
        logger.error("分段文本时出错: %s", e)
        return [text]

def extract_content(pdf_path: str, toc: List[Dict]) -> List[Dict]:
    """从 PDF 提取内容并组织为知识片段。"""
    try:
        doc = fitz.open(pdf_path)
        knowledge_fragments = []
        
        for chapter in toc:
            chapter_title = chapter["chapter"]
            start_page = chapter["start_page"] - 1
            end_page = chapter["end_page"] - 1
            
            chapter_text = ""
            for page_num in tqdm(range(start_page, end_page + 1), desc=f"提取 {chapter_title} 页面"):
                try:
                    page = doc[page_num]
                    chapter_text += page.get_text("text") + "\n"
                except Exception as e:
                    logger.warning("提取 %s 页面 %d 时出错: %s", pdf_path, page_num + 1, e)
                    continue
            
            for subsection in chapter["subsections"]:
                subsection_title = subsection["subsection"]
                sub_start_page = subsection["start_page"] - 1
                sub_end_page = subsection["end_page"] - 1
                
                subsection_text = ""
                for page_num in range(sub_start_page, sub_end_page + 1):
                    try:
                        page = doc[page_num]
                        subsection_text += page.get_text("text") + "\n"
                    except Exception as e:
                        logger.warning("提取 %s 子章节页面 %d 时出错: %s", pdf_path, page_num + 1, e)
                        continue
                
                paragraphs = segment_text_to_paragraphs(subsection_text)
                logger.info("从 %s 的 %s 生成 %d 个段落", pdf_path, subsection_title, len(paragraphs))
                
                for para_idx, paragraph in enumerate(paragraphs, 1):
                    fragment_id = str(uuid.uuid4())
                    fragment = {
                        "id": fragment_id,
                        "textbook": os.path.basename(pdf_path),
                        "chapter": chapter_title,
                        "subsection": subsection_title,
                        "paragraph_id": para_idx,
                        "text": paragraph,
                        "relations": {
                            "parent_chapter": chapter_title,
                            "sibling_paragraphs": [
                                str(i) for i in range(1, len(paragraphs) + 1) if i != para_idx
                            ]
                        }
                    }
                    knowledge_fragments.append(fragment)
        
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
        
        # Get list of PDF files
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