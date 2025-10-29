import os
import zipfile
import uuid
import logging
from typing import Dict, List, Tuple, Iterable
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _read_zip_text(zf: zipfile.ZipFile, path: str) -> str:
    with zf.open(path) as fp:
        return fp.read().decode("utf-8", errors="ignore")


def _find_opf_path(zf: zipfile.ZipFile) -> str:
    # META-INF/container.xml -> rootfile full-path
    content = _read_zip_text(zf, "META-INF/container.xml")
    soup = BeautifulSoup(content, "xml")
    root = soup.find("rootfile")
    if not root:
        raise RuntimeError("EPUB 缺少 rootfile 定义")
    return root.get("full-path")


def _parse_opf(zf: zipfile.ZipFile, opf_path: str) -> Tuple[Dict, Dict, List[str]]:
    opf_dir = os.path.dirname(opf_path)
    opf_xml = _read_zip_text(zf, opf_path)
    soup = BeautifulSoup(opf_xml, "xml")

    # metadata
    md = soup.find("metadata")
    title = (md.find("dc:title").text if md and md.find("dc:title") else "")
    creator = (md.find("dc:creator").text if md and md.find("dc:creator") else "")
    pubdate = (md.find("dc:date").text if md and md.find("dc:date") else "")
    metadata = {
        "title": title.strip(),
        "creator": creator.strip(),
        "publish_date": pubdate.strip(),
    }

    # manifest: id -> href
    manifest = {}
    for item in soup.find_all("item"):
        manifest[item.get("id")] = os.path.normpath(os.path.join(opf_dir, item.get("href")))

    # spine: ordered itemrefs
    spine_hrefs: List[str] = []
    for ref in soup.find("spine").find_all("itemref"):
        item_id = ref.get("idref")
        if item_id in manifest:
            spine_hrefs.append(manifest[item_id])

    return metadata, manifest, spine_hrefs


def _extract_text_from_xhtml(html: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html, "lxml")
    # Collect headings for path clues
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        txt = " ".join(tag.get_text(" ", strip=True).split())
        if txt:
            headings.append(txt)
    # Visible text blocks
    paras: List[str] = []
    for p in soup.find_all(["p", "li"]):
        txt = " ".join(p.get_text(" ", strip=True).split())
        if txt:
            paras.append(txt)
    content = "\n".join(paras)
    return content, headings


def iter_epub_blocks(epub_path: str, min_para_len: int = 50) -> Iterable[Dict]:
    """按 spine 顺序遍历 xhtml，输出块：{chapter_path, title, text, page_hint}。
    page_hint 为近似页码（EPUB 无硬页概念，这里用 spine 序号）。"""
    with zipfile.ZipFile(epub_path) as zf:
        opf_path = _find_opf_path(zf)
        metadata, manifest, spine = _parse_opf(zf, opf_path)

        chapter_stack: List[str] = []
        import pdb; pdb.set_trace()
        for idx, href in enumerate(spine, start=1):
            html = _read_zip_text(zf, href)
            content, headings = _extract_text_from_xhtml(html)
            if headings:
                # 更新章节路径（保留最近的 3 级标题作为路径）
                chapter_stack = (chapter_stack + headings)[-3:]
            chapter_path = " > ".join(chapter_stack) if chapter_stack else metadata.get("title") or "EPUB"

            # 切段
            for para in content.split("\n"):
                para = para.strip()
                if len(para) < min_para_len:
                    continue
                yield {
                    "id": str(uuid.uuid4()),
                    "title": chapter_stack[-1] if chapter_stack else (metadata.get("title") or ""),
                    "chapter_path": chapter_path,
                    "text": para,
                    "source_meta": metadata,
                    "source_file": os.path.basename(epub_path),
                    "page_hint": idx,
                }


