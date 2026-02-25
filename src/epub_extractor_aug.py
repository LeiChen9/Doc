import json
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

@dataclass
class TocEntry:
    title_chain: List[str]
    href: str
    fragment: Optional[str] = None


def _read_text(zf: zipfile.ZipFile, path: str) -> str:
    with zf.open(path) as fp:
        return fp.read().decode("utf-8", errors="ignore")


def _opf_path(zf: zipfile.ZipFile) -> str:
    content = _read_text(zf, "META-INF/container.xml")
    soup = BeautifulSoup(content, "xml")
    root = soup.find("rootfile")
    if not root:
        raise RuntimeError("未找到 container.xml 中的 rootfile 定义")
    return root.get("full-path")


def _parse_manifest(
    zf: zipfile.ZipFile, opf_path: str
) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    opf_dir = os.path.dirname(opf_path)
    opf_xml = _read_text(zf, opf_path)
    soup = BeautifulSoup(opf_xml, "xml")

    manifest: Dict[str, str] = {}
    nav_href: Optional[str] = None
    ncx_href: Optional[str] = None

    for item in soup.find_all("item"):
        item_id = item.get("id")
        href = item.get("href")
        if not item_id or not href:
            continue
        manifest[item_id] = os.path.normpath(os.path.join(opf_dir, href))
        props = item.get("properties", "") or ""
        if "nav" in props.split():
            nav_href = manifest[item_id]
        media_type = (item.get("media-type") or "").lower()
        if media_type == "application/x-dtbncx+xml":
            ncx_href = manifest[item_id]

    if not nav_href:
        # 常见回退：manifest 中文件名包含 nav.xhtml
        for path in manifest.values():
            if path.lower().endswith("nav.xhtml"):
                nav_href = path
                break

    return manifest, nav_href, ncx_href


def _pick_nav(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    nav = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav", role="doc-toc")
    return nav or soup.find("nav")


def _collect_toc(nav: BeautifulSoup, base_dir: str) -> List[TocEntry]:
    entries: List[TocEntry] = []

    def walk(li_tag, chain: List[str]):
        link = li_tag.find("a", recursive=False)
        text = ""
        href = ""
        if link:
            text = " ".join(link.get_text(" ", strip=True).split())
            href = link.get("href", "")
        else:
            text = " ".join(li_tag.get_text(" ", strip=True).split())
        if not text:
            return

        href_file, frag = _split_href(href)
        resolved = os.path.normpath(os.path.join(base_dir, href_file)) if href_file else ""
        new_chain = chain + [text]
        entries.append(TocEntry(title_chain=new_chain, href=resolved, fragment=frag))

        child_list = li_tag.find(["ol", "ul"], recursive=False)
        if child_list:
            for child_li in child_list.find_all("li", recursive=False):
                walk(child_li, new_chain)

    root_list = nav.find(["ol", "ul"])
    if not root_list:
        return entries

    for li in root_list.find_all("li", recursive=False):
        walk(li, [])
    return entries


def _collect_toc_from_ncx(ncx_xml: str, base_dir: str) -> List[TocEntry]:
    soup = BeautifulSoup(ncx_xml, "xml")
    nav_map = soup.find("navMap")
    entries: List[TocEntry] = []

    def walk_navpoint(navpoint, chain: List[str]):
        label = navpoint.find("text")
        content = navpoint.find("content")
        if not content:
            return
        text = _normalize(label.get_text(" ", strip=True)) if label else ""
        src = content.get("src", "")
        href_file, frag = _split_href(src)
        resolved = os.path.normpath(os.path.join(base_dir, href_file)) if href_file else ""
        new_chain = chain + ([text] if text else [])
        entries.append(TocEntry(title_chain=new_chain, href=resolved, fragment=frag))
        for child in navpoint.find_all("navPoint", recursive=False):
            walk_navpoint(child, new_chain)

    if not nav_map:
        return entries
    for navpoint in nav_map.find_all("navPoint", recursive=False):
        walk_navpoint(navpoint, [])
    return entries


def _infer_level(title: str) -> int:
    """基于常见中文标题习惯推测层级，默认最浅为 1。"""
    t = title.strip()
    if not t:
        return 1
    if "章" in t:
        return 1
    if "节" in t:
        return 2
    if "附录" in t:
        return 1
    # 其他标题按 3 级处理
    return 3


def _find_start_node(soup: BeautifulSoup, entry: TocEntry) -> Optional[BeautifulSoup]:
    # 优先 fragment
    if entry.fragment:
        target = soup.find(id=entry.fragment) or soup.find(attrs={"name": entry.fragment})
        if target:
            return target
    # 次选标题匹配
    if entry.title_chain:
        title = entry.title_chain[-1]
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            if _normalize(tag.get_text(" ", strip=True)) == _normalize(title):
                return tag
    # 默认 body 起始
    return soup.body


def _extract_range(start_node: BeautifulSoup, end_node: Optional[BeautifulSoup]) -> Tuple[List[str], List[Dict[str, str]]]:
    paragraphs: List[str] = []
    images: List[Dict[str, str]] = []

    def visit(node):
        if getattr(node, "name", None) in ("p", "li"):
            txt = _normalize(node.get_text(" ", strip=True))
            if txt:
                paragraphs.append(txt)
        if getattr(node, "name", None) == "img":
            src = node.get("src")
            if src:
                images.append({"src": src, "alt": node.get("alt", "")})

    cursor = start_node
    while cursor and cursor != end_node:
        visit(cursor)
        cursor = cursor.next_element
    return paragraphs, images


def _split_href(href: str) -> Tuple[str, Optional[str]]:
    if not href:
        return "", None
    if "#" in href:
        file_part, frag = href.split("#", 1)
        return file_part, frag or None
    return href, None


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _select_scope(soup: BeautifulSoup, fragment: Optional[str]):
    if fragment:
        target = soup.find(id=fragment) or soup.find(attrs={"name": fragment})
        if target:
            return target.find_parent(["section", "article"]) or target.parent or target
    return soup.body or soup


# 修改后的函数：生成search_guide，使用_infer_level推断层级构建toc_tree
def _generate_search_guide(toc_entries: List[TocEntry], opf_soup: BeautifulSoup) -> Dict[str, Dict]:
    toc_tree: Dict[str, Dict[str, List[str]]] = {}
    stack: List[Tuple[int, Any]] = []  # (level, node) where node is dict for level 1, list for level 2

    for entry in toc_entries:
        if not entry.title_chain:
            continue
        title = entry.title_chain[-1]
        if title == "目录":
            continue
        level = _infer_level(title)

        while stack and stack[-1][0] >= level:
            stack.pop()

        if level == 1:
            node: Dict[str, List[str]] = {}
            toc_tree[title] = node
            stack.append((1, node))
        elif level == 2:
            if not stack or stack[-1][0] != 1:
                # 如果没有章，创建默认章
                if not stack:
                    untitled_chapter = "Untitled Chapter"
                    chapter_node: Dict[str, List[str]] = {}
                    toc_tree[untitled_chapter] = chapter_node
                    stack.append((1, chapter_node))
            chapter_node = stack[-1][1]
            node: List[str] = []
            chapter_node[title] = node
            stack.append((2, node))
        else:  # level >= 3
            if not stack or stack[-1][0] != 2:
                # 如果没有节，创建默认节；如果没有章，先创建章
                if not stack or stack[-1][0] != 1:
                    if not stack:
                        untitled_chapter = "Untitled Chapter"
                        chapter_node: Dict[str, List[str]] = {}
                        toc_tree[untitled_chapter] = chapter_node
                        stack.append((1, chapter_node))
                chapter_node = stack[-1][1]
                untitled_section = "Untitled Section"
                section_node: List[str] = []
                chapter_node[untitled_section] = section_node
                stack.append((2, section_node))
            section_node = stack[-1][1]
            section_node.append(title)

    return {"toc_tree": toc_tree}


def parse_epub(epub_path: str) -> Dict[str, Dict]:
    """解析 EPUB，输出按目录嵌套且有序的 dict，并添加search_guide。"""
    with zipfile.ZipFile(epub_path) as zf:
        opf_path = _opf_path(zf)
        manifest, nav_href, ncx_href = _parse_manifest(zf, opf_path)

        # 加载OPF soup，用于提取metadata（尽管overview已移除，但保留以防未来使用）
        opf_xml = _read_text(zf, opf_path)
        opf_soup = BeautifulSoup(opf_xml, "xml")

        toc_entries: List[TocEntry] = []
        base_dir = ""

        if nav_href:
            nav_html = _read_text(zf, nav_href)
            nav_soup = BeautifulSoup(nav_html, "xml")
            nav_tag = _pick_nav(nav_soup)
            if not nav_tag:
                raise RuntimeError("nav.xhtml 缺少 <nav> 节点")
            base_dir = os.path.dirname(nav_href)
            toc_entries = _collect_toc(nav_tag, base_dir)
        elif ncx_href:
            ncx_xml = _read_text(zf, ncx_href)
            base_dir = os.path.dirname(ncx_href)
            toc_entries = _collect_toc_from_ncx(ncx_xml, base_dir)
        else:
            raise RuntimeError("OPF 中未找到 nav.xhtml 或 toc.ncx")
        soup_cache: Dict[str, BeautifulSoup] = {}

        book_tree: Dict[str, Dict] = {}

        # 记录原始顺序，按 href 分组以便分段提取
        href_groups: Dict[str, List[int]] = {}
        for idx, entry in enumerate(toc_entries):
            href_groups.setdefault(entry.href, []).append(idx)

        extracted: Dict[int, Tuple[List[str], List[Dict[str, str]]]] = {}

        for href, idx_list in href_groups.items():
            if not href:
                continue
            if href not in soup_cache:
                html = _read_text(zf, href)
                soup_cache[href] = BeautifulSoup(html, "xml")
            soup = soup_cache[href]

            starts = []
            for i in idx_list:
                starts.append(_find_start_node(soup, toc_entries[i]))

            for local_pos, i in enumerate(idx_list):
                start_node = starts[local_pos]
                end_node = starts[local_pos + 1] if local_pos + 1 < len(starts) else None
                if not start_node:
                    extracted[i] = ([], [])
                    continue
                extracted[i] = _extract_range(start_node, end_node)

        # 按 toc 顺序构建分层树（使用简单层级推断）
        stack: List[Tuple[int, Dict]] = []

        for idx, entry in enumerate(toc_entries):
            title = entry.title_chain[-1] if entry.title_chain else "Untitled"
            level = _infer_level(title)
            paragraphs, images = extracted.get(idx, ([], []))

            # 不让“目录”节点吞并其他内容
            if title == "目录":
                continue

            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_children = book_tree if not stack else stack[-1][1]["children"]

            node = parent_children.setdefault(
                title,
                {"href": os.path.basename(entry.href), "paragraphs": [], "images": [], "children": {}},
            )
            if not node["href"]:
                node["href"] = os.path.basename(entry.href)
            node["paragraphs"].extend(paragraphs)
            node["images"].extend(images)

            stack.append((level, node))

        # 生成search_guide（只包含toc_tree）
        search_guide = _generate_search_guide(toc_entries, opf_soup)

        # 返回扩展的dict
        return {"book_tree": book_tree, "search_guide": search_guide}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EPUB 结构化解析器")
    parser.add_argument(
        "--name",
        nargs='?',
        default='merck_guide'
    )
    args = parser.parse_args()
    args.epub = f"../data/raw/{args.name}.epub"
    args.output = f"../data/know/{args.name}_aug.json"

    data = parse_epub(args.epub)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"输出已保存到 {args.output}")