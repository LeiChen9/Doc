import glob
import os
import logging
from typing import Dict, Iterable

from .config import CONFIG
from .epub_parser import iter_epub_blocks
from .chunker import iter_clean_blocks
from .entity_extractor import attach_entity
from .embedder import get_embedder
from .db import dump_jsonl, persist_postgres


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_records(epub_path: str) -> Iterable[Dict]:
    # 1) 解析 + 粗切
    blocks = iter_epub_blocks(epub_path, min_para_len=CONFIG.min_chunk_len)
    blocks = iter_clean_blocks(blocks)

    # 2) 实体
    blocks = (attach_entity(b) for b in blocks)

    # 3) 嵌入
    embedder = get_embedder(CONFIG.embedding_backend, CONFIG.embedding_dim)
    buffer = []
    for b in blocks:
        buffer.append(b)
        if len(buffer) >= 64:
            texts = [x["text"] for x in buffer]
            vecs = embedder.encode(texts)
            for x, v in zip(buffer, vecs):
                yield _to_record(x, v)
            buffer = []
    if buffer:
        texts = [x["text"] for x in buffer]
        vecs = embedder.encode(texts)
        for x, v in zip(buffer, vecs):
            yield _to_record(x, v)


def _to_record(block: Dict, vector) -> Dict:
    meta = {
        "publish_date": block.get("source_meta", {}).get("publish_date"),
        "creator": block.get("source_meta", {}).get("creator"),
        "page_hint": block.get("page_hint"),
    }
    return {
        "id": block["id"],
        "entity": block["entity"],
        "title": block.get("title") or "",
        "text": block["text"],
        "source": block.get("source_file") or "",
        "chapter_path": block.get("chapter_path") or "",
        "tags": block.get("tags", []),
        "metadata": meta,
        "vector": vector,
    }


def run_pipeline():
    # 定位唯一 epub
    pattern = os.path.join(CONFIG.data_dir, CONFIG.epub_glob)
    epub_files = sorted(glob.glob(pattern))
    if not epub_files:
        raise FileNotFoundError(f"未在 {CONFIG.data_dir} 发现 EPUB 文件")
    if len(epub_files) > 1:
        logger.warning("发现多个 EPUB，仅处理第一个: %s", epub_files[0])
    epub_path = epub_files[0]

    # 生成记录，先落 JSONL
    records_iter = build_records(epub_path)
    # 同时写 JSONL（便于调试）
    cnt = dump_jsonl(CONFIG.json_records, records_iter)
    logger.info("记录生成完成: %d", cnt)

    # 可选：写入 PostgreSQL
    if CONFIG.pg_dsn:
        logger.info("检测到 PG_DSN，写入数据库…")
        # 重新读取 JSONL 流式写库
        def _iter_again():
            import json
            with open(CONFIG.json_records, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        persist_postgres(_iter_again(), CONFIG.pg_dsn, CONFIG.embedding_dim, CONFIG.create_indexes)
    else:
        logger.info("未配置 PG_DSN，跳过落库，仅输出 JSONL：%s", CONFIG.json_records)


if __name__ == "__main__":
    run_pipeline()


