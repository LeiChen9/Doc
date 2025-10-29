import json
import logging
from typing import Dict, Iterable, List, Optional
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_records (
    id UUID PRIMARY KEY,
    entity VARCHAR(200) NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    chapter_path TEXT NOT NULL,
    tags JSONB NOT NULL,
    metadata JSONB NOT NULL,
    vector VECTOR(%(dim)s)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge_records(entity);
CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_records(source);
CREATE INDEX IF NOT EXISTS idx_knowledge_chapter_path ON knowledge_records(chapter_path);
"""


def dump_jsonl(path: str, records: Iterable[Dict]) -> int:
    cnt = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cnt += 1
    logger.info("写出 JSONL: %s (%d 条)", path, cnt)
    return cnt


def persist_postgres(records: Iterable[Dict], dsn: str, dim: int = 384, create_indexes: bool = True) -> int:
    import psycopg2
    from psycopg2.extras import Json

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cur = conn.cursor()

    # schema
    cur.execute(SCHEMA_SQL, {"dim": dim})

    sql = (
        "INSERT INTO knowledge_records (id, entity, title, text, source, chapter_path, tags, metadata, vector) "
        "VALUES (%(id)s, %(entity)s, %(title)s, %(text)s, %(source)s, %(chapter_path)s, %(tags)s, %(metadata)s, %(vector)s) "
        "ON CONFLICT (id) DO UPDATE SET entity=EXCLUDED.entity, title=EXCLUDED.title, text=EXCLUDED.text, "
        "source=EXCLUDED.source, chapter_path=EXCLUDED.chapter_path, tags=EXCLUDED.tags, metadata=EXCLUDED.metadata, vector=EXCLUDED.vector"
    )

    count = 0
    try:
        for r in records:
            vec = r.get("vector")
            cur.execute(
                sql,
                {
                    "id": r["id"],
                    "entity": r["entity"],
                    "title": r["title"],
                    "text": r["text"],
                    "source": r["source"],
                    "chapter_path": r["chapter_path"],
                    "tags": Json(r.get("tags", [])),
                    "metadata": Json(r.get("metadata", {})),
                    "vector": vec if vec is None else list(vec),
                },
            )
            count += 1
        conn.commit()
        logger.info("已写入 PostgreSQL: %d 条", count)
    except Exception as e:
        conn.rollback()
        logger.exception("写入 PostgreSQL 失败: %s", e)
        raise
    finally:
        cur.close()
        conn.close()
    return count


