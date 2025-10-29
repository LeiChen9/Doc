import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class KnowledgeConfig:
    # Data
    data_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    epub_glob: str = "*.epub"

    # Chunking
    min_chunk_len: int = 50
    max_chunk_len: int = 500

    # Embedding
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "dummy")  # dummy | bge-small-zh | sentence-transformers
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # DB
    pg_dsn: Optional[str] = os.getenv("PG_DSN")  # e.g. postgresql://user:pass@localhost:5432/db
    create_indexes: bool = True

    # IO
    out_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    jsonl_fragments: str = os.path.join(out_dir, "epub_fragments.jsonl")
    json_records: str = os.path.join(out_dir, "knowledge_records.jsonl")


CONFIG = KnowledgeConfig()


