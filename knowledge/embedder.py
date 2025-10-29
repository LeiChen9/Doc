import hashlib
from typing import List


def _hash_to_unit_vector(text: str, dim: int = 384) -> List[float]:
    """无依赖占位嵌入：哈希映射到固定维度的稀疏向量，再做单位化。"""
    buckets = [0.0] * dim
    # 使用多个哈希种子
    for salt in ("a", "b", "c"):
        h = hashlib.md5((salt + text).encode("utf-8")).hexdigest()
        for i in range(0, len(h), 8):
            idx = int(h[i:i+8], 16) % dim
            buckets[idx] += 1.0
    # 单位化
    norm = sum(x * x for x in buckets) ** 0.5 or 1.0
    return [x / norm for x in buckets]


class DummyEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [_hash_to_unit_vector(t, self.dim) for t in texts]


def get_embedder(backend: str = "dummy", dim: int = 384):
    # 预留：可扩展为加载 sentence-transformers 或 BGE-small-zh
    return DummyEmbedder(dim)


