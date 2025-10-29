from typing import Dict, Iterable


def normalize_block(block: Dict) -> Dict:
    """对块做最小清洗；保留路径、标题、文本。"""
    text = block.get("text", "").strip()
    text = " ".join(text.split())
    block["text"] = text
    return block


def iter_clean_blocks(blocks: Iterable[Dict]) -> Iterable[Dict]:
    for b in blocks:
        b2 = normalize_block(dict(b))
        if len(b2.get("text", "")) >= 50:
            yield b2


