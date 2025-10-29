from typing import Dict, List, Tuple
import re


def simple_medical_entity(text: str) -> Tuple[str, List[str]]:
    """占位实体抽取：
    - 规则：提取最长的中文连续词串或大写英文词作为主实体
    - 标签：基于简单关键词映射（极简）
    实际项目中，应替换为 LLM 或专用 NER 模型。
    """
    # 优先中文实体
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    entity = max(zh, key=len) if zh else ""
    if not entity:
        en = re.findall(r"[A-Za-z]{3,}", text)
        entity = max(en, key=len) if en else text[:10]

    tags: List[str] = []
    low = text.lower()
    if any(k in low for k in ["症状", "表现", "symptom"]):
        tags.append("症状")
    if any(k in low for k in ["机制", "pathogenesis", "mechanism"]):
        tags.append("机制")
    if any(k in low for k in ["治疗", "用药", "therapy", "treatment"]):
        tags.append("治疗")
    if any(k in low for k in ["诊断", "标准", "diagnosis"]):
        tags.append("诊断")
    return entity, list(set(tags))


def attach_entity(block: Dict) -> Dict:
    entity, tags = simple_medical_entity(block["text"])
    block["entity"] = entity[:200]
    block["tags"] = tags
    return block


