import argparse
import os
from pathlib import Path

from openai import OpenAI


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://chatapi.littlewheat.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-3o7OxkzUFZeQDbUFLIEgiXYXpulbKnKOJ7OAoiMyepxhbnYK")
DEFAULT_JSON = Path(__file__).resolve().parents[1] / "data" / "know" / "internal.json"

def llm_query(question: str) -> str: 
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    prompt = """
根据人民卫生出版社的《默克家庭诊疗手册（插图版）》ISBN: 9787117076401，关于问题 '''{question}''''，应该查阅第几章第几节第几段的内容以获得关于该问题的全面理解？
请仅输出 JSON，格式严格遵守如下定义：
```json
{{
    "chapter": "第几章",
    "chapter_name": "章节名称",
    "section": "第几节",
    "section_name": "节名称",
    "paragraph": "第几段"
    "reasoning": "为什么选择这个章节、节、段落？请详细阐述。" 
}}
"""
    print(prompt.format(question=question))
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search engine")
    parser.add_argument("--question", type=str, default="上牙膛破了怎么办")
    args = parser.parse_args()
    print(f"Question: {args.question}")
    query = llm_query(args.question)
    print(f"Query: {query}")