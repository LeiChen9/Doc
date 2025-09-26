import json
import logging
import gc
import psutil
import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import scoring
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uuid
from tqdm import tqdm
from openai import OpenAI  # For LLM query rewrite
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Log memory usage
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info("当前内存使用: %.2f MB", mem_info.rss / 1024 / 1024)

class KnowledgeSearchEngine:
    def __init__(self, json_path: str):
        self.json_path = json_path
        # Initialize OpenAI client with secret API key from environment variable
        self.client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
                             api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = []  # Flat list: {'id': str, 'textbook': str, 'title_path': list, 'page_range': list, 'text': str}
        self.load_and_flatten()
        self.build_indexes()
    
    def load_and_flatten(self):
        """Flatten JSON to documents with metadata."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def flatten_toc(node: dict, current_path: list, textbook: str):
            title_path = current_path + [node["title"]]
            for para_idx, para in enumerate(node["content"], 1):
                self.documents.append({
                    "id": str(uuid.uuid4()),
                    "textbook": textbook,
                    "title_path": title_path,
                    "page_range": [node["start_page"], node["end_page"]],
                    "text": para
                })
            for sub_title, sub_node in node["subsections"].items():
                flatten_toc(sub_node, title_path, textbook)
        
        for textbook in data["textbooks"]:
            for title, node in textbook.get("toc", {}).items():
                flatten_toc(node, [], textbook["textbook_name"])
        
        logger.info(f"Flattened {len(self.documents)} documents")
        log_memory_usage()
    
    def build_indexes(self):
        """Build Whoosh (BM25) and FAISS (semantic) indexes."""
        # Whoosh for BM25
        schema = Schema(id=ID(stored=True), text=TEXT(stored=True))
        if not os.path.exists("whoosh_index"):
            os.mkdir("whoosh_index")
        self.ix = create_in("whoosh_index", schema)
        writer = self.ix.writer()
        for doc in self.documents:
            writer.add_document(id=doc["id"], text=doc["text"])
        writer.commit()
        logger.info("Built Whoosh BM25 index")
        
        # Sentence Transformers + FAISS for semantic
        self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
        embeddings = self.model.encode([doc["text"] for doc in tqdm(self.documents, desc="Encoding texts")], batch_size=32)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        logger.info("Built FAISS semantic index")
        log_memory_usage()
    
    def rewrite_query(self, user_query: str) -> str:
        """Use custom OpenAI gateway for query rewrite and semantic expansion."""
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites user queries for better search. Expand with synonyms, related terms, and rephrase for precision in Chinese or English as appropriate."},
                    {"role": "user", "content": f"Rewrite this query for keyword and semantic search: {user_query}. Provide the rewritten query."}
                ],
                temperature=0.5
            )
            rewritten = completion.choices[0].message.content.strip()
            logger.info(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return user_query  # Fallback to original query
    
    def bm25_search(self, query: str, top_k: int = 50) -> list:
        """BM25 search with Whoosh."""
        with self.ix.searcher(weighting=scoring.BM25F) as searcher:
            parser = QueryParser("text", self.ix.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=top_k)
            return [hit["id"] for hit in results]
    
    def semantic_search(self, query: str, top_k: int = 50) -> list:
        """Semantic search with FAISS."""
        query_emb = self.model.encode([query])
        _, indices = self.faiss_index.search(np.array(query_emb).astype('float32'), top_k)
        return [self.documents[i]["id"] for i in indices[0]]
    
    def rrf_fusion(self, bm25_ids: list, semantic_ids: list, k: int = 60) -> list:
        """Reciprocal Rank Fusion for hybrid ranking."""
        scores = {}
        for rank, doc_id in enumerate(bm25_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        for rank, doc_id in enumerate(semantic_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return sorted_ids
    
    def search(self, user_query: str, top_n: int = 5) -> list:
        """Main search function."""
        rewritten_query = self.rewrite_query(user_query)
        bm25_results = self.bm25_search(rewritten_query)
        semantic_results = self.semantic_search(rewritten_query)
        fused_ids = self.rrf_fusion(bm25_results, semantic_results)
        
        results = []
        for doc_id in fused_ids[:top_n]:
            doc = next(d for d in self.documents if d["id"] == doc_id)
            path_str = " > ".join(doc["title_path"])
            results.append({
                "textbook": doc["textbook"],
                "path": path_str,
                "page_range": f"第 {doc['page_range'][0]}-{doc['page_range'][1]} 页"
            })
        
        logger.info(f"Found {len(results)} results for query: {user_query}")
        log_memory_usage()
        gc.collect()
        return results

# Example usage
if __name__ == "__main__":
    # Note: Set OPENAI_API_KEY in .env file
    engine = KnowledgeSearchEngine("../data/sample_knowledge_fragments.json")
    query = "骨骼系统的功能是什么？"
    results = engine.search(query)
    for res in results:
        print(f"教材: {res['textbook']}, 路径: {res['path']}, 页码: {res['page_range']}")