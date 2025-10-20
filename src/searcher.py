import json
import logging
import gc
import psutil
import os
import re
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import scoring
import uuid
from difflib import SequenceMatcher  # For edit distance (similarity ratio)
from collections import defaultdict
import time

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
        self.documents = []  # Flat list: {'id': str, 'textbook': str, 'title_path': list, 'page_range': list, 'text': str}
        self.document_map = {}  # id -> document mapping for fast lookup
        self.text_lower_map = {}  # id -> lowercased text for fast search
        self.keyword_index = defaultdict(set)  # keyword -> set of document ids
        self.load_and_flatten()
        self.build_indexes()
    
    def load_and_flatten(self):
        """Flatten JSON to documents with metadata."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def flatten_toc(node: dict, current_path: list, textbook: str):
            title_path = current_path + [node["title"]]
            for para_idx, para in enumerate(node["content"], 1):
                doc_id = str(uuid.uuid4())
                doc = {
                    "id": doc_id,
                    "textbook": textbook,
                    "title_path": title_path,
                    "page_range": [node["start_page"], node["end_page"]],
                    "text": para
                }
                self.documents.append(doc)
                self.document_map[doc_id] = doc
                self.text_lower_map[doc_id] = para.lower()
                
                # Build keyword index for fast lookup
                words = re.findall(r'\w+', para.lower())
                for word in words:
                    if len(word) > 2:  # Only index words longer than 2 characters
                        self.keyword_index[word].add(doc_id)
            
            for sub_title, sub_node in node["subsections"].items():
                flatten_toc(sub_node, title_path, textbook)
        
        for textbook in data["textbooks"]:
            for title, node in textbook.get("toc", {}).items():
                flatten_toc(node, [], textbook["textbook_name"])
        
        logger.info(f"Flattened {len(self.documents)} documents")
        log_memory_usage()
    
    def build_indexes(self):
        """Build only Whoosh (BM25) index - no embedding models."""
        # Whoosh for BM25
        schema = Schema(id=ID(stored=True), text=TEXT(stored=True))
        if not os.path.exists("src/whoosh_index"):
            os.makedirs("src/whoosh_index", exist_ok=True)
        
        # Use existing index if available
        try:
            from whoosh.index import open_dir
            self.ix = open_dir("src/whoosh_index")
            logger.info("Using existing Whoosh BM25 index")
        except:
            self.ix = create_in("src/whoosh_index", schema)
            writer = self.ix.writer()
            for doc in self.documents:
                writer.add_document(id=doc["id"], text=doc["text"])
            writer.commit()
            logger.info("Built new Whoosh BM25 index")
        
        log_memory_usage()
    
    def expand_query(self, user_query: str) -> list:
        """Simple query expansion without LLM - extract key terms."""
        # Extract Chinese and English words
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', user_query)
        english_words = re.findall(r'[a-zA-Z]+', user_query.lower())
        
        # Combine all terms
        all_terms = chinese_words + english_words
        
        # Add original query as whole
        expanded_terms = [user_query.lower()] + all_terms
        
        # Remove duplicates and empty strings
        return list(set([term for term in expanded_terms if term.strip()]))
    
    def calculate_text_relevance_score(self, text: str, query_terms: list) -> float:
        """Calculate relevance score using keyword matching and edit distance."""
        text_lower = text.lower()
        score = 0.0
        
        # 1. Exact keyword matches (highest weight)
        for term in query_terms:
            if term.lower() in text_lower:
                score += 2.0
        
        # 2. Edit distance similarity for query string
        edit_sim = self.edit_distance_similarity(query_terms[0], text) if query_terms else 0.0
        score += edit_sim
        
        # 3. Word overlap bonus
        query_words = set(re.findall(r'\w+', ' '.join(query_terms).lower()))
        text_words = set(re.findall(r'\w+', text_lower))
        overlap = len(query_words.intersection(text_words))
        score += overlap * 0.5
        
        return min(score, 10.0)  # Cap at 10.0
    
    def keyword_search(self, query_terms: list, top_k: int = 50) -> dict:
        """Fast keyword search using pre-built keyword index."""
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Direct keyword match
            if term_lower in self.keyword_index:
                for doc_id in self.keyword_index[term_lower]:
                    doc_scores[doc_id] += 3.0  # High score for exact keyword match
            
            # Partial match in keyword index
            for keyword in self.keyword_index:
                if term_lower in keyword or keyword in term_lower:
                    for doc_id in self.keyword_index[keyword]:
                        doc_scores[doc_id] += 1.0  # Lower score for partial match
        
        # Sort by score and return top results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_docs[:top_k])
    
    def simple_grep_search(self, query: str, top_k: int = 50) -> list:
        """Simple grep-like search: check if query in text (case-insensitive)."""
        matches = []
        lower_query = query.lower()
        for doc_id, text_lower in self.text_lower_map.items():
            if lower_query in text_lower:
                matches.append(doc_id)
            if len(matches) >= top_k:
                break
        return matches
    
    def edit_distance_similarity(self, str1: str, str2: str) -> float:
        """Compute similarity ratio using SequenceMatcher (edit distance based)."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def bm25_search(self, query: str, top_k: int = 50) -> list:
        """BM25 search with Whoosh."""
        with self.ix.searcher(weighting=scoring.BM25F) as searcher:
            parser = QueryParser("text", self.ix.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=top_k)
            return [hit["id"] for hit in results]
    
    def edit_distance_search(self, query: str, top_k: int = 50, threshold: float = 0.3) -> dict:
        """Search using edit distance similarity."""
        doc_scores = {}
        query_lower = query.lower()
        
        # Only search through a subset for performance
        search_limit = min(1000, len(self.documents))  # Limit search for speed
        
        for i, (doc_id, text_lower) in enumerate(list(self.text_lower_map.items())[:search_limit]):
            # Use shorter text for faster comparison
            short_text = text_lower[:500]  # Limit text length for speed
            similarity = self.edit_distance_similarity(query_lower, short_text)
            
            if similarity >= threshold:
                doc_scores[doc_id] = similarity
        
        # Sort by score and return top results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_docs[:top_k])
    
    def merge_search_results(self, *result_dicts) -> dict:
        """Merge multiple search result dictionaries by combining scores."""
        merged_scores = defaultdict(float)
        
        for result_dict in result_dicts:
            for doc_id, score in result_dict.items():
                merged_scores[doc_id] += score
        
        # Sort by combined score
        sorted_results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_results)
    
    def evaluate_results(self, result_dict: dict, query_terms: list, min_score_threshold: float = 2.0) -> bool:
        """Evaluate if results are good enough using keyword matching and edit distance."""
        if not result_dict:
            return False
        
        # Check if top results have good scores
        top_scores = list(result_dict.values())[:5]
        if not top_scores:
            return False
        
        avg_score = sum(top_scores) / len(top_scores)
        max_score = max(top_scores)
        
        logger.info(f"Avg score: {avg_score:.2f}, Max score: {max_score:.2f}")
        return max_score >= min_score_threshold
    
    def search(self, user_query: str, top_n: int = 5) -> list:
        """Fast search using keyword matching and edit distance - no LLM or embeddings."""
        start_time = time.time()
        
        # Expand query to get search terms
        query_terms = self.expand_query(user_query)
        logger.info(f"Search terms: {query_terms}")
        
        # Stage 1: Fast keyword search
        keyword_results = self.keyword_search(query_terms, top_k=100)
        
        # Stage 2: Simple grep search for exact matches
        grep_ids = self.simple_grep_search(user_query, top_k=50)
        grep_results = {doc_id: 5.0 for doc_id in grep_ids}  # High score for exact matches
        
        # Stage 3: BM25 search as backup
        bm25_ids = self.bm25_search(user_query, top_k=50)
        bm25_results = {}
        for i, doc_id in enumerate(bm25_ids):
            bm25_results[doc_id] = 3.0 - (i * 0.1)  # Decreasing score by rank
        
        # Merge all results
        final_results = self.merge_search_results(keyword_results, grep_results, bm25_results)
        
        # If results are not good enough, try edit distance search on top results
        if not self.evaluate_results(final_results, query_terms, min_score_threshold=1.0):
            logger.info("Enhancing with edit distance search")
            edit_results = self.edit_distance_search(user_query, top_k=30, threshold=0.2)
            final_results = self.merge_search_results(final_results, edit_results)
        
        # Convert to final format
        results = []
        for doc_id, score in list(final_results.items())[:top_n * 2]:  # Get more candidates
            doc = self.document_map[doc_id]
            
            # Add relevance score for final ranking
            relevance_score = self.calculate_text_relevance_score(doc["text"], query_terms)
            final_score = score + relevance_score
            
            path_str = " > ".join(doc["title_path"])
            results.append({
                "textbook": doc["textbook"],
                "path": path_str,
                "page_range": f"第 {doc['page_range'][0]}-{doc['page_range'][1]} 页",
                "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "score": final_score
            })
        
        # Sort by final score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        final_results_list = results[:top_n]
        
        # Remove score from final output (internal use only)
        for result in final_results_list:
            result.pop("score", None)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Found {len(final_results_list)} results for query: {user_query} in {elapsed_time:.2f}s")
        log_memory_usage()
        
        return final_results_list

# Example usage
if __name__ == "__main__":
    engine = KnowledgeSearchEngine("../data/sample_knowledge_fragments.json")
    query = "骨骼系统的功能是什么？"
    results = engine.search(query)
    for res in results:
        print(f"教材: {res['textbook']}, 路径: {res['path']}, 页码: {res['page_range']}")
        if 'text' in res:
            print(f"内容预览: {res['text']}")
        print("-" * 50)