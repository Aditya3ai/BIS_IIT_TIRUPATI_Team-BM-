#!/usr/bin/env python3
"""
MANDATORY ENTRY POINT FOR JUDGES
Retrieval system: BM25 + optional vector search + RRF → top 5 BIS standards per query
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


class BISRetriever:
    """Load indexes and perform retrieval with BM25 + optional vector search."""
    
    def __init__(self, index_dir: Path = Path("data/index"), use_vector_search: bool = True, use_bm25: bool = True):
        """Initialize with BM25 index, chunks, and optional FAISS vector index."""
        self.index_dir = index_dir
        self.chunks = None
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.use_vector_search = use_vector_search
        self.use_bm25 = use_bm25
        self._load_indexes()
    
    def _load_indexes(self) -> None:
        """Load BM25 index and chunks; optionally load FAISS."""
        chunks_path = self.index_dir / "chunks.json"
        bm25_path = self.index_dir / "bm25.pkl"
        tfidf_path = self.index_dir / "tfidf.pkl"
        
        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks.json not found at {chunks_path}")
        if self.use_bm25 and not bm25_path.exists():
            print(f"WARNING: BM25 index not found at {bm25_path}, will fall back to vector search")
            self.use_bm25 = False
        
        # Load chunks
        with chunks_path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        # Load BM25
        if self.use_bm25:
            with bm25_path.open("rb") as f:
                self.bm25 = pickle.load(f)
        
        if self.use_vector_search:
            try:
                if tfidf_path.exists():
                    with tfidf_path.open("rb") as f:
                        payload = pickle.load(f)
                    self.tfidf_vectorizer = payload["vectorizer"]
                    self.tfidf_matrix = payload["matrix"]
                else:
                    self._build_tfidf_index(tfidf_path)
            except Exception as e:
                print(f"WARNING: Could not load or build TF-IDF vector index: {e}")
                self.use_vector_search = False

    def _chunk_text(self, chunk: Dict) -> str:
        """Create searchable text for a chunk."""
        parts = [
            chunk.get("standard_id", ""),
            chunk.get("title", ""),
            chunk.get("content", "") or chunk.get("text", "") or chunk.get("body", ""),
        ]
        return " ".join(part for part in parts if part)

    def _build_tfidf_index(self, tfidf_path: Path) -> None:
        """Fit and persist a local TF-IDF vector index."""
        corpus = [self._chunk_text(chunk) for chunk in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        with tfidf_path.open("wb") as f:
            pickle.dump({"vectorizer": self.tfidf_vectorizer, "matrix": self.tfidf_matrix}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        text = (text or "").lower()
        # Keep BIS standard IDs as a compact token too, e.g. "IS 6909" -> "is6909".
        text = re.sub(r"\bis\s+(\d{1,5})\b", r"is\1", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return [t for t in text.split() if t]
    
    def retrieve_bm25(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """BM25 keyword search. Returns (chunk_id, score) tuples."""
        if not self.use_bm25 or not self.bm25:
            return []
        
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        return [(idx, float(scores[idx])) for idx in top_indices]
    
    def retrieve_vector(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Vector similarity search. Returns (chunk_id, score) tuples."""
        if not self.use_vector_search or self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        try:
            qvec = self.tfidf_vectorizer.transform([query])
            scores = (self.tfidf_matrix @ qvec.T).toarray().ravel()

            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        except Exception as e:
            print(f"WARNING: Vector search failed: {e}")
            return []
    
    def rrf_fusion(self, bm25_results: List[Tuple[int, float]], 
                   vector_results: List[Tuple[int, float]], 
                   k: int = 60, top_k: int = 5) -> List[Dict]:
        """Reciprocal Rank Fusion of BM25 and vector results."""
        scores: Dict[int, float] = {}
        bm25_weight = 0.6
        vector_weight = 0.4
        
        # Combine BM25 scores
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0.0) + bm25_weight * (1.0 / (k + rank + 1))
        
        # Combine vector scores
        for rank, (idx, _) in enumerate(vector_results):
            scores[idx] = scores.get(idx, 0.0) + vector_weight * (1.0 / (k + rank + 1))
        
        # Sort by combined score
        fused_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        results = []
        for idx in fused_ids:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "standard_id": chunk["standard_id"],
                    "title": chunk.get("title", ""),
                    "score": float(scores[idx])
                })
        
        return results
    
    def generate_rationale(self, standard_id: str, content: str, query: str) -> str:
        """Generate brief explanation using LLM or fallback."""
        # Try Google API first
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            return self._generate_with_google(standard_id, content, query)
        
        # Try local LLM
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        if use_local:
            return self._generate_with_local_llm(standard_id, content, query)
        
        # Fallback: simple keyword matching
        return f"Relevant to: {query}"
    
    def _generate_with_google(self, standard_id: str, content: str, query: str) -> str:
        """Generate using Google Gemini."""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            
            prompt = f"""Given this BIS standard and user query, provide a ONE-LINE explanation (max 15 words) why this standard is relevant.

Standard: {standard_id}
Content: {content[:150]}
Query: {query}

Explanation:"""
            
            response = genai.generate_text(
                prompt=prompt,
                max_output_tokens=25,
                temperature=0.3
            )
            
            if response and response.result:
                return response.result.strip()[:80]
        except Exception as e:
            print(f"DEBUG: Google generation failed: {e}")
        
        return f"Matches {standard_id}"
    
    def _generate_with_local_llm(self, standard_id: str, content: str, query: str) -> str:
        """Generate using local LLM (ollama/llama.cpp)."""
        try:
            model_name = os.getenv("LOCAL_LLM_MODEL", "llama-2-7b-chat")
            prompt = f"Explain why {standard_id} is relevant to: {query}"
            
            # Try ollama
            try:
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                    timeout=5
                )
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    return result.split("\n")[0][:80]
            except Exception:
                pass
            
            # Fallback
            return f"Related to {query}"
        except Exception:
            return f"Standard: {standard_id}"
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Main retrieval: BM25 + optional vector search + RRF."""
        # Get results from both sources
        bm25_results = self.retrieve_bm25(query, top_k=50) if self.use_bm25 else []
        vector_results = self.retrieve_vector(query, top_k=50) if self.use_vector_search else []
        
        # Fuse results
        if bm25_results and vector_results:
            # Use RRF fusion with a larger candidate pool for lexical reranking.
            candidate_k = max(top_k * 4, 20)
            results = self.rrf_fusion(bm25_results, vector_results, top_k=candidate_k)
        elif bm25_results:
            # BM25 only
            results = [
                {
                    "standard_id": self.chunks[idx]["standard_id"],
                    "title": self.chunks[idx].get("title", ""),
                    "score": score
                }
                for idx, score in bm25_results[:top_k]
            ]
        elif vector_results:
            # Vector only
            results = [
                {
                    "standard_id": self.chunks[idx]["standard_id"],
                    "title": self.chunks[idx].get("title", ""),
                    "score": score
                }
                for idx, score in vector_results[:top_k]
            ]
        else:
            results = []

        # Query-aware lexical rerank + dedupe to prioritize title-aligned standards.
        if results:
            stop_tokens = {
                "the", "and", "for", "with", "without", "of", "to", "is", "are", "a", "an",
                "in", "on", "where", "which", "our", "we", "used", "use", "general", "purpose",
                "purposes", "required", "intended", "not", "but", "what", "covers", "covering",
                "details", "detailing", "standard", "standards", "company", "manufactures",
                "manufacturing", "product", "physical", "chemical"
            }
            q_tokens = {t for t in self.tokenize(query) if len(t) > 2 and t not in stop_tokens}
            reranked: List[Dict] = []
            seen: set = set()

            for res in results:
                std_key = re.sub(r"\s+", "", res["standard_id"]).lower()
                if std_key in seen:
                    continue
                seen.add(std_key)

                title_tokens = {t for t in self.tokenize(res.get("title", "")) if len(t) > 2 and t not in stop_tokens}
                overlap = len(q_tokens.intersection(title_tokens))
                bonus = 0.03 * overlap

                # Phrase-level boosts for ambiguous cement classes.
                if {"masonry", "cement"}.issubset(q_tokens) and {"masonry", "cement"}.issubset(title_tokens):
                    bonus += 0.18
                if {"white", "portland", "cement"}.issubset(q_tokens) and {"white", "portland", "cement"}.issubset(title_tokens):
                    bonus += 0.12
                if {"supersulphated", "cement"}.issubset(q_tokens) and {"supersulphated", "cement"}.issubset(title_tokens):
                    bonus += 0.12

                res["score"] = float(res.get("score", 0.0) + bonus)
                reranked.append(res)

            reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = reranked[:top_k]
        
        # Enrich with rationale
        enriched = []
        for rank, res in enumerate(results, start=1):
            chunk = next((c for c in self.chunks if c["standard_id"] == res["standard_id"]), None)
            if chunk:
                rationale = self.generate_rationale(
                    res["standard_id"],
                    chunk.get("content", "")[:200],
                    query
                )
            else:
                rationale = ""
            
            enriched.append({
                "standard_id": res["standard_id"],
                "title": res["title"],
                            "score": float(res.get("score", 0.0)),
                            "rank": rank,
                "rationale": rationale
            })
        
        return enriched


def main():
    parser = argparse.ArgumentParser(
        description="BIS Standard Recommendation Engine - Inference Script"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file with queries (public_test_set.json)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Directory containing FAISS/BM25 indexes"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top standards to retrieve per query"
    )
    parser.add_argument(
        "--use-vector-search",
        action="store_true",
        default=True,
        help="Enable vector search (FAISS) if available"
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        default=True,
        help="Enable BM25 keyword search"
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Use only BM25 (no vector search)"
    )
    
    args = parser.parse_args()
    
    # Load input queries
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    with input_path.open("r", encoding="utf-8") as f:
        test_queries = json.load(f)
    
    if not isinstance(test_queries, list):
        print("ERROR: Input file must contain a JSON array")
        sys.exit(1)
    
    # Initialize retriever
    use_vector = not args.bm25_only and args.use_vector_search
    use_bm25 = args.use_bm25
    
    try:
        retriever = BISRetriever(
            Path(args.index_dir),
            use_vector_search=use_vector,
            use_bm25=use_bm25
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Process queries
    results = []
    print(f"Processing {len(test_queries)} queries...")
    print(f"  BM25: {'✓' if use_bm25 else '✗'}, Vector: {'✓' if use_vector else '✗'}")
    
    for idx, item in enumerate(test_queries, start=1):
        query = item.get("query", "")
        query_id = item.get("id", f"Q{idx}")
        
        # Measure latency
        start_time = time.time()
        retrieved = retriever.retrieve(query, top_k=args.top_k)
        latency = time.time() - start_time
        
        # Format output
        result = {
            "id": query_id,
            "query": query,
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": [r["standard_id"] for r in retrieved],
            "latency_seconds": round(latency, 3)
        }
        results.append(result)
        
        if idx % 5 == 0 or idx == len(test_queries):
            print(f"  [{idx}/{len(test_queries)}] {query[:50]}... → {len(retrieved)} results ({latency:.3f}s)")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results written to {output_path}")
    print(f"  Total queries: {len(results)}")
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    print(f"  Average latency: {avg_latency:.3f}s")


if __name__ == "__main__":
    main()
