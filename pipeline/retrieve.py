import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pipeline.utils import normalize_text, tokenize


def _embed_query(text: str, mode: str) -> np.ndarray:
    if mode == "google":
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query",
        )
        return np.asarray(result["embedding"], dtype=np.float32)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def _rrf_fusion(bm25_ids: List[int], vec_ids: List[int], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for rank, idx in enumerate(bm25_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(vec_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieve(query: str, index_dir: Path, embeddings: str, top_k: int = 5) -> List[Dict]:
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    bm25 = pickle.loads((index_dir / "bm25.pkl").read_bytes())

    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ids = list(np.argsort(bm25_scores)[::-1][:50])

    if embeddings == "none":
        fused = bm25_ids[:top_k]
        return [chunks[i] for i in fused]

    import faiss

    index = faiss.read_index(str(index_dir / "faiss.index"))
    qvec = _embed_query(normalize_text(query), embeddings).astype("float32")
    faiss.normalize_L2(qvec.reshape(1, -1))
    scores, ids = index.search(qvec.reshape(1, -1), 50)
    vec_ids = [int(i) for i in ids[0] if i >= 0]

    fused = _rrf_fusion(bm25_ids, vec_ids)
    return [chunks[i] for i in fused[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top standards from indexes.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--embeddings", choices=["google", "sbert", "none"], default="google")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve(args.query, Path(args.index_dir), args.embeddings, args.top_k)
    for item in results:
        print(item["standard_id"], "-", item.get("title", ""))


if __name__ == "__main__":
    main()
