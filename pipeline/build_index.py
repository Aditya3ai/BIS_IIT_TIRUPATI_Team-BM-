import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from pipeline.utils import normalize_text, tokenize


_IS_TOKEN_RE = re.compile(r"\bIS\s+(\d{1,5})\b", flags=re.IGNORECASE)


def _build_retrieval_text(chunk: Dict, title_weight: int = 3) -> str:
    """Create BM25 text with title boosting and compact IS token support."""
    standard_id = normalize_text(chunk.get("standard_id", ""))
    title = normalize_text(chunk.get("title", ""))
    content = normalize_text(chunk.get("content", ""))

    compact_standard_id = _IS_TOKEN_RE.sub(lambda m: f"IS{m.group(1)}", standard_id)
    boosted_title = " ".join([title] * max(1, title_weight)) if title else ""

    return normalize_text(f"{standard_id} {compact_standard_id} {boosted_title} {content}")


def _load_chunks(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _embed_with_google(texts: List[str]) -> np.ndarray:
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    genai.configure(api_key=api_key)

    vectors = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        vectors.append(result["embedding"])
    return np.asarray(vectors, dtype=np.float32)


def _embed_with_sentence_transformers(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def _build_faiss(vectors: np.ndarray) -> "faiss.Index":
    import faiss

    if vectors.size == 0:
        raise RuntimeError("No vectors to index")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def _build_bm25(texts: List[str]):
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def build_indexes(chunks_path: Path, out_dir: Path, embeddings: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = _load_chunks(chunks_path)
    texts = [_build_retrieval_text(c, title_weight=3) for c in chunks]

    if embeddings == "google":
        vectors = _embed_with_google(texts)
    elif embeddings == "sbert":
        vectors = _embed_with_sentence_transformers(texts)
    else:
        vectors = np.zeros((len(texts), 1), dtype=np.float32)

    if embeddings != "none":
        index = _build_faiss(vectors)
        import faiss

        faiss.write_index(index, str(out_dir / "faiss.index"))
        np.save(out_dir / "vectors.npy", vectors)

    bm25 = _build_bm25(texts)
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    with (out_dir / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes from chunks.")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON")
    parser.add_argument("--out-dir", required=True, help="Output directory for indexes")
    parser.add_argument("--embeddings", choices=["google", "sbert", "none"], default="google")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    out_dir = Path(args.out_dir)
    build_indexes(chunks_path, out_dir, args.embeddings)
    print(f"Indexes saved to {out_dir}")


if __name__ == "__main__":
    main()
