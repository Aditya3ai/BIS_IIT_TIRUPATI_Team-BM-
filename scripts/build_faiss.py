#!/usr/bin/env python3
"""
Build FAISS index and save embeddings/ids for hybrid retrieval.
Saves to `data/index/faiss.index`, `data/index/embeddings.npy`, `data/index/ids.pkl`.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np


def extract_text(chunk):
    for key in ("text", "content", "chunk", "body", "doc"):
        if key in chunk and isinstance(chunk[key], str):
            return chunk[key]
    # fallback: join string fields
    parts = [str(v) for v in chunk.values() if isinstance(v, str)]
    return "\n".join(parts)


def main():
    from sentence_transformers import SentenceTransformer
    import faiss

    base = Path(__file__).resolve().parents[1]
    chunks_path = base / "data" / "chunks.json"
    if not chunks_path.exists():
        print("ERROR: data/chunks.json not found. Run quick_pipeline.py first.")
        return

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    texts = []
    ids = []
    for i, c in enumerate(chunks):
        ids.append(c.get("standard_id") or c.get("id") or str(i))
        texts.append(extract_text(c))

    print(f"Loaded {len(texts)} chunks. Building embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity (inner product)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    ids_int = np.arange(len(ids)).astype("int64")
    index.add_with_ids(embeddings, ids_int)

    out_dir = base / "data" / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "ids.pkl", "wb") as f:
        pickle.dump(ids, f)

    print("FAISS index, embeddings and ids written to data/index/")


if __name__ == '__main__':
    main()
