#!/usr/bin/env python3
"""
Repository-root inference.py (copy of src/inference.py) with hallucination guard.
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

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


# (Copy BISRetriever implementation from src)
from src.inference import BISRetriever  # reuse class from src to avoid duplication


def post_filter_rationale(rationale: str, standard_id: str) -> str:
    """Remove any IS standard mentions that are not the current `standard_id`.
    Keeps rationale concise.
    """
    # Remove tokens like 'IS 1234' if not equal to standard_id
    def replace_match(m):
        s = m.group(0)
        if standard_id.replace(' ', '') in s.replace(' ', ''):
            return s
        return ''
    filtered = re.sub(r"IS\s*\d{1,5}(?:\s*\(Part\s*\d+\))?(?::\s*\d{4})?", replace_match, rationale, flags=re.IGNORECASE)
    # Collapse whitespace
    return re.sub(r"\s+", " ", filtered).strip()


def main():
    parser = argparse.ArgumentParser(description="Repo-root inference wrapper")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    # Load queries
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    queries = json.loads(input_path.read_text(encoding="utf-8"))

    retriever = BISRetriever(Path(args.index_dir), use_vector_search=not args.bm25_only, use_bm25=True)

    results = []
    for item in queries:
        q = item.get("query", "")
        qid = item.get("id", "")
        start = time.time()
        retrieved = retriever.retrieve(q, top_k=args.top_k)
        latency = time.time() - start

        # Apply hallucination guard to rationale
        for r in retrieved:
            if "rationale" in r and r["rationale"]:
                r["rationale"] = post_filter_rationale(r["rationale"], r["standard_id"]) or f"Relevant to: {q}"

        results.append({
            "id": qid,
            "query": q,
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": [r["standard_id"] for r in retrieved],
            "latency_seconds": round(latency, 3),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {out_path}")


if __name__ == '__main__':
    main()
