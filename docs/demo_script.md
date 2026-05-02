Demo Script — BIS Recommendation Engine

1. Intro (20s)
- One-line problem statement: mapping product descriptions to BIS standards is slow.
- One-line solution: RAG-powered retriever that returns top-3 standards with rationale.

2. Quick architecture (30s)
- Ingest PDFs → chunk standards → build BM25 + FAISS indexes.
- Use RRF fusion for final ranking; rationale by LLM with hallucination guard.

3. Live demo (2:30)
- Show `inference.py` running against a sample query (or `public_test_set.json`).
- Open `data/chunks.json` to show a retrieved standard and its text.
- Show output JSON highlighting `retrieved_standards` and latency.

4. Evaluation (30s)
- Show `eval_script.py` results: HitRate@3 and MRR.

5. Wrap-up & next steps (30s)
- Improvements: UI, deployment, stronger hallucination filters, more embeddings.

Recording instructions:
- Use screen recorder (OBS or native tool) and a microphone.
- Record 1–2 takes; aim for <=7 minutes total.
- Export MP4 and attach to submission.
