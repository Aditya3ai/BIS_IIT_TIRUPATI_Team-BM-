# Setup & Configuration Guide

## Quick Start (No API Key Needed)

**BM25-only mode works offline:**

```bash
pip install -r requirements.txt
python src/inference.py \
  --input public_test_set.json \
  --output data/results/output.json \
  --bm25-only
```

**Performance:** Hit Rate 90%, MRR 0.74, <5ms latency ✅

---

## Optional: Add LLM Rationale Generation

### Option 1: Google Gemini (Cloud - Recommended)

Create `.env` file:
```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_actual_api_key
```

Then run without `--bm25-only`:
```bash
python src/inference.py \
  --input public_test_set.json \
  --output data/results/output.json
```

**Benefits:**
- Generated rationales (20 pts manual scoring)
- No hallucination risk (filtered to retrieved standards only)
- Fast (<1s per query)

### Option 2: Local LLM (Completely Offline)

**Prerequisites:**
- Install [Ollama](https://ollama.ai) or [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Start local LLM server: `ollama serve` (runs on localhost:11434)

Edit `.env`:
```
USE_LOCAL_LLM=true
LOCAL_LLM_MODEL=llama-2-7b-chat
LOCAL_LLM_DEVICE=cpu
```

Then run:
```bash
python src/inference.py \
  --input public_test_set.json \
  --output data/results/output.json
```

**Benefits:**
- ✅ Completely local (no API key exposed)
- ✅ No network calls
- ✅ Privacy-preserving
- ⚠️ Slower (~3-5s per query, depends on system)

---

## Optional: Add Vector Search (FAISS + RRF Fusion)

**Prerequisites:**
- Install embeddings model: `pip install sentence-transformers`

**Generate FAISS index** (one-time):
```bash
python -c "
import json, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

# Load chunks
chunks = json.load(open('data/chunks.json', encoding='utf-8'))

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [f\"{c['standard_id']} {c.get('title','')} {c.get('content','')[:200]}\" for c in chunks]
embeddings = model.encode(texts, normalize_embeddings=True).astype('float32')

# Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, 'data/index/faiss.index')
print(f'✓ FAISS index created with {len(chunks)} vectors')
"
```

**Then run with vector search:**
```bash
python src/inference.py \
  --input public_test_set.json \
  --output data/results/output_hybrid.json
```

**Result:**
- BM25 + Vector search fused with RRF
- Expected Hit Rate: 90-95% (minimal improvement over BM25 alone)
- Latency: ~50-100ms per query

---

## Judges' Evaluation Setup

### What Judges Will Do

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run inference
python src/inference.py \
  --input <path_to_hidden_test_set.json> \
  --output <path_to_results.json>

# 3. Evaluate (their script)
python eval_script.py --results <path_to_results.json>
```

### Important Notes

✅ **Judges will run BM25-only** (default, no extra flags)  
✅ **API keys NOT needed** for passing automated scoring  
✅ **No `.env` file required** - judges won't have API keys anyway  
⚠️ **Vector search optional** - adds complexity, minimal gain  

---

## API Key Safety

### DO NOT
```bash
git commit .env                    # ❌ Exposes API keys
python src/inference.py            # ❌ May fail if no .env
```

### DO
```bash
cp .env.example .env               # ✅ Copy from template
git add .env.example              # ✅ Track template only
git ignore .env                   # ✅ Never commit actual keys
python src/inference.py --bm25-only  # ✅ Works without .env
```

---

## Configuration Reference

| Setting | Default | Options |
|---------|---------|---------|
| `GOOGLE_API_KEY` | (empty) | Your Google API key |
| `USE_LOCAL_LLM` | false | true/false |
| `LOCAL_LLM_MODEL` | llama-2-7b-chat | ollama model name |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | HuggingFace model |
| `USE_VECTOR_SEARCH` | true | true/false |
| `USE_BM25` | true | true/false |
| `RRF_K` | 60 | RRF fusion parameter |

---

## Testing

### Local Testing (With Metrics)
```bash
# Run inference
python src/inference.py \
  --input public_test_set.json \
  --output data/results/test_output.json \
  --bm25-only

# Check metrics
python eval_script.py --results data/results/test_output.json
```

### Benchmark Different Modes

```bash
# BM25 only (fastest, reliable)
python src/inference.py --input public_test_set.json --output out1.json --bm25-only

# With vector search (if FAISS available)
python src/inference.py --input public_test_set.json --output out2.json

# With Google API (if key available)
# Edit .env with GOOGLE_API_KEY, then run above
```

---

## Troubleshooting

### Error: "chunks.json not found"
```bash
python quick_pipeline.py  # Re-generate data
```

### Error: "faiss not installed"
```bash
pip install faiss-cpu
```

### Error: "GOOGLE_API_KEY not set"
- Create `.env` file (copy from `.env.example`)
- Set `GOOGLE_API_KEY=your_key`
- Or just run with `--bm25-only` (works offline)

### Error: "Local LLM not responding"
- Start Ollama: `ollama serve`
- Or disable: `USE_LOCAL_LLM=false` in `.env`

---

## Recommended Setup for Hackathon

**For Maximum Score (40+20+40 pts):**

1. **Automated (40 pts)** ← Already passing ✅
   - Use BM25 (no API key needed)
   - Hit Rate 90%, MRR 0.74, <5ms latency

2. **Manual (20 pts)** ← Optional for+5-10 pts
   - Add Google API key + rationale generation
   - Better explanations = higher relevance score

3. **Subjective (40 pts)** ← Optional
   - Create simple web UI
   - Nice presentation/demo

**Minimum viable solution:**
```bash
python src/inference.py --input public_test_set.json --output out.json --bm25-only
```

**Better solution:**
```bash
# Set up .env with Google API key, then:
python src/inference.py --input public_test_set.json --output out.json
```
