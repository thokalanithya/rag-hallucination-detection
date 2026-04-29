# RAG Hallucination Detector

Detects and classifies hallucinations in RAG-generated answers. Given a question, the system retrieves relevant passages from a document corpus, generates an answer with GPT-4o, then runs a multi-gate detector to score how well the answer is grounded in the retrieved context.

---

## How it works

```
Question
  │
  ├─ [1] ChromaDB vector search        top-20 candidate chunks (all-MiniLM-L6-v2)
  ├─ [2] Cross-encoder re-rank         top-5 final chunks (ms-marco-MiniLM-L-6-v2)
  ├─ [3] GPT-4o answer generation      cited answer from retrieved context only
  │       └─ web fallback              DuckDuckGo search if context is insufficient
  └─ [4] Hallucination detection
          ├─ Gate 1  cosine(sentence, merged context) ≥ 0.65 → grounded
          ├─ Gate 2  max cosine(sentence, each chunk) ≥ 0.50 → grounded
          ├─ Gate 3  claim verification (numbers, names, dates must appear in context)
          └─ GPT-4o type classifier    only fires when hallucination is detected
```

Hallucination score = `1 − avg(per-sentence support scores)`. Scores ≥ 0.5 are flagged as hallucinated.

---

## Hallucination types

| Type | Description |
|---|---|
| `grounded` | Fully supported by the retrieved context |
| `unsupported_hallucination` | Contains information absent from context |
| `context_conflicting_hallucination` | Directly contradicts the context |
| `fabricated_detail_or_entity` | Invents names, numbers, or citations |
| `overgeneralization_or_extrapolation` | Draws broader conclusions than context supports |
| `partially_supported_hallucination` | Partially grounded; some details invented |

---

## Project structure

```
rag-hallucination-detection/
├── src/
│   ├── app.py                        # FastAPI server (SSE streaming, web UI)
│   ├── config.py                     # Thresholds, model names, paths
│   ├── agent/
│   │   └── hallucination_agent.py    # Multi-gate hallucination detector
│   ├── rag/
│   │   ├── pipeline.py               # End-to-end RAG pipeline
│   │   ├── vector_store.py           # ChromaDB index build + retrieval
│   │   ├── embedder.py               # Sentence-transformer embeddings
│   │   ├── reranker.py               # Cross-encoder re-ranking
│   │   ├── generator.py              # GPT-4o answer + citation generation
│   │   ├── web_search.py             # DuckDuckGo web fallback
│   │   ├── trust_scorer.py           # Domain trust scoring (Tranco list)
│   │   └── chunker.py                # Token-aware document chunking
│   ├── tools/
│   │   ├── similarity_checker.py     # Bi-encoder cosine similarity (Gates 1 & 2)
│   │   ├── claim_extractor.py        # Extract verifiable facts (Gate 3)
│   │   ├── find_in_context.py        # Exact/token/fuzzy claim lookup (Gate 3)
│   │   └── sentence_splitter.py      # NLTK sentence tokenizer
│   ├── data/
│   │   ├── qasper_loader.py          # Load allenai/qasper from HuggingFace
│   │   ├── halueval_loader.py        # Load HaluEval QA pairs
│   │   └── pptx_loader.py            # Extract text from .pptx files
│   └── static/                       # Web UI assets
├── data/
│   ├── qasper_cache/                 # HuggingFace dataset cache
│   ├── qasper_test_examples.json     # 20 test examples from 20 different papers
│   ├── qa_data.json                  # HaluEval QA pairs
│   └── pptx/                         # Uploaded PowerPoint files
├── chroma_db/                        # Persistent ChromaDB vector store
├── tests/
│   └── smoke_test_e2e.py             # End-to-end smoke test (RAG + web cases)
├── scripts/
│   └── ingest_pptx.py                # Ingest .pptx files into ChromaDB
├── requirements.txt
└── .env                              # API keys (not committed)
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv src/venv
source src/venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env   # then fill in your keys
```

Required `.env` keys:

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

---

## Running the app

```bash
source src/venv/bin/activate
cd src
uvicorn app:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Ingesting documents

**Qasper academic papers** (loaded automatically on first run via HuggingFace):

```bash
cd src
python -c "
from data.qasper_loader import load_qasper
from rag.vector_store import build_index
docs, _ = load_qasper(max_papers=50)
build_index(docs)
"
```

**PowerPoint files** — drop `.pptx` files into `data/pptx/` then run:

```bash
python scripts/ingest_pptx.py
```

---

## Running tests

```bash
source src/venv/bin/activate
python tests/smoke_test_e2e.py
```

Runs 20 RAG cases from the Qasper corpus and 5 web-fallback cases, reporting pass/fail for each.

---

## Key configuration (`src/config.py`)

| Parameter | Value | Purpose |
|---|---|---|
| `HALLUCINATION_THRESHOLD` | `0.5` | Score ≥ 0.5 → hallucinated |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Bi-encoder for Gates 1 & 2 |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder re-ranker |
| `RETRIEVAL_CANDIDATES` | `20` | Chunks pulled from ChromaDB (broad pass) |
| `RETRIEVAL_TOP_K` | `5` | Chunks kept after re-ranking (final) |
| `CHUNK_SIZE` | `600` | Tokens per chunk |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Web results used when RAG is insufficient |
