# CSE 594 — RAG Hallucination Detector

A diagnostic framework that detects and classifies hallucinations in RAG-generated answers using an independent LLM-based verifier.

---

## Project Structure

```
cse594/
├── data/
│   └── sample_dataset.json       # 10 labeled scientific Q&A entries
├── src/
│   ├── config.py                 # Constants, thresholds, hallucination taxonomy
│   ├── dataset_schema.py         # Dataclasses: DatasetEntry, ContextChunk, etc.
│   └── data_loader.py            # Load, validate, filter, inspect dataset
├── tests/                        # Unit tests (added from Week 3 onward)
├── results/                      # Evaluation outputs (added Week 6)
├── requirements.txt
└── README.md
```

---

## Hallucination Taxonomy

| Label | Description |
|---|---|
| `grounded` | Fully supported by the retrieved context |
| `unsupported_hallucination` | Contains information absent from context |
| `context_conflicting_hallucination` | Contradicts information in context |
| `fabricated_detail_or_entity` | Invents names, numbers, citations |
| `overgeneralization_or_extrapolation` | Draws broader conclusions than context supports |
| `partially_supported_hallucination` | Partially grounded; some details invented |

---

## Weekly Milestones

| Week | Focus | Status |
|---|---|---|
| 1 | Environment setup, schema design, seed dataset | ✅ Complete |
| 2 | RAG pipeline (FAISS + HuggingFace embeddings) | 🔲 |
| 3 | LLM verifier — answer-level scoring | 🔲 |
| 4 | Sentence-level hallucination detection | 🔲 |
| 5 | Hallucination type classification | 🔲 |
| 6 | Evaluation on full dataset | 🔲 |
| 7 | Threshold tuning & prompt refinement | 🔲 |
| 8 | Final report & presentation | 🔲 |

---

## Setup

```bash
# 1. Clone / navigate to the project
cd cse594

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. Run Week 1 smoke test
cd src
python data_loader.py
```

---

## Key Configuration (src/config.py)

| Parameter | Value | Purpose |
|---|---|---|
| `HALLUCINATION_THRESHOLD` | 0.5 | Score ≥ 0.5 → hallucinated |
| `GROUNDED_UPPER` | 0.25 | Score < 0.25 → fully grounded |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | RAG retriever |
| `VERIFIER_MODEL` | `gpt-4o-mini` | LLM verifier |
| `RETRIEVAL_TOP_K` | 3 | Context chunks retrieved per query |
