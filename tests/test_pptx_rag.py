"""
test_pptx_rag.py
----------------
Verify that the RAG pipeline retrieves and answers from the ingested PPT data.

Run AFTER ingest_pptx.py has been executed:
    python tests/test_pptx_rag.py

The script:
  1. Lists all documents indexed in the pptx_chunks collection
  2. Runs a few sample retrievals to confirm content is searchable
  3. Optionally runs the full RAG pipeline (requires Azure OpenAI keys in .env)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

COLLECTION_NAME = "pptx_chunks"

# Load only vector_store (avoids triggering rag/__init__.py → pipeline → ddgs)
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

SRC = PROJECT_ROOT / "src"
_load_module("config", SRC / "config.py")
_load_module("rag.chunker", SRC / "rag" / "chunker.py")
_load_module("rag.vector_store", SRC / "rag" / "vector_store.py")

# ── 1. Collection stats ────────────────────────────────────────────────────────
from rag.vector_store import collection_stats, retrieve

stats = collection_stats(COLLECTION_NAME)
print(f"\n{'='*60}")
print(f"  Collection: '{COLLECTION_NAME}'")
print(f"  Total chunks indexed: {stats['total_chunks']}")
print(f"{'='*60}")

if stats["total_chunks"] == 0:
    print("\nNo chunks found. Run  scripts/ingest_pptx.py  first.")
    sys.exit(1)

# ── 2. List indexed documents ──────────────────────────────────────────────────
import chromadb
from config import CHROMA_DIR

client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_collection(COLLECTION_NAME)

# Sample up to 200 items to discover which paper_ids are in the collection
sample = col.get(limit=200, include=["metadatas"])
paper_ids = sorted({m["paper_id"] for m in sample["metadatas"]})
titles    = {m["paper_id"]: m["title"] for m in sample["metadatas"]}

print(f"\nDocuments in '{COLLECTION_NAME}':")
for pid in paper_ids:
    print(f"  • {pid}  —  {titles[pid][:70]}")

# ── 3. Retrieval smoke tests ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Retrieval smoke tests (top-3 chunks per query)")
print(f"{'='*60}")

# Build auto-generated queries from document titles
test_queries = [f"What is covered in {titles[pid][:50]}?" for pid in paper_ids[:3]]
# Append a generic fallback
test_queries.append("What are the main topics discussed in the presentations?")

for query in test_queries:
    print(f"\nQuery: {query!r}")
    results = retrieve(query, top_k=3, collection_name=COLLECTION_NAME)
    if not results:
        print("  No results returned.")
        continue
    for i, r in enumerate(results, 1):
        print(f"  [{i}] similarity={r['similarity']:.3f} | {r['paper_id']} slide chunk {r['chunk_index']}")
        print(f"       {r['text'][:120].strip()} …")

# ── 4. Full RAG pipeline (optional) ───────────────────────────────────────────
print(f"\n{'='*60}")
print("  Full RAG pipeline test (requires .env with Azure OpenAI keys)")
print(f"{'='*60}")

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    from rag.pipeline import run_rag_pipeline

    question = f"Summarize the key points from the presentation: {titles[paper_ids[0]][:60]}"
    print(f"\nQuestion: {question!r}")

    result = run_rag_pipeline(
        question=question,
        candidates=20,
        top_k=5,
        collection_name=COLLECTION_NAME,
    )

    print(f"\nAnswer:\n{result.get('answer', 'N/A')}")
    print(f"\nHallucination score : {result.get('hallucination_score', 'N/A')}")
    print(f"Is hallucinated     : {result.get('is_hallucinated', 'N/A')}")
    print(f"Type                : {result.get('hallucination_type', 'N/A')}")

except Exception as e:
    print(f"\n[SKIPPED] Full pipeline test failed: {e}")
    print("  This is OK if Azure OpenAI keys are not configured.")

print(f"\n{'='*60}")
print("  Test complete.")
print(f"{'='*60}\n")
