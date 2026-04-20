"""
test_pipeline_web_fallback.py
------------------------------
Runs the full RAG pipeline with the first GPT call (generate_answer_with_citations)
patched to return "I don't have enough information to answer this."

This forces the web search fallback path and exercises:
    Step 3b — DuckDuckGo web search
    Step 4  — Hallucination detection against web snippets

Run from the project root:
    python tests/test_pipeline_web_fallback.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from unittest.mock import patch

from rag.pipeline import run_rag_pipeline

# ── Test question ──────────────────────────────────────────────────────────────
QUESTION = "What is retrieval-augmented generation and how does it reduce hallucinations?"

# ── Simulated insufficient RAG response ───────────────────────────────────────
INSUFFICIENT_ANSWER = """\
ANSWER
------
I don't have enough information to answer this.

SUPPORTING CONTEXT
------------------

REFERENCES
----------
"""

# Fake chunks returned by the reranker (pipeline needs non-empty list to proceed)
FAKE_CHUNKS = [
    {
        "text":        "placeholder chunk",
        "paper_id":    "test_paper",
        "title":       "Test Paper",
        "chunk_index": 0,
        "token_start": 0,
        "token_end":   10,
        "rerank_score": 0.5,
        "similarity":  0.5,
    }
]

print("=" * 60)
print("FULL PIPELINE — WEB FALLBACK TEST")
print("=" * 60)
print(f"\nQuestion : {QUESTION}")
print(f"\nPatching generate_answer_with_citations → insufficient answer")
print(f"Patching rerank → single fake chunk (bypass ChromaDB reranker)")
print("=" * 60)

with patch("rag.pipeline.generate_answer_with_citations",
           return_value=(INSUFFICIENT_ANSWER, "fake context", {})), \
     patch("rag.pipeline.rerank",
           return_value=FAKE_CHUNKS), \
     patch("rag.pipeline.retrieve",
           return_value=FAKE_CHUNKS):

    result = run_rag_pipeline(question=QUESTION)

print("\n" + "=" * 60)
print("PIPELINE RESULT")
print("=" * 60)
print(f"  Source           : {result['source']}")
print(f"  Score            : {result['score']:.4f}")
print(f"  Is Hallucinated  : {result['is_hallucinated']}")
print(f"  Hallucination Type: {result['hallucination_type']}")
print(f"  Explanation      : {result['explanation']}")

if result["web_citations"]:
    print(f"\n  Web Citations ({len(result['web_citations'])}):")
    for num, cite in result["web_citations"].items():
        print(f"    [{num}] {cite['title']}")
        print(f"         {cite['url']}")

print(f"\n  Answer:\n")
print(result["answer"])
print("=" * 60)
