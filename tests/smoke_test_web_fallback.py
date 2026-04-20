"""
smoke_test_web_fallback.py
--------------------------
Smoke test for the web search fallback path.

Simulates a RAG answer of "I don't have enough information to answer this."
then verifies that:
    1. is_insufficient_answer() correctly detects it
    2. search_web() returns results from DuckDuckGo
    3. generate_answer_from_web() produces a cited answer via GPT-4o

Run from the project root:
    python tests/smoke_test_web_fallback.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from rag.generator import is_insufficient_answer, generate_answer_from_web
from rag.web_search import search_web

# ── Test question — something clearly outside a Qasper academic corpus ────────
QUESTION = "What is the latest stable version of Python and when was it released?"

# ── Simulated GPT output that triggers the fallback ───────────────────────────
SIMULATED_RAG_ANSWER = """\
ANSWER
------
I don't have enough information to answer this.

SUPPORTING CONTEXT
------------------

REFERENCES
----------
"""

print("=" * 60)
print("WEB SEARCH FALLBACK — SMOKE TEST")
print("=" * 60)
print(f"\nQuestion : {QUESTION}")
print(f"Simulated RAG answer:\n{SIMULATED_RAG_ANSWER.strip()}\n")

# ── Step 1: Verify trigger detection ──────────────────────────────────────────
print("-" * 60)
print("Step 1 — is_insufficient_answer() detection")
triggered = is_insufficient_answer(SIMULATED_RAG_ANSWER)
print(f"  Triggered : {triggered}")
assert triggered, "FAIL: is_insufficient_answer() should have returned True"
print("  PASS")

# ── Step 2: DuckDuckGo search ─────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Step 2 — DuckDuckGo web search")
web_results = search_web(QUESTION, max_results=5)
print(f"  Results returned : {len(web_results)}")
for r in web_results:
    print(f"  [{r['number']}] {r['title']}")
    print(f"       URL     : {r['url']}")
    print(f"       Snippet : {r['snippet'][:120]}...")
assert len(web_results) > 0, "FAIL: DuckDuckGo returned no results"
print("  PASS")

# ── Step 3: GPT answer from web results ───────────────────────────────────────
print("\n" + "-" * 60)
print("Step 3 — generate_answer_from_web() via GPT-4o")
answer, web_citations = generate_answer_from_web(QUESTION, web_results)

print(f"\n  Full answer:\n")
print(answer)

print(f"\n  Parsed citations ({len(web_citations)}):")
for num, cite in web_citations.items():
    print(f"    [{num}] {cite['title']}")
    print(f"         {cite['url']}")

assert len(answer) > 0, "FAIL: generate_answer_from_web() returned empty answer"
print("\n  PASS")

print("\n" + "=" * 60)
print("ALL STEPS PASSED")
print("=" * 60)
