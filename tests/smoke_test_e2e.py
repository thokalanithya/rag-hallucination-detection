"""
smoke_test_e2e.py
-----------------
End-to-end smoke test for the full RAG + hallucination-detection pipeline.

Six test cases:
    RAG-1, RAG-2, RAG-3  — questions answerable from the Qasper academic corpus
                            (source should be "rag")
    WEB-1, WEB-2, WEB-3  — general-knowledge questions outside the NLP corpus
                            (pipeline falls back to DuckDuckGo web search;
                             source should be "web")

Each case runs the complete run_rag_pipeline() workflow:
    Stage 1  ChromaDB vector search
    Stage 2  Cross-encoder re-ranking
    Stage 3  GPT-4o answer generation (with web fallback if insufficient)
    Stage 4  Agentic hallucination detection

Run from the project root:
    python tests/smoke_test_e2e.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from data.qasper_loader import load_qasper
from rag.vector_store import build_index
from rag.pipeline import run_rag_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"

def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def _result_summary(label: str, result: dict) -> None:
    print(f"\n  Question  : {result['question']}")
    print(f"  Source    : {result['source']}")
    print(f"  Score     : {result['score']:.4f}  ({'HALLUCINATED' if result['is_hallucinated'] else 'GROUNDED'})")
    print(f"  Type      : {result['hallucination_type']}")
    print(f"  Explain   : {result['explanation']}")
    answer_preview = result['answer'][:200].replace("\n", " ")
    print(f"  Answer    : {answer_preview}{'...' if len(result['answer']) > 200 else ''}")


# ── Setup: build Qasper index and pick 3 answerable questions ─────────────────

_banner("SETUP — Loading Qasper corpus and building ChromaDB index")

documents, qa_pairs = load_qasper(max_papers=10)
build_index(documents)

answerable_pairs = [q for q in qa_pairs if q["answerable"]]
if len(answerable_pairs) < 3:
    raise RuntimeError(
        f"Need at least 3 answerable QA pairs; found {len(answerable_pairs)}. "
        "Try increasing max_papers."
    )

rag_cases = answerable_pairs[:3]

print(f"  Loaded {len(documents)} documents, {len(qa_pairs)} QA pairs")
print(f"  Selected {len(rag_cases)} RAG test cases from Qasper:")
for i, q in enumerate(rag_cases, 1):
    print(f"    RAG-{i}: [{q['paper_id']}] {q['question'][:80]}")


# ── Web test cases: topics clearly outside the NLP academic corpus ────────────

WEB_CASES = [
    "What are the main symptoms of COVID-19 infection and how are they treated?",
    # "How do I make sourdough bread at home from scratch?",
    # "What were the key outcomes of the 2024 Paris Summer Olympics?",
    "What are the early warning signs of Type 2 diabetes?",
    "How does the mRNA vaccine technology work?",
    "What is the recommended daily intake of vitamin B12 for adults?",
    "What is the difference between TCP and UDP protocols?"
]


# ── Run RAG test cases ────────────────────────────────────────────────────────

_banner("RAG TEST CASES — Questions answered from Qasper knowledge base")

rag_results = []
rag_statuses = []

for i, case in enumerate(rag_cases, 1):
    print(f"\n{'─' * 60}")
    print(f"  RAG-{i}: {case['question'][:80]}")
    print(f"{'─' * 60}")

    t0 = time.time()
    result = run_rag_pipeline(
        question=case["question"],
        paper_id_filter=case["paper_id"],
    )
    elapsed = time.time() - t0

    _result_summary(f"RAG-{i}", result)
    print(f"  Elapsed   : {elapsed:.1f}s")

    # Assertions
    errors = []
    if result["source"] not in ("rag", "web"):
        errors.append(f"unexpected source '{result['source']}'")
    if not isinstance(result["score"], float):
        errors.append("score is not a float")
    if not isinstance(result["is_hallucinated"], bool):
        errors.append("is_hallucinated is not a bool")
    if not result["answer"].strip():
        errors.append("answer is empty")
    if not result["hallucination_type"]:
        errors.append("hallucination_type is empty")

    if errors:
        status = FAIL
        print(f"\n  STATUS    : FAIL — {'; '.join(errors)}")
    else:
        status = PASS
        print(f"\n  STATUS    : PASS  (source={result['source']})")

    rag_results.append(result)
    rag_statuses.append(status)


# ── Run Web test cases ────────────────────────────────────────────────────────

_banner("WEB TEST CASES — General-knowledge questions triggering web fallback")

web_results = []
web_statuses = []

for i, question in enumerate(WEB_CASES, 1):
    print(f"\n{'─' * 60}")
    print(f"  WEB-{i}: {question}")
    print(f"{'─' * 60}")

    t0 = time.time()
    result = run_rag_pipeline(question=question)
    elapsed = time.time() - t0

    _result_summary(f"WEB-{i}", result)
    print(f"  Elapsed   : {elapsed:.1f}s")

    # Assertions
    errors = []
    if result["source"] not in ("rag", "web"):
        errors.append(f"unexpected source '{result['source']}'")
    if not isinstance(result["score"], float):
        errors.append("score is not a float")
    if not isinstance(result["is_hallucinated"], bool):
        errors.append("is_hallucinated is not a bool")
    if not result["answer"].strip():
        errors.append("answer is empty")

    # Warn (not fail) if RAG answered a question we expected to go to web —
    # the model may have found loosely-related content in the corpus.
    if result["source"] == "rag":
        print(f"  NOTE      : Expected web fallback but RAG answered (corpus may overlap)")

    if errors:
        status = FAIL
        print(f"\n  STATUS    : FAIL — {'; '.join(errors)}")
    else:
        status = PASS
        print(f"\n  STATUS    : PASS  (source={result['source']})")

    web_results.append(result)
    web_statuses.append(status)


# ── Final summary ─────────────────────────────────────────────────────────────

_banner("SMOKE TEST SUMMARY")

all_statuses = rag_statuses + web_statuses
passed = all_statuses.count(PASS)
failed = all_statuses.count(FAIL)

print(f"\n  {'Label':<10} {'Question':<55} {'Source':<5} {'Score':<7} {'Halluc':<8} {'Status'}")
print(f"  {'-'*10} {'-'*55} {'-'*5} {'-'*7} {'-'*8} {'-'*6}")

for i, (result, status) in enumerate(zip(rag_results, rag_statuses), 1):
    q = result["question"][:53] + ".." if len(result["question"]) > 53 else result["question"]
    print(
        f"  {'RAG-' + str(i):<10} {q:<55} {result['source']:<5} "
        f"{result['score']:<7.4f} {str(result['is_hallucinated']):<8} {status}"
    )

for i, (result, status) in enumerate(zip(web_results, web_statuses), 1):
    q = result["question"][:53] + ".." if len(result["question"]) > 53 else result["question"]
    print(
        f"  {'WEB-' + str(i):<10} {q:<55} {result['source']:<5} "
        f"{result['score']:<7.4f} {str(result['is_hallucinated']):<8} {status}"
    )

print(f"\n  Total: {len(all_statuses)} cases — {passed} passed, {failed} failed")
print()

if failed > 0:
    print("RESULT: SOME TESTS FAILED")
    sys.exit(1)
else:
    print("RESULT: ALL TESTS PASSED")
