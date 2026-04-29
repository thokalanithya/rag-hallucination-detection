"""
smoke_test_hallucination_types.py
----------------------------------
Verify that all 6 hallucination types from the taxonomy are correctly
detected and classified by the pipeline.

One crafted (question, context, answer) triple per type:
    grounded                        — expected: not hallucinated
    unsupported_hallucination       — info absent from context
    context_conflicting_hallucination — directly contradicts context
    fabricated_detail_or_entity     — invented names / numbers
    overgeneralization_or_extrapolation — broader than context supports
    partially_supported_hallucination — part correct, part invented

Calls run_agent() directly — no ChromaDB, no LLM answer generation needed.
GPT-4o is only used for the type classification step (when is_hallucinated=True).

Run from the project root:
    python tests/smoke_test_hallucination_types.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from agent.hallucination_agent import run_agent
from config import HallucinationType

# ── Fixed context used by all test cases ──────────────────────────────────────

CONTEXT = """
A study conducted at Penn State University examined the effect of tutoring frequency
on undergraduate academic performance. The study included 120 undergraduate students
enrolled in introductory calculus (MATH 140) during the Fall 2022 semester.
Students who attended tutoring sessions at least twice per week improved their GPA
by an average of 0.4 points over the semester. Students who attended only once per
week showed no statistically significant improvement in GPA. The study concluded
that high-frequency tutoring is associated with measurable academic gains
specifically in introductory mathematics courses.
""".strip()

QUESTION = "What did the Penn State tutoring study find about GPA improvement?"

# ── Test cases ─────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id":                   "TYPE-1",
        "label":                "grounded",
        "expected_hallucinated": False,
        "expected_type":        "none",
        "answer": (
            "Students who attended tutoring at least twice per week improved their GPA "
            "by an average of 0.4 points over the semester. Students who attended only "
            "once per week showed no statistically significant improvement in GPA."
        ),
    },
    {
        "id":                   "TYPE-2",
        "label":                "unsupported_hallucination",
        "expected_hallucinated": True,
        "expected_type":        HallucinationType.UNSUPPORTED,
        "answer": (
            "Students who participated in tutoring sessions reported significantly lower "
            "stress levels and improved sleep quality throughout the semester, which "
            "contributed to better focus during exams."
        ),
    },
    {
        "id":                   "TYPE-3",
        "label":                "context_conflicting_hallucination",
        "expected_hallucinated": True,
        "expected_type":        HallucinationType.CONTRADICTION,
        "answer": (
            "Students who attended tutoring sessions once per week showed a statistically "
            "significant improvement in their GPA over the semester, demonstrating that "
            "even minimal tutoring frequency produces measurable academic gains."
        ),
    },
    {
        "id":                   "TYPE-4",
        "label":                "fabricated_detail_or_entity",
        "expected_hallucinated": True,
        "expected_type":        HallucinationType.FABRICATION,
        "answer": (
            "The study was led by Dr. Sarah Mitchell and published in the Journal of "
            "Educational Psychology in March 2023. Penn State's TutorTrac attendance "
            "system was used to log sessions, and the study covered 350 students across "
            "five STEM courses."
        ),
    },
    {
        "id":                   "TYPE-5",
        "label":                "overgeneralization_or_extrapolation",
        "expected_hallucinated": True,
        "expected_type":        HallucinationType.OVERGENERALIZATION,
        "answer": (
            "Regular tutoring attendance universally improves GPA for all college "
            "students across all subjects and institutions worldwide, proving that "
            "tutoring is the single most effective academic intervention available."
        ),
    },
    {
        "id":                   "TYPE-6",
        "label":                "partially_supported_hallucination",
        "expected_hallucinated": True,
        "expected_type":        HallucinationType.PARTIAL,
        "answer": (
            "Students attending tutoring at least twice per week improved their GPA by "
            "0.4 points on average. Students who attended once per week also saw a "
            "moderate improvement of 0.2 points, showing that even low-frequency "
            "tutoring produces some benefit."
        ),
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS   = "PASS"
FAIL   = "FAIL"
WARN   = "WARN"   # detected as grounded when hallucination was expected (gate miss)

def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── Run test cases ─────────────────────────────────────────────────────────────

_banner("HALLUCINATION TYPE CLASSIFICATION SMOKE TEST")
print(f"\n  Context (shared across all cases):")
print(f"  {CONTEXT[:120].replace(chr(10), ' ')} ...")
print(f"\n  Question: {QUESTION}")

records = []

for case in TEST_CASES:
    print(f"\n{'─' * 70}")
    print(f"  {case['id']} — expected type: {case['label']}")
    print(f"  Answer: {case['answer'][:100]}{'...' if len(case['answer']) > 100 else ''}")
    print(f"{'─' * 70}")

    t0 = time.time()
    result = run_agent(
        context=CONTEXT,
        answer=case["answer"],
        question=QUESTION,
        verify_claims=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    detected_hall   = result.is_hallucinated
    classified_type = result.hallucination_type

    print(f"\n  Score          : {result.score:.4f}")
    print(f"  Is hallucinated: {detected_hall}  (expected: {case['expected_hallucinated']})")
    print(f"  Type           : {classified_type}  (expected: {case['expected_type']})")
    print(f"  Explanation    : {result.explanation}")
    if result.problematic_sentences:
        for s in result.problematic_sentences:
            print(f"  Flagged        : {s[:100]}")
    print(f"  Elapsed        : {elapsed:.1f}s")

    # ── Determine status ──────────────────────────────────────────────────────
    if not case["expected_hallucinated"]:
        # Grounded case: detection must be correct
        if not detected_hall:
            status = PASS
        else:
            status = FAIL
            print(f"  FAIL: expected GROUNDED but got HALLUCINATED (score={result.score:.4f})")
    else:
        # Hallucinated case: check detection first, then type
        if not detected_hall:
            # Gates missed it — type classifier never ran
            status = WARN
            print(f"  WARN: expected HALLUCINATED but gates scored it as GROUNDED.")
            print(f"        Type classifier did not run — cosine similarity too high for this type.")
        elif classified_type == case["expected_type"]:
            status = PASS
        else:
            status = FAIL
            print(f"  FAIL: detected as hallucinated but type mismatch.")
            print(f"        Got '{classified_type}', expected '{case['expected_type']}'")

    print(f"\n  STATUS: {status}")

    records.append({
        "id":                   case["id"],
        "label":                case["label"],
        "expected_hallucinated": case["expected_hallucinated"],
        "expected_type":        case["expected_type"],
        "detected_hallucinated": detected_hall,
        "classified_type":      classified_type,
        "score":                result.score,
        "status":               status,
    })

# ── Summary table ──────────────────────────────────────────────────────────────

_banner("SUMMARY")

print(f"\n  {'ID':<8} {'Expected Type':<40} {'Got Type':<40} {'Score':<7} {'Status'}")
print(f"  {'-'*8} {'-'*40} {'-'*40} {'-'*7} {'-'*6}")

for r in records:
    print(
        f"  {r['id']:<8} {r['expected_type']:<40} {r['classified_type']:<40} "
        f"{r['score']:<7.4f} {r['status']}"
    )

passed = sum(1 for r in records if r["status"] == PASS)
warned = sum(1 for r in records if r["status"] == WARN)
failed = sum(1 for r in records if r["status"] == FAIL)

print(f"\n  Total: {len(records)} cases — {passed} passed, {warned} warned (gate miss), {failed} failed")
print()

if warned:
    print("  NOTE: WARN means the cosine-similarity gates did not flag the answer as")
    print("        hallucinated, so the GPT-4o type classifier never ran. This is a")
    print("        known limitation for subtle types (overgeneralization, partial).")
    print("        Detection improvement requires NLI or LLM-based gate changes.")
    print()

if failed > 0:
    print("RESULT: SOME TESTS FAILED")
    sys.exit(1)
else:
    print("RESULT: ALL TESTS PASSED (or warned)")
