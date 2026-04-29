"""
hallucination_agent.py
-----------------------
Two-gate hallucination detector — no LLM involved.

Architecture
------------
  Step 1 → split_sentences(answer)
            Split the answer into individual sentences (NLTK).

  Step 2 → Gate 1: check_sentence_support(sentence, context)
            Bi-encoder cosine similarity [0.0, 1.0].
            ≥ 0.65 → grounded, skip Gate 2.

  Step 3 → Gate 2 (only for sentences that failed Gate 1):
            Per-passage max cosine similarity.
            Context is split into individual numbered passages; check_sentence_support
            is called on each passage individually and the max score is taken.
            If max score ≥ 0.65 → grounded.
            Fixes context-dilution false positives: a single supporting passage
            can rescue a sentence that scored low against the full merged context
            because irrelevant chunks diluted the embedding.

  Step 4 → Gate 3: claim verification (RAG answers only, opt-in via verify_claims=True)
            extract_claims(sentence) pulls out specific verifiable facts
            (numbers, names, dates, percentages, p-values).
            find_in_context(claim, context) checks each via exact → token → fuzzy match.
            If any claim is absent from context → sentence demoted to no_support,
            type set to "contradicting". Catches factual substitutions that
            cosine and NLI both miss (e.g. "92% accuracy" when context says "87%").

  Step 5 → verdict
            Hallucination score  = 1 − avg(effective sentence support scores).
            is_hallucinated      = score ≥ THRESHOLD (0.5).
            Problematic sentences = those that failed all applicable gates.

Usage
-----
  # Single sample
  python -m agent.hallucination_agent detect \\
      --context "..." --answer "..." --verbose

  # Batch evaluation on qa_data.json
  python -m agent.hallucination_agent evaluate --n_samples 20

  # Full HaluEval pipeline
  python -m agent.hallucination_agent pipeline --n_samples 50
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Add src/ to path so all packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.halueval_loader import load_halueval_qa
from tools import split_sentences, check_sentence_support, extract_claims, find_in_context
from config import HallucinationType

THRESHOLD = 0.5


# ── GPT-4o hallucination type classifier ─────────────────────────────────────

_TYPE_PROMPT = """\
A hallucination has been detected in a RAG-generated answer. \
Classify it into exactly one of the types below.

Types:
{type_descriptions}

Question:
{question}

Retrieved context (truncated to 2000 chars):
{context}

Generated answer:
{answer}

Flagged sentences (not supported by context):
{flagged}

Reply with ONLY the type label — no explanation, no punctuation. \
Choose from:
{type_labels}"""


def _classify_type_with_llm(
    question: str,
    context: str,
    answer: str,
    problematic_sentences: list[str],
) -> str:
    """
    Call GPT-4o to classify which hallucination type applies.
    Falls back to 'unsupported_hallucination' on any error.
    """
    import os
    try:
        from openai import AzureOpenAI
    except ImportError:
        return HallucinationType.UNSUPPORTED

    api_key  = os.environ.get("AZURE_OPENAI_API_KEY", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    model    = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if not api_key or not endpoint:
        return HallucinationType.UNSUPPORTED

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=endpoint,
    )

    # Build descriptions for all non-grounded types
    non_grounded = [t for t in HallucinationType.ALL_TYPES if t != HallucinationType.GROUNDED]
    type_descriptions = "\n".join(
        f"- {t}: {HallucinationType.DESCRIPTIONS[t]}" for t in non_grounded
    )
    flagged_text = "\n".join(f"  - {s}" for s in problematic_sentences) \
        if problematic_sentences else "  (full answer is unsupported)"

    prompt = _TYPE_PROMPT.format(
        type_descriptions=type_descriptions,
        question=question,
        context=context[:2000],
        answer=answer,
        flagged=flagged_text,
        type_labels="\n".join(f"- {t}" for t in non_grounded),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=30,
        )
        raw = response.choices[0].message.content.strip().lower()
        for t in non_grounded:
            if t in raw:
                return t
        return HallucinationType.UNSUPPORTED
    except Exception:
        return HallucinationType.UNSUPPORTED

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    question: str
    context: str
    answer: str
    score: float
    is_hallucinated: bool
    hallucination_type: str
    problematic_sentences: list
    reasoning: str
    explanation: str
    ground_truth_label: Optional[int] = None
    agent_steps: int = 0


# ── Hallucination detection ───────────────────────────────────────────────────


def run_agent(
    context: str,
    answer: str,
    question: str = "",
    max_steps: int = 15,  # unused — kept for interface compatibility
    verbose: bool = False,
    verify_claims: bool = False,
    chunks: list[dict] | None = None,
) -> DetectionResult:
    """
    Detect hallucination using a two-gate pipeline.

    Gate 1 — cosine(sentence, full_merged_context): fast first pass.
              ≥ 0.65 → grounded immediately.

    Gate 2 — max cosine(sentence, each_chunk_individually): fires only when
              Gate 1 fails. Uses the raw retrieved chunks passed via `chunks`
              so each chunk is compared in isolation — no dilution from
              unrelated chunks being merged into one embedding. A single
              supporting chunk is enough to rescue the sentence.

    Gate 3 (RAG only, verify_claims=True) — claim verification: sentences that
              passed Gates 1 or 2 are checked for specific verifiable facts
              (numbers, names, dates); any claim absent from context is a hard
              veto → sentence demoted to no_support, type = "contradicting".

    Parameters
    ----------
    context       : Merged context string sent to the LLM (used for Gate 1 and Gate 3)
    answer        : Generated answer to evaluate
    question      : Original question (for record-keeping)
    max_steps     : Unused — kept so callers don't need to change their signature
    verbose       : Print per-sentence scores
    verify_claims : Run Gate 3 claim verification (enable for RAG answers only)
    chunks        : Individual retrieved chunk dicts (must have a "text" key).
                    When provided, Gate 2 compares against each chunk text
                    separately. Falls back to splitting context by [N] headers
                    if not provided.

    Returns
    -------
    DetectionResult with score, type, problematic sentences, reasoning, explanation
    """
    sentences = split_sentences(answer)

    if not sentences:
        return DetectionResult(
            question=question, context=context, answer=answer,
            score=0.0, is_hallucinated=False, hallucination_type="none",
            problematic_sentences=[], reasoning="No sentences found.",
            explanation="Empty answer — treated as grounded.", agent_steps=0,
        )

    # Build the list of individual texts for Gate 2.
    # Prefer the raw chunk texts (passed via `chunks`) — each chunk is a clean,
    # focused passage with no dilution from other chunks. Fall back to splitting
    # the merged context string by [N] headers if chunks are not provided.
    if chunks:
        _chunk_texts = [c["text"].strip() for c in chunks if c.get("text", "").strip()]
    else:
        import re as _re
        _passages = _re.split(r'\n\n(?=\[\d+\])', context.strip())
        _chunk_texts = [p.strip() for p in _passages if p.strip()] or [context]

    support_scores: list[float] = []
    flagged: list[tuple[str, float]] = []
    nli_rescued: int = 0
    claims_failed: int = 0

    _ANAPHORIC = {"this", "these", "it", "they", "their", "its", "such", "here"}

    for i, sentence in enumerate(sentences):
        result = check_sentence_support(sentence, context)
        score  = result["support_score"]
        signal = result["signal"]

        # Gate 2: max cosine against individual chunks (only for Gate 1 failures).
        # Each chunk is compared in isolation so unrelated chunks cannot dilute
        # the score. A single chunk with cosine ≥ 0.65 rescues the sentence.
        if signal != "supported":
            import logging as _logging
            _log = _logging.getLogger("rag_app")
            # For anaphoric sentences ("This...", "These...", "It...") the
            # bi-encoder cannot resolve the pronoun without context. Prepend
            # the previous sentence so the model sees the full referential chain.
            first_word = sentence.strip().split()[0].lower() if sentence.strip() else ""
            gate2_text = (
                sentences[i - 1] + " " + sentence if (i > 0 and first_word in _ANAPHORIC)
                else sentence
            )
            chunk_scores = [
                check_sentence_support(gate2_text, text)["support_score"]
                for text in _chunk_texts
            ]
            max_chunk_score = max(chunk_scores)
            _log.warning(
                f"Gate2 | sentence: {sentence[:70]!r} | "
                f"chunk scores: {[round(s,3) for s in chunk_scores]} | "
                f"max={max_chunk_score:.3f}"
            )
            # Gate 2 threshold (0.50) is lower than Gate 1 (0.65): individual
            # chunk texts are longer than the answer sentence, so the pooled
            # embedding naturally scores lower even for semantically aligned pairs.
            if max_chunk_score >= 0.50:
                score  = max_chunk_score
                signal = "supported"
                nli_rescued += 1

        # Gate 3: claim verification (RAG only, only for sentences that passed Gates 1+2)
        if verify_claims and signal == "supported":
            claim_result = extract_claims(sentence)
            if claim_result["has_claims"]:
                for claim_item in claim_result["claims"]:
                    lookup = find_in_context(claim_item["value"], context)
                    if not lookup["found"]:
                        score  = 0.35  # demote to no_support — specific fact not in context
                        signal = "contradicting"
                        claims_failed += 1
                        break

        support_scores.append(score)

        if verbose:
            print(f"    [{signal}] {score:.3f} | {sentence[:80]}")

        if signal != "supported":
            flagged.append((sentence, score))

    avg_support = sum(support_scores) / len(support_scores)
    hall_score  = round(max(0.0, min(1.0, 1.0 - avg_support)), 4)
    is_hall     = hall_score >= THRESHOLD

    n_no_support  = sum(1 for s in support_scores if s < 0.40)
    n_low_support = sum(1 for s in support_scores if 0.40 <= s < 0.65)

    problematic_sentences = [s for s, _ in flagged]

    if not is_hall:
        hall_type = "none"
    else:
        hall_type = _classify_type_with_llm(
            question=question,
            context=context,
            answer=answer,
            problematic_sentences=problematic_sentences,
        )

    score_str = " | ".join(f"{s:.2f}" for s in support_scores)
    reasoning = (
        f"Per-sentence support scores: [{score_str}]. "
        f"Avg support: {avg_support:.3f}. "
        f"Supported: {len(support_scores) - n_no_support - n_low_support} "
        f"(incl. {nli_rescued} rescued by NLI gate), "
        f"Low: {n_low_support}, No support: {n_no_support} "
        f"(incl. {claims_failed} failed claim verification)."
    )

    if is_hall:
        explanation = (
            f"{len(flagged)} of {len(sentences)} sentence(s) scored below the "
            f"support threshold (avg cosine similarity={avg_support:.3f})."
        )
    else:
        explanation = (
            f"All sentences are semantically aligned with the retrieved context "
            f"(avg cosine similarity={avg_support:.3f})."
        )

    return DetectionResult(
        question=question,
        context=context,
        answer=answer,
        score=hall_score,
        is_hallucinated=is_hall,
        hallucination_type=hall_type,
        problematic_sentences=problematic_sentences,
        reasoning=reasoning,
        explanation=explanation,
        agent_steps=len(sentences),
    )


# ── Type normalisation (for evaluate_qa_data) ─────────────────────────────────

TYPE_NORMALISE: dict[str, str] = {
    "none":               "grounded",
    "unsupported":        "unsupported_hallucination",
    "contradicting":      "context_conflicting_hallucination",
    "fabricated":         "fabricated_detail_or_entity",
    "overgeneralization": "overgeneralization_or_extrapolation",
    "partial":            "partially_supported_hallucination",
    # identity mappings for full label strings
    "grounded":                           "grounded",
    "unsupported_hallucination":          "unsupported_hallucination",
    "context_conflicting_hallucination":  "context_conflicting_hallucination",
    "fabricated_detail_or_entity":        "fabricated_detail_or_entity",
    "overgeneralization_or_extrapolation":"overgeneralization_or_extrapolation",
    "partially_supported_hallucination":  "partially_supported_hallucination",
}

ALL_LABELS = [
    "grounded",
    "unsupported_hallucination",
    "context_conflicting_hallucination",
    "fabricated_detail_or_entity",
    "overgeneralization_or_extrapolation",
    "partially_supported_hallucination",
]


def _normalise_type(raw: str) -> str:
    return TYPE_NORMALISE.get(raw.lower().strip(), raw.lower().strip())


# ── QA-data evaluation (qa_data.json) ─────────────────────────────────────────

def evaluate_qa_data(
    dataset_path: Optional[str] = None,
    n_samples: int = 20,
    seed: int = 42,
    verbose: bool = False,
    output_csv: str = "results_qa_eval.csv",
) -> "pd.DataFrame":
    """
    Evaluate binary hallucination detection on a subset of qa_data.json.

    Each line has: knowledge, question, right_answer (label=0),
    hallucinated_answer (label=1).

    Builds a balanced sample, runs the agent, and reports binary accuracy
    + hallucination type distribution.
    """
    import random
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )

    if dataset_path is None:
        dataset_path = str(Path(__file__).parent.parent / "data" / "qa_data.json")

    raw_entries = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_entries.append(json.loads(line))

    random.seed(seed)
    random.shuffle(raw_entries)

    half = n_samples // 2
    samples = []
    g_count = h_count = 0
    for entry in raw_entries:
        if g_count < half:
            samples.append({
                "question": entry["question"],
                "context":  entry["knowledge"],
                "answer":   entry["right_answer"],
                "label":    0,
            })
            g_count += 1
        if h_count < half:
            samples.append({
                "question": entry["question"],
                "context":  entry["knowledge"],
                "answer":   entry["hallucinated_answer"],
                "label":    1,
            })
            h_count += 1
        if g_count >= half and h_count >= half:
            break

    print(f"\nEvaluating {len(samples)} samples from qa_data.json "
          f"({g_count} grounded + {h_count} hallucinated)\n")

    records = []

    for i, sample in enumerate(samples):
        true_str = "HALL" if sample["label"] == 1 else "OK  "
        print(f"[{i+1}/{len(samples)}] true={true_str}  "
              f"Q: {sample['question'][:55]}…", end=" ", flush=True)

        result = run_agent(
            context=sample["context"],
            answer=sample["answer"],
            question=sample["question"],
            verbose=verbose,
        )

        pred       = int(result.is_hallucinated)
        is_correct = pred == sample["label"]
        hall_type  = result.hallucination_type if result.is_hallucinated else "none"

        print(
            f"pred={'HALL' if pred else 'OK  '}  "
            f"score={result.score:.2f}  "
            f"type={hall_type:<20}  "
            f"{'✓' if is_correct else '✗'}"
        )

        records.append({
            "question":          sample["question"],
            "context":           sample["context"],
            "answer":            sample["answer"],
            "ground_truth":      sample["label"],
            "predicted":         pred,
            "score":             result.score,
            "hallucination_type": hall_type,
            "problematic_sentences": "; ".join(result.problematic_sentences),
            "explanation":       result.explanation,
            "correct":           is_correct,
        })

    df = pd.DataFrame(records)
    y_true, y_pred = df["ground_truth"], df["predicted"]

    print("\n" + "=" * 60)
    print("QA DATA EVALUATION RESULTS")
    print("=" * 60)
    print(f"Provider          : cosine similarity (sentence-transformers)")
    print(f"Samples           : {len(df)}  ({g_count} grounded + {h_count} hallucinated)")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision         : {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall            : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1                : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["Grounded", "Hallucinated"]))

    type_counts = df[df["predicted"] == 1]["hallucination_type"].value_counts()
    if not type_counts.empty:
        print("Hallucination type distribution (predicted positives):")
        print(type_counts.to_string())
    print("=" * 60)

    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    return df


# ── HaluEval batch pipeline ────────────────────────────────────────────────────

def run_pipeline(
    n_samples: int = 50,
    threshold: float = THRESHOLD,
    seed: int = 42,
    output_csv: str = "results_agent.csv",
    verbose: bool = False,
) -> "pd.DataFrame":
    """Run the agent over the HaluEval QA dataset and evaluate performance."""
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )

    df = load_halueval_qa(n_samples=n_samples, seed=seed)
    records = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Analysing …", end=" ", flush=True)

        result = run_agent(
            context=row["context"],
            answer=row["answer"],
            question=row.get("question", ""),
            verbose=verbose,
        )
        result.ground_truth_label = int(row["label"])

        print(
            f"score={result.score:.2f}  "
            f"pred={'HAL' if result.is_hallucinated else 'OK ':3s}  "
            f"true={'HAL' if result.ground_truth_label else 'OK ':3s}  "
            f"type={result.hallucination_type}"
        )
        records.append(asdict(result))

    results_df = pd.DataFrame(records)
    y_true = results_df["ground_truth_label"]
    y_pred = results_df["is_hallucinated"].astype(int)

    print("\n" + "=" * 60)
    print("AGENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Provider          : cosine similarity (sentence-transformers)")
    print(f"Samples evaluated : {len(results_df)}")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision         : {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall            : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1                : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["Grounded", "Hallucinated"]))

    type_counts = results_df[results_df["is_hallucinated"]]["hallucination_type"].value_counts()
    if not type_counts.empty:
        print("Hallucination type distribution:")
        print(type_counts.to_string())
    print("=" * 60)

    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    return results_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cosine-similarity hallucination detector"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── single detect ─────────────────────────────────────────────────────────
    det = subparsers.add_parser("detect", help="Detect hallucination in one sample")
    det.add_argument("--context",  required=True)
    det.add_argument("--answer",   required=True)
    det.add_argument("--question", default="")
    det.add_argument("--verbose",  action="store_true")

    # ── qa_data evaluation ────────────────────────────────────────────────────
    evl = subparsers.add_parser("evaluate", help="Evaluate on qa_data.json subset")
    evl.add_argument("--dataset",   default=None)
    evl.add_argument("--n_samples", type=int,   default=20)
    evl.add_argument("--seed",      type=int,   default=42)
    evl.add_argument("--output",    default="results_qa_eval.csv")
    evl.add_argument("--verbose",   action="store_true")

    # ── HaluEval pipeline ─────────────────────────────────────────────────────
    pip = subparsers.add_parser("pipeline", help="Run full HaluEval pipeline")
    pip.add_argument("--n_samples", type=int,   default=50)
    pip.add_argument("--threshold", type=float, default=THRESHOLD)
    pip.add_argument("--seed",      type=int,   default=42)
    pip.add_argument("--output",    default="results_agent.csv")
    pip.add_argument("--verbose",   action="store_true")

    args = parser.parse_args()

    if args.command == "detect":
        result = run_agent(
            context=args.context,
            answer=args.answer,
            question=args.question,
            verbose=args.verbose,
        )
        print("\n── Detection Result ──────────────────────────────────────")
        print(f"Score                : {result.score:.3f}")
        print(f"Hallucinated         : {result.is_hallucinated}")
        print(f"Type                 : {result.hallucination_type}")
        print(f"Problematic sentences: {result.problematic_sentences}")
        print(f"Explanation          : {result.explanation}")
        print(f"Reasoning            : {result.reasoning[:300]}…"
              if len(result.reasoning) > 300 else f"Reasoning: {result.reasoning}")
        print(f"Agent steps          : {result.agent_steps}")

    elif args.command == "evaluate":
        evaluate_qa_data(
            dataset_path=args.dataset,
            n_samples=args.n_samples,
            seed=args.seed,
            verbose=args.verbose,
            output_csv=args.output,
        )

    elif args.command == "pipeline":
        run_pipeline(
            n_samples=args.n_samples,
            threshold=args.threshold,
            seed=args.seed,
            output_csv=args.output,
            verbose=args.verbose,
        )
