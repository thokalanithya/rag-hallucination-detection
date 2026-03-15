"""
CSE 594 - Hallucination Detection for RAG Systems
data_loader.py: Load, validate, save, and inspect dataset entries.
"""

import json
import os
import sys
from typing import List, Optional, Tuple

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import (
    SAMPLE_DATASET_PATH,
    HALLUCINATION_THRESHOLD,
    GROUNDED_UPPER,
    MIN_CONTEXT_WORDS,
    MIN_ANSWER_WORDS,
    HallucinationType,
)
from dataset_schema import Dataset, DatasetEntry, ContextChunk, SentenceEvaluation


# ─────────────────────────────────────────────
# Load & Save
# ─────────────────────────────────────────────

def load_dataset(path: str = SAMPLE_DATASET_PATH) -> Dataset:
    """Load a Dataset from a JSON file and return it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return Dataset.load(path)


def save_dataset(dataset: Dataset, path: str = SAMPLE_DATASET_PATH) -> None:
    """Save a Dataset to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.save(path)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_entry(entry: DatasetEntry) -> Tuple[bool, List[str]]:
    """
    Validates a single DatasetEntry.

    Returns:
        (is_valid: bool, issues: List[str])
        is_valid is True only if issues is empty.
    """
    issues = []

    # Question check
    if not entry.question or not entry.question.strip():
        issues.append("Missing or empty question.")

    # Context check
    if not entry.retrieved_context:
        issues.append("No retrieved context chunks provided.")
    else:
        for i, chunk in enumerate(entry.retrieved_context):
            if chunk.word_count() < MIN_CONTEXT_WORDS:
                issues.append(
                    f"Context chunk {i} is too short "
                    f"({chunk.word_count()} words, minimum {MIN_CONTEXT_WORDS})."
                )

    # Answer check
    if not entry.generated_answer or not entry.generated_answer.strip():
        issues.append("Missing or empty generated answer.")
    elif len(entry.generated_answer.split()) < MIN_ANSWER_WORDS:
        issues.append(
            f"Generated answer is too short "
            f"({len(entry.generated_answer.split())} words, minimum {MIN_ANSWER_WORDS})."
        )

    # Label validity
    if entry.ground_truth_label and entry.ground_truth_label not in HallucinationType.ALL_TYPES:
        issues.append(
            f"Unknown ground_truth_label: {entry.ground_truth_label!r}. "
            f"Valid labels: {HallucinationType.ALL_TYPES}"
        )

    # Score range
    if entry.answer_score is not None:
        if not (0.0 <= entry.answer_score <= 1.0):
            issues.append(
                f"answer_score {entry.answer_score} is out of range [0.0, 1.0]."
            )

    return len(issues) == 0, issues


def validate_dataset(dataset: Dataset) -> Tuple[int, int, dict]:
    """
    Validates all entries in a Dataset.

    Returns:
        (valid_count, invalid_count, error_report)
        error_report maps entry_id → list of issues
    """
    valid_count = 0
    invalid_count = 0
    error_report = {}

    for entry in dataset.entries:
        is_valid, issues = validate_entry(entry)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            error_report[entry.metadata.entry_id] = issues

    return valid_count, invalid_count, error_report


# ─────────────────────────────────────────────
# Filtering Helpers
# ─────────────────────────────────────────────

def filter_by_label(dataset: Dataset, label: str) -> List[DatasetEntry]:
    """Return all entries matching the given ground-truth label."""
    return dataset.get_by_label(label)


def filter_hallucinated(dataset: Dataset) -> List[DatasetEntry]:
    """Return all entries whose label is NOT grounded."""
    return [e for e in dataset.entries if e.ground_truth_label != HallucinationType.GROUNDED]


def filter_scored(dataset: Dataset) -> List[DatasetEntry]:
    """Return entries that already have a verifier score."""
    return [e for e in dataset.entries if e.is_scored()]


def filter_unscored(dataset: Dataset) -> List[DatasetEntry]:
    """Return entries that still need to be scored."""
    return [e for e in dataset.entries if not e.is_scored()]


# ─────────────────────────────────────────────
# Statistics & Reporting
# ─────────────────────────────────────────────

def dataset_summary(dataset: Dataset) -> dict:
    """
    Returns a summary dictionary with dataset-level statistics.
    Useful for quick inspection and for building evaluation reports.
    """
    total = len(dataset.entries)
    labeled = sum(1 for e in dataset.entries if e.is_labeled())
    scored = sum(1 for e in dataset.entries if e.is_scored())

    dist = dataset.label_distribution()

    summary = {
        "total_entries": total,
        "labeled_entries": labeled,
        "unlabeled_entries": total - labeled,
        "scored_entries": scored,
        "unscored_entries": total - scored,
        "label_distribution": dist,
    }

    # If any entries have been scored, compute average scores per label
    if scored > 0:
        scores_by_label: dict = {}
        for e in dataset.entries:
            if e.is_scored() and e.ground_truth_label:
                scores_by_label.setdefault(e.ground_truth_label, []).append(e.answer_score)
        summary["avg_score_by_label"] = {
            lbl: round(sum(vals) / len(vals), 4)
            for lbl, vals in scores_by_label.items()
        }

    return summary


def print_summary(dataset: Dataset) -> None:
    """Pretty-print the dataset summary to stdout."""
    s = dataset_summary(dataset)
    print("\n" + "=" * 50)
    print(f"  DATASET SUMMARY  (v{dataset.version})")
    print("=" * 50)
    print(f"  Total entries   : {s['total_entries']}")
    print(f"  Labeled         : {s['labeled_entries']}")
    print(f"  Scored          : {s['scored_entries']}")
    print()
    print("  Label distribution:")
    for label, count in s["label_distribution"].items():
        bar = "█" * count
        print(f"    {label:<45} {count:>2}  {bar}")
    if "avg_score_by_label" in s:
        print()
        print("  Average verifier score by label:")
        for label, avg in s["avg_score_by_label"].items():
            print(f"    {label:<45} {avg:.4f}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# Entry-Level Inspection
# ─────────────────────────────────────────────

def inspect_entry(entry: DatasetEntry) -> None:
    """Print a detailed view of a single DatasetEntry."""
    print("\n" + "─" * 60)
    print(f"  Entry ID : {entry.metadata.entry_id}")
    print(f"  Domain   : {entry.metadata.domain}")
    print(f"  Difficulty: {entry.metadata.difficulty or 'N/A'}")
    print(f"  Label    : {entry.ground_truth_label or 'UNLABELED'}")
    print(f"  Score    : {entry.answer_score if entry.is_scored() else 'NOT SCORED'}")
    print("─" * 60)
    print(f"  Question :\n    {entry.question}")
    print()
    print(f"  Context  ({len(entry.retrieved_context)} chunk(s)):")
    for i, c in enumerate(entry.retrieved_context):
        print(f"    [{i+1}] {c.source}")
        print(f"         {c.text[:120]}...")
    print()
    print(f"  Answer   :\n    {entry.generated_answer}")
    if entry.ground_truth_note:
        print(f"\n  GT Note  : {entry.ground_truth_note}")
    if entry.sentence_scores:
        print(f"\n  Sentence Evaluations ({len(entry.sentence_scores)}):")
        for ss in entry.sentence_scores:
            print(f"    [{ss.hallucination_score:.2f}] ({ss.hallucination_type}) {ss.sentence[:80]}")
    print("─" * 60)


# ─────────────────────────────────────────────
# Quick Test — run as script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("[data_loader] Running Week 1 smoke test...\n")

    # 1. Load the sample dataset
    ds = load_dataset()

    # 2. Validate all entries
    valid, invalid, errors = validate_dataset(ds)
    print(f"Validation: {valid} valid, {invalid} invalid.")
    if errors:
        for eid, issues in errors.items():
            print(f"  Entry {eid}: {issues}")

    # 3. Print summary
    print_summary(ds)

    # 4. Inspect one entry from each hallucination type
    for label in HallucinationType.ALL_TYPES:
        matches = filter_by_label(ds, label)
        if matches:
            print(f"\n--- Example of [{label}] ---")
            inspect_entry(matches[0])
