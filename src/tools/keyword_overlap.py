"""
keyword_overlap.py
-------------------
Token-level word overlap between an answer and the retrieved context.

Complements cosine similarity: while similarity captures topic-level
closeness, keyword overlap catches word-for-word mismatches — e.g.
"Mumbai" vs "Delhi" in an otherwise identical sentence.

Metric: F1 word overlap (same as SQuAD evaluation)
  precision = |answer_tokens ∩ context_tokens| / |answer_tokens|
  recall    = |answer_tokens ∩ context_tokens| / |context_tokens_unique|
  f1        = 2 * precision * recall / (precision + recall)

Stopwords are removed so common words don't inflate scores.
"""

import re
from typing import Set

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "and", "or", "but", "not", "it",
    "its", "this", "that", "these", "those", "i", "we", "you", "he",
    "she", "they", "as", "also", "which", "who", "what", "when", "where",
    "about", "after", "before", "during", "into", "through", "than",
    "so", "if", "because", "while", "although",
}


def _tokens(text: str) -> Set[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def keyword_overlap(answer: str, context: str) -> dict:
    """
    Compute token-level F1 overlap between *answer* and *context*.

    Parameters
    ----------
    answer  : The generated answer (or a single sentence from it).
    context : The retrieved context passage.

    Returns
    -------
    dict with keys:
        f1            : float [0.0, 1.0] — higher = more word overlap
        precision     : float — fraction of answer tokens found in context
        recall        : float — fraction of context tokens found in answer
        missing_tokens: list[str] — answer tokens NOT in context
        signal        : str — "high_overlap" | "partial_overlap" | "low_overlap"
        detail        : str — human-readable summary for the agent
    """
    answer_tokens  = _tokens(answer)
    context_tokens = _tokens(context)

    if not answer_tokens:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "missing_tokens": [],
            "signal": "low_overlap",
            "detail": "Answer has no content tokens to compare.",
        }

    common    = answer_tokens & context_tokens
    precision = len(common) / len(answer_tokens) if answer_tokens else 0.0
    recall    = len(common) / len(context_tokens) if context_tokens else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    missing = sorted(answer_tokens - context_tokens)

    if f1 >= 0.60:
        signal = "high_overlap"
        detail = (
            f"F1={f1:.3f} — most answer tokens appear in context. "
            "Answer is likely well-grounded at word level."
        )
    elif f1 >= 0.30:
        signal = "partial_overlap"
        detail = (
            f"F1={f1:.3f} — partial word overlap. "
            f"Tokens in answer but NOT in context: {missing[:8]}. "
            "Check whether these missing tokens are hallucinated facts."
        )
    else:
        signal = "low_overlap"
        detail = (
            f"F1={f1:.3f} — low word overlap. "
            f"Most answer tokens absent from context: {missing[:8]}. "
            "Strong signal of hallucination."
        )

    return {
        "f1":             round(f1, 4),
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "missing_tokens": missing[:15],   # cap list for readability
        "signal":         signal,
        "detail":         detail,
    }
