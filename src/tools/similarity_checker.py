"""
similarity_checker.py
----------------------
Compute semantic similarity between a single sentence and a context passage
using sentence-transformers (all-MiniLM-L6-v2).

A low score (< 0.5) signals that the sentence may not be grounded in the
context — giving the agent real evidence to reason over.

Note: similarity ≠ factual correctness. A sentence can be topically similar
but contradict the context (e.g. a wrong number). The agent uses this score
as one input to its reasoning, not as a final verdict.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

# ── Bi-encoder singleton (load once, reuse across calls) ─────────────────────
_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Run: pip install sentence-transformers"
            )
    return _model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_sentence_support(sentence: str, context: str) -> dict:
    """
    Compute how semantically similar *sentence* is to *context*.

    Parameters
    ----------
    sentence : A single sentence from the generated answer.
    context  : The full retrieved context passage.

    Returns
    -------
    dict with keys:
        support_score : float [0.0, 1.0] — higher = more semantically similar
        signal        : str — "supported" | "low_support" | "no_support"
        detail        : str — human-readable interpretation for the agent
    """
    sentence = sentence.strip()
    context  = context.strip()

    if not sentence or not context:
        return {
            "support_score": 0.0,
            "signal": "no_support",
            "detail": "Empty input — cannot compute similarity.",
        }

    model = _get_model()
    embeddings = model.encode([sentence, context], convert_to_numpy=True)
    score = _cosine(embeddings[0], embeddings[1])
    # Clamp to [0, 1] (cosine can be slightly negative for very dissimilar texts)
    score = max(0.0, min(1.0, score))

    if score >= 0.65:
        signal = "supported"
        detail = (
            f"Similarity {score:.3f} — sentence appears semantically aligned "
            "with the context. Verify specific facts manually."
        )
    elif score >= 0.40:
        signal = "low_support"
        detail = (
            f"Similarity {score:.3f} — sentence has partial overlap with context. "
            "Check for added details, overgeneralization, or subtle contradictions."
        )
    else:
        signal = "no_support"
        detail = (
            f"Similarity {score:.3f} — sentence is semantically distant from "
            "context. Likely contains unsupported or fabricated information."
        )

    return {
        "support_score": round(score, 4),
        "signal": signal,
        "detail": detail,
    }
