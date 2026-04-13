"""
find_in_context.py
-------------------
Check whether a specific claim, entity, or value appears in the context.

This is the key tool for catching factual substitutions — cases where the
hallucinated answer swaps one fact for another (e.g. "Mumbai" instead of
"Delhi"). Cosine similarity cannot catch these because both words are
semantically close; direct lookup can.

Matching strategy (in order of strictness):
  1. Exact substring match (case-insensitive)
  2. All tokens of the claim appear in context (word-level)
  3. Fuzzy ratio match (difflib) ≥ 0.80 for multi-word claims
"""

import re
import difflib
from typing import List


def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def find_in_context(claim: str, context: str) -> dict:
    """
    Search for *claim* inside *context* using three matching strategies.

    Parameters
    ----------
    claim   : A specific fact, entity, number, or phrase to look up.
    context : The full retrieved context passage.

    Returns
    -------
    dict with keys:
        found          : bool — True if the claim was located by any strategy
        match_type     : str  — "exact", "all_tokens", "fuzzy", or "not_found"
        match_score    : float — 1.0 for exact/all_tokens, fuzzy ratio otherwise
        context_snippet: str  — surrounding text where claim was found (or "")
        detail         : str  — human-readable explanation for the agent
    """
    claim   = claim.strip()
    context = context.strip()

    if not claim or not context:
        return {
            "found": False,
            "match_type": "not_found",
            "match_score": 0.0,
            "context_snippet": "",
            "detail": "Empty claim or context.",
        }

    claim_lower   = claim.lower()
    context_lower = context.lower()

    # ── Strategy 1: exact substring ───────────────────────────────────────────
    if claim_lower in context_lower:
        idx = context_lower.index(claim_lower)
        start = max(0, idx - 40)
        end   = min(len(context), idx + len(claim) + 40)
        snippet = "…" + context[start:end].strip() + "…"
        return {
            "found": True,
            "match_type": "exact",
            "match_score": 1.0,
            "context_snippet": snippet,
            "detail": f'Exact match found: "{claim}" appears in context.',
        }

    # ── Strategy 2: all tokens present ───────────────────────────────────────
    claim_tokens   = set(_tokenise(claim))
    context_tokens = set(_tokenise(context))
    if claim_tokens and claim_tokens.issubset(context_tokens):
        return {
            "found": True,
            "match_type": "all_tokens",
            "match_score": 1.0,
            "context_snippet": "",
            "detail": (
                f'All words of "{claim}" appear in context '
                f"(tokens: {', '.join(sorted(claim_tokens))})."
            ),
        }

    # ── Strategy 3: fuzzy match (multi-word claims only) ─────────────────────
    if len(claim.split()) > 1:
        # Slide a window of roughly claim-length words over context
        context_words = context.split()
        claim_words   = claim.split()
        window_size   = len(claim_words)
        best_ratio    = 0.0
        best_window   = ""
        for i in range(max(1, len(context_words) - window_size + 1)):
            window = " ".join(context_words[i: i + window_size])
            ratio  = difflib.SequenceMatcher(
                None, claim_lower, window.lower()
            ).ratio()
            if ratio > best_ratio:
                best_ratio  = ratio
                best_window = window
        if best_ratio >= 0.80:
            return {
                "found": True,
                "match_type": "fuzzy",
                "match_score": round(best_ratio, 3),
                "context_snippet": best_window,
                "detail": (
                    f'Fuzzy match (score={best_ratio:.2f}) for "{claim}". '
                    f'Closest context phrase: "{best_window}".'
                ),
            }

    # ── Not found ─────────────────────────────────────────────────────────────
    return {
        "found": False,
        "match_type": "not_found",
        "match_score": 0.0,
        "context_snippet": "",
        "detail": (
            f'"{claim}" was NOT found in the context by any matching strategy. '
            "This claim is likely unsupported or fabricated."
        ),
    }
