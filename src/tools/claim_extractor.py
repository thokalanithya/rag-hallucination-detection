"""
claim_extractor.py
-------------------
Extract verifiable factual claims from a sentence using regex patterns.

The agent calls this on low-support sentences to get a precise list of
specific claims (numbers, names, dates, citations, percentages) that can
be compared against the retrieved context.

This is deterministic — no LLM involved — so the agent can trust the output
as a direct reflection of what the sentence actually says.
"""

import re
from typing import List, Dict


# ── Regex patterns for different claim types ──────────────────────────────────

_PATTERNS: List[Dict] = [
    {
        "type": "citation",
        "pattern": r"[A-Z][a-z]+ et al\.?,?\s*\(?\d{4}\)?",
        "description": "Academic citation (Author et al., Year)",
    },
    {
        "type": "statistic",
        "pattern": r"\b\d+\.?\d*\s*%",
        "description": "Percentage or statistic",
    },
    {
        "type": "number",
        "pattern": r"\b(?<!\w)\d+(?:[,\.]\d+)*(?:\s*(?:million|billion|thousand|km|mg|kg|ml|cm|m|ft|lb|Hz|USD|EUR))?\b",
        "description": "Numeric value (possibly with unit)",
    },
    {
        "type": "year",
        "pattern": r"\b(1[0-9]{3}|20[0-9]{2})\b",
        "description": "Four-digit year",
    },
    {
        "type": "proper_noun",
        "pattern": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        "description": "Proper noun / named entity (multi-word)",
    },
    {
        "type": "p_value",
        "pattern": r"p\s*[<>=≤≥]\s*0\.\d+",
        "description": "Statistical p-value",
    },
    {
        "type": "comparison",
        "pattern": r"\b(?:more|less|greater|fewer|higher|lower|larger|smaller)\s+than\b",
        "description": "Comparative claim",
    },
]


def extract_claims(sentence: str) -> Dict:
    """
    Extract verifiable factual claims from *sentence*.

    Parameters
    ----------
    sentence : A single sentence from the generated answer.

    Returns
    -------
    dict with keys:
        sentence      : original sentence
        claims        : list of dicts {type, value, description}
        summary       : human-readable summary for the agent
        has_claims    : bool — True if any verifiable claims were found
    """
    sentence = sentence.strip()
    found: List[Dict] = []
    seen: set = set()

    for pat in _PATTERNS:
        matches = re.findall(pat["pattern"], sentence)
        for match in matches:
            value = match.strip() if isinstance(match, str) else " ".join(match).strip()
            if value and value not in seen:
                seen.add(value)
                found.append({
                    "type": pat["type"],
                    "value": value,
                    "description": pat["description"],
                })

    if found:
        claim_list = "; ".join(f'"{c["value"]}" ({c["type"]})' for c in found)
        summary = (
            f"Found {len(found)} verifiable claim(s): {claim_list}. "
            "Check each against the retrieved context."
        )
    else:
        summary = (
            "No specific verifiable claims detected (no numbers, names, dates, or citations). "
            "The sentence may still be an overgeneralization or paraphrase error."
        )

    return {
        "sentence":   sentence,
        "claims":     found,
        "summary":    summary,
        "has_claims": bool(found),
    }
