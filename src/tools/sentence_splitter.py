"""
sentence_splitter.py
---------------------
Split a block of text into individual sentences using NLTK.

The agent calls this first on the generated answer so it can
check each sentence independently for hallucination signals.
"""

import re
from typing import List


def _download_punkt_if_needed() -> None:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def split_sentences(text: str) -> List[str]:
    """
    Split *text* into a list of individual sentences.

    Uses NLTK's sentence tokenizer with a regex fallback if NLTK
    is unavailable.

    Parameters
    ----------
    text : The answer or passage to split.

    Returns
    -------
    List of non-empty sentence strings, preserving original text.
    """
    text = text.strip()
    if not text:
        return []

    try:
        _download_punkt_if_needed()
        import nltk
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Regex fallback: split on ". ", "! ", "? "
        sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if s.strip()]
