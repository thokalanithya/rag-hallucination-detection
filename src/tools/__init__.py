"""
tools/
------
Deterministic utility tools used by the hallucination detection agent.

Tools
-----
split_sentences       : split text into individual sentences (NLTK)
check_sentence_support: semantic similarity via sentence-transformers
extract_claims        : extract verifiable facts via regex
find_in_context       : direct lookup of a claim inside the context
"""
from .sentence_splitter  import split_sentences
from .similarity_checker import check_sentence_support
from .claim_extractor    import extract_claims
from .find_in_context    import find_in_context

__all__ = [
    "split_sentences",
    "check_sentence_support",
    "extract_claims",
    "find_in_context",
]
