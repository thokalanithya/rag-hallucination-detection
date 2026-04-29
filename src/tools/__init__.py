"""
tools/
------
Deterministic utility tools used by the hallucination detection agent.
GPT-4o calls these to gather evidence; it does the reasoning itself.

Tools
-----
split_sentences       : split text into individual sentences (NLTK)
check_sentence_support: semantic similarity via sentence-transformers
extract_claims        : extract verifiable facts via regex
find_in_context       : direct lookup of a claim inside the context
keyword_overlap       : token-level F1 word overlap (answer vs context)
"""
from .sentence_splitter  import split_sentences
from .similarity_checker import check_sentence_support, check_nli_entailment
from .claim_extractor    import extract_claims
from .find_in_context    import find_in_context
from .keyword_overlap    import keyword_overlap

__all__ = [
    "split_sentences",
    "check_sentence_support",
    "check_nli_entailment",
    "extract_claims",
    "find_in_context",
    "keyword_overlap",
]
