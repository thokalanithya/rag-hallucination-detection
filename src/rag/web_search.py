"""
web_search.py
-------------
DuckDuckGo web search fallback for the RAG pipeline.

Called when the local vector store does not contain sufficient context
(i.e. GPT returned "I don't have enough information to answer this.").

Results are filtered to WEB_SEARCH_TRUSTED_DOMAINS defined in config.py.
No API key or account required.

Usage
-----
    from rag.web_search import search_web

    results = search_web("What is contrastive learning?", max_results=5)
    # [{"number": 1, "title": "...", "url": "...", "snippet": "..."}, ...]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddgs import DDGS
from config import WEB_SEARCH_TRUSTED_DOMAINS, WEB_SEARCH_FETCH_MULTIPLIER


def _is_trusted(url: str, trusted_domains: list[str]) -> bool:
    """Return True if the URL belongs to one of the trusted domains."""
    return any(domain in url for domain in trusted_domains)


def search_web(
    query: str,
    max_results: int = 5,
    trusted_domains: list[str] = WEB_SEARCH_TRUSTED_DOMAINS,
) -> list[dict]:
    """
    Query DuckDuckGo, filter to trusted domains, and return structured results.

    Fetches max_results * WEB_SEARCH_FETCH_MULTIPLIER candidates from
    DuckDuckGo so there are enough to filter down to max_results trusted hits.

    Parameters
    ----------
    query          : the search query (typically the user's question)
    max_results    : maximum number of trusted results to return
    trusted_domains: list of allowed domain strings; defaults to
                     WEB_SEARCH_TRUSTED_DOMAINS from config.py

    Returns
    -------
    List of result dicts (up to max_results) with keys:
        number  : 1-indexed citation number
        title   : page title
        url     : source URL (always from a trusted domain)
        snippet : short excerpt from the page
    """
    fetch_count = max_results * WEB_SEARCH_FETCH_MULTIPLIER
    trusted = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=fetch_count):
            url = r.get("href", "")
            if not _is_trusted(url, trusted_domains):
                continue
            trusted.append({
                "number":  len(trusted) + 1,
                "title":   r.get("title", ""),
                "url":     url,
                "snippet": r.get("body", ""),
            })
            if len(trusted) == max_results:
                break

    return trusted
