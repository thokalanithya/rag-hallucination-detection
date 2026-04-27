"""
web_search.py
-------------
DuckDuckGo web search fallback for the RAG pipeline.

Called when the local vector store does not contain sufficient context
(i.e. GPT returned "I don't have enough information to answer this.").

Results are scored for trustworthiness using:
  - Tranco top-1M domain list  (no API key required)
  - OpenPageRank API           (optional — set OPENPAGE_RANK_API_KEY in .env)

Only results scoring at or above WEB_SEARCH_MIN_TRUST_SCORE are returned,
ranked by trust score descending. This replaces the old static whitelist.

Usage
-----
    from rag.web_search import search_web

    results = search_web("What is contrastive learning?", max_results=5)
    # [{"number": 1, "title": "...", "url": "...", "snippet": "...", "trust_score": 0.5}, ...]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from ddgs import DDGS
from config import WEB_SEARCH_FETCH_MULTIPLIER, WEB_SEARCH_MIN_TRUST_SCORE
from .trust_scorer import score_source

logger = logging.getLogger("rag_app")


def search_web(
    query: str,
    max_results: int = 5,
    min_trust_score: float = WEB_SEARCH_MIN_TRUST_SCORE,
) -> list[dict]:
    """
    Query DuckDuckGo, score each result for trustworthiness, and return the
    top results above the minimum trust threshold.

    Fetches max_results * WEB_SEARCH_FETCH_MULTIPLIER candidates from
    DuckDuckGo, scores all of them, filters by min_trust_score, sorts by
    score descending, and returns up to max_results.

    Parameters
    ----------
    query          : the search query (typically the user's question)
    max_results    : maximum number of results to return
    min_trust_score: minimum trust score (0.0–1.0) to include a result;
                     defaults to WEB_SEARCH_MIN_TRUST_SCORE from config

    Returns
    -------
    List of result dicts (up to max_results), sorted by trust score desc:
        number      : 1-indexed citation number
        title       : page title
        url         : source URL
        snippet     : short excerpt from the page
        trust_score : float 0.0–1.0 (higher = more trustworthy)
    """
    fetch_count = max_results * WEB_SEARCH_FETCH_MULTIPLIER
    candidates  = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=fetch_count):
            url = r.get("href", "")
            if not url:
                continue
            candidates.append({
                "title":   r.get("title", ""),
                "url":     url,
                "snippet": r.get("body", ""),
            })

    logger.info(f"  DDG QUERY  : {query!r}")
    logger.info(f"  DDG FETCHED: {len(candidates)} candidates (fetch_count={fetch_count})")
    for c in candidates:
        logger.debug(f"    RAW  {c['url']}")

    # Score all candidates
    logger.debug(f"  TRUST SCORING ({len(candidates)} candidates, threshold≥{min_trust_score}):")
    scored = []
    dropped = []
    for c in candidates:
        score = score_source(c["url"])
        if score >= min_trust_score:
            scored.append({**c, "trust_score": score})
        else:
            dropped.append((score, c["url"]))

    if dropped:
        logger.debug(f"  DROPPED {len(dropped)} below threshold {min_trust_score}:")
        for score, url in sorted(dropped):
            logger.debug(f"    DROP  score={score:.2f}  {url}")

    # Sort by trust score descending, take top max_results, renumber
    scored.sort(key=lambda x: x["trust_score"], reverse=True)
    results = []
    for i, r in enumerate(scored[:max_results], start=1):
        results.append({
            "number":      i,
            "title":       r["title"],
            "url":         r["url"],
            "snippet":     r["snippet"],
            "trust_score": r["trust_score"],
        })

    logger.info(f"  DDG KEPT   : {len(results)}/{len(candidates)} results after trust filter")
    for r in results:
        logger.info(
            f"    [{r['number']}] trust={r['trust_score']:.2f}  "
            f"{r['title'][:55]}  |  {r['url']}"
        )

    return results
