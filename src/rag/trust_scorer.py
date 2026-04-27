"""
trust_scorer.py
---------------
Dynamic source trustworthiness scoring using two free signals:

  1. Tranco top-1M domain list  — measures how established / high-traffic a
     domain is, based on DNS query data aggregated across multiple sources.
     Downloaded once and cached locally; refreshed every 7 days.
     No API key required.

  2. OpenPageRank API (optional) — link-graph authority score (0–10) derived
     from Common Crawl. Adds up to 0.4 extra points on top of the Tranco
     score. Only used when OPENPAGE_RANK_API_KEY is set in the environment.
     Free tier: 100 requests/day at openpagerank.com.

Score range: 0.0 (unknown / suspicious) → 1.0 (globally authoritative)

  0.0 – 0.25  unknown domain, not in Tranco top-1M
  0.25 – 0.40  low-traffic but ranked
  0.40 – 0.55  mid-tier site (top 500K)
  0.55 – 0.75  established, high-traffic (top 100K)
  0.75 – 1.0   highly authoritative (top 10K, e.g. wikipedia, bbc, github)
"""

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import requests
import tldextract

from config import TRANCO_CACHE_PATH, TRANCO_CACHE_MAX_AGE_DAYS

logger = logging.getLogger("rag_app")

_TRANCO_API = "https://tranco-list.eu/api/lists/date/latest"
_OPR_URL    = "https://openpagerank.com/api/v1.0/getPageRank"

# Platforms where any subdomain can host user content — treat the full
# subdomain as the lookup key so random blogs don't inherit the platform score.
_HOSTED_PLATFORMS = {
    "blogspot.com", "wordpress.com", "medium.com", "tumblr.com",
    "substack.com", "github.io", "netlify.app", "vercel.app",
    "wixsite.com", "squarespace.com", "weebly.com", "sites.google.com",
}

_tranco_cache: dict[str, int] | None = None


# ── Domain helpers ──────────────────────────────────────────────────────────

def _lookup_domain(url: str) -> str:
    """
    Return the domain string to use for Tranco / OPR lookups.

    For normal sites: registered domain (e.g. bbc.co.uk, nih.gov).
    For user-content platforms (blogspot, wordpress …): full subdomain
    (e.g. some-random-blog.blogspot.com) so random blogs don't inherit the
    platform's high rank.
    """
    ext = tldextract.extract(url)
    registered = f"{ext.domain}.{ext.suffix}".lower() if ext.domain and ext.suffix else ""
    if registered in _HOSTED_PLATFORMS:
        # use full host so the specific blog/page is scored, not the platform
        subdomain = ext.subdomain.lower()
        return f"{subdomain}.{registered}" if subdomain else registered
    return registered or url  # fallback: raw url


# ── Tranco list management ──────────────────────────────────────────────────

def _fetch_tranco_download_url() -> str:
    """Ask Tranco API for the latest list's download URL."""
    resp = requests.get(_TRANCO_API, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    url  = data.get("download")
    if not url:
        raise ValueError(f"Tranco API returned no download URL: {data}")
    return url


def _parse_tranco_response(content: bytes) -> dict[str, int]:
    """Parse Tranco response bytes — handles both ZIP and plain CSV."""
    if content[:2] == b"PK":  # ZIP magic bytes
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            with zf.open(zf.namelist()[0]) as csv_file:
                lines = csv_file.read().decode().splitlines()
    else:
        lines = content.decode().splitlines()

    tranco: dict[str, int] = {}
    for line in lines:
        parts = line.strip().split(",", 1)
        if len(parts) == 2:
            try:
                tranco[parts[1].lower()] = int(parts[0])
            except ValueError:
                pass
    return tranco


def _load_tranco() -> dict[str, int]:
    """
    Return the Tranco domain→rank dict.
    Loads from local cache when fresh; downloads and caches otherwise.
    Falls back to an empty dict if the download fails.
    """
    global _tranco_cache
    if _tranco_cache is not None:
        return _tranco_cache

    cache_path = Path(TRANCO_CACHE_PATH)
    max_age    = TRANCO_CACHE_MAX_AGE_DAYS * 86_400

    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < max_age:
        with open(cache_path) as f:
            _tranco_cache = json.load(f)
        print(f"[trust_scorer] Tranco list loaded from cache ({len(_tranco_cache):,} domains)")
        return _tranco_cache

    print("[trust_scorer] Downloading Tranco top-1M list (this happens once)...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_url = _fetch_tranco_download_url()
        print(f"[trust_scorer] Fetching: {download_url}")
        resp = requests.get(download_url, timeout=120)
        resp.raise_for_status()
        _tranco_cache = _parse_tranco_response(resp.content)
        with open(cache_path, "w") as f:
            json.dump(_tranco_cache, f)
        print(f"[trust_scorer] Tranco list cached ({len(_tranco_cache):,} domains) → {cache_path}")
    except Exception as exc:
        print(f"[trust_scorer] Tranco download failed: {exc}. Scoring without it.")
        _tranco_cache = {}

    return _tranco_cache


# ── Individual scorers ──────────────────────────────────────────────────────

def _tranco_score(domain: str) -> float:
    """Return 0.0–0.6 based on Tranco global rank."""
    rank = _load_tranco().get(domain)
    if rank is None:
        return 0.0
    if rank <= 1_000:
        return 0.60
    if rank <= 10_000:
        return 0.50
    if rank <= 100_000:
        return 0.40
    if rank <= 500_000:
        return 0.30
    return 0.15   # ranked but below 500K


def _opr_score(domain: str) -> float:
    """
    Query OpenPageRank API and return 0.0–0.4 contribution.
    Returns 0.0 silently if the API key is absent or the call fails.
    """
    api_key = os.environ.get("OPENPAGE_RANK_API_KEY", "")
    if not api_key:
        return 0.0
    try:
        resp = requests.get(
            _OPR_URL,
            params={"domains[]": domain},
            headers={"API-OPR": api_key},
            timeout=5,
        )
        if resp.status_code == 200:
            results = resp.json().get("response", [])
            if results:
                opr = float(results[0].get("page_rank_decimal", 0))
                return round((opr / 10.0) * 0.4, 4)
    except Exception:
        pass
    return 0.0


# ── Public API ──────────────────────────────────────────────────────────────

def score_source(url: str) -> float:
    """
    Return a trust score 0.0–1.0 for a web result URL.

    Combines:
      - Tranco rank   → up to 0.6  (always available, no key needed)
      - OpenPageRank  → up to 0.4  (only when OPENPAGE_RANK_API_KEY is set)

    When only Tranco is available, the max possible score is 0.6.
    """
    domain  = _lookup_domain(url)
    t_score = _tranco_score(domain)
    o_score = _opr_score(domain)
    total   = min(1.0, round(t_score + o_score, 4))

    tranco_rank = _load_tranco().get(domain)
    rank_str    = f"rank #{tranco_rank:,}" if tranco_rank else "not in Tranco top-1M"
    opr_note    = f"opr={o_score:.2f}" if o_score > 0 else "opr=n/a (no API key)"
    logger.debug(
        f"    SCORE  {domain:45s}  tranco={t_score:.2f} ({rank_str})  "
        f"{opr_note}  → total={total:.2f}"
    )

    return total
