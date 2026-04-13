"""
reranker.py
-----------
Cross-encoder re-ranking for the two-stage RAG retrieval pipeline.

Stage 1 (ChromaDB)  : fast bi-encoder vector search → top-20 candidates (broad recall)
Stage 2 (this file) : slow cross-encoder joint scoring → top-5 final  (precision)

The cross-encoder scores each (query, chunk) pair together — unlike the
bi-encoder which encodes them independently — capturing fine-grained
relevance signals that cosine similarity misses.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking (real search queries)
    - Runs fully offline, no API key needed
    - Already available via sentence-transformers

Usage
-----
    from rag.reranker import rerank

    candidates = retrieve(query, top_k=20)   # Stage 1: ChromaDB
    final      = rerank(query, candidates, top_k=5)  # Stage 2: cross-encoder
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RETRIEVAL_TOP_K


# ── Singleton — load model once ───────────────────────────────────────────────

_cross_encoder = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[reranker] Loading cross-encoder: {RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


# ── Public API ────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Re-score and re-order candidate chunks using a cross-encoder.

    Parameters
    ----------
    query      : the user's question
    candidates : list of chunk dicts from vector_store.retrieve()
                 each must have a "text" field
    top_k      : number of chunks to return after re-ranking

    Returns
    -------
    Top-k chunks re-ordered by cross-encoder score (highest first).
    Each chunk dict gets two new fields added:
        rerank_score      : float — raw cross-encoder logit (higher = more relevant)
        rerank_position   : int   — rank position after re-ranking (0-based)
    Original fields (paper_id, title, chunk_index, similarity, etc.) are preserved.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()

    # Build (query, chunk_text) pairs for joint scoring
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs — returns a numpy array of logits
    scores = model.predict(pairs)

    # Attach score to each candidate
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    # Sort by cross-encoder score descending
    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    # Add rank position for transparency
    for position, chunk in enumerate(reranked):
        chunk["rerank_position"] = position

    return reranked[:top_k]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.qasper_loader import load_qasper
    from src.rag.vector_store import build_index, retrieve

    # Build index on 5 papers
    documents, qa_pairs = load_qasper(max_papers=5)
    build_index(documents)

    # Pick a real question
    qa = next(q for q in qa_pairs if q["answerable"])
    query = qa["question"]

    print(f"Query: {query}\n")

    # Stage 1 — ChromaDB: broad retrieval
    candidates = retrieve(query, top_k=20)
    print(f"Stage 1 — ChromaDB returned {len(candidates)} candidates:")
    for i, c in enumerate(candidates[:5]):
        print(
            f"  [{i+1}] similarity={c['similarity']:.4f} | "
            f"{c['paper_id']} chunk_{c['chunk_index']} | "
            f"{c['text'][:70].strip()}..."
        )

    print()

    # Stage 2 — Re-rank
    final = rerank(query, candidates, top_k=5)
    print(f"Stage 2 — Re-ranked top-5:")
    for i, c in enumerate(final):
        print(
            f"  [{i+1}] rerank_score={c['rerank_score']:.4f} | "
            f"was_rank={candidates.index(c) + 1 if c in candidates else '?'} | "
            f"{c['paper_id']} chunk_{c['chunk_index']} | "
            f"{c['text'][:70].strip()}..."
        )

    # Show ranking shifts
    print("\nRanking shifts (ChromaDB position → Re-rank position):")
    for chunk in final:
        orig = next(
            (i + 1 for i, c in enumerate(candidates) if c["chunk_index"] == chunk["chunk_index"]
             and c["paper_id"] == chunk["paper_id"]),
            "?"
        )
        print(
            f"  {c['paper_id']} chunk_{chunk['chunk_index']} : "
            f"#{orig} → #{chunk['rerank_position'] + 1}  "
            f"(sim={chunk['similarity']:.4f} → rerank={chunk['rerank_score']:.4f})"
        )
