from __future__ import annotations

"""
vector_store.py
---------------
ChromaDB-backed vector store for the Qasper RAG pipeline.

Responsibilities:
    1. build_index()  — chunk each paper, embed with all-MiniLM-L6-v2,
                        persist to ChromaDB with full citation metadata
    2. retrieve()     — embed a query, run similarity search, return ranked
                        chunks with source citations

Metadata stored per chunk (enables citation in generated answers):
    paper_id     : Qasper paper ID  (e.g. "1909.00694")
    title        : paper title
    chunk_index  : position of this chunk within the paper
    token_start  : first token index in the original document
    token_end    : last token index in the original document
    token_count  : number of tokens in this chunk

Usage
-----
    from rag.vector_store import build_index, retrieve

    # One-time indexing
    build_index(documents)

    # Query at any time
    results = retrieve("How does the model handle unanswerable questions?", top_k=5)
    for r in results:
        print(r["text"], r["paper_id"], r["chunk_index"])
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
)
from rag.chunker import chunk_document


# ── Module-level singletons (loaded once) ─────────────────────────────────────

_embedding_model = None
_chroma_client   = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[vector_store] Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


# ── Public API ────────────────────────────────────────────────────────────────

def build_index(
    documents: list[dict],
    collection_name: str = CHROMA_COLLECTION,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    reset: bool = False,
) -> chromadb.Collection:
    """
    Chunk, embed, and store all documents in ChromaDB.

    Parameters
    ----------
    documents       : list of dicts from qasper_loader.load_qasper()
                      each must have keys: paper_id, title, full_text
    collection_name : ChromaDB collection to write into
    chunk_size      : tokens per chunk
    overlap         : overlapping tokens between chunks
    reset           : if True, drop and recreate the collection first

    Returns
    -------
    The ChromaDB collection object.
    """
    client = _get_client()
    model  = _get_embedding_model()

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[vector_store] Dropped existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(
            f"[vector_store] Collection '{collection_name}' already has "
            f"{existing_count} chunks — skipping indexing. "
            f"Pass reset=True to rebuild."
        )
        return collection

    print(f"\n[vector_store] Indexing {len(documents)} documents into '{collection_name}' …")

    total_chunks = 0
    for doc in tqdm(documents, desc="Indexing papers"):
        paper_id  = doc["paper_id"]
        title     = doc["title"]
        full_text = doc["full_text"]

        chunks = chunk_document(full_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue

        texts      = [c["text"] for c in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

        ids = [f"{paper_id}__chunk_{c['index']}" for c in chunks]

        metadatas = [
            {
                "paper_id":    paper_id,
                "title":       title,
                "chunk_index": c["index"],
                "token_start": c["token_start"],
                "token_end":   c["token_end"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"[vector_store] Done — {total_chunks} chunks indexed across {len(documents)} papers.")
    return collection


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    collection_name: str = CHROMA_COLLECTION,
    paper_id_filter: str | None = None,
) -> list[dict]:
    """
    Embed query and return the top-k most similar chunks from ChromaDB.

    Parameters
    ----------
    query           : the user's question
    top_k           : number of chunks to return
    collection_name : ChromaDB collection to search
    paper_id_filter : if set, restrict search to one specific paper

    Returns
    -------
    List of dicts (sorted by relevance, best first), each with:
        text        : chunk text
        paper_id    : source paper ID
        title       : paper title
        chunk_index : chunk position within the paper
        token_start : starting token offset
        token_end   : ending token offset
        distance    : cosine distance (lower = more similar)
        similarity  : 1 - distance  (higher = more similar)
    """
    client = _get_client()
    model  = _get_embedding_model()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    where = {"paper_id": paper_id_filter} if paper_id_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":        text,
            "paper_id":    meta["paper_id"],
            "title":       meta["title"],
            "chunk_index": meta["chunk_index"],
            "token_start": meta["token_start"],
            "token_end":   meta["token_end"],
            "distance":    round(dist, 4),
            "similarity":  round(1 - dist, 4),
        })

    return chunks


def collection_stats(collection_name: str = CHROMA_COLLECTION) -> dict:
    """Return basic stats about the indexed collection."""
    client     = _get_client()
    collection = client.get_or_create_collection(collection_name)
    count      = collection.count()
    return {"collection": collection_name, "total_chunks": count}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.qasper_loader import load_qasper

    # Load 5 papers and build index
    documents, _ = load_qasper(max_papers=5)
    build_index(documents, reset=True)

    print(f"\nCollection stats: {collection_stats()}")

    # Run sample retrievals
    test_queries = [
        "How does the model handle unanswerable questions?",
        "What dataset was used for training?",
        "What is the evaluation metric?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"\n  Rank {i+1} | similarity={r['similarity']:.4f}")
            print(f"  Paper : {r['paper_id']} — {r['title'][:60]}")
            print(f"  Chunk : {r['chunk_index']} (tokens {r['token_start']}–{r['token_end']})")
            print(f"  Text  : {r['text'][:120].strip()}...")
