"""
embedder.py
-----------
Embeds document chunks and a query, then retrieves the top-K most
relevant chunks using cosine similarity.

Uses:
    EMBEDDING_MODEL  = "all-MiniLM-L6-v2"  (from config.py)
    RETRIEVAL_TOP_K  = 3                    (from config.py)

Usage
-----
    from chunker import chunk_document
    from embedder import embed_chunks, retrieve_top_k

    chunks   = chunk_document(document_text)
    embedded = embed_chunks(chunks)
    top_k    = retrieve_top_k(question, embedded)
    context  = "\n\n".join(c["text"] for c in top_k)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, RETRIEVAL_TOP_K


# Load model once at import time (avoid reloading on every call)
_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' field to each chunk dict.

    Parameters
    ----------
    chunks : output of chunk_document() — list of dicts with a 'text' field

    Returns
    -------
    Same list with an added 'embedding' key (numpy array) per chunk.
    """
    texts = [c["text"] for c in chunks]
    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


def retrieve_top_k(question: str, embedded_chunks: list[dict], top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """
    Embed the question and return the top-K most similar chunks.

    Similarity is measured by cosine similarity between the question
    embedding and each chunk embedding.

    Parameters
    ----------
    question        : the user's question string
    embedded_chunks : output of embed_chunks()
    top_k           : number of chunks to return (default: RETRIEVAL_TOP_K = 3)

    Returns
    -------
    List of top-K chunk dicts (sorted by similarity, highest first),
    each with an added 'similarity' field.
    """
    question_embedding = _model.encode(question, convert_to_numpy=True)

    # Compute cosine similarity between question and each chunk
    for chunk in embedded_chunks:
        chunk["similarity"] = _cosine_similarity(question_embedding, chunk["embedding"])

    # Sort by similarity descending, return top-K
    ranked = sorted(embedded_chunks, key=lambda c: c["similarity"], reverse=True)
    return ranked[:top_k]


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot   = np.dot(vec_a, vec_b)
    norms = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norms) if norms > 0 else 0.0


if __name__ == "__main__":
    from rag.chunker import chunk_document

    document = """
    Photosynthesis is the process by which green plants, algae, and some bacteria
    convert light energy into chemical energy stored in glucose. This process occurs
    primarily in the chloroplasts, which contain a green pigment called chlorophyll.
    Chlorophyll absorbs sunlight — mostly in the red and blue wavelengths — and uses
    that energy to drive a series of chemical reactions. The overall equation for
    photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This means plants
    take in carbon dioxide and water, and with the help of sunlight, produce glucose
    and release oxygen as a byproduct. The glucose produced is used by the plant as
    an energy source for growth, reproduction, and other metabolic processes. Without
    photosynthesis, most life on Earth would not be possible, as it forms the base of
    nearly all food chains and is responsible for the oxygen in our atmosphere.
    Photosynthesis occurs in two main stages: the light-dependent reactions and the
    light-independent reactions (Calvin cycle). In the light-dependent reactions,
    sunlight is used to split water molecules and generate ATP and NADPH. In the Calvin
    cycle, these energy carriers are used to fix carbon dioxide into organic molecules.
    """ * 3

    question = "How do plants produce glucose?"

    print(f"Question: {question}\n")

    # Step 1: chunk
    chunks = chunk_document(document, chunk_size=50, overlap=10)
    print(f"Total chunks: {len(chunks)}")

    # Step 2: embed
    embedded = embed_chunks(chunks)
    print(f"Embeddings shape: {embedded[0]['embedding'].shape}\n")

    # Step 3: retrieve top-K
    top_k = retrieve_top_k(question, embedded, top_k=3)

    print("Top-3 most relevant chunks:")
    for i, c in enumerate(top_k):
        print(f"\n  Rank {i+1} | Chunk {c['index']} | similarity={c['similarity']:.4f}")
        print(f"  {c['text'][:120]}...")
