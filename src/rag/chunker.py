"""
chunker.py
----------
Splits a document into overlapping fixed-size token chunks.

Uses CHUNK_SIZE, CHUNK_OVERLAP, and CHUNK_ENCODING from config.py.

Switching from word-based to token-based splitting ensures chunk sizes are
consistent with the embedding model's actual context window, and aligns with
the 500–800 token target for the Qasper RAG pipeline.

Usage
-----
    from rag.chunker import chunk_document

    chunks = chunk_document("Your long document text here...")
    for chunk in chunks:
        print(chunk["index"], chunk["token_count"], chunk["text"][:80])
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_ENCODING


def chunk_document(
    document: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    encoding_name: str = CHUNK_ENCODING,
) -> list[dict]:
    """
    Split a document into overlapping token-based chunks.

    Parameters
    ----------
    document      : the full document text
    chunk_size    : number of tokens per chunk  (default: CHUNK_SIZE = 600)
    overlap       : number of tokens shared between consecutive chunks
                    (default: CHUNK_OVERLAP = 100)
    encoding_name : tiktoken encoding to use   (default: CHUNK_ENCODING = "cl100k_base")

    Returns
    -------
    List of dicts, each with:
        index        : int  — chunk number (0-based)
        text         : str  — decoded chunk text
        token_start  : int  — starting token index in the original document
        token_end    : int  — ending token index in the original document
        token_count  : int  — number of tokens in this chunk
    """
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(document)
    total_tokens = len(token_ids)

    chunks = []
    start = 0
    index = 0
    step = chunk_size - overlap  # how far to advance after each chunk

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_token_ids = token_ids[start:end]

        chunks.append({
            "index":       index,
            "text":        enc.decode(chunk_token_ids),
            "token_start": start,
            "token_end":   end,
            "token_count": len(chunk_token_ids),
        })

        index += 1

        if end == total_tokens:
            break

        start += step

    return chunks


if __name__ == "__main__":
    sample = (
        "Attention mechanisms have become an integral part of compelling sequence "
        "modeling and transduction models in various tasks, allowing modeling of "
        "dependencies without regard to their distance in the input or output "
        "sequences. In all but a few cases, however, such attention mechanisms are "
        "used in conjunction with a recurrent network. The Transformer model "
        "architecture eschews recurrence and instead relies entirely on an attention "
        "mechanism to draw global dependencies between input and output. "
    ) * 30  # ~1 600 tokens — enough to see multiple chunks

    chunks = chunk_document(sample, chunk_size=600, overlap=100)

    print(f"Document tokens : {len(tiktoken.get_encoding(CHUNK_ENCODING).encode(sample))}")
    print(f"Chunks created  : {len(chunks)}\n")
    for c in chunks:
        print(
            f"  Chunk {c['index']:02d} | tokens {c['token_start']}–{c['token_end']} "
            f"({c['token_count']} tokens)"
        )
        print(f"    {c['text'][:90].strip()}...")
        print()
