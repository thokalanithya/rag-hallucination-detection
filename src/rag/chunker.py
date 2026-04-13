"""
chunker.py
----------
Splits a document into overlapping fixed-size word chunks.

Uses CHUNK_SIZE and CHUNK_OVERLAP from config.py.

Usage
-----
    from chunker import chunk_document

    chunks = chunk_document("Your long document text here...")
    for chunk in chunks:
        print(chunk["index"], chunk["text"])
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(document: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a document into overlapping word-based chunks.

    Parameters
    ----------
    document   : the full document text
    chunk_size : number of words per chunk  (default: CHUNK_SIZE = 300)
    overlap    : number of words shared between consecutive chunks (default: CHUNK_OVERLAP = 50)

    Returns
    -------
    List of dicts, each with:
        index      : int  — chunk number (0-based)
        text       : str  — chunk text
        word_start : int  — starting word index in the original document
        word_end   : int  — ending word index in the original document
        word_count : int  — number of words in this chunk
    """
    words = document.split()
    total_words = len(words)

    chunks = []
    start = 0
    index = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]

        chunks.append({
            "index":      index,
            "text":       " ".join(chunk_words),
            "word_start": start,
            "word_end":   end,
            "word_count": len(chunk_words),
        })

        index += 1

        # If we've reached the end, stop
        if end == total_words:
            break

        # Move forward by (chunk_size - overlap) so chunks share 'overlap' words
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample = """
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
    """ * 3  # repeat to get enough words to see multiple chunks

    chunks = chunk_document(sample, chunk_size=50, overlap=10)

    print(f"Document words : {len(sample.split())}")
    print(f"Chunks created : {len(chunks)}\n")
    for c in chunks:
        print(f"  Chunk {c['index']} | words {c['word_start']}–{c['word_end']} ({c['word_count']} words)")
        print(f"    {c['text'][:80]}...")
        print()
