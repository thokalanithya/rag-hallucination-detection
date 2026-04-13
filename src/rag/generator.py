"""
generator.py
------------
Generates an answer from top-K retrieved chunks using GPT-4o (Azure OpenAI).

Pipeline position:
    chunk_document() → embed_chunks() → retrieve_top_k() → generate_answer()

Usage
-----
    from chunker import chunk_document
    from embedder import embed_chunks, retrieve_top_k
    from generator import generate_answer

    chunks   = chunk_document(document_text)
    embedded = embed_chunks(chunks)
    top_k    = retrieve_top_k(question, embedded)
    answer, context = generate_answer(question, top_k)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

import os
from openai import AzureOpenAI

_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context provided below.
Do not add any information that is not present in the context.
If the context does not contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, top_k_chunks: list[dict]) -> tuple[str, str]:
    """
    Generate an answer from the top-K retrieved chunks.

    Parameters
    ----------
    question      : the user's question
    top_k_chunks  : output of retrieve_top_k() — list of chunk dicts with 'text' field

    Returns
    -------
    (answer, context)
        answer  : GPT-4o's generated answer
        context : the concatenated top-K chunk texts used as input
    """
    context  = "\n\n".join(c["text"] for c in top_k_chunks)
    prompt   = ANSWER_PROMPT.format(context=context, question=question)
    response = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    answer = response.choices[0].message.content.strip()
    return answer, context


if __name__ == "__main__":
    from rag.chunker import chunk_document
    from rag.embedder import embed_chunks, retrieve_top_k

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
    """

    question = "How do plants produce glucose?"

    print(f"Question: {question}\n")

    chunks   = chunk_document(document, chunk_size=50, overlap=10)
    embedded = embed_chunks(chunks)
    top_k    = retrieve_top_k(question, embedded, top_k=3)

    print("Retrieved context chunks:")
    for i, c in enumerate(top_k):
        print(f"  [{i+1}] (similarity={c['similarity']:.4f}) {c['text'][:100]}...")

    print("\nGenerating answer...\n")
    answer, context = generate_answer(question, top_k)

    print(f"Answer:\n{answer}")
    print(f"\nContext used:\n{context}")
