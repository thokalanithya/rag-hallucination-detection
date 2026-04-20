"""
generator.py
------------
Generates answers from top-K retrieved chunks using GPT-4o (Azure OpenAI).

Two functions:
    generate_answer()               — plain answer, no citations
    generate_answer_with_citations()— answer with inline [1][2] citations
                                      + structured citation map

Pipeline position:
    retrieve() → rerank() → generate_answer_with_citations()

Usage
-----
    from rag.generator import generate_answer_with_citations

    answer, context, citations = generate_answer_with_citations(question, chunks)
    # citations = {1: {paper_id, title, chunk_index, tokens}, 2: {...}, ...}
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


WEB_CITATION_PROMPT = """You are a research assistant. Answer the question using ONLY the web search results provided below.

Rules:
- Cite every claim inline using the source number in brackets, e.g. [1] or [2][3].
- Do NOT include any information not present in the search results.
- If no result answers the question, say so clearly.

Respond in exactly this format:

ANSWER
------
<your answer here with inline citations like [1] or [2]>

REFERENCES
----------
For each citation used, list:
[N] <title>
    URL: <url>

---

Web search results:
{context}

Question: {question}"""


CITATION_PROMPT = """You are a research assistant. Answer the question using ONLY the numbered context passages below.

Rules:
- Cite every claim inline with the passage number in brackets, e.g. [1] or [2][3].
- Do NOT include any information not present in the passages.
- If no passage answers the question, write "I don't have enough information to answer this." in the ANSWER section and leave the other sections empty.

Respond in exactly this format:

ANSWER
------
<your answer here with inline citations like [1] or [2]>

SUPPORTING CONTEXT
------------------
For each citation used, quote the exact sentence(s) from the passage that support your answer:
[N] "<exact quote from passage N>"

"""


# def generate_answer(question: str, top_k_chunks: list[dict]) -> tuple[str, str]:
#     """
#     Generate an answer from the top-K retrieved chunks.

#     Parameters
#     ----------
#     question      : the user's question
#     top_k_chunks  : output of retrieve_top_k() — list of chunk dicts with 'text' field

#     Returns
#     -------
#     (answer, context)
#         answer  : GPT-4o's generated answer
#         context : the concatenated top-K chunk texts used as input
#     """
#     context  = "\n\n".join(c["text"] for c in top_k_chunks)
#     prompt   = ANSWER_PROMPT.format(context=context, question=question)
#     response = _client.chat.completions.create(
#         model=_model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=512,
#     )
#     answer = response.choices[0].message.content.strip()
#     return answer, context


def generate_answer_with_citations(
    question: str,
    top_k_chunks: list[dict],
) -> tuple[str, str, dict]:
    """
    Generate an answer with inline citations pointing to source chunks.

    Each chunk is numbered [1], [2], ... in the prompt. The LLM is instructed
    to cite inline. A citation map is returned so callers can resolve [1] →
    exact paper, chunk, and token range.

    Parameters
    ----------
    question     : the user's question
    top_k_chunks : re-ranked chunk dicts — must have keys:
                   text, paper_id, title, chunk_index, token_start, token_end

    Returns
    -------
    (answer, context, citations)
        answer    : full structured LLM response (ANSWER + SUPPORTING CONTEXT + REFERENCES)
        context   : the numbered context string sent to the LLM
        citations : dict mapping citation number (int) to chunk metadata:
                    {1: {paper_id, title, chunk_index, token_start, token_end, text}, ...}
    """
    # Build numbered context block and citation map in one pass
    context_parts = []
    citations = {}

    for number, chunk in enumerate(top_k_chunks, start=1):
        header = (
            f"[{number}] Source: {chunk['paper_id']} — "
            f"{chunk['title'][:60]} | "
            f"chunk {chunk['chunk_index']} | "
            f"tokens {chunk['token_start']}–{chunk['token_end']}"
        )
        context_parts.append(f"{header}\n{chunk['text'].strip()}")

        citations[number] = {
            "paper_id":    chunk["paper_id"],
            "title":       chunk["title"],
            "chunk_index": chunk["chunk_index"],
            "token_start": chunk["token_start"],
            "token_end":   chunk["token_end"],
            "text":        chunk["text"],
        }

    context = "\n\n".join(context_parts)
    prompt  = CITATION_PROMPT.format(context=context, question=question)

    response = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500,
    )
    answer = response.choices[0].message.content.strip()
    return answer, context, citations


_INSUFFICIENT_PHRASE = "i don't have enough information to answer this"


def is_insufficient_answer(answer: str) -> bool:
    """
    Return True if the LLM signalled it could not answer from the retrieved context.

    Checks for the exact phrase the CITATION_PROMPT instructs the model to use
    when no passage answers the question.
    """
    return _INSUFFICIENT_PHRASE in answer.lower()


def generate_answer_from_web(
    question: str,
    web_results: list[dict],
) -> tuple[str, dict]:
    """
    Generate a cited answer from Bing web search results.

    Parameters
    ----------
    question    : the user's question
    web_results : output of web_search.search_bing() —
                  list of {number, title, url, snippet}

    Returns
    -------
    (answer, web_citations)
        answer        : structured LLM response (ANSWER + REFERENCES)
        web_citations : dict mapping citation number (int) to
                        {title, url, snippet}
    """
    context_parts = []
    web_citations = {}

    for r in web_results:
        n = r["number"]
        context_parts.append(
            f"[{n}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"{r['snippet']}"
        )
        web_citations[n] = {
            "title":   r["title"],
            "url":     r["url"],
            "snippet": r["snippet"],
        }

    context = "\n\n".join(context_parts)
    prompt  = WEB_CITATION_PROMPT.format(context=context, question=question)

    response = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500,
    )
    answer = response.choices[0].message.content.strip()
    return answer, web_citations


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
