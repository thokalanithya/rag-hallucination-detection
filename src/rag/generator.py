"""
generator.py
------------
Generates answers from top-K retrieved chunks using GPT-4o (Azure OpenAI).

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

REFERENCES
----------
[N] "<title>," arXiv preprint arXiv:<paper_id>, sec. <chunk_index>.

---

Context passages:
{context}

Question: {question}"""


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
    # Group chunks by paper_id (preserving first-appearance order),
    # then build one numbered context entry per paper with merged text.
    from collections import OrderedDict
    papers: OrderedDict = OrderedDict()
    for chunk in top_k_chunks:
        pid = chunk["paper_id"]
        if pid not in papers:
            papers[pid] = {"paper_id": pid, "title": chunk["title"], "chunks": []}
        papers[pid]["chunks"].append(chunk)

    context_parts = []
    citations = {}

    for number, paper in enumerate(papers.values(), start=1):
        merged_text = "\n\n".join(c["text"].strip() for c in paper["chunks"])
        header = f"[{number}] Source: {paper['paper_id']} — {paper['title'][:60]}"
        context_parts.append(f"{header}\n{merged_text}")
        citations[number] = {
            "paper_id": paper["paper_id"],
            "title":    paper["title"],
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

    # Renumber citations sequentially from 1 using only what the LLM actually cited.
    import re as _re
    used = sorted({int(m) for m in _re.findall(r'\[(\d+)\]', answer)})
    if used != list(range(1, len(used) + 1)):
        remap = {old: new for new, old in enumerate(used, start=1)}
        # Replace in answer text (use placeholder to avoid double-replacing)
        def _sub(m):
            return f"[\u00ab{remap[int(m.group(1))]}\u00bb]"
        answer = _re.sub(r'\[(\d+)\]', _sub, answer)
        answer = answer.replace("[\u00ab", "[").replace("\u00bb]", "]")
        citations = {remap[old]: citations[old] for old in used if old in citations}

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


