"""
pipeline.py
-----------
Full RAG pipeline with ChromaDB retrieval and hallucination detection.

Flow:
    question
        → vector_store.retrieve()   query ChromaDB for top-k relevant chunks
        → generate_answer()         GPT-4o answers using only retrieved context
        → run_agent()               agentic detector scores and classifies the answer

Documents must be indexed into ChromaDB via vector_store.build_index()
before running this pipeline.

Usage
-----
    python -m rag.pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

import re

from .vector_store import build_index, retrieve
from .reranker import rerank
from .generator import generate_answer_with_citations, is_insufficient_answer, generate_answer_from_web
from .web_search import search_web
from agent.hallucination_agent import run_agent as run_hallucination_agent
from config import RETRIEVAL_CANDIDATES, RETRIEVAL_TOP_K, CHROMA_COLLECTION, WEB_SEARCH_MAX_RESULTS


def _extract_answer_text(full_response: str) -> str:
    """
    Pull just the ANSWER section from the structured LLM response.

    The response format is:
        ANSWER
        ------
        <answer text with inline citations>

        SUPPORTING CONTEXT
        ...

    Returns the answer text only — citations stripped — so the hallucination
    agent compares clean prose against the retrieved context.
    """
    # Match everything between the ANSWER header and the next section header
    match = re.search(
        r"ANSWER\s*[-─]+\s*(.*?)(?=\n\s*(?:SUPPORTING CONTEXT|REFERENCES|$))",
        full_response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        # Also strip inline citation markers like [1][2]
        text = match.group(1).strip()
        text = re.sub(r'\[\d+\]', '', text).strip()
        return text
    # Fallback: return the full response if structure not found
    return full_response


def run_rag_pipeline(
    question: str,
    candidates: int = RETRIEVAL_CANDIDATES,
    top_k: int = RETRIEVAL_TOP_K,
    collection_name: str = CHROMA_COLLECTION,
    paper_id_filter: str | None = None,
    status_callback=None,
) -> dict:
    """
    Run the full RAG pipeline for a question against the ChromaDB index.

    Two-stage retrieval:
        Stage 1 — ChromaDB vector search : broad recall  (top candidates)
        Stage 2 — Cross-encoder rerank   : precision     (top_k final)

    Parameters
    ----------
    question        : user's question
    candidates      : chunks to pull from ChromaDB in stage 1 (broad)
    top_k           : chunks to keep after re-ranking in stage 2 (precise)
    collection_name : ChromaDB collection to search
    paper_id_filter : restrict retrieval to a single paper (optional)

    Returns
    -------
    dict with keys:
        question             : the input question
        context              : numbered context string sent to the LLM
        answer               : GPT-4o answer with inline [1][2] citations
        citations            : {1: {paper_id, title}, ...}  (one entry per unique paper)
        retrieved_chunks     : final re-ranked chunks with full citation metadata
        score                : hallucination score (0.0 = grounded, 1.0 = hallucinated)
        is_hallucinated      : bool
        hallucination_type   : type string (or "grounded" if not hallucinated)
        explanation          : plain-English summary from the judge
        problematic_sentences: list of flagged sentences
    """

    def _emit(event_type: str, message: str, data=None):
        if status_callback:
            status_callback(event_type, message, data)

    print("\n" + "="*60)
    print("RAG PIPELINE")
    print("="*60)

    # ── Step 1: Retrieve candidates from ChromaDB ─────────────────
    print(f"\n[1/4] Stage 1 — ChromaDB vector search (top-{candidates} candidates)")
    print(f"      Question: \"{question}\"")
    if paper_id_filter:
        print(f"      Filter  : paper_id = {paper_id_filter}")

    _emit("status", "Searching knowledge base...")
    candidate_chunks = retrieve(
        query=question,
        top_k=candidates,
        collection_name=collection_name,
        paper_id_filter=paper_id_filter,
    )

    if not candidate_chunks:
        print("      → No chunks found in index.")
        return {
            "question":              question,
            "context":               "",
            "answer":                "No relevant context found in the index.",
            "citations":             {},
            "retrieved_chunks":      [],
            "source":                "none",
            "web_citations":         {},
            "score":                 0.0,
            "is_hallucinated":       False,
            "hallucination_type":    "grounded",
            "explanation":           "No chunks retrieved.",
            "problematic_sentences": [],
        }

    print(f"      → {len(candidate_chunks)} candidates retrieved")
    for i, c in enumerate(candidate_chunks[:3]):
        print(
            f"         [{i+1}] sim={c['similarity']:.4f} | "
            f"{c['paper_id']} chunk_{c['chunk_index']} | "
            f"{c['text'][:60].strip()}..."
        )
    if len(candidate_chunks) > 3:
        print(f"         ... and {len(candidate_chunks) - 3} more")

    # ── Step 2: Re-rank with cross-encoder ────────────────────────
    print(f"\n[2/4] Stage 2 — Cross-encoder re-ranking (top-{top_k} final)")
    _emit("status", "Re-ranking results...")
    retrieved_chunks = rerank(question, candidate_chunks, top_k=top_k)

    for i, c in enumerate(retrieved_chunks):
        print(
            f"      [{i+1}] rerank={c['rerank_score']:.4f} | "
            f"{c['paper_id']} chunk_{c['chunk_index']} | "
            f"{c['text'][:60].strip()}..."
        )

    # ── Step 3: Generate answer with citations ────────────────────
    print(f"\n[3/4] Generating answer with citations (GPT-4o)")
    _emit("status", "Generating answer from knowledge base...")
    answer, context, citations = generate_answer_with_citations(question, retrieved_chunks)
    print(f"      → {answer[:120]}{'...' if len(answer) > 120 else ''}")
    print(f"      → {len(citations)} source(s) cited")

    # ── Step 3b: Web search fallback ──────────────────────────────
    source = "rag"
    web_citations = {}

    if is_insufficient_answer(answer):
        print(f"\n[3b] Insufficient RAG context — falling back to DuckDuckGo web search")
        _emit("web_fallback", "Did not find information in store, searching the web...")
        try:
            web_results = search_web(question, max_results=WEB_SEARCH_MAX_RESULTS)
            if web_results:
                print(f"      → {len(web_results)} web result(s) retrieved")
                for r in web_results:
                    print(f"         [{r['number']}] trust={r.get('trust_score', 0):.2f} | {r['title'][:50]} | {r['url'][:55]}")
                answer, web_citations = generate_answer_from_web(question, web_results)
                source = "web"
                context = "\n\n".join(
                    f"[{r['number']}] {r['snippet']}" for r in web_results
                )
                citations = {}
                print(f"      → Web answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            else:
                print(f"      → Web search returned no results. Keeping original response.")
        except Exception as e:
            print(f"      → Web search failed: {e}. Keeping original response.")

    # ── Step 4: Detect hallucination ──────────────────────────────
    print(f"\n[4/4] Detecting hallucination (GPT-4o agent)")
    _emit("status", "Checking for hallucinations...")
    answer_text = _extract_answer_text(answer)

    # Web-fallback answers that explicitly decline to answer are grounded by
    # construction: the LLM is correctly refusing to fabricate. Comparing a
    # disclaimer like "The search results do not contain..." against irrelevant
    # web snippets (YouTube, Steam, etc.) produces near-zero cosine scores and
    # causes false-positive hallucination flags. Skip detection for these cases.
    _WEB_DECLINE_PHRASES = (
        "search results provided do not contain",
        "search results do not provide",
        "search results do not contain",
        "cannot answer your question based on the provided search results",
        "cannot answer based on the provided search results",
        "no relevant information found",
        "no relevant references were found",
        "cannot provide an answer to this question based on the available",
        "cannot confirm",
    )
    _is_web_decline = (
        source == "web"
        and any(p in answer_text.lower() for p in _WEB_DECLINE_PHRASES)
    )

    if _is_web_decline:
        print(f"      → Web answer is a grounded declination — skipping hallucination check")
        from agent.hallucination_agent import DetectionResult
        result = DetectionResult(
            question=question, context=context, answer=answer_text,
            score=0.0, is_hallucinated=False, hallucination_type="grounded",
            problematic_sentences=[],
            reasoning="Web answer explicitly declines to answer — treated as grounded.",
            explanation="Answer correctly states that the web search did not return relevant information.",
            agent_steps=0,
        )
    else:
        # Always use the full retrieved context so the agent verifies claims against
        # the complete source material, not just the LLM's own cherry-picked quotes.
        result = run_hallucination_agent(
            context=context,
            answer=answer_text,
            question=question,
            verify_claims=(source == "rag"),
            chunks=retrieved_chunks if source == "rag" else None,
        )

    print(f"      → Score        : {result.score:.4f}")
    print(f"      → Hallucinated : {result.is_hallucinated}")
    print(f"      → Type         : {result.hallucination_type}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULT SUMMARY")
    print("="*60)
    print(f"  Question : {question}")
    print(f"  Answer   : {answer}")
    print(f"  Score    : {result.score:.4f}  ({'HALLUCINATED' if result.is_hallucinated else 'GROUNDED'})")
    if result.is_hallucinated:
        print(f"  Type     : {result.hallucination_type}")
        print(f"  Flagged  : {result.problematic_sentences}")
    print(f"  Explain  : {result.explanation}")
    print(f"\n  Answer:\n  {answer}")
    print(f"\n  Citations:")
    for num, cite in citations.items():
        print(f"    [{num}] {cite['paper_id']} — {cite['title'][:55]}")
    print("="*60)

    return {
        "question":              question,
        "context":               context,
        "answer":                answer,
        "citations":             citations,
        "retrieved_chunks":      retrieved_chunks,
        "source":                source,
        "web_citations":         web_citations,
        "score":                 result.score,
        "is_hallucinated":       result.is_hallucinated,
        "hallucination_type":    result.hallucination_type,
        "explanation":           result.explanation,
        "problematic_sentences": result.problematic_sentences,
    }


if __name__ == "__main__":
    from data.qasper_loader import load_qasper

    # Ensure index is built
    documents, qa_pairs = load_qasper(max_papers=5)
    build_index(documents)

    # Pick two real Qasper questions — one answerable, one not
    answerable   = next(q for q in qa_pairs if q["answerable"])
    unanswerable = next(q for q in qa_pairs if not q["answerable"])

    # Answerable — should be grounded
    run_rag_pipeline(
        question=answerable["question"],
        paper_id_filter=answerable["paper_id"],
    )

    print("\n\n")

    # Unanswerable — should flag as hallucination if model invents an answer
    run_rag_pipeline(
        question=unanswerable["question"],
        paper_id_filter=unanswerable["paper_id"],
    )
