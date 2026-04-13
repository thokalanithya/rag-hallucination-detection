"""
pipeline.py
-----------
Full RAG pipeline with agentic hallucination detection.

Flow:
    document + question
        → chunk_document()   split into overlapping word chunks
        → embed_chunks()     encode chunks into vectors
        → retrieve_top_k()   find most relevant chunks via cosine similarity
        → generate_answer()  GPT-4o answers using only retrieved context
        → run_agent()        agentic detector scores and classifies the answer

Usage
-----
    python -m rag.pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

from docx import Document

from .chunker import chunk_document
from .embedder import embed_chunks, retrieve_top_k
from .generator import generate_answer
from agent.hallucination_agent import run_agent as run_gpt4o_agent
from agent.prometheus_judge import run_prometheus_judge
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K


def load_docx(path: str) -> str:
    """Extract plain text from a .docx file."""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def run_rag_pipeline(
    document: str,
    question: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    top_k: int = RETRIEVAL_TOP_K,
    judge: str = "prometheus",   # "prometheus" or "gpt4o"
) -> dict:
    """
    Run the full RAG pipeline on a document and question.

    Parameters
    ----------
    document   : full document text to use as the knowledge source
    question   : user's question
    chunk_size : words per chunk
    overlap    : overlapping words between chunks
    top_k      : number of chunks to retrieve

    Returns
    -------
    dict with keys:
        question             : the input question
        context              : concatenated top-K chunks used for generation
        answer               : GPT-4o generated answer
        score                : hallucination score (0.0 = grounded, 1.0 = hallucinated)
        is_hallucinated      : bool
        hallucination_type   : type string (or "none" if grounded)
        explanation          : plain-English summary from the agent
        problematic_sentences: list of flagged sentences
        top_k_chunks         : list of retrieved chunk dicts with similarity scores
    """

    print("\n" + "="*60)
    print("RAG PIPELINE")
    print("="*60)

    # ── Step 1: Chunk ──────────────────────────────────────────────
    print(f"\n[1/5] Chunking document  (chunk_size={chunk_size}, overlap={overlap})")
    chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    print(f"      → {len(chunks)} chunks created")

    # ── Step 2: Embed ──────────────────────────────────────────────
    print(f"\n[2/5] Embedding chunks   (model: all-MiniLM-L6-v2)")
    embedded = embed_chunks(chunks)
    print(f"      → {len(embedded)} chunks embedded (dim={embedded[0]['embedding'].shape[0]})")

    # ── Step 3: Retrieve ───────────────────────────────────────────
    print(f"\n[3/5] Retrieving top-{top_k} chunks for question:")
    print(f"      \"{question}\"")
    top_k_chunks = retrieve_top_k(question, embedded, top_k=top_k)
    for i, c in enumerate(top_k_chunks):
        print(f"      [{i+1}] chunk_{c['index']} similarity={c['similarity']:.4f} | {c['text'][:80]}...")

    # ── Step 4: Generate ───────────────────────────────────────────
    print(f"\n[4/5] Generating answer  (GPT-4o)")
    answer, context = generate_answer(question, top_k_chunks)
    print(f"      → Answer: {answer}")

    # ── Step 5: Detect hallucination ──────────────────────────────
    print(f"\n[5/5] Detecting hallucination  (judge={judge})")
    if judge == "prometheus":
        result = run_prometheus_judge(question=question, context=context, answer=answer)
    else:
        result = run_gpt4o_agent(context=context, answer=answer, question=question)

    print(f"      → Score          : {result.score:.4f}")
    print(f"      → Hallucinated   : {result.is_hallucinated}")
    print(f"      → Type           : {result.hallucination_type}")
    print(f"      → Explanation    : {result.explanation}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULT SUMMARY")
    print("="*60)
    print(f"  Question  : {question}")
    print(f"  Answer    : {answer}")
    print(f"  Score     : {result.score:.4f}  ({'HALLUCINATED' if result.is_hallucinated else 'GROUNDED'})")
    if result.is_hallucinated:
        print(f"  Type      : {result.hallucination_type}")
        print(f"  Sentences : {result.problematic_sentences}")
    print(f"  Explanation: {result.explanation}")
    print(f"\n  Context used ({len(top_k_chunks)} chunks):")
    for i, c in enumerate(top_k_chunks):
        print(f"    [{i+1}] {c['text'][:100]}...")
    print("="*60)

    return {
        "question":              question,
        "context":               context,
        "answer":                answer,
        "score":                 result.score,
        "is_hallucinated":       result.is_hallucinated,
        "hallucination_type":    result.hallucination_type,
        "explanation":           result.explanation,
        "problematic_sentences": result.problematic_sentences,
        "top_k_chunks":          top_k_chunks,
    }


if __name__ == "__main__":
    import os

    docx_path = os.path.join(os.path.dirname(__file__), "..", "..", "RAG-test.docx")
    document  = load_docx(docx_path)

    print(f"Loaded document: {docx_path}")
    print(f"Words: {len(document.split())}\n")

    question_1 = "What happens after an instructor submits an AI violation form?"
    question_2 = "What is the penalty for plagiarism at Harvard University?"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="prometheus", choices=["prometheus", "gpt4o"],
                        help="Judge model to use for hallucination detection")
    args = parser.parse_args()

    print(f"Using judge: {args.judge}\n")

    # Grounded question — answer is in the document
    run_rag_pipeline(document, question_1, judge=args.judge)

    print("\n\n")

    # Out-of-scope question — answer is NOT in the document
    run_rag_pipeline(document, question_2, judge=args.judge)
