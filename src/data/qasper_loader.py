"""
qasper_loader.py
----------------
Loads the allenai/qasper dataset from HuggingFace and transforms it into
two reusable structures for the RAG pipeline:

  1. documents  — full paper texts, ready to chunk and embed
  2. qa_pairs   — question + ground-truth answer + evidence spans, used for
                  retrieval evaluation and Ragas scoring

Qasper structure (per paper):
    id            → unique paper ID
    title         → paper title
    abstract      → abstract text
    full_text     → list of sections, each with:
                        section_name : str
                        paragraphs   : list[str]
    qas           → list of Q&A dicts, each with:
                        question : str
                        answers  : list of answer objects

Each answer object contains:
    unanswerable      : bool
    extractive_spans  : list[str]   ← exact text pulled from the paper
    yes_no            : bool | None
    free_form_answer  : str         ← abstractive answer
    evidence          : list[str]   ← supporting paragraph(s)
    highlighted_evidence : list[str]

Usage
-----
    from data.qasper_loader import load_qasper

    documents, qa_pairs = load_qasper(max_papers=20)

    # documents  → list of dicts: {paper_id, title, full_text}
    # qa_pairs   → list of dicts: {paper_id, question, answer, evidence, answerable}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from datasets import load_dataset

from config import QASPER_SPLIT, QASPER_MAX_PAPERS, QASPER_CACHE_DIR


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_full_text(paper: dict) -> str:
    """
    Concatenate title + abstract + all section paragraphs into a single string.

    Qasper stores full_text in columnar format:
        full_text["section_name"] : list[str]        — one entry per section
        full_text["paragraphs"]   : list[list[str]]  — parallel list of paragraph lists

    Layout:
        <title>
        Abstract
        <abstract text>
        <Section Name>
        <paragraph 1>
        <paragraph 2>
        ...
    """
    parts = [paper["title"].strip()]

    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        parts.append("Abstract")
        parts.append(abstract)

    full_text = paper.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraphs    = full_text.get("paragraphs") or []

    for name, paras in zip(section_names, paragraphs):
        name = (name or "").strip()
        if name:
            parts.append(name)
        for para in paras or []:
            para = (para or "").strip()
            if para:
                parts.append(para)

    return "\n\n".join(parts)


def _extract_best_answer(answer_obj: dict) -> tuple[str, list[str]]:
    """
    Pull the most usable answer text and evidence list from one answer object.

    Priority:
        1. free_form_answer  (abstractive, most complete)
        2. extractive_spans joined into a sentence
        3. "Yes" / "No" for boolean questions
        4. "" for unanswerable questions

    Returns
    -------
    (answer_text, evidence_list)
    """
    if answer_obj.get("unanswerable"):
        return "", []

    evidence = answer_obj.get("evidence") or []

    free_form = (answer_obj.get("free_form_answer") or "").strip()
    if free_form:
        return free_form, evidence

    spans = [s.strip() for s in (answer_obj.get("extractive_spans") or []) if s.strip()]
    if spans:
        return " ... ".join(spans), evidence

    yes_no = answer_obj.get("yes_no")
    if yes_no is not None:
        return "Yes" if yes_no else "No", evidence

    return "", evidence


# ── Public API ────────────────────────────────────────────────────────────────

def load_qasper(
    split: str = QASPER_SPLIT,
    max_papers: Optional[int] = QASPER_MAX_PAPERS,
    cache_dir: str = QASPER_CACHE_DIR,
) -> tuple[list[dict], list[dict]]:
    """
    Load Qasper from HuggingFace and return documents + Q&A pairs.

    Parameters
    ----------
    split      : dataset split — "train", "validation", or "test"
    max_papers : cap on the number of papers to load (None = all)
    cache_dir  : local directory for HuggingFace cache

    Returns
    -------
    documents : list[dict]
        Each dict has:
            paper_id  : str   — unique Qasper paper ID
            title     : str   — paper title
            full_text : str   — concatenated title + abstract + sections

    qa_pairs : list[dict]
        Each dict has:
            paper_id   : str        — links back to a document
            question   : str        — the research question
            answer     : str        — best available answer text
            evidence   : list[str]  — supporting paragraphs from the paper
            answerable : bool       — False if the paper cannot answer this question
    """
    print(f"[qasper_loader] Loading allenai/qasper ({split} split) …")
    dataset = load_dataset("allenai/qasper", split=split, cache_dir=cache_dir, trust_remote_code=True)

    if max_papers is not None:
        dataset = dataset.select(range(min(max_papers, len(dataset))))

    print(f"[qasper_loader] Processing {len(dataset)} papers …")

    documents: list[dict] = []
    qa_pairs:  list[dict] = []

    for paper in dataset:
        paper_id  = paper["id"]
        title     = (paper.get("title") or "").strip()
        full_text = _build_full_text(paper)

        documents.append({
            "paper_id":  paper_id,
            "title":     title,
            "full_text": full_text,
        })

        # qas is columnar: dict of lists
        qas         = paper.get("qas") or {}
        questions   = qas.get("question") or []
        answers_col = qas.get("answers") or []   # parallel list of annotator-answer lists

        for question, ann_list in zip(questions, answers_col):
            question = (question or "").strip()
            if not question:
                continue

            # ann_list is a dict of lists (columnar again):
            #   {"answer": [answer_obj, ...]}  — one per annotator
            answer_text = ""
            evidence    = []
            answerable  = False

            ann_answers = ann_list.get("answer") or []
            for ans_obj in ann_answers:
                if not ans_obj.get("unanswerable", True):
                    text, ev = _extract_best_answer(ans_obj)
                    if text:
                        answer_text = text
                        evidence    = ev
                        answerable  = True
                        break

            # Include unanswerable questions too (useful for hallucination tests)
            if not answerable:
                answer_text = "This question cannot be answered from the paper."

            qa_pairs.append({
                "paper_id":   paper_id,
                "question":   question,
                "answer":     answer_text,
                "evidence":   evidence,
                "answerable": answerable,
            })

    answerable_count   = sum(1 for q in qa_pairs if q["answerable"])
    unanswerable_count = len(qa_pairs) - answerable_count

    print(
        f"[qasper_loader] Done — "
        f"{len(documents)} documents | "
        f"{len(qa_pairs)} Q&A pairs "
        f"({answerable_count} answerable, {unanswerable_count} unanswerable)"
    )

    return documents, qa_pairs


def preview(documents: list[dict], qa_pairs: list[dict], n: int = 2) -> None:
    """Pretty-print a sample of loaded data for quick sanity checking."""
    print("\n" + "="*70)
    print("DOCUMENT PREVIEW")
    print("="*70)
    for doc in documents[:n]:
        print(f"\nPaper ID : {doc['paper_id']}")
        print(f"Title    : {doc['title']}")
        snippet  = doc["full_text"][:300].replace("\n", " ")
        print(f"Text     : {snippet}{'…' if len(doc['full_text']) > 300 else ''}")
        print(f"Length   : {len(doc['full_text'].split())} words")

    print("\n" + "="*70)
    print("Q&A PAIR PREVIEW")
    print("="*70)
    shown = 0
    for qa in qa_pairs:
        if shown >= n * 2:
            break
        print(f"\nPaper ID   : {qa['paper_id']}")
        print(f"Question   : {qa['question']}")
        print(f"Answer     : {qa['answer'][:200]}{'…' if len(qa['answer']) > 200 else ''}")
        print(f"Answerable : {qa['answerable']}")
        if qa["evidence"]:
            ev = qa["evidence"][0][:150]
            print(f"Evidence   : {ev}…")
        shown += 1
    print("="*70)


if __name__ == "__main__":
    documents, qa_pairs = load_qasper(max_papers=5)
    preview(documents, qa_pairs, n=2)

    print("\nDocument columns :", list(documents[0].keys()))
    print("QA pair columns  :", list(qa_pairs[0].keys()))
