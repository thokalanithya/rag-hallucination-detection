"""
pptx_loader.py
--------------
Extract text from PowerPoint (.pptx) files and convert to the document format
expected by build_index():  {paper_id, title, full_text}

Each slide's title + body text is extracted and concatenated with a
"--- Slide N ---" separator so the chunker can split naturally.
"""

from __future__ import annotations

import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt


def _slide_text(slide) -> str:
    """Return all text from a single slide, title first."""
    parts = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            line = " ".join(run.text for run in para.runs).strip()
            if line:
                parts.append(line)
    return "\n".join(parts)


def load_pptx(file_path: str | Path) -> dict:
    """
    Load a single .pptx file and return a document dict.

    Returns
    -------
    {
        "paper_id":  str  — filename without extension
        "title":     str  — first slide title or filename
        "full_text": str  — all slide text concatenated
    }
    """
    file_path = Path(file_path)
    prs = Presentation(str(file_path))

    paper_id = file_path.stem  # filename without extension
    title = paper_id           # fallback if no title slide

    slide_blocks = []
    for i, slide in enumerate(prs.slides, start=1):
        text = _slide_text(slide).strip()
        if not text:
            continue

        # Use the first non-empty slide text as document title
        if i == 1 and title == paper_id:
            first_line = text.splitlines()[0]
            if first_line:
                title = first_line

        slide_blocks.append(f"--- Slide {i} ---\n{text}")

    full_text = "\n\n".join(slide_blocks)
    return {"paper_id": paper_id, "title": title, "full_text": full_text}


def load_pptx_dir(directory: str | Path) -> list[dict]:
    """
    Load all .pptx files from a directory.

    Returns a list of document dicts, one per file.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.pptx")) + sorted(directory.glob("*.ppt"))

    if not files:
        print(f"[pptx_loader] No .pptx files found in {directory}")
        return []

    documents = []
    for f in files:
        try:
            doc = load_pptx(f)
            print(f"[pptx_loader] Loaded: {f.name!r} — {len(doc['full_text'])} chars, title={doc['title'][:60]!r}")
            documents.append(doc)
        except Exception as e:
            print(f"[pptx_loader] WARNING: Failed to load {f.name!r}: {e}")

    print(f"[pptx_loader] Total: {len(documents)} presentations loaded.")
    return documents
