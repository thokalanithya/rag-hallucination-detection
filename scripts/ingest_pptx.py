"""
ingest_pptx.py
--------------
Ingest PowerPoint files from data/pptx/ into ChromaDB.

Usage:
    cd <project_root>
    python scripts/ingest_pptx.py

Options (edit at the top of this file):
    PPTX_DIR        — folder containing .pptx files
    COLLECTION_NAME — ChromaDB collection to write into
    RESET           — set True to wipe and rebuild from scratch
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PPTX_DIR        = PROJECT_ROOT / "data" / "pptx"
COLLECTION_NAME = "pptx_chunks"
RESET           = False      # set True to wipe existing data and re-index

# ── Main ───────────────────────────────────────────────────────────────────────
import importlib.util, types

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

SRC = PROJECT_ROOT / "src"
_load_module("config", SRC / "config.py")
_load_module("rag.chunker", SRC / "rag" / "chunker.py")
_load_module("rag.vector_store", SRC / "rag" / "vector_store.py")

from data.pptx_loader import load_pptx_dir
from rag.vector_store import build_index, collection_stats


def main():
    print(f"\n{'='*60}")
    print(f"  PPT Ingestion → ChromaDB collection: '{COLLECTION_NAME}'")
    print(f"  Source directory: {PPTX_DIR}")
    print(f"{'='*60}\n")

    if not PPTX_DIR.exists():
        print(f"ERROR: {PPTX_DIR} does not exist.")
        print("Create the folder and place your .pptx files inside, then re-run.")
        sys.exit(1)

    documents = load_pptx_dir(PPTX_DIR)
    if not documents:
        print("No documents to index. Exiting.")
        sys.exit(0)

    print(f"\n[ingest] Building ChromaDB index for {len(documents)} presentations …")
    build_index(documents, collection_name=COLLECTION_NAME, reset=RESET)

    stats = collection_stats(COLLECTION_NAME)
    print(f"\n[ingest] Done. Collection '{COLLECTION_NAME}' now has {stats['total_chunks']} chunks.")


if __name__ == "__main__":
    main()
