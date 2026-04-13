"""
halueval_loader.py
------------------
HaluEval dataset loader for CSE 594 hallucination detection project.

Downloads qa_data.json from the HaluEval GitHub repository and transforms
it into a unified DataFrame with the schema expected by the hallucination
detector:
    question  | context  | answer  | label  | answer_type

Each sample in qa_data.json provides:
    - knowledge          → context
    - question           → question
    - right_answer       → grounded answer  (label = 0)
    - hallucinated_answer→ hallucinated answer (label = 1)

Both answer variants are kept as separate rows so the detector sees a
balanced mix of hallucinated and non-hallucinated examples.

Usage
-----
    from halueval_loader import load_halueval_qa

    df = load_halueval_qa(n_samples=200, seed=42)
    print(df.head())
    print(df["label"].value_counts())
"""

import json
import random
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

HALUEVAL_QA_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
)
DEFAULT_CACHE_PATH = Path("data/qa_data.json")


# ── Download helper ───────────────────────────────────────────────────────────

def download_halueval_qa(cache_path: Path = DEFAULT_CACHE_PATH) -> Path:
    """
    Download qa_data.json from the HaluEval GitHub repo if not already cached.

    Parameters
    ----------
    cache_path : Path
        Local path where the file will be saved.

    Returns
    -------
    Path
        Path to the cached file.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[halueval_loader] Using cached file: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[halueval_loader] Downloading HaluEval QA data to {cache_path} …")
    urllib.request.urlretrieve(HALUEVAL_QA_URL, cache_path)
    print(f"[halueval_loader] Download complete ({cache_path.stat().st_size / 1_048_576:.1f} MB).")
    return cache_path


# ── Core loader ───────────────────────────────────────────────────────────────

def load_halueval_qa(
    n_samples: Optional[int] = None,
    seed: int = 42,
    cache_path: Path = DEFAULT_CACHE_PATH,
    balanced: bool = True,
) -> pd.DataFrame:
    """
    Load HaluEval QA data and return a unified DataFrame.

    Each original record produces TWO rows:
        • one row with right_answer   → label = 0  (grounded / no hallucination)
        • one row with hallucinated_answer → label = 1  (hallucination)

    Parameters
    ----------
    n_samples : int, optional
        Total number of rows to return after expansion (both labels combined).
        If None, return all rows (~20 000 for the full 10 K QA file).
    seed : int
        Random seed for reproducible sampling.
    cache_path : Path
        Where to cache / read the raw JSON file.
    balanced : bool
        If True (default), sample equal numbers of label-0 and label-1 rows
        when n_samples is set.

    Returns
    -------
    pd.DataFrame
        Columns: question, context, answer, label, answer_type
        label      : int  — 0 = grounded, 1 = hallucinated
        answer_type: str  — "right_answer" or "hallucinated_answer"
    """
    path = download_halueval_qa(cache_path)

    # Parse JSONL format (one JSON object per line)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                record = json.loads(line)
                question = record["question"]
                context  = record["knowledge"]

                # Grounded answer row
                rows.append({
                    "question":    question,
                    "context":     context,
                    "answer":      record["right_answer"],
                    "label":       0,
                    "answer_type": "right_answer",
                })

                # Hallucinated answer row
                rows.append({
                    "question":    question,
                    "context":     context,
                    "answer":      record["hallucinated_answer"],
                    "label":       1,
                    "answer_type": "hallucinated_answer",
                })

    df = pd.DataFrame(rows)

    if n_samples is not None:
        rng = random.Random(seed)
        if balanced:
            half = n_samples // 2
            grounded     = df[df["label"] == 0].sample(n=half,                    random_state=seed)
            hallucinated = df[df["label"] == 1].sample(n=n_samples - half,        random_state=seed)
            df = pd.concat([grounded, hallucinated]).sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print(
        f"[halueval_loader] Loaded {len(df)} rows | "
        f"grounded={len(df[df['label']==0])} | "
        f"hallucinated={len(df[df['label']==1])}"
    )
    return df


# ── Convenience: get a small preview ─────────────────────────────────────────

def preview(df: pd.DataFrame, n: int = 3) -> None:
    """Pretty-print a few rows for quick sanity checking."""
    for i, row in df.head(n).iterrows():
        print(f"\n{'='*70}")
        print(f"[{i}] label={row['label']} ({row['answer_type']})")
        print(f"QUESTION : {row['question']}")
        print(f"CONTEXT  : {row['context'][:200]}{'…' if len(row['context']) > 200 else ''}")
        print(f"ANSWER   : {row['answer']}")
    print(f"\n{'='*70}")


# ── Main: quick smoke-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_halueval_qa(n_samples=100, seed=42)
    print("\nSchema:")
    print(df.dtypes)
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    preview(df, n=2)
