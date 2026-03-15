"""
CSE 594 - Hallucination Detection for RAG Systems
dataset_schema.py: Dataclasses defining every layer of the dataset structure.

Schema hierarchy:
    DatasetEntry
    ├── question          (str)
    ├── retrieved_context (List[ContextChunk])
    │     └── ContextChunk: text, source, relevance_score
    ├── generated_answer  (str)
    ├── sentence_scores   (List[SentenceEvaluation])   ← filled by verifier
    │     └── SentenceEvaluation: sentence, score, hallucination_type, explanation
    ├── answer_score      (float | None)               ← aggregate score
    ├── ground_truth_label (str | None)                ← for evaluation
    └── metadata          (EntryMetadata)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import uuid
from datetime import datetime

from config import HallucinationType


# ─────────────────────────────────────────────
# 1. Context Chunk
# ─────────────────────────────────────────────
@dataclass
class ContextChunk:
    """
    One retrieved passage from the document corpus.
    A single DatasetEntry may have multiple ContextChunks (top-k retrieval).
    """
    text: str
    source: str                          # e.g. paper title, DOI, filename
    chunk_index: int = 0                 # Position within the source document
    relevance_score: Optional[float] = None  # Cosine similarity from retriever

    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# 2. Sentence-Level Evaluation
# ─────────────────────────────────────────────
@dataclass
class SentenceEvaluation:
    """
    Verifier output for a single sentence in the generated answer.
    Populated during Week 3 (scoring) and Week 5 (classification).
    """
    sentence: str
    hallucination_score: Optional[float] = None   # 0.0 → 1.0
    hallucination_type: Optional[str] = None       # From HallucinationType
    explanation: Optional[str] = None              # Verifier's reasoning
    is_hallucinated: Optional[bool] = None         # Derived from threshold

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# 3. Entry Metadata
# ─────────────────────────────────────────────
@dataclass
class EntryMetadata:
    """
    Bookkeeping information attached to every dataset entry.
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    domain: str = "scientific_academic"
    difficulty: Optional[str] = None        # "easy" | "medium" | "hard"
    notes: Optional[str] = None             # Free-text annotation

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# 4. Top-Level Dataset Entry
# ─────────────────────────────────────────────
@dataclass
class DatasetEntry:
    """
    The core unit of the dataset.

    Lifecycle:
        Week 1  → question, retrieved_context, generated_answer,
                  ground_truth_label, metadata are populated manually / via pipeline.
        Week 3  → answer_score is filled by the LLM verifier.
        Week 4  → sentence_scores list is filled (sentence-level scoring).
        Week 5  → hallucination_type on each SentenceEvaluation is classified.
    """
    question: str
    retrieved_context: List[ContextChunk]
    generated_answer: str

    # Ground truth for evaluation (set manually in the sample dataset)
    ground_truth_label: Optional[str] = None   # From HallucinationType.ALL_TYPES
    ground_truth_note: Optional[str] = None    # Human explanation for the label

    # Verifier outputs (empty until Week 3+)
    answer_score: Optional[float] = None
    predicted_label: Optional[str] = None
    sentence_scores: List[SentenceEvaluation] = field(default_factory=list)

    metadata: EntryMetadata = field(default_factory=EntryMetadata)

    # ── Convenience helpers ──────────────────

    def combined_context_text(self) -> str:
        """Concatenates all context chunks into a single string for the verifier prompt."""
        return "\n\n".join(
            f"[Source {i+1}: {c.source}]\n{c.text}"
            for i, c in enumerate(self.retrieved_context)
        )

    def is_labeled(self) -> bool:
        """Returns True if a ground-truth label has been assigned."""
        return self.ground_truth_label is not None

    def is_scored(self) -> bool:
        """Returns True if the verifier has produced an answer-level score."""
        return self.answer_score is not None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "retrieved_context": [c.to_dict() for c in self.retrieved_context],
            "generated_answer": self.generated_answer,
            "ground_truth_label": self.ground_truth_label,
            "ground_truth_note": self.ground_truth_note,
            "answer_score": self.answer_score,
            "predicted_label": self.predicted_label,
            "sentence_scores": [s.to_dict() for s in self.sentence_scores],
            "metadata": self.metadata.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "DatasetEntry":
        """Deserialize a DatasetEntry from a plain dictionary (e.g. loaded from JSON)."""
        return DatasetEntry(
            question=d["question"],
            retrieved_context=[
                ContextChunk(**c) for c in d.get("retrieved_context", [])
            ],
            generated_answer=d["generated_answer"],
            ground_truth_label=d.get("ground_truth_label"),
            ground_truth_note=d.get("ground_truth_note"),
            answer_score=d.get("answer_score"),
            predicted_label=d.get("predicted_label"),
            sentence_scores=[
                SentenceEvaluation(**s) for s in d.get("sentence_scores", [])
            ],
            metadata=EntryMetadata(**d["metadata"]) if "metadata" in d else EntryMetadata(),
        )

    def __repr__(self) -> str:
        label = self.ground_truth_label or "unlabeled"
        scored = f", score={self.answer_score:.2f}" if self.is_scored() else ""
        return (
            f"DatasetEntry(id={self.metadata.entry_id!r}, "
            f"label={label!r}{scored}, "
            f"q={self.question[:60]!r})"
        )


# ─────────────────────────────────────────────
# 5. Dataset Container
# ─────────────────────────────────────────────
@dataclass
class Dataset:
    """
    Collection of DatasetEntry objects with save/load utilities.
    """
    entries: List[DatasetEntry] = field(default_factory=list)
    version: str = "1.0"

    def add(self, entry: DatasetEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def label_distribution(self) -> dict:
        """Returns a count of each ground-truth label in the dataset."""
        dist = {}
        for e in self.entries:
            lbl = e.ground_truth_label or "unlabeled"
            dist[lbl] = dist.get(lbl, 0) + 1
        return dist

    def get_by_label(self, label: str) -> List[DatasetEntry]:
        return [e for e in self.entries if e.ground_truth_label == label]

    def save(self, path: str) -> None:
        payload = {
            "version": self.version,
            "total_entries": len(self.entries),
            "label_distribution": self.label_distribution(),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Dataset] Saved {len(self.entries)} entries → {path}")

    @staticmethod
    def load(path: str) -> "Dataset":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        entries = [DatasetEntry.from_dict(e) for e in payload.get("entries", [])]
        ds = Dataset(entries=entries, version=payload.get("version", "1.0"))
        print(f"[Dataset] Loaded {len(ds.entries)} entries from {path}")
        return ds
