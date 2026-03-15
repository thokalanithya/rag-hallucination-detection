"""
CSE 594 - Hallucination Detection for RAG Systems
config.py: Central configuration, constants, and hallucination taxonomy
"""

# ─────────────────────────────────────────────
# Project Metadata
# ─────────────────────────────────────────────
PROJECT_NAME = "RAG Hallucination Detector"
VERSION = "0.1.0"
DOMAIN = "scientific_academic"

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DATASET_PATH = os.path.join(DATA_DIR, "sample_dataset.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────
# Hallucination Detection Thresholds
# ─────────────────────────────────────────────
# Score range: 0.0 (fully grounded) → 1.0 (full hallucination)
HALLUCINATION_THRESHOLD = 0.5   # Scores >= this are classified as hallucinated
PARTIAL_SUPPORT_LOWER  = 0.25   # Between PARTIAL_SUPPORT_LOWER and THRESHOLD = partial
GROUNDED_UPPER         = 0.25   # Scores < this are considered fully grounded

# ─────────────────────────────────────────────
# Hallucination Type Taxonomy
# ─────────────────────────────────────────────
class HallucinationType:
    """
    Taxonomy of hallucination categories used throughout the project.
    Each type represents a distinct failure mode in RAG-generated answers.
    """

    GROUNDED = "grounded"
    """Answer is directly and fully supported by the retrieved context."""

    UNSUPPORTED = "unsupported_hallucination"
    """
    Answer contains information that is simply absent from the retrieved context.
    The model introduces facts or claims that were never stated.
    Example: Context discusses drug A; answer also mentions drug B with no basis.
    """

    CONTRADICTION = "context_conflicting_hallucination"
    """
    Answer directly contradicts information in the retrieved context.
    Example: Context says 'p < 0.05'; answer states 'results were not significant'.
    """

    FABRICATION = "fabricated_detail_or_entity"
    """
    Answer introduces specific non-existent names, numbers, citations, or entities.
    Example: Citing a paper 'Smith et al. 2019' that does not appear in the context.
    """

    OVERGENERALIZATION = "overgeneralization_or_extrapolation"
    """
    Answer draws broader conclusions than the context supports.
    Example: Context describes one study's findings; answer claims it is universally true.
    """

    PARTIAL = "partially_supported_hallucination"
    """
    Part of the answer is grounded but extra invented details are added.
    Example: Core finding is correct but a specific statistic is fabricated.
    """

    # Ordered list for iteration / display
    ALL_TYPES = [
        GROUNDED,
        UNSUPPORTED,
        CONTRADICTION,
        FABRICATION,
        OVERGENERALIZATION,
        PARTIAL,
    ]

    # Human-readable descriptions (for reports and prompts)
    DESCRIPTIONS = {
        GROUNDED:          "Fully supported by the retrieved context.",
        UNSUPPORTED:       "Contains information absent from the context.",
        CONTRADICTION:     "Contradicts information stated in the context.",
        FABRICATION:       "Introduces non-existent names, numbers, or citations.",
        OVERGENERALIZATION:"Draws broader conclusions than the context justifies.",
        PARTIAL:           "Partially supported; some details are invented.",
    }

# ─────────────────────────────────────────────
# RAG Pipeline Settings (Week 2)
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # HuggingFace sentence-transformers
RETRIEVAL_TOP_K = 3                      # Number of context chunks to retrieve
CHUNK_SIZE      = 300                    # Words per context chunk
CHUNK_OVERLAP   = 50                     # Overlap between consecutive chunks

# ─────────────────────────────────────────────
# Verifier LLM Settings (Week 3)
# ─────────────────────────────────────────────
VERIFIER_MODEL       = "gpt-4o-mini"
VERIFIER_TEMPERATURE = 0.0              # Deterministic scoring
VERIFIER_MAX_TOKENS  = 512
OPENAI_API_KEY_ENV   = "OPENAI_API_KEY" # Name of the environment variable

# ─────────────────────────────────────────────
# Dataset Settings
# ─────────────────────────────────────────────
DATASET_VERSION = "1.0"
MIN_CONTEXT_WORDS = 30    # Minimum words a valid context chunk must have
MIN_ANSWER_WORDS  = 5     # Minimum words a valid answer must have
