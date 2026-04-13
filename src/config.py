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
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────
# Hallucination Detection Thresholds
# ─────────────────────────────────────────────
# Score range: 0.0 (fully grounded) → 1.0 (full hallucination)
HALLUCINATION_THRESHOLD = 0.5   # Scores >= this are classified as hallucinated; scores < this are grounded

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
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"              # HuggingFace sentence-transformers
RETRIEVAL_CANDIDATES  = 20                               # First-pass: chunks pulled from ChromaDB (broad recall)
RETRIEVAL_TOP_K       = 5                                # Final: chunks kept after re-ranking (precision)
CHUNK_SIZE            = 600                              # Tokens per context chunk (target: 500-800)
CHUNK_OVERLAP         = 100                              # Overlapping tokens between consecutive chunks
CHUNK_ENCODING        = "cl100k_base"                    # tiktoken encoding (matches GPT-4 / ada-002)

# ── Re-ranking ─────────────────────────────────────────────────────────────
RERANKER_MODEL        = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Offline cross-encoder

# ─────────────────────────────────────────────
# Qasper Dataset Settings
# ─────────────────────────────────────────────
QASPER_SPLIT       = "train"             # HuggingFace split to load ("train" / "validation" / "test")
QASPER_MAX_PAPERS  = 50                  # Max papers to load (None = full dataset)
QASPER_CACHE_DIR   = os.path.join(DATA_DIR, "qasper_cache")  # Local HF cache

# ─────────────────────────────────────────────
# ChromaDB Vector Store Settings
# ─────────────────────────────────────────────
CHROMA_DIR        = os.path.join(BASE_DIR, "chroma_db")   # Persistent storage path
CHROMA_COLLECTION = "qasper_chunks"                        # Collection name

# ─────────────────────────────────────────────
# Verifier LLM Settings — Azure OpenAI (PRIMARY)
# ─────────────────────────────────────────────
VERIFIER_MODEL            = "gpt-4o"
VERIFIER_TEMPERATURE      = 0.0              # Deterministic scoring
VERIFIER_MAX_TOKENS       = 512
AZURE_API_KEY_ENV         = "AZURE_OPENAI_API_KEY"
AZURE_ENDPOINT_ENV        = "AZURE_OPENAI_ENDPOINT"
AZURE_DEPLOYMENT_ENV      = "AZURE_OPENAI_DEPLOYMENT"

# ─────────────────────────────────────────────
# Dataset Settings
# ─────────────────────────────────────────────
DATASET_VERSION = "1.0"
MIN_CONTEXT_WORDS = 30    # Minimum words a valid context chunk must have
MIN_ANSWER_WORDS  = 5     # Minimum words a valid answer must have
