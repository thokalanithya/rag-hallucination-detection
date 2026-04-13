"""
prometheus_judge.py
--------------------
Hallucination judge powered by Prometheus via Ollama (local, no API cost).

Prometheus is a model fine-tuned specifically for LLM evaluation — it scores
responses on a 1–5 rubric and provides detailed feedback. Unlike GPT-4o,
it has no self-evaluation bias since it never generated the answer.

Architecture
------------
  GPT-4o  →  generates answer         (generator.py)
  Prometheus (Ollama)  →  judges it   (this file)

Prometheus prompt format
------------------------
Prometheus expects a structured rubric-based prompt:
  - Instruction  : the question
  - Response     : the answer to judge
  - Context      : the retrieved passage
  - Score rubric : faithfulness criteria (1=hallucinated, 5=grounded)

Output: Feedback text + [RESULT] <1–5 score>

Score → hallucination mapping
------------------------------
  5  →  0.0  (fully grounded)
  4  →  0.2
  3  →  0.5
  2  →  0.8
  1  →  1.0  (fully hallucinated)

Setup
-----
  ollama pull prometheus    # ~4GB
  ollama serve

Usage
-----
  from agent.prometheus_judge import run_prometheus_judge

  result = run_prometheus_judge(
      question="What happens after ...",
      context="The retrieved passage ...",
      answer="The generated answer ...",
  )
  print(result.score, result.hallucination_type, result.explanation)
"""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI   # Ollama exposes an OpenAI-compatible API

# ── Abstention detection ─────────────────────────────────────────────────────

ABSTENTION_PHRASES = [
    "i don't have enough information",
    "i do not have enough information",
    "i cannot answer",
    "i can't answer",
    "not enough information",
    "the context does not",
    "the context doesn't",
    "no information",
    "i don't know",
    "cannot be determined",
    "not mentioned in the context",
]

def is_abstention(answer: str) -> bool:
    """Return True if the answer is a refusal/abstention rather than a real answer."""
    return any(phrase in answer.lower() for phrase in ABSTENTION_PHRASES)


# ── Ollama client ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
PROMETHEUS_MODEL = os.environ.get("PROMETHEUS_MODEL", "mistral")

_client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

THRESHOLD = 0.5

# ── Score → hallucination score mapping ───────────────────────────────────────

SCORE_MAP = {5: 0.0, 4: 0.2, 3: 0.5, 2: 0.8, 1: 1.0}

# ── Prometheus rubric prompt ──────────────────────────────────────────────────

PROMETHEUS_PROMPT = """###Task Description:
An instruction (the question), a response to evaluate, the retrieved context, and a score rubric are given.
1. Write detailed feedback assessing the faithfulness of the response to the retrieved context.
2. After writing feedback, write a score that is an integer between 1 and 5.
3. Output format: "Feedback: (your feedback) [RESULT] (integer 1-5)"
4. Do not generate any other text outside this format.

###Instruction (Question):
{question}

###Retrieved Context:
{context}

###Response to Evaluate:
{answer}

###Score Rubric:
[Is the response faithful and grounded in the retrieved context?]
Score 1: The response contains major hallucinations — significant information not present in or contradicting the context.
Score 2: The response contains noticeable hallucinations or unsupported claims that go beyond the context.
Score 3: The response is mostly faithful but includes minor details not fully supported by the context.
Score 4: The response is largely faithful with only negligible deviations from the context.
Score 5: The response is completely faithful and grounded — every claim is directly supported by the context.

###Feedback:"""

# ── Hallucination type inference from feedback ────────────────────────────────

TYPE_KEYWORDS = {
    "contradict":       "contradicting",
    "conflicts":        "contradicting",
    "opposite":         "contradicting",
    "fabricat":         "fabricated",
    "invent":           "fabricated",
    "not mentioned":    "unsupported",
    "not present":      "unsupported",
    "absent":           "unsupported",
    "not in the":       "unsupported",
    "broader":          "overgeneralization",
    "generaliz":        "overgeneralization",
    "extrapolat":       "overgeneralization",
    "partially":        "partial",
    "some details":     "partial",
}


def _infer_type(feedback: str, score: int) -> str:
    """Infer hallucination type from the feedback text and score."""
    if score >= 4:
        return "none"
    lower = feedback.lower()
    for keyword, hall_type in TYPE_KEYWORDS.items():
        if keyword in lower:
            return hall_type
    return "unsupported"   # default for hallucinated answers


# ── Result dataclass (same interface as GPT-4o agent) ─────────────────────────

@dataclass
class DetectionResult:
    question:              str
    context:               str
    answer:                str
    score:                 float          # 0.0 = grounded, 1.0 = hallucinated
    is_hallucinated:       bool
    hallucination_type:    str
    problematic_sentences: list
    reasoning:             str            # raw feedback from Prometheus
    explanation:           str            # short summary
    ground_truth_label:    Optional[int] = None
    prometheus_score:      int = 0        # raw 1–5 score


# ── Core judge function ───────────────────────────────────────────────────────

def run_prometheus_judge(
    question: str,
    context:  str,
    answer:   str,
) -> DetectionResult:
    """
    Use Prometheus (via Ollama) to judge whether the answer is hallucinated.

    Parameters
    ----------
    question : the original question
    context  : retrieved context passage
    answer   : generated answer to evaluate

    Returns
    -------
    DetectionResult with score, type, explanation (same interface as GPT-4o agent)
    """
    # ── Abstention check — skip judge entirely ────────────────────────────────
    if is_abstention(answer):
        return DetectionResult(
            question=question,
            context=context,
            answer=answer,
            score=0.0,
            is_hallucinated=False,
            hallucination_type="abstention",
            problematic_sentences=[],
            reasoning="Model declined to answer.",
            explanation="Model correctly abstained — answer was not in the context.",
            prometheus_score=0,
        )

    prompt = PROMETHEUS_PROMPT.format(
        question=question,
        context=context.strip(),
        answer=answer.strip(),
    )

    try:
        response = _client.chat.completions.create(
            model=PROMETHEUS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        # Graceful fallback if Ollama is unreachable
        return DetectionResult(
            question=question, context=context, answer=answer,
            score=0.0, is_hallucinated=False,
            hallucination_type="none",
            problematic_sentences=[],
            reasoning=f"Prometheus unavailable: {e}",
            explanation="Could not reach Prometheus judge.",
            prometheus_score=0,
        )

    # Parse: "Feedback: ... [RESULT] <1-5>"
    feedback = raw
    prometheus_score = 3    # default middle score

    result_match = re.search(r"\[RESULT\]\s*([1-5])", raw)
    if result_match:
        prometheus_score = int(result_match.group(1))
        feedback = raw[:result_match.start()].replace("Feedback:", "").strip()

    hall_score  = SCORE_MAP.get(prometheus_score, 0.5)
    hall_type   = _infer_type(feedback, prometheus_score)
    is_hallucinated = hall_score >= THRESHOLD

    # Build a short explanation from the first sentence of feedback
    first_sentence = feedback.split(".")[0].strip() + "." if feedback else "No feedback."

    return DetectionResult(
        question=question,
        context=context,
        answer=answer,
        score=hall_score,
        is_hallucinated=is_hallucinated,
        hallucination_type=hall_type,
        problematic_sentences=[],
        reasoning=feedback,
        explanation=first_sentence,
        prometheus_score=prometheus_score,
    )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    context = """
    Photosynthesis is the process by which green plants convert light energy into
    glucose. The overall equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.
    Plants use chlorophyll in the chloroplasts to absorb sunlight.
    """

    # Grounded answer
    result = run_prometheus_judge(
        question="How do plants produce glucose?",
        context=context,
        answer="Plants produce glucose through photosynthesis using CO2, water, and sunlight via chlorophyll.",
    )
    print("── Grounded answer ──────────────────────────────")
    print(f"  Prometheus score : {result.prometheus_score}/5")
    print(f"  Hallucination    : {result.score:.2f}  ({'HALLUCINATED' if result.is_hallucinated else 'GROUNDED'})")
    print(f"  Type             : {result.hallucination_type}")
    print(f"  Explanation      : {result.explanation}")
    print(f"  Feedback         : {result.reasoning[:200]}")

    print()

    # Hallucinated answer
    result2 = run_prometheus_judge(
        question="How do plants produce glucose?",
        context=context,
        answer="Plants produce glucose using mitochondria and nitrogen from the soil, discovered by Newton in 1704.",
    )
    print("── Hallucinated answer ──────────────────────────")
    print(f"  Prometheus score : {result2.prometheus_score}/5")
    print(f"  Hallucination    : {result2.score:.2f}  ({'HALLUCINATED' if result2.is_hallucinated else 'GROUNDED'})")
    print(f"  Type             : {result2.hallucination_type}")
    print(f"  Explanation      : {result2.explanation}")
    print(f"  Feedback         : {result2.reasoning[:200]}")
