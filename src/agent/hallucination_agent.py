"""
hallucination_agent.py
-----------------------
Proper agentic hallucination detector powered by Azure OpenAI (GPT-4o).

Architecture
------------
GPT-4o is the sole reasoner. It uses three deterministic utility tools to
gather evidence, then reasons over that evidence itself to produce a verdict.

  Tool 1 → split_sentences(answer)
            Splits the answer into individual sentences (NLTK).

  Tool 2 → check_sentence_support(sentence, context)
            Returns a semantic similarity score [0.0, 1.0] using
            sentence-transformers. Low score = likely unsupported.

  Tool 3 → extract_claims(sentence)
            Extracts specific verifiable facts (numbers, names, dates,
            citations) from a sentence for targeted verification.

Agent loop
----------
  1. GPT-4o calls split_sentences → gets sentence list
  2. GPT-4o calls check_sentence_support for each sentence
  3. For low-support sentences, GPT-4o calls extract_claims
  4. GPT-4o reasons over all evidence and emits a structured JSON verdict:
       {
         "reasoning":             "<chain-of-thought>",
         "is_hallucinated":       <bool>,
         "score":                 <float 0.0–1.0>,
         "hallucination_type":    "<type or none>",
         "problematic_sentences": ["<sentence>", ...],
         "explanation":           "<1–2 sentence summary>"
       }

Provider
--------
Azure OpenAI (GPT-4o) — reads credentials from .env

Usage
-----
  # Single sample
  python -m agent.hallucination_agent detect \\
      --context "..." --answer "..." --verbose

  # Batch evaluation on qa_data.json
  python -m agent.hallucination_agent evaluate --n_samples 20

  # Full HaluEval pipeline
  python -m agent.hallucination_agent pipeline --n_samples 50
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Add src/ to path so all packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AzureOpenAI
from data.halueval_loader import load_halueval_qa

_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
from tools import split_sentences, check_sentence_support, extract_claims, find_in_context, keyword_overlap

# ── Hallucination taxonomy ────────────────────────────────────────────────────

HALLUCINATION_TYPES = [
    "unsupported",        # info not in context
    "contradicting",      # directly contradicts context
    "fabricated",         # invented names / numbers / citations
    "overgeneralization", # broader than context supports
    "partial",            # part grounded, extra details invented
    "none",               # not hallucinated
]

THRESHOLD = 0.5

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    question: str
    context: str
    answer: str
    score: float
    is_hallucinated: bool
    hallucination_type: str
    problematic_sentences: list
    reasoning: str
    explanation: str
    ground_truth_label: Optional[int] = None
    agent_steps: int = 0


# ── Tool definitions (OpenAI function-calling schema) ─────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "split_sentences",
            "description": (
                "Split the generated answer into individual sentences. "
                "Always call this first before any other tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The generated answer to split.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_overlap",
            "description": (
                "Supplementary sanity check. Compute token-level F1 word overlap between "
                "the full answer and the context. Catches gross entity substitutions that "
                "semantic similarity misses — e.g. 'Mumbai' vs 'Delhi'. "
                "Returns f1 score, precision, recall, and missing_tokens. "
                "Call once on the full answer only — not per-sentence. "
                "Treat missing_tokens as a hint to investigate, not as proof of hallucination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer":  {"type": "string", "description": "Answer text or sentence."},
                    "context": {"type": "string", "description": "Retrieved context passage."},
                },
                "required": ["answer", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_in_context",
            "description": (
                "String lookup for a single atomic fact inside the context. "
                "Call ONLY on individual claims returned by extract_claims — "
                "one specific number, one proper name, one date per call. "
                "Do NOT pass phrases, ranges, or paraphrase words. "
                "Returns found=True/False, match_type, and a context snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim":   {"type": "string", "description": "The specific fact or entity to look up."},
                    "context": {"type": "string", "description": "The retrieved context passage."},
                },
                "required": ["claim", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_sentence_support",
            "description": (
                "PRIMARY grounding check. Compute semantic similarity between a sentence "
                "and the context. Returns support_score [0.0, 1.0] and signal: "
                "'supported' (≥0.65) — sentence is grounded, no further checks needed; "
                "'low_support' (0.40–0.65) — borderline, verify specific facts; "
                "'no_support' (<0.40) — likely unsupported, verify specific facts. "
                "Call on every sentence before any other per-sentence tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string", "description": "A sentence from the answer."},
                    "context":  {"type": "string", "description": "The retrieved context passage."},
                },
                "required": ["sentence", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_claims",
            "description": (
                "Extract individual atomic verifiable facts (numbers, percentages, dates, "
                "proper names, citations, p-values) from a sentence. "
                "Call ONLY on sentences that scored below 0.65 in check_sentence_support. "
                "Each returned item is one atomic claim to pass separately to find_in_context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string", "description": "A suspicious sentence."},
                },
                "required": ["sentence"],
            },
        },
    },
]

# ── Tool dispatch (real utility tools — no LLM calls) ─────────────────────────

def _dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Execute the named utility tool and return a JSON string result."""
    if tool_name == "split_sentences":
        result = split_sentences(tool_args["text"])
        return json.dumps({"sentences": result, "count": len(result)})

    if tool_name == "keyword_overlap":
        result = keyword_overlap(tool_args["answer"], tool_args["context"])
        return json.dumps(result)

    if tool_name == "find_in_context":
        result = find_in_context(tool_args["claim"], tool_args["context"])
        return json.dumps(result)

    if tool_name == "check_sentence_support":
        result = check_sentence_support(tool_args["sentence"], tool_args["context"])
        return json.dumps(result)

    if tool_name == "extract_claims":
        result = extract_claims(tool_args["sentence"])
        return json.dumps(result)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Agent system prompt ────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a hallucination detection agent for RAG (Retrieval-Augmented Generation) systems.

Your goal: determine whether a generated answer is hallucinated relative to the retrieved context, and classify the exact type.

You have FIVE tools that provide evidence. You do ALL the reasoning yourself.

─── TOOLS ───────────────────────────────────────────────────────────
1. split_sentences(text)
   → Split answer into sentences. Call FIRST, always.

2. check_sentence_support(sentence, context)
   → Semantic similarity score [0.0–1.0]. PRIMARY gate for every sentence.
   → signal: "supported" (≥0.65), "low_support" (0.40–0.65), "no_support" (<0.40).
   → Call on EVERY sentence. The result determines whether deeper checks run.

3. extract_claims(sentence)
   → Pull out atomic verifiable facts: numbers, proper names, dates, percentages,
     citations. Each is returned as a separate item.
   → Call ONLY on sentences that scored below 0.65 in check_sentence_support.

4. find_in_context(claim, context)
   → String lookup: does this exact fact exist in the context?
   → Call ONLY on individual atomic claims from extract_claims — one claim per call.
   → NEVER call it on a phrase, a range, or a paraphrase word.

5. keyword_overlap(answer, context)
   → Token-level F1. Use as a supplementary sanity check on the full answer only.
   → Helps spot gross entity substitutions (Mumbai vs Delhi). Do not use per-sentence.

─── RECOMMENDED WORKFLOW ────────────────────────────────────────────
Step 1: split_sentences(answer)
Step 2: For EACH sentence → check_sentence_support(sentence, context)
          ● score ≥ 0.65 ("supported") → sentence is GROUNDED. Stop here for this sentence.
          ● score 0.40–0.65 ("low_support") → run Steps 3–4 on this sentence.
          ● score < 0.40 ("no_support") → run Steps 3–4 on this sentence.
Step 3: extract_claims(sentence)  ← only for sentences below 0.65
Step 4: find_in_context(claim, context)  ← one call per atomic claim from Step 3
Step 5: keyword_overlap(full_answer, context)  ← optional sanity check on the whole answer
Step 6: Reason over ALL evidence → emit final JSON verdict

─── CRITICAL REASONING RULES ────────────────────────────────────────
★ check_sentence_support ≥ 0.65 = GROUNDED. Do not override it.
  If a sentence scores "supported", it is semantically aligned with the context.
  The LLM always paraphrases — "surgical details" means the same as "how far
  from the tumor is cut during surgery." Do NOT call find_in_context on
  supported sentences. Do NOT flag paraphrase as overgeneralization.

★ find_in_context verifies ATOMIC facts only — never phrases or ranges.
  Extract individual atoms from extract_claims first, then check each one.
  "0.556" and "0.731" are valid individual claims to check.
  "0.556 to 0.731" is a paraphrased range — do NOT pass it to find_in_context.
  "leveraging" is a paraphrase verb — do NOT pass it to find_in_context.

★ A sentence needs BOTH low semantic support AND a missing specific fact
  to be classified as hallucinated.
  Low similarity alone → possible paraphrase, not hallucination.
  Missing specific fact alone on a supported sentence → not hallucination.
  Low similarity + find_in_context(specific_fact) = False → hallucination.

★ keyword_overlap missing_tokens are background signal only.
  Most missing tokens are paraphrase words. Only treat a missing token as
  suspicious if it is a proper name, number, or date AND the sentence already
  scored below 0.65 in check_sentence_support.

★ Short answers need extra care.
  A one-word answer that does not appear in context = hallucinated.
  A one-word answer that appears exactly in context = grounded.

─── HALLUCINATION TYPES ─────────────────────────────────────────────
- unsupported       : claim absent from context (not mentioned at all)
- contradicting     : claim directly contradicts something in context
- fabricated        : invented name, number, citation, or entity
- overgeneralization: broader conclusion than context supports
- partial           : part grounded, but extra invented details added
- none              : answer is fully grounded

─── FINAL OUTPUT ─────────────────────────────────────────────────────
After ALL tool calls, output ONLY this JSON (no markdown fences, no extra text):

{
  "reasoning": "<step-by-step: what tools showed, what you concluded>",
  "is_hallucinated": <true or false>,
  "score": <float 0.0–1.0; 0.0=fully grounded, 1.0=fully hallucinated>,
  "hallucination_type": "<one of the 6 types>",
  "problematic_sentences": ["<exact sentence>", ...],
  "explanation": "<1–2 sentence plain-English finding>"
}"""


# ── Core agent loop ────────────────────────────────────────────────────────────

def run_agent(
    context: str,
    answer: str,
    question: str = "",
    max_steps: int = 15,
    verbose: bool = False,
) -> DetectionResult:
    """
    Run the hallucination detection agent for a single (context, answer) pair.

    GPT-4o uses split_sentences → check_sentence_support → extract_claims
    to gather evidence, then reasons over it to produce a structured verdict.

    Parameters
    ----------
    context   : Retrieved context passage
    answer    : Generated answer to evaluate
    question  : Original question (for record-keeping)
    max_steps : Maximum tool-call iterations
    verbose   : Print each agent step

    Returns
    -------
    DetectionResult with score, type, problematic sentences, reasoning, explanation
    """
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"Generated answer:\n{answer}"
            ),
        },
    ]

    # Default verdict values
    score                 = 0.0
    hall_type             = "none"
    problematic_sentences = []
    reasoning             = ""
    explanation           = "Answer is grounded."
    steps                 = 0

    for step in range(max_steps):
        if verbose:
            print(f"    [step {step + 1}] calling GPT-4o …")

        raw_resp  = _client.chat.completions.create(
            model=_model, messages=messages, tools=TOOLS,
            tool_choice="auto", temperature=0.0,
        )
        msg = raw_resp.choices[0].message
        response = {
            "role":    msg.role,
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ],
        }
        messages.append(response)

        tool_calls = response.get("tool_calls", [])

        # No tool calls → GPT-4o produced its final verdict
        if not tool_calls:
            raw = (response.get("content") or "").strip()
            if verbose:
                print(f"    [verdict]\n{raw}")

            # Strip accidental markdown fences
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break

            try:
                verdict               = json.loads(raw)
                score                 = float(verdict.get("score", score))
                hall_type             = verdict.get("hallucination_type", hall_type)
                problematic_sentences = verdict.get("problematic_sentences", [])
                reasoning             = verdict.get("reasoning", "")
                explanation           = verdict.get("explanation", explanation)
                # Reconcile score with is_hallucinated so both agree.
                # If is_hallucinated=False but score is high, cap it down.
                # If is_hallucinated=True but score is low, bump it up.
                is_hall = verdict.get("is_hallucinated", score >= THRESHOLD)
                if not is_hall and score >= THRESHOLD:
                    score = min(score, THRESHOLD - 0.01)
                elif is_hall and score < THRESHOLD:
                    score = max(score, THRESHOLD)
            except (json.JSONDecodeError, ValueError):
                pass  # use defaults

            steps = step + 1
            break

        # Execute each tool call with the real utility functions
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            if verbose:
                arg_preview = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v)
                               for k, v in fn_args.items()}
                print(f"    [tool] {fn_name}({arg_preview})")

            tool_result = _dispatch_tool(fn_name, fn_args)

            if verbose:
                result_preview = tool_result[:120] + "…" if len(tool_result) > 120 else tool_result
                print(f"    [result] {result_preview}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      tool_result,
            })

        steps = step + 1
        time.sleep(0.05)

    return DetectionResult(
        question=question,
        context=context,
        answer=answer,
        score=score,
        is_hallucinated=score >= THRESHOLD,
        hallucination_type=hall_type,
        problematic_sentences=problematic_sentences,
        reasoning=reasoning,
        explanation=explanation,
        agent_steps=steps,
    )


# ── Type normalisation (for evaluate_qa_data) ─────────────────────────────────

TYPE_NORMALISE: dict[str, str] = {
    "none":               "grounded",
    "unsupported":        "unsupported_hallucination",
    "contradicting":      "context_conflicting_hallucination",
    "fabricated":         "fabricated_detail_or_entity",
    "overgeneralization": "overgeneralization_or_extrapolation",
    "partial":            "partially_supported_hallucination",
    # identity mappings for full label strings
    "grounded":                           "grounded",
    "unsupported_hallucination":          "unsupported_hallucination",
    "context_conflicting_hallucination":  "context_conflicting_hallucination",
    "fabricated_detail_or_entity":        "fabricated_detail_or_entity",
    "overgeneralization_or_extrapolation":"overgeneralization_or_extrapolation",
    "partially_supported_hallucination":  "partially_supported_hallucination",
}

ALL_LABELS = [
    "grounded",
    "unsupported_hallucination",
    "context_conflicting_hallucination",
    "fabricated_detail_or_entity",
    "overgeneralization_or_extrapolation",
    "partially_supported_hallucination",
]


def _normalise_type(raw: str) -> str:
    return TYPE_NORMALISE.get(raw.lower().strip(), raw.lower().strip())


# ── QA-data evaluation (qa_data.json) ─────────────────────────────────────────

def evaluate_qa_data(
    dataset_path: Optional[str] = None,
    n_samples: int = 20,
    seed: int = 42,
    verbose: bool = False,
    output_csv: str = "results_qa_eval.csv",
) -> "pd.DataFrame":
    """
    Evaluate binary hallucination detection on a subset of qa_data.json.

    Each line has: knowledge, question, right_answer (label=0),
    hallucinated_answer (label=1).

    Builds a balanced sample, runs the agent, and reports binary accuracy
    + hallucination type distribution.
    """
    import random
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )

    if dataset_path is None:
        dataset_path = str(Path(__file__).parent.parent / "data" / "qa_data.json")

    raw_entries = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_entries.append(json.loads(line))

    random.seed(seed)
    random.shuffle(raw_entries)

    half = n_samples // 2
    samples = []
    g_count = h_count = 0
    for entry in raw_entries:
        if g_count < half:
            samples.append({
                "question": entry["question"],
                "context":  entry["knowledge"],
                "answer":   entry["right_answer"],
                "label":    0,
            })
            g_count += 1
        if h_count < half:
            samples.append({
                "question": entry["question"],
                "context":  entry["knowledge"],
                "answer":   entry["hallucinated_answer"],
                "label":    1,
            })
            h_count += 1
        if g_count >= half and h_count >= half:
            break

    print(f"\nEvaluating {len(samples)} samples from qa_data.json "
          f"({g_count} grounded + {h_count} hallucinated)\n")

    records = []

    for i, sample in enumerate(samples):
        true_str = "HALL" if sample["label"] == 1 else "OK  "
        print(f"[{i+1}/{len(samples)}] true={true_str}  "
              f"Q: {sample['question'][:55]}…", end=" ", flush=True)

        result = run_agent(
            context=sample["context"],
            answer=sample["answer"],
            question=sample["question"],
            verbose=verbose,
        )

        pred       = int(result.is_hallucinated)
        is_correct = pred == sample["label"]
        hall_type  = result.hallucination_type if result.is_hallucinated else "none"

        print(
            f"pred={'HALL' if pred else 'OK  '}  "
            f"score={result.score:.2f}  "
            f"type={hall_type:<20}  "
            f"{'✓' if is_correct else '✗'}"
        )

        records.append({
            "question":          sample["question"],
            "context":           sample["context"],
            "answer":            sample["answer"],
            "ground_truth":      sample["label"],
            "predicted":         pred,
            "score":             result.score,
            "hallucination_type": hall_type,
            "problematic_sentences": "; ".join(result.problematic_sentences),
            "explanation":       result.explanation,
            "correct":           is_correct,
        })

    df = pd.DataFrame(records)
    y_true, y_pred = df["ground_truth"], df["predicted"]

    print("\n" + "=" * 60)
    print("QA DATA EVALUATION RESULTS")
    print("=" * 60)
    print(f"Provider          : Azure OpenAI ({_model})")
    print(f"Samples           : {len(df)}  ({g_count} grounded + {h_count} hallucinated)")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision         : {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall            : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1                : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["Grounded", "Hallucinated"]))

    type_counts = df[df["predicted"] == 1]["hallucination_type"].value_counts()
    if not type_counts.empty:
        print("Hallucination type distribution (predicted positives):")
        print(type_counts.to_string())
    print("=" * 60)

    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    return df


# ── HaluEval batch pipeline ────────────────────────────────────────────────────

def run_pipeline(
    n_samples: int = 50,
    threshold: float = THRESHOLD,
    seed: int = 42,
    output_csv: str = "results_agent.csv",
    verbose: bool = False,
) -> "pd.DataFrame":
    """Run the agent over the HaluEval QA dataset and evaluate performance."""
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )

    df = load_halueval_qa(n_samples=n_samples, seed=seed)
    records = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Analysing …", end=" ", flush=True)

        result = run_agent(
            context=row["context"],
            answer=row["answer"],
            question=row.get("question", ""),
            verbose=verbose,
        )
        result.ground_truth_label = int(row["label"])

        print(
            f"score={result.score:.2f}  "
            f"pred={'HAL' if result.is_hallucinated else 'OK ':3s}  "
            f"true={'HAL' if result.ground_truth_label else 'OK ':3s}  "
            f"type={result.hallucination_type}"
        )
        records.append(asdict(result))

    results_df = pd.DataFrame(records)
    y_true = results_df["ground_truth_label"]
    y_pred = results_df["is_hallucinated"].astype(int)

    print("\n" + "=" * 60)
    print("AGENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Provider          : Azure OpenAI ({_model})")
    print(f"Samples evaluated : {len(results_df)}")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision         : {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall            : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1                : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["Grounded", "Hallucinated"]))

    type_counts = results_df[results_df["is_hallucinated"]]["hallucination_type"].value_counts()
    if not type_counts.empty:
        print("Hallucination type distribution:")
        print(type_counts.to_string())
    print("=" * 60)

    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    return results_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Azure OpenAI hallucination detection agent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── single detect ─────────────────────────────────────────────────────────
    det = subparsers.add_parser("detect", help="Detect hallucination in one sample")
    det.add_argument("--context",  required=True)
    det.add_argument("--answer",   required=True)
    det.add_argument("--question", default="")
    det.add_argument("--verbose",  action="store_true")

    # ── qa_data evaluation ────────────────────────────────────────────────────
    evl = subparsers.add_parser("evaluate", help="Evaluate on qa_data.json subset")
    evl.add_argument("--dataset",   default=None)
    evl.add_argument("--n_samples", type=int,   default=20)
    evl.add_argument("--seed",      type=int,   default=42)
    evl.add_argument("--output",    default="results_qa_eval.csv")
    evl.add_argument("--verbose",   action="store_true")

    # ── HaluEval pipeline ─────────────────────────────────────────────────────
    pip = subparsers.add_parser("pipeline", help="Run full HaluEval pipeline")
    pip.add_argument("--n_samples", type=int,   default=50)
    pip.add_argument("--threshold", type=float, default=THRESHOLD)
    pip.add_argument("--seed",      type=int,   default=42)
    pip.add_argument("--output",    default="results_agent.csv")
    pip.add_argument("--verbose",   action="store_true")

    args = parser.parse_args()

    if args.command == "detect":
        result = run_agent(
            context=args.context,
            answer=args.answer,
            question=args.question,
            verbose=args.verbose,
        )
        print("\n── Detection Result ──────────────────────────────────────")
        print(f"Score                : {result.score:.3f}")
        print(f"Hallucinated         : {result.is_hallucinated}")
        print(f"Type                 : {result.hallucination_type}")
        print(f"Problematic sentences: {result.problematic_sentences}")
        print(f"Explanation          : {result.explanation}")
        print(f"Reasoning            : {result.reasoning[:300]}…"
              if len(result.reasoning) > 300 else f"Reasoning: {result.reasoning}")
        print(f"Agent steps          : {result.agent_steps}")

    elif args.command == "evaluate":
        evaluate_qa_data(
            dataset_path=args.dataset,
            n_samples=args.n_samples,
            seed=args.seed,
            verbose=args.verbose,
            output_csv=args.output,
        )

    elif args.command == "pipeline":
        run_pipeline(
            n_samples=args.n_samples,
            threshold=args.threshold,
            seed=args.seed,
            output_csv=args.output,
            verbose=args.verbose,
        )
