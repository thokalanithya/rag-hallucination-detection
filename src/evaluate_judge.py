"""
evaluate_judge.py
-----------------
Evaluate how well a judge model detects hallucinations on the HaluEval QA dataset.

Supports two judges:
  --judge prometheus   Mistral via Ollama with Prometheus rubric prompt (local, free)
  --judge gpt4o        GPT-4o via Azure OpenAI (cloud)

Each sample in HaluEval has:
  context  : the knowledge passage
  question : the question
  answer   : right_answer (label=0) or hallucinated_answer (label=1)

The judge predicts 0 (grounded) or 1 (hallucinated).
We compare against ground truth and report binary classification metrics.

Usage
-----
  python src/evaluate_judge.py --judge prometheus --n_samples 50
  python src/evaluate_judge.py --judge gpt4o --n_samples 50
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from data.halueval_loader import load_halueval_qa


def evaluate(judge: str, n_samples: int, seed: int, output_csv: str) -> pd.DataFrame:

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading {n_samples} samples from HaluEval QA dataset...")
    df = load_halueval_qa(n_samples=n_samples, seed=seed)

    # ── Load judge ─────────────────────────────────────────────────────────────
    if judge == "prometheus":
        from agent.prometheus_judge import run_prometheus_judge
        judge_fn  = lambda q, c, a: run_prometheus_judge(question=q, context=c, answer=a)
        judge_label = "Mistral (Prometheus rubric, Ollama)"
    else:
        from agent.hallucination_agent import run_agent
        judge_fn  = lambda q, c, a: run_agent(context=c, answer=a, question=q)
        judge_label = "GPT-4o (Azure OpenAI)"

    print(f"Judge : {judge_label}")
    print(f"Samples: {len(df)}  ({len(df[df['label']==0])} grounded + {len(df[df['label']==1])} hallucinated)\n")
    print("-" * 70)

    records = []

    for i, row in df.iterrows():
        true_label = row["label"]
        true_str   = "HALL" if true_label == 1 else "OK  "

        print(f"[{i+1:>3}/{len(df)}] true={true_str}  Q: {row['question'][:50]}...", end=" ", flush=True)

        result = judge_fn(row["question"], row["context"], row["answer"])

        pred      = int(result.is_hallucinated)
        correct   = pred == true_label
        hall_type = result.hallucination_type if result.is_hallucinated else "none"

        print(
            f"pred={'HALL' if pred else 'OK  '}  "
            f"score={result.score:.2f}  "
            f"type={hall_type:<20}  "
            f"{'✓' if correct else '✗'}"
        )

        records.append({
            "question":          row["question"],
            "context":           row["context"],
            "answer":            row["answer"],
            "answer_type":       row["answer_type"],
            "ground_truth":      true_label,
            "predicted":         pred,
            "score":             result.score,
            "hallucination_type": hall_type,
            "explanation":       result.explanation,
            "correct":           correct,
        })

        time.sleep(0.1)  # polite pause

    results_df = pd.DataFrame(records)
    y_true     = results_df["ground_truth"]
    y_pred     = results_df["predicted"]

    # ── Metrics ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Judge             : {judge_label}")
    print(f"Samples           : {len(results_df)}  "
          f"({len(results_df[results_df['ground_truth']==0])} grounded + "
          f"{len(results_df[results_df['ground_truth']==1])} hallucinated)")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision         : {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall            : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1                : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("                  Pred:OK   Pred:HALL")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  True: OK        {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  True: HALL      {cm[1][0]:>6}    {cm[1][1]:>6}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["Grounded", "Hallucinated"]))

    type_counts = results_df[results_df["predicted"] == 1]["hallucination_type"].value_counts()
    if not type_counts.empty:
        print("Hallucination type distribution (predicted positives):")
        print(type_counts.to_string())

    print("=" * 60)

    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hallucination judge on HaluEval")
    parser.add_argument("--judge",     default="prometheus", choices=["prometheus", "gpt4o"],
                        help="Judge to evaluate (default: prometheus)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default="results_judge_eval.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    evaluate(
        judge      = args.judge,
        n_samples  = args.n_samples,
        seed       = args.seed,
        output_csv = args.output,
    )
