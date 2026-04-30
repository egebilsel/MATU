"""Evaluate the included SAUP-Multiple quick-start baselines."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


QUICK_START = Path(__file__).resolve().parents[1]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def accuracy_from_labels(labels) -> float:
    values = list(labels)
    return sum(str(x).lower() == "correct" for x in values) / max(len(values), 1)


def auroc_binary(y_error: np.ndarray, uncertainty: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_error, uncertainty))
    except Exception:
        pos = uncertainty[y_error == 1]
        neg = uncertainty[y_error == 0]
        wins = 0.0
        total = 0.0
        for p in pos:
            wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
            total += len(neg)
        return wins / total if total else float("nan")


def auarc(accuracy: np.ndarray, uncertainty: np.ndarray) -> float:
    order = np.argsort(uncertainty)
    sorted_acc = accuracy[order]
    coverages = np.arange(1, len(sorted_acc) + 1, dtype=float) / len(sorted_acc)
    acc_curve = np.cumsum(sorted_acc) / np.arange(1, len(sorted_acc) + 1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(acc_curve, coverages))
    return float(np.trapz(acc_curve, coverages))


def evaluate(name: str, scores: dict, labels: dict, *, invert: bool = False) -> None:
    keys = sorted(set(scores) & set(labels))
    uncertainty = []
    accuracy = []
    for key in keys:
        values = np.asarray(scores[key], dtype=float)
        score = float(np.nanmean(values))
        uncertainty.append(-score if invert else score)
        accuracy.append(accuracy_from_labels(labels[key]))

    uncertainty_arr = np.asarray(uncertainty, dtype=float)
    accuracy_arr = np.asarray(accuracy, dtype=float)
    y_error = (accuracy_arr < 1.0).astype(int)
    print(f"{name}:")
    print(f"  tasks: {len(keys)}")
    print(f"  AUROC: {auroc_binary(y_error, uncertainty_arr):.4f}")
    print(f"  AUARC: {auarc(accuracy_arr, uncertainty_arr):.4f}")


def evaluate_math_qwen() -> None:
    labels = load_pickle(QUICK_START / "results" / "accuracy_dict_Math_qwen2.5.pkl")
    saup = load_pickle(QUICK_START / "results" / "saup_scores_Math_qwen2.5.pkl")
    evaluate("MATH + Qwen2.5 + SAUP-Multiple", saup, labels, invert=True)


def evaluate_mmlu_autogen_qwen() -> None:
    labels = load_pickle(QUICK_START / "results" / "accuracy_dict_MMLU_Autogen_qwen2.5.pkl")
    saup = load_pickle(QUICK_START / "results" / "saup_scores_MMLU_Autogen_qwen2.5.pkl")
    evaluate("MMLU + AutoGen + Qwen2.5 + SAUP-Multiple", saup, labels, invert=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate included SAUP-Multiple quick-start baselines.")
    parser.add_argument(
        "--sample",
        choices=["math-qwen", "mmlu-autogen-qwen", "all"],
        default="all",
        help="Baseline sample to evaluate.",
    )
    args = parser.parse_args()

    if args.sample in {"math-qwen", "all"}:
        evaluate_math_qwen()
    if args.sample == "all":
        print()
    if args.sample in {"mmlu-autogen-qwen", "all"}:
        evaluate_mmlu_autogen_qwen()


if __name__ == "__main__":
    main()
