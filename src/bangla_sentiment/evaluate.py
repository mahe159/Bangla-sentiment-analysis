"""Evaluation: accuracy, macro-F1, per-class scores, confusion matrix, majority baseline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import LABELS


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Metrics for 3-class sentiment. Macro-F1 is the headline number because the
    classes are very imbalanced, which makes plain accuracy look better than it is."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    classes = sorted(LABELS)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            LABELS[c]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
        "class_order": [LABELS[c] for c in classes],
    }


def majority_baseline(y_train: list[int], y_true: list[int]) -> dict:
    """Metrics for always predicting the most common training class."""
    majority = Counter(y_train).most_common(1)[0][0]
    out = compute_metrics(y_true, [majority] * len(y_true))
    out["predicts"] = LABELS[majority]
    return out


def save_report(metrics: dict, out_dir: str | Path, name: str = "metrics") -> list[Path]:
    """Write ``<name>.json`` and ``<name>_confusion_matrix.png``; return the paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{name}.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = np.array(metrics["confusion_matrix"])
    order = metrics["class_order"]
    row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (row %)")
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center",
                    color="white" if row_pct[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    png_path = out / f"{name}_confusion_matrix.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [json_path, png_path]
