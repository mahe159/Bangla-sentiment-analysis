"""Command line: train, evaluate and predict.

    python -m bangla_sentiment train    --data-dir path/to/BanglaBook/data/csv
    python -m bangla_sentiment evaluate --data-dir path/to/BanglaBook/data/csv
    python -m bangla_sentiment predict  "বইটি খুব ভালো লেগেছে"
"""

from __future__ import annotations

import argparse
import sys
import time

from . import evaluate as ev
from . import model as mdl
from .data import DataError, find_split, load_split

DEFAULT_MODEL = "models/sentiment.joblib"


def _train(args) -> int:
    texts, labels = load_split(find_split(args.data_dir, "train"), args.max_train)
    print(f"Training on {len(texts)} reviews ({args.kind})...")
    started = time.time()
    pipeline = mdl.train(texts, labels, kind=args.kind)
    seconds = time.time() - started
    mdl.save(pipeline, args.model, meta={"kind": args.kind, "n_train": len(texts)})
    print(f"Saved {args.model} (training took {seconds:.0f}s)")
    return 0


def _evaluate(args) -> int:
    pipeline, meta = mdl.load(args.model)
    split = "validation" if args.split == "validation" else "test"
    texts, labels = load_split(find_split(args.data_dir, split), args.max_rows)
    metrics = ev.compute_metrics(labels, mdl.predict(pipeline, texts))
    train_texts, train_labels = load_split(find_split(args.data_dir, "train"), args.max_rows)
    baseline = ev.majority_baseline(train_labels, labels)
    metrics["model_kind"] = meta.get("kind")
    metrics["baseline_majority"] = {
        k: baseline[k] for k in ("accuracy", "macro_f1", "predicts")
    }
    paths = ev.save_report(metrics, args.out_dir, name=f"metrics_{split}")
    print(f"{split}: n={metrics['n']}  accuracy={metrics['accuracy']:.4f}  "
          f"macro-F1={metrics['macro_f1']:.4f}")
    print(f"majority baseline ({baseline['predicts']}): accuracy={baseline['accuracy']:.4f}  "
          f"macro-F1={baseline['macro_f1']:.4f}")
    for name, row in metrics["per_class"].items():
        print(f"  {name:9s} P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f} (n={row['support']})")
    print("Wrote:", ", ".join(str(p) for p in paths))
    return 0


def _predict(args) -> int:
    pipeline, _ = mdl.load(args.model)
    result = mdl.explain(pipeline, args.text)
    print(f"Prediction: {result['label']}")
    for name, value in result["scores"].items():
        print(f"  {name:9s} {value:.3f}")
    if result["top_features"]:
        print("Top supporting words:", ", ".join(f for f, _ in result["top_features"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bangla_sentiment", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train and save a model")
    t.add_argument("--data-dir", required=True)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--kind", choices=mdl.MODEL_KINDS, default="logreg")
    t.add_argument("--max-train", type=int, default=0, help="use a random sample (0 = all)")
    t.set_defaults(func=_train)

    e = sub.add_parser("evaluate", help="score a saved model on validation or test data")
    e.add_argument("--data-dir", required=True)
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--split", choices=["validation", "test"], default="test")
    e.add_argument("--max-rows", type=int, default=0)
    e.add_argument("--out-dir", default="docs")
    e.set_defaults(func=_evaluate)

    p = sub.add_parser("predict", help="classify one review")
    p.add_argument("text")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(func=_predict)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DataError, mdl.ModelError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
