"""TF-IDF + linear classifier for 3-class Bangla sentiment, with simple explanations."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .data import LABELS
from .preprocess import normalize

# \w does not match Bengali vowel signs, which would split words apart,
# so tokens are runs of word characters or anything in the Bengali Unicode block.
TOKEN_PATTERN = r"[\w\u0980-\u09FF]+"
MODEL_KINDS = ("logreg", "svm")


class ModelError(Exception):
    """Raised for unusable model files or inputs."""


def build_pipeline(
    kind: str = "logreg",
    max_word_features: int = 300_000,
    max_char_features: int = 150_000,
    min_df: int = 2,
) -> Pipeline:
    """Create an untrained pipeline. ``class_weight='balanced'`` is used because
    BanglaBook is heavily skewed towards positive reviews."""
    if kind not in MODEL_KINDS:
        raise ModelError(f"Unknown model kind {kind!r}; choose from {MODEL_KINDS}")
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=max_word_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    min_df=min_df,
                    max_features=max_char_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    if kind == "logreg":
        clf = LogisticRegression(max_iter=300, class_weight="balanced", C=2.0)
    else:
        clf = LinearSVC(class_weight="balanced", C=0.5)
    return Pipeline([("features", features), ("clf", clf)])


def train(texts: list[str], labels: list[int], kind: str = "logreg", **kwargs) -> Pipeline:
    if len(texts) != len(labels) or not texts:
        raise ModelError("texts and labels must be non-empty and the same length")
    if len(set(labels)) < 2:
        raise ModelError("Need at least two different classes to train")
    pipeline = build_pipeline(kind, **kwargs)
    pipeline.fit(texts, labels)
    return pipeline


def scores(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    """Per-class scores, one row per text. Probabilities for logistic regression,
    softmax-normalised margins for the SVM (not calibrated probabilities)."""
    clean = [normalize(t) for t in texts]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba(clean)
    margins = pipeline.decision_function(clean)
    margins = margins - margins.max(axis=1, keepdims=True)
    exp = np.exp(margins)
    return exp / exp.sum(axis=1, keepdims=True)


def predict(pipeline: Pipeline, texts: list[str]) -> list[int]:
    return pipeline.predict([normalize(t) for t in texts]).tolist()


def explain(pipeline: Pipeline, text: str, top_n: int = 8) -> dict:
    """Predict one text and list the word/phrase features that pushed hardest towards
    the predicted class (TF-IDF value x classifier weight). Word features only, because
    character n-grams are not readable."""
    if not normalize(text):
        raise ModelError("Please enter some text.")
    clean = normalize(text)
    feats = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    row = feats.transform([clean])
    prob = scores(pipeline, [text])[0]
    class_index = int(np.argmax(prob))
    label = int(clf.classes_[class_index])

    names = feats.get_feature_names_out()
    coef_row = clf.coef_[class_index] if clf.coef_.shape[0] > 1 else clf.coef_[0]
    contrib = row.multiply(coef_row).tocoo()
    items = [
        (str(names[j]).removeprefix("word__"), float(v))
        for j, v in zip(contrib.col, contrib.data)
        if str(names[j]).startswith("word__")
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return {
        "label": LABELS[label],
        "scores": {LABELS[int(c)]: float(p) for c, p in zip(clf.classes_, prob)},
        "top_features": items[:top_n],
    }


def top_features_per_class(pipeline: Pipeline, top_n: int = 15) -> dict[str, list[str]]:
    """Highest-weighted word features for each class (for the README / docs)."""
    feats = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    names = np.array([str(n) for n in feats.get_feature_names_out()])
    is_word = np.array([n.startswith("word__") for n in names])
    result = {}
    for idx, cls in enumerate(clf.classes_):
        weights = np.where(is_word, clf.coef_[idx], -np.inf)
        best = np.argsort(weights)[::-1][:top_n]
        result[LABELS[int(cls)]] = [names[i].removeprefix("word__") for i in best]
    return result


def save(pipeline: Pipeline, path: str | Path, meta: dict | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "meta": meta or {}}, out)


def load(path: str | Path) -> tuple[Pipeline, dict]:
    """Load a model saved by :func:`save`. Only open files you created yourself:
    joblib files are pickles and can run code when loaded."""
    p = Path(path)
    if not p.is_file():
        raise ModelError(f"Model file not found: {p}. Train one first (see README).")
    try:
        blob = joblib.load(p)
        return blob["pipeline"], blob.get("meta", {})
    except Exception as exc:
        raise ModelError(f"Could not load model {p.name}: {exc}") from exc
