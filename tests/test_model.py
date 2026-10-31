import pytest

from bangla_sentiment import ModelError, explain, load, predict, save, scores, train
from bangla_sentiment.model import TOKEN_PATTERN, build_pipeline, top_features_per_class

import re


def test_token_pattern_keeps_bengali_words_whole():
    # \\w alone would split at vowel signs; our pattern must not.
    assert re.findall(TOKEN_PATTERN, "অসাধারণ বইটি") == ["অসাধারণ", "বইটি"]


@pytest.mark.parametrize("kind", ["logreg", "svm"])
def test_model_fits_toy_data_and_predicts_valid_labels(toy_data, kind):
    texts, labels = toy_data
    model = train(texts, labels, kind=kind, min_df=1)
    preds = predict(model, texts)
    assert set(preds) <= {0, 1, 2}
    assert sum(p == y for p, y in zip(preds, labels)) / len(labels) > 0.8  # sanity only


def test_scores_are_probability_like(toy_data):
    texts, labels = toy_data
    model = train(texts, labels, min_df=1)
    s = scores(model, ["বইটি খুব ভালো", "একদম বাজে"])
    assert s.shape == (2, 3)
    assert s.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_explain_returns_label_scores_and_words(toy_data):
    texts, labels = toy_data
    model = train(texts, labels, min_df=1)
    out = explain(model, "অসাধারণ চমৎকার বই")
    assert out["label"] == "positive"
    assert abs(sum(out["scores"].values()) - 1) < 1e-6
    assert out["top_features"] and all(isinstance(w, str) for w, _ in out["top_features"])


def test_explain_rejects_empty_text(toy_data):
    texts, labels = toy_data
    model = train(texts, labels, min_df=1)
    with pytest.raises(ModelError):
        explain(model, "   ")


def test_train_validates_inputs():
    with pytest.raises(ModelError):
        train([], [])
    with pytest.raises(ModelError):
        train(["ক", "খ"], [1, 1])
    with pytest.raises(ModelError):
        build_pipeline("deep-magic")


def test_save_and_load_round_trip(tmp_path, toy_data):
    texts, labels = toy_data
    model = train(texts, labels, min_df=1)
    save(model, tmp_path / "m.joblib", meta={"kind": "logreg"})
    loaded, meta = load(tmp_path / "m.joblib")
    assert meta["kind"] == "logreg"
    assert predict(loaded, texts) == predict(model, texts)


def test_load_missing_or_corrupt_file(tmp_path):
    with pytest.raises(ModelError, match="not found"):
        load(tmp_path / "none.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelError, match="Could not load"):
        load(bad)


def test_top_features_per_class_lists_all_classes(toy_data):
    texts, labels = toy_data
    model = train(texts, labels, min_df=1)
    top = top_features_per_class(model, top_n=3)
    assert set(top) == {"negative", "neutral", "positive"} and all(len(v) == 3 for v in top.values())
