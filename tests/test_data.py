import pandas as pd
import pytest

from bangla_sentiment import DataError, load_split


def test_load_split_returns_clean_aligned_lists(toy_csv_dir):
    texts, labels = load_split(toy_csv_dir / "train.csv")
    assert len(texts) == len(labels) == 24
    assert set(labels) == {0, 1, 2}


def test_rows_with_bad_labels_or_empty_text_are_dropped(tmp_path):
    pd.DataFrame(
        {"Review": ["ভালো", "   ", "খারাপ", "অন্য"], "label": [2, 1, 9, 0]}
    ).to_csv(tmp_path / "x.csv", index=False)
    texts, labels = load_split(tmp_path / "x.csv")
    assert texts == ["ভালো", "অন্য"] and labels == [2, 0]


def test_max_rows_samples_reproducibly(toy_csv_dir):
    a = load_split(toy_csv_dir / "train.csv", max_rows=5)
    b = load_split(toy_csv_dir / "train.csv", max_rows=5)
    assert len(a[0]) == 5 and a == b


def test_missing_file_and_missing_columns(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_split(tmp_path / "nope.csv")
    pd.DataFrame({"text": ["x"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(DataError, match="columns"):
        load_split(tmp_path / "bad.csv")
