import json

import pytest

from bangla_sentiment.cli import main
from bangla_sentiment.evaluate import compute_metrics, majority_baseline, save_report


def test_metrics_perfect_and_imbalanced_cases():
    perfect = compute_metrics([0, 1, 2, 2], [0, 1, 2, 2])
    assert perfect["accuracy"] == 1.0 and perfect["macro_f1"] == 1.0
    # Always predicting the majority class: high accuracy, low macro-F1.
    skewed = compute_metrics([2] * 8 + [0, 1], [2] * 10)
    assert skewed["accuracy"] == 0.8 and skewed["macro_f1"] < 0.4


def test_majority_baseline_uses_training_majority():
    out = majority_baseline([2, 2, 2, 0], [0, 1, 2])
    assert out["predicts"] == "positive" and out["accuracy"] == pytest.approx(1 / 3)


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([1], [1, 2])


def test_save_report_writes_json_and_png(tmp_path):
    paths = save_report(compute_metrics([0, 1, 2], [0, 2, 2]), tmp_path)
    assert all(p.exists() for p in paths)
    assert json.loads(paths[0].read_text())["n"] == 3


def test_cli_train_evaluate_predict_end_to_end(toy_csv_dir, tmp_path, capsys):
    model = str(tmp_path / "m.joblib")
    assert main(["train", "--data-dir", str(toy_csv_dir), "--model", model]) == 0
    assert main(["evaluate", "--data-dir", str(toy_csv_dir), "--model", model,
                 "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "metrics_test.json").exists()
    assert main(["predict", "অসাধারণ বই", "--model", model]) == 0
    assert "Prediction:" in capsys.readouterr().out


def test_cli_reports_errors_instead_of_crashing(tmp_path, capsys):
    assert main(["train", "--data-dir", str(tmp_path)]) == 1
    assert "Expected" in capsys.readouterr().err
    assert main(["predict", "ভালো", "--model", str(tmp_path / "missing.joblib")]) == 1
