from pathlib import Path

from streamlit.testing.v1 import AppTest

from bangla_sentiment import save, train

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_shows_training_help_when_model_missing(tmp_path):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    # default model path is relative to the working directory; none exists in tests
    at.sidebar.text_input[0].set_value(str(tmp_path / "missing.joblib")).run()
    assert not at.exception
    assert any("Model file not found" in e.value for e in at.error)


def test_app_analyzes_text_with_a_trained_model(tmp_path, toy_data):
    texts, labels = toy_data
    model_file = tmp_path / "m.joblib"
    save(train(texts, labels, min_df=1), model_file, meta={"kind": "logreg"})

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].set_value(str(model_file)).run()
    at.text_area[0].set_value("অসাধারণ চমৎকার বই").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Prediction: positive" in s.value for s in at.subheader)
