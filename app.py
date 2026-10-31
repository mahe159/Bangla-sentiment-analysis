"""Streamlit app for Bangla review sentiment.  Run:  streamlit run app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from bangla_sentiment import LABELS, ModelError, explain, load, scores  # noqa: E402

DEFAULT_MODEL = "models/sentiment.joblib"
MAX_BATCH_ROWS = 5000

st.set_page_config(page_title="Bangla Sentiment Analysis", page_icon="💬")
st.title("💬 Bangla Sentiment Analysis")
st.caption("Classifies Bangla text as negative, neutral or positive. Runs locally on CPU.")


@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str):
    return load(path)


model_path = st.sidebar.text_input("Model file", DEFAULT_MODEL)
try:
    pipeline, meta = get_model(model_path)
except ModelError as exc:
    st.error(str(exc))
    st.code(
        "python -m bangla_sentiment train --data-dir path/to/BanglaBook/data/csv",
        language="bash",
    )
    st.stop()

st.sidebar.write(f"Model type: {meta.get('kind', 'unknown')}")
st.sidebar.caption(
    "Labels in the training data come from star ratings, so the model learns "
    "'rating-like' sentiment, which is not always the same as the wording."
)

single, batch = st.tabs(["Single text", "Batch (CSV)"])

with single:
    text = st.text_area("Bangla text", height=120, placeholder="এখানে রিভিউ লিখুন...")
    if st.button("Analyze", type="primary"):
        try:
            result = explain(pipeline, text)
        except ModelError as exc:
            st.warning(str(exc))
        else:
            st.subheader(f"Prediction: {result['label']}")
            st.bar_chart(pd.Series(result["scores"], name="score"))
            if result["top_features"]:
                st.write("Words and phrases that supported this prediction:")
                st.table(
                    pd.DataFrame(result["top_features"], columns=["feature", "weight"])
                )

with batch:
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()
        if df.empty:
            st.warning("The file has no rows.")
            st.stop()
        column = st.selectbox("Column containing the text", list(df.columns))
        if len(df) > MAX_BATCH_ROWS:
            st.warning(f"Only the first {MAX_BATCH_ROWS} rows will be scored.")
            df = df.head(MAX_BATCH_ROWS)
        if st.button("Score file"):
            texts = df[column].fillna("").astype(str).tolist()
            probs = scores(pipeline, texts)
            classes = list(pipeline.named_steps["clf"].classes_)
            out = df.copy()
            out["predicted_sentiment"] = [
                LABELS[int(classes[i])] for i in probs.argmax(axis=1)
            ]
            out["confidence"] = probs.max(axis=1).round(3)
            st.dataframe(out.head(50))
            st.download_button(
                "Download results (CSV)",
                out.to_csv(index=False).encode("utf-8-sig"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
