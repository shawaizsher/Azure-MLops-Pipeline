from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.train import build_pipeline, evaluate_model, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, pd.Series]:
    return load_dataset()


def load_saved_model() -> object | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_saved_metrics() -> dict[str, float] | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def train_and_save_model(features: pd.DataFrame, target: pd.Series) -> tuple[object, dict[str, float]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(x_train, y_train)

    metrics = evaluate_model(pipeline, x_test, y_test)

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return pipeline, metrics


def render_metrics(metrics: dict[str, float]) -> None:
    st.subheader("Latest Evaluation Metrics")
    columns = st.columns(5)
    metric_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for idx, metric_name in enumerate(metric_order):
        if metric_name in metrics:
            columns[idx].metric(metric_name.upper(), f"{metrics[metric_name]:.3f}")


def main() -> None:
    st.set_page_config(page_title="Loan Default Model Demo", page_icon="💳", layout="wide")

    st.title("Loan Default Prediction Demo")
    st.caption("Streamlit showcase for a simple Logistic Regression pipeline using the German Credit dataset.")

    features, target = get_dataset()

    if "model" not in st.session_state:
        saved_model = load_saved_model()
        if saved_model is not None:
            st.session_state["model"] = saved_model
            st.session_state["metrics"] = load_saved_metrics()

    with st.sidebar:
        st.header("Model Controls")

        if st.button("Train / Refresh Model", use_container_width=True):
            with st.spinner("Training model and saving artifacts..."):
                model, metrics = train_and_save_model(features, target)
            st.session_state["model"] = model
            st.session_state["metrics"] = metrics
            st.success("Model trained and saved to outputs/.")

        if st.button("Load Saved Model", use_container_width=True):
            model = load_saved_model()
            if model is None:
                st.error("No saved model found. Train the model first.")
            else:
                st.session_state["model"] = model
                st.session_state["metrics"] = load_saved_metrics()
                st.success("Saved model loaded.")

        if not MODEL_PATH.exists():
            st.info("No model artifact yet. Click 'Train / Refresh Model' to create one.")

    summary_cols = st.columns(3)
    summary_cols[0].metric("Dataset Rows", f"{len(features):,}")
    summary_cols[1].metric("Dataset Features", f"{features.shape[1]}")
    summary_cols[2].metric("Default Rate", f"{target.mean():.1%}")

    model = st.session_state.get("model")
    if model is None:
        st.warning("Model is not loaded. Use sidebar controls to train or load a saved model.")
        st.stop()

    metrics = st.session_state.get("metrics")
    if metrics:
        render_metrics(metrics)

    sample_tab, custom_tab = st.tabs(["Sample Prediction", "Custom Applicant"])

    with sample_tab:
        st.subheader("Predict Using a Real Dataset Row")
        sample_index = st.slider("Choose a row index", min_value=0, max_value=len(features) - 1, value=0)
        sample_row = features.iloc[[sample_index]].copy()

        sample_prediction = int(model.predict(sample_row)[0])
        sample_probability = float(model.predict_proba(sample_row)[0, 1])
        actual_label = int(target.iloc[sample_index])

        sample_cols = st.columns(3)
        sample_cols[0].metric("Prediction", "Default Risk" if sample_prediction == 1 else "Non-default")
        sample_cols[1].metric("Default Probability", f"{sample_probability:.1%}")
        sample_cols[2].metric("Actual Label", "Default Risk" if actual_label == 1 else "Non-default")

        st.caption("Selected applicant feature values")
        st.dataframe(sample_row.T, use_container_width=True)

    with custom_tab:
        st.subheader("Predict Using Custom Applicant Inputs")
        numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

        custom_values: dict[str, object] = {}
        left_col, right_col = st.columns(2)

        with st.form("custom_prediction_form"):
            for idx, feature_name in enumerate(numeric_features):
                default_value = float(features[feature_name].median())
                current_col = left_col if idx % 2 == 0 else right_col
                with current_col:
                    custom_values[feature_name] = st.number_input(
                        feature_name,
                        value=default_value,
                        step=1.0,
                        format="%.2f",
                    )

            for idx, feature_name in enumerate(categorical_features):
                options = sorted(features[feature_name].dropna().astype(str).unique().tolist())
                current_col = left_col if (idx + len(numeric_features)) % 2 == 0 else right_col
                with current_col:
                    custom_values[feature_name] = st.selectbox(feature_name, options=options, index=0)

            submitted = st.form_submit_button("Predict Default Risk", use_container_width=True)

        if submitted:
            custom_frame = pd.DataFrame([custom_values], columns=features.columns)
            custom_prediction = int(model.predict(custom_frame)[0])
            custom_probability = float(model.predict_proba(custom_frame)[0, 1])

            result_cols = st.columns(2)
            result_cols[0].metric("Prediction", "Default Risk" if custom_prediction == 1 else "Non-default")
            result_cols[1].metric("Default Probability", f"{custom_probability:.1%}")


if __name__ == "__main__":
    main()
