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


def predict_default_risk(model: object, data: pd.DataFrame) -> tuple[int, float]:
    prediction = int(model.predict(data)[0])
    probabilities = model.predict_proba(data)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Model predict_proba output is not a valid binary probability matrix.")

    return prediction, float(probabilities[0, 1])


def simplify_feature_name(feature_name: str) -> str:
    if "__" in feature_name:
        return feature_name.split("__", maxsplit=1)[1]
    return feature_name


def get_feature_contributions(
    model: object,
    input_row: pd.DataFrame,
    top_n: int = 6,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["model"]

        transformed = preprocessor.transform(input_row)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        row_values = transformed[0]
        coefficients = classifier.coef_[0]
        feature_names = preprocessor.get_feature_names_out()

        if len(row_values) != len(coefficients):
            return None, None

        contribution_frame = pd.DataFrame(
            {
                "Feature": [simplify_feature_name(name) for name in feature_names],
                "Contribution": row_values * coefficients,
            }
        )

        increasing_risk = (
            contribution_frame[contribution_frame["Contribution"] > 0]
            .sort_values("Contribution", ascending=False)
            .head(top_n)
            .copy()
        )
        decreasing_risk = (
            contribution_frame[contribution_frame["Contribution"] < 0]
            .sort_values("Contribution", ascending=True)
            .head(top_n)
            .copy()
        )

        increasing_risk["Contribution"] = increasing_risk["Contribution"].round(4)
        decreasing_risk["Contribution"] = decreasing_risk["Contribution"].round(4)

        return increasing_risk[["Feature", "Contribution"]], decreasing_risk[["Feature", "Contribution"]]
    except Exception:
        return None, None


def render_prediction_explainer(
    model: object,
    input_row: pd.DataFrame,
    prediction: int,
    default_probability: float,
) -> None:
    threshold = 0.5
    confidence = default_probability if prediction == 1 else 1 - default_probability

    st.markdown("### Why this prediction happened")
    st.progress(
        min(max(default_probability, 0.0), 1.0),
        text=f"Default risk score: {default_probability:.1%} (decision threshold: {threshold:.0%})",
    )

    detail_cols = st.columns(3)
    detail_cols[0].metric("Decision Threshold", f"{threshold:.0%}")
    detail_cols[1].metric("Distance From Threshold", f"{abs(default_probability - threshold):.1%}")
    detail_cols[2].metric("Prediction Confidence", f"{confidence:.1%}")

    higher_risk_df, lower_risk_df = get_feature_contributions(model, input_row)
    if higher_risk_df is None or lower_risk_df is None:
        st.info("Feature-level contribution view is unavailable for this model artifact.")
        return

    contribution_cols = st.columns(2)
    with contribution_cols[0]:
        st.markdown("#### Features Increasing Default Risk")
        if higher_risk_df.empty:
            st.caption("No positive-risk feature contributions for this input.")
        else:
            st.dataframe(higher_risk_df, use_container_width=True)

    with contribution_cols[1]:
        st.markdown("#### Features Reducing Default Risk")
        if lower_risk_df.empty:
            st.caption("No negative-risk feature contributions for this input.")
        else:
            st.dataframe(lower_risk_df, use_container_width=True)


def load_saved_model(features: pd.DataFrame) -> tuple[object | None, str | None]:
    if not MODEL_PATH.exists():
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        # Smoke-test inference to catch version-mismatched serialized models early.
        predict_default_risk(model, features.iloc[[0]].copy())
        return model, None
    except Exception as exc:
        return (
            None,
            "Saved model is incompatible with the current environment. "
            "Use 'Train / Refresh Model' to regenerate it. "
            f"Details: {type(exc).__name__}: {exc}",
        )


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

    with st.expander("What the app is doing under the hood", expanded=True):
        st.markdown(
            "1. Uses the German Credit dataset where `bad` means default risk and `good` means non-default.\n"
            "2. Builds a pipeline: impute missing values, scale numeric columns, one-hot encode categories.\n"
            "3. Trains Logistic Regression to output a default probability score.\n"
            "4. Uses a 50% threshold: score >= 50% => Default Risk, otherwise Non-default.\n"
            "5. Displays top feature contributions that push the score up or down for each prediction."
        )

    features, target = get_dataset()

    if "model" not in st.session_state:
        saved_model, load_error = load_saved_model(features)
        if saved_model is not None:
            st.session_state["model"] = saved_model
            st.session_state["metrics"] = load_saved_metrics()
        elif load_error:
            st.warning(load_error)

    with st.sidebar:
        st.header("Model Controls")

        if st.button("Train / Refresh Model", use_container_width=True):
            with st.spinner("Training model and saving artifacts..."):
                model, metrics = train_and_save_model(features, target)
            st.session_state["model"] = model
            st.session_state["metrics"] = metrics
            st.success("Model trained and saved to outputs/.")

        if st.button("Load Saved Model", use_container_width=True):
            model, load_error = load_saved_model(features)
            if model is None:
                if load_error:
                    st.error(load_error)
                else:
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

        try:
            sample_prediction, sample_probability = predict_default_risk(model, sample_row)
        except Exception as exc:
            st.error(
                "Model inference failed. Use 'Train / Refresh Model' to regenerate a compatible model. "
                f"Details: {type(exc).__name__}: {exc}"
            )
            st.stop()

        actual_label = int(target.iloc[sample_index])

        sample_cols = st.columns(3)
        sample_cols[0].metric("Prediction", "Default Risk" if sample_prediction == 1 else "Non-default")
        sample_cols[1].metric("Default Probability", f"{sample_probability:.1%}")
        sample_cols[2].metric("Actual Label", "Default Risk" if actual_label == 1 else "Non-default")

        render_prediction_explainer(model, sample_row, sample_prediction, sample_probability)

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

            try:
                custom_prediction, custom_probability = predict_default_risk(model, custom_frame)
            except Exception as exc:
                st.error(
                    "Model inference failed. Use 'Train / Refresh Model' to regenerate a compatible model. "
                    f"Details: {type(exc).__name__}: {exc}"
                )
                st.stop()

            result_cols = st.columns(2)
            result_cols[0].metric("Prediction", "Default Risk" if custom_prediction == 1 else "Non-default")
            result_cols[1].metric("Default Probability", f"{custom_probability:.1%}")

            render_prediction_explainer(model, custom_frame, custom_prediction, custom_probability)

            st.caption("Submitted custom feature values")
            st.dataframe(custom_frame.T, use_container_width=True)


if __name__ == "__main__":
    main()
