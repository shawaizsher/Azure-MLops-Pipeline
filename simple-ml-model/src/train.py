from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")

    features = dataset.data
    target = dataset.target.map({"good": 0, "bad": 1})

    if target.isnull().any():
        raise ValueError("Unexpected target values in dataset. Expected only 'good' and 'bad'.")

    return features, target.astype(int)


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    features, target = load_dataset()

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

    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print("Saved model to:", output_dir / "model.joblib")
    print("Saved metrics to:", output_dir / "metrics.json")
    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
