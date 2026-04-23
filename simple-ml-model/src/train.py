from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
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


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model: object,
) -> Pipeline:
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


def configure_mlflow(project_root: Path) -> Path:
    tracking_dir = project_root / "src" / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("loan-default-prediction")

    return tracking_dir


def train_and_log_model(
    *,
    model_key: str,
    registered_model_name: str,
    estimator: object,
    numeric_features: list[str],
    categorical_features: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> tuple[Pipeline, dict[str, float], str, str | None]:
    pipeline = build_pipeline(numeric_features, categorical_features, estimator)
    run_name = f"{model_key}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        base_params = {
            "model_type": model_key,
            "dataset": "credit-g",
            "test_size": 0.2,
            "random_state": 42,
        }
        estimator_params = estimator.get_params()
        interesting_keys = ["max_iter", "n_estimators", "max_depth", "min_samples_leaf", "n_jobs"]
        for key in interesting_keys:
            if key in estimator_params and estimator_params[key] is not None:
                base_params[key] = estimator_params[key]

        mlflow.log_params(base_params)

        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        mlflow.log_metrics(metrics)

        model_path = output_dir / f"model_{model_key}.joblib"
        metrics_path = output_dir / f"metrics_{model_key}.json"

        joblib.dump(pipeline, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

        model_registry_name: str | None = registered_model_name
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                registered_model_name=registered_model_name,
            )
        except Exception:
            # Fallback for tracking stores where model registry is unavailable.
            model_registry_name = None
            mlflow.sklearn.log_model(sk_model=pipeline, name="model")

        active_run = mlflow.active_run()
        run_id = ""
        if active_run is not None:
            run_id = active_run.info.run_id

    return pipeline, metrics, run_id, model_registry_name


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = configure_mlflow(project_root)

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

    model_configs: dict[str, tuple[object, str]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            "loan-default-classifier-logreg",
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "loan-default-classifier-random-forest",
        ),
    }

    comparison_results: dict[str, dict[str, float]] = {}
    run_ids: dict[str, str] = {}
    registered_models: dict[str, str | None] = {}
    baseline_model_key = "logistic_regression"

    for model_key, (estimator, registry_name) in model_configs.items():
        pipeline, metrics, run_id, registered_model_name = train_and_log_model(
            model_key=model_key,
            registered_model_name=registry_name,
            estimator=estimator,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            output_dir=output_dir,
        )
        comparison_results[model_key] = metrics
        run_ids[model_key] = run_id
        registered_models[model_key] = registered_model_name

        if model_key == baseline_model_key:
            joblib.dump(pipeline, output_dir / "model.joblib")
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison_path = output_dir / "metrics_comparison.json"
    comparison_path.write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")

    print("Training complete for model comparison.")
    print("MLflow tracking directory:", tracking_dir)
    print("Saved baseline model to:", output_dir / "model.joblib")
    print("Saved baseline metrics to:", output_dir / "metrics.json")
    print("Saved comparison metrics to:", comparison_path)
    print("MLflow run ids:")
    for model_key, run_id in run_ids.items():
        print(f"  {model_key}: {run_id}")
    print("Registered models:")
    for model_key, registered_model_name in registered_models.items():
        print(f"  {model_key}: {registered_model_name}")
    print("Model metrics:")
    for model_key, metrics in comparison_results.items():
        print(
            f"  {model_key}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
