from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_openml


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "outputs" / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Run `python src/train.py` first to create outputs/model.joblib."
        )

    model = joblib.load(model_path)

    dataset = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    sample: pd.DataFrame = dataset.data.head(5).copy()

    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)[:, 1]

    print("Sample predictions (1 = default risk, 0 = non-default):")
    for idx, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
        print(f"  row_{idx}: prediction={int(pred)}, default_probability={prob:.4f}")


if __name__ == "__main__":
    main()
