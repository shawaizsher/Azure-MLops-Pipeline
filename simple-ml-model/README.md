# Simple ML Model: Loan Default Prediction

This project is a minimal, interview-friendly machine learning example focused on the model only.

## Problem
Binary classification: predict whether a borrower is likely to default.

## Dataset
- **Name:** German Credit (`credit-g`)
- **Source:** OpenML (dataset id available through `sklearn.datasets.fetch_openml`)
- **Target:** `class` mapped to:
  - `1` = default risk (`bad`)
  - `0` = non-default (`good`)

## Model
- Preprocessing:
  - Numeric features: median imputation + standard scaling
  - Categorical features: most-frequent imputation + one-hot encoding
- Algorithm: Logistic Regression

## Quick Start
```bash
cd simple-ml-model
pip install -r requirements.txt
python src/train.py
python src/predict_sample.py
streamlit run app.py
mlflow ui --backend-store-uri .\\src\\mlruns --host 127.0.0.1 --port 5000 -w 1
```

## Output
Running training will create:
- `outputs/model.joblib` (trained pipeline)
- `outputs/metrics.json` (evaluation metrics)

The training script also logs to MLflow:
- Experiment: `loan-default-prediction`
- Registered model: `loan-default-classifier`

Open MLflow at `http://127.0.0.1:5000` to inspect runs and model versions.

## Notes
- Run `python src/train.py` first so the model artifact exists for inference.
- `src/predict_sample.py` prints predictions and default probabilities for 5 sample rows.

## Streamlit Web App
The app provides a simple UI to showcase the model:
- Train or reload the saved model from the sidebar
- View dataset summary and latest evaluation metrics
- Predict default risk for an existing sample row
- Predict default risk for custom applicant inputs

Run it with:
```bash
cd simple-ml-model
streamlit run app.py
```
