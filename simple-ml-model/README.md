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
```

## Output
Running training will create:
- `outputs/model.joblib` (trained pipeline)
- `outputs/metrics.json` (evaluation metrics)
