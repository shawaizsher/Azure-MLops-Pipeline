# Azure-MLops-Pipeline

Minimal loan default prediction project with MLflow tracking and model comparison.

## Project Layout

- `simple-ml-model/`: main application and training code
- `simple-ml-model/src/train.py`: trains and logs models to MLflow
- `simple-ml-model/app.py`: Streamlit demo app

## What It Does

- Loads the German Credit dataset (`credit-g`) from OpenML
- Trains two classification models:
	- Logistic Regression
	- Random Forest
- Logs parameters, metrics, and model artifacts to MLflow
- Registers model versions in the MLflow Model Registry
- Saves output artifacts under `simple-ml-model/outputs/`

## Quick Start

```bash
cd simple-ml-model
pip install -r requirements.txt
python src/train.py
```

## Run the UI Apps

### 1) MLflow UI (compare model runs)

```bash
cd simple-ml-model
mlflow ui --backend-store-uri .\\src\\mlruns --host 127.0.0.1 --port 5000 -w 1
```

Open: `http://127.0.0.1:5000`

In the **loan-default-prediction** experiment, compare metrics for both runs.

### 2) Streamlit App

```bash
cd simple-ml-model
streamlit run app.py
```

## Notes

- On Windows, using `-w 1` for MLflow UI avoids intermittent multi-worker socket issues.
- MLflow filesystem backend works for local development; consider SQLite backend for longer-term usage.

## Azure App Service Container Deployment

- Use a Linux App Service configured for **Container** publishing.
- The workflow builds and pushes a container image to GHCR, then points the web app to that image.
- If you still see `Container Image: mcr.microsoft.com/appsvc/staticsite:latest`, the deployment did not update the web app image yet.
- Ensure the deployed GHCR package is pullable by App Service:
	- Preferred: make the package public.
	- Or configure container registry credentials for GHCR in App Service container settings.

- By Shawaiz Sher