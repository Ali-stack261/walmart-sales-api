# 🛒 Walmart Sales Forecasting — MLOps Project

An end-to-end Machine Learning and MLOps application that predicts Walmart
weekly sales using historical sales data and external economic indicators.
Built with production-ready practices: automated testing, experiment
tracking, containerization, CI/CD, and live cloud deployment — including a
public API and an interactive UI.

## 🌐 Live Demo

Component
URL

**Interactive UI** (Streamlit)
https://walmart-sales-api-euwaxsx5clntsoxmcvrdff.streamlit.app

**API** (FastAPI Cloud)
https://walmart-sales-api-ed2ca0cc.fastapicloud.dev

**API docs** (Swagger UI)
https://walmart-sales-api-ed2ca0cc.fastapicloud.dev/docs

> Both are hosted on free tiers. The API may take a few seconds to respond on
the first request after a period of inactivity.

## 🎯 Features

- **Machine Learning Pipeline** — trains and compares Random Forest, XGBoost,
and LightGBM regressors, automatically selecting the best model by
validation RMSE.
- **MLOps Integration** — experiment tracking and model logging via MLflow.
- **API & Serving** — REST API built with FastAPI: single predictions, batch
predictions, health checks, and a lightweight data-drift report.
- **Interactive UI** — a Streamlit app for exploring predictions without
writing any code: sliders for single predictions, CSV upload for batch
scoring, and a drift-report viewer.
- **Engineering Best Practices** — Docker containerization, automated tests
via Pytest, and CI/CD via GitHub Actions (runs the full pipeline —
ingest → preprocess → feature engineering → train — before tests on every
push).
- **Monitoring** — prediction logging (`logs/predictions.jsonl`) and a
drift-report endpoint comparing live inputs against the training
distribution.

## 🏗️ Architecture

```
graph TD;
    GitHub-->GitHubActions[GitHub Actions CI];
    GitHubActions-->Pipeline[Ingest → Preprocess → Features → Train];
    Pipeline-->MLflow[MLflow Tracking];
    Pipeline-->Model[(best_model.joblib)];
    GitHubActions-->Pytest;
    GitHub-->FastAPICloud[FastAPI Cloud];
    Model-->FastAPICloud;
    FastAPICloud-->API[REST API: /predict, /predict/batch, /drift, /health];
    API-->StreamlitUI[Streamlit UI];
    GitHub-->Docker[Dockerfile];
    Docker-->SelfHost[Self-hosted / any container platform];
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Git
- Docker (optional, for containerized runs)

### 2. Clone and install

```
git clone https://github.com/Ali-stack261/walmart-sales-api.git
cd walmart-sales-api
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt
```

> If you prefer the `pyproject.toml` project format instead of
> `requirements.txt`, a modern toolchain such as `uv` or `pip install .` will
> pick up dependencies from `[project]` in `pyproject.toml`.

### 3. Environment configuration

The app runs with sensible defaults out of the box — no `.env` file is
required. If you want to override anything (e.g. `MLFLOW_TRACKING_URI`),
create a `.env` file at the repo root; see `src/config.py` for the full list
of supported variables.

### 4. Run the pipeline

```
python -m src.ingest              # load raw data
python -m src.preprocessing       # clean data
python -m src.feature_engineering # engineer features, train/val/test split
python -m src.train                # train models, log to MLflow, save best model
```

This produces `models/best_model.joblib` and `models/feature_columns.json`,
which the API needs to serve predictions.

### 5. Run the API locally

```
uvicorn main:app --reload
```

(equivalent to `uvicorn api.main:app --reload` — see [Project Structure](https://claude.ai/chat/130736cc-1681-48dd-ac32-fa6495ed3cd4#-project-structure) for why both exist)

The API will be live at `http://127.0.0.1:8000`, with interactive docs at
`http://127.0.0.1:8000/docs`.

### 6. Run the UI locally

```
pip install streamlit requests
API_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py
```

### Other useful commands

```
black .                    # format code
ruff check .                # lint
pytest                      # run tests
mlflow ui                   # view experiment tracking dashboard
python -m src.drift_monitor # generate a drift report manually
```

## 📡 API Endpoints

Method
Path
Description

`GET`
`/health`
Service status, model name, and target variable

`POST`
`/predict`
Single prediction

`POST`
`/predict/batch`
Batch prediction (JSON array of requests)

`GET`
`/drift`
Lightweight drift report vs. training data

### Example requests

```
curl https://walmart-sales-api-ed2ca0cc.fastapicloud.dev/health

curl -X POST https://walmart-sales-api-ed2ca0cc.fastapicloud.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"Store":1,"Holiday_Flag":0,"Temperature":55,"Fuel_Price":3.4,"CPI":210,"Unemployment":7.5,"Month":8,"WeekOfYear":32,"Year":2024}'

curl -X POST https://walmart-sales-api-ed2ca0cc.fastapicloud.dev/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"Store":1,"Holiday_Flag":0,"Temperature":55,"Fuel_Price":3.4,"CPI":210,"Unemployment":7.5,"Month":8,"WeekOfYear":32,"Year":2024},{"Store":2,"Holiday_Flag":1,"Temperature":60,"Fuel_Price":3.6,"CPI":215,"Unemployment":8.1,"Month":9,"WeekOfYear":35,"Year":2024}]'

curl https://walmart-sales-api-ed2ca0cc.fastapicloud.dev/drift
```

Replace the host with `http://127.0.0.1:8000` when running locally.

## 🐳 Docker

```
docker build -t walmart-sales-api .
docker run -p 8000:8000 walmart-sales-api
```

## 🧪 Tests

```
pytest
```

CI runs the full data pipeline and training step before tests on every push
to `main`, so the model artifact tests depend on is always freshly produced
in CI — see `.github/workflows/ci.yml`.

## 🛤️ Project Phases

### Phase 1 — Environment Setup & Data Ingestion

Configured the Python environment, dependencies, and code quality tools
(Ruff, Black). Converted raw CSV data into compressed Parquet for faster
loading.

### Phase 2 — Exploratory Data Analysis & Cleaning

Statistical analysis of Walmart sales and economic factors. Handled missing
values, outliers, and incorrect data types.

### Phase 3 — Feature Engineering

Engineered time-based features (week of year, month, holiday flags), applied
scaling, and split data into reproducible train/validation/test sets.

### Phase 4 — Model Training & Experiment Tracking

Trained Random Forest, XGBoost, and LightGBM. Used **MLflow** to track
hyperparameters, metrics (RMSE, MAE, R²), and model artifacts. Saved the
best-performing model as `best_model.joblib`.

### Phase 5 — API Development

Built a **FastAPI** application exposing the trained model via REST, with
Pydantic request validation using realistic business-range constraints,
batch prediction support, prediction logging, and a drift-report endpoint.

### Phase 6 — CI/CD & Deployment

Containerized with Docker. Added a GitHub Actions workflow that runs the
full pipeline and test suite on every push. Deployed the API to **FastAPI
Cloud** and the UI to **Streamlit Community Cloud** — see [Live
Demo](https://claude.ai/chat/130736cc-1681-48dd-ac32-fa6495ed3cd4#-live-demo) above.

## 📁 Project Structure

```
walmart-sales-api/
├── .github/workflows/     # CI pipeline (GitHub Actions)
├── api/
│   └── main.py            # FastAPI app: routes, schemas, prediction logic
├── config/                # config.yaml — paths, hyperparameters, split sizes
├── data/
│   ├── raw/                # original dataset
│   ├── interim/             # cleaned data (generated)
│   └── processed/           # engineered features, train/val/test splits (generated)
├── models/                # best_model.joblib, feature_columns.json
├── src/
│   ├── ingest.py            # data ingestion
│   ├── preprocessing.py     # cleaning pipeline
│   ├── feature_engineering.py # feature creation + splits
│   ├── train.py              # model training + MLflow logging
│   ├── drift_monitor.py      # drift report generator
│   ├── config.py             # centralized config loader
│   ├── logger.py             # structured logging
│   └── utils.py               # shared helpers
├── tests/                  # pytest suite (API + training)
├── streamlit_app.py        # interactive UI (single/batch prediction, drift viewer)
├── main.py                 # root-level shim re-exporting api.main:app,
│                            # required for FastAPI Cloud's auto-discovery
├── Dockerfile
├── requirements.txt
└── pyproject.toml          # dependencies ([project]) + tool config (Black, Ruff, Pytest)
```

## 📊 Dataset

- **Task:** Regression
- **Target variable:** `Weekly_Sales`
- **Key features:**

- `Store` — store number
- `Date` — week of sales
- `Holiday_Flag` — whether the week contains a special holiday
- `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` — external economic
factors

## 📝 Notes

- The API loads `models/best_model.joblib` and `models/feature_columns.json`
at startup — both must exist before the API can serve predictions. Run the
pipeline (Step 4 above) if they're missing.
- Predictions are logged to `logs/predictions.jsonl` for auditability.
- `main.py` at the repo root exists purely so FastAPI Cloud's `fastapi run`
auto-discovery can find the app; the actual implementation lives in
`api/main.py`, and that's what Docker, tests, and local development use
directly.
