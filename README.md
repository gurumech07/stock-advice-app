# Stock Advice App

A FastAPI-based stock analysis demo that combines fundamentals-based heuristics with an optional fine-tuned price-based classifier and SHAP explainability.

**Tech Stack**: Python (FastAPI, yfinance, pandas, scikit-learn, joblib, shap), Plotly, Docker, Kubernetes (kind for local dev).

## Highlights (what's new)

- Lightweight fine-tuning: `app/train_finetune.py` trains a simple price-based classifier and writes a model bundle to `app/finetuned_model_bundle.pkl`.
- Explainability: server-side SHAP summaries and a SHAP chart are produced when a compatible model bundle and `shap` are available.
- Richer JSON: `app/analyze` now returns `fundamentals`, `technical`, `model` and a `chart_stats` diagnostic object (last/adj/min/max/currency/info_price/info_ratio/warning).
- New visualization endpoint: `/analyze-charts-html` combines fundamentals, technicals, model probabilities and the full chart set in one page.
- Smoke test: `app/smoke_test.py` for quick programmatic validation of the API and bundle.

## Quick Start

### Prerequisites
- Python 3.10+ (or compatible)
- pip
- Docker (optional)

### Install dependencies
```bash
pip install -r requirements.txt
# if you plan to use SHAP and train locally: pip install shap
```

### Run the API locally
```bash
uvicorn app.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs for the Swagger UI.

### Helpful endpoints

- POST `/analyze` — main JSON analysis (symbol in body). Returns `fundamentals`, `technical`, `model`, `charts` and `chart_stats`.
- GET `/charts-html?symbol=XXX` — standalone charts HTML view.
- GET `/analyze-charts-html?symbol=XXX` — consolidated page with name, current price, fundamentals table, model summary and charts.

### Analyze a ticker (curl example)
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

## Fine-tuning (optional)

The repository includes a lightweight trainer to demonstrate fine-tuning using historical price features.

Train locally (example):
```bash
pip install -r requirements.txt
python -m app.train_finetune
```

This will fetch historical price data (via `yfinance`), train a `RandomForestClassifier`, and save a model bundle to `app/finetuned_model_bundle.pkl`.

Bundle format (joblib dict saved to `app/finetuned_model_bundle.pkl`):
- `model` — fitted scikit-learn estimator (e.g., RandomForest)
- `features` — list of feature names expected by the model
- `background` — small background sample used for SHAP (optional)
- `scaler` — fitted scaler applied to features (optional)

Notes:
- Training requires network access to Yahoo Finance and may take several minutes depending on your machine and connection.
- The trainer is intentionally simple (price-only features, heuristic labels) for demonstration purposes. For production, use richer features, cross-validation, and a robust labeling strategy.

## Explainability (SHAP)

- If `shap` is installed and a compatible model bundle exists, the app will attempt to compute SHAP values and include a SHAP chart in the charts set.
- The code contains defensive fallbacks when SHAP or certain numpy features aren't available; for best results install `shap` and a recent `numpy`.

## Diagnostics & chart debugging

- `analyze` now includes a `chart_stats` object with numeric diagnostics: `last`, `adj`, `min`, `max`, `currency`, `info_price`, `info_ratio`, and `warning` when there are large mismatches. This helps detect issues like Close vs Adj Close mismatches or currency/scale problems.

## Smoke test

Quickly validate the app (and the model bundle if present):
```bash
python -m app.smoke_test
```

The smoke test asserts the presence of key fields in the `/analyze` response and reports basic bundle health.

## Project Layout

- `app/` — application code
  - `main.py` — FastAPI app and endpoints
  - `analysis.py` — analysis logic (rules + optional fine-tuned model)
  - `train_finetune.py` — training script (creates `finetuned_model_bundle.pkl`)
  - `models.py` — Pydantic schemas
  - `charts.py` — Plotly chart generators and SHAP wiring
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build
- `K8s/` — Kubernetes manifests for deployment

## Disclaimer

This project is for EDUCATIONAL purposes only and is NOT financial advice. Do not make investment decisions based solely on outputs from this application.

## License

MIT

---
Updated: January 2026
