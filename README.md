# Stock Advice App

A FastAPI-based stock analysis demo that combines simple fundamentals-based scoring with an optional fine-tuned price-based classifier for improved recommendations.

**Tech Stack**: Python (FastAPI, yfinance, pandas, scikit-learn), Docker, Kubernetes (kind for local dev).

## What changed

- Added a training script at `app/train_finetune.py` that builds a simple supervised dataset from historical prices, trains a `RandomForestClassifier`, and writes a model bundle to `app/finetuned_model_bundle.pkl`.
- `app/analysis.py` now attempts to load that bundle and use it during inference (with graceful fallback to the original rules-based score).

## Quick Start

### Prerequisites
- Python 3.10+ (or compatible)
- pip
- Docker (optional for container runs)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the API locally
```bash
uvicorn app.main:app --reload --port 8000
```
Then open http://127.0.0.1:8000/docs for the Swagger UI.

### Analyze a ticker (curl example)
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

## Fine-tuning (optional)

The repository includes a lightweight trainer to demonstrate how to fine-tune the app using historical price features.

Train locally (example):
```bash
pip install -r requirements.txt
python -m app.train_finetune
```

This will fetch historical price data for a seed list of tickers, train a `RandomForestClassifier`, and save a model bundle to `app/finetuned_model_bundle.pkl`.

Notes:
- Training requires network access to Yahoo Finance via `yfinance` and may take several minutes depending on your machine and connection.
- The trainer is intentionally simple (price-only features, heuristic labels) and intended as a demo; for production fine-tuning use more features, cross-validation, and a robust labeling strategy.

## How inference integrates the fine-tuned model

- When `app/analysis.py` starts, it looks for `app/finetuned_model_bundle.pkl` and, if present, will use price-derived features to produce a probability-based score and label.
- If the bundle is missing or inference fails, the function falls back to the original fundamentals-based scoring so the API remains available.

## Project Layout

- `app/` — application code
  - `main.py` — FastAPI app and endpoints
  - `analysis.py` — analysis logic (rules + optional fine-tuned model)
  - `train_finetune.py` — training script (creates `finetuned_model_bundle.pkl`)
  - `models.py` — Pydantic schemas
  - `charts.py` — Plotly chart generators
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build
- `K8s/` — Kubernetes manifests for deployment

## Disclaimer

This project is for EDUCATIONAL purposes only and is NOT financial advice. Do not make investment decisions based solely on outputs from this application.

## License

MIT

---
Updated: January 2026
