Stock Advice App
A FastAPI-based stock analysis application that performs fundamental analysis on user-provided ticker symbols. It fetches data using yfinance, computes key metrics (P/E, debt-to-equity, ROE, EPS growth), generates interactive Plotly charts, and provides a simple buy/sell/hold rating based on rules-based scoring.

Tech Stack: Python (FastAPI, yfinance, pandas, plotly), Docker, Kubernetes (kind for local dev).

[ [

Features
Input: POST /analyze with {"symbol": "AAPL"}

Output: Metrics, score (0-100), rating (Buy/Hold/Sell), Plotly chart JSON

Local K8s deployment via kind

Educational demo for DevSecOps portfolios

Quick Start
Prerequisites
Docker Desktop

kubectl

kind (install: brew install kind)

Local Development
Clone & install:

bash
git clone <repo>
cd stock-advice-app
pip install -r app/requirements.txt
uvicorn app.main:app --reload
Test API:

bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
Docker
bash
docker build -t stock-advice .
docker run -p 8000:8000 stock-advice
Kubernetes (kind)
Create cluster:

bash
kind create cluster --config kind-config.yaml
kind load docker-image stock-advice:latest
Deploy:

bash
kubectl apply -f k8s/
Access: http://localhost:8000/docs

Project Structure
text
stock-advice-app/
├── app/                 # FastAPI source
│   ├── main.py
│   ├── models.py
│   ├── analysis.py      # yfinance logic + scoring
│   └── charts.py
├── Dockerfile
├── k8s/                 # Deployment manifests
├── kind-config.yaml
└── README.md
API Example Response
json
{
  "score": 82,
  "rating": "Buy",
  "metrics": {"forwardPE": 12.5, "debtToEquity": 45, ...},
  "chart": "<plotly json>",
  "disclaimer": "Educational tool only. Not financial advice."
}
Fundamental Analysis Logic
Data: yfinance (free, public Yahoo Finance API wrapper)

Metrics: Forward P/E <15 (+25), Debt/Equity <50 (+25), ROE >15% (+25), EPS growth >10% (+25)

Charts: Plotly bar/line for ratios/trends

Extend with scikit-learn for ML predictions

Deployment Notes
Optimized for kind (multi-node config included)

Scale: kubectl scale deployment/stock-advice --replicas=3

Prod: Use managed K8s (EKS/GKE), add HTTPS/Ingress

License
MIT License – see LICENSE for details.

🚨 IMPORTANT DISCLAIMERS
This is an EDUCATIONAL DEMO PROJECT ONLY. Read before use:

1. Not Financial Advice
Analysis and ratings (buy/sell/hold) are automated, rules-based, and for learning purposes.

Do NOT base trades on this app. Past performance ≠ future results. Markets involve substantial risk of loss.

Consult a licensed financial advisor. Developers assume no responsibility for financial decisions or losses.

2. Data Source Limitations
Uses yfinance (Yahoo Finance data).

Personal, non-commercial use only. Yahoo TOS prohibits redistribution, commercial apps, or high-volume scraping.

Data may be delayed, incomplete, or erroneous. No warranty on accuracy.

3. No Liability
Use at your own risk. No guarantees on uptime, security, or results.

In EU/Germany: This is not MiFID II-compliant investment advice (per BaFin guidelines).

4. Regulatory Compliance
For demo/portfolio use (e.g., GitHub). Do not monetize or offer as a service without legal review.

Add your own API keys/rate limiting for production data sources (e.g., Alpha Vantage, Polygon.io).

By using this app, you agree to these terms.

Contributing
Fork, PRs welcome for features like ML models or cloud CI/CD (GitHub Actions).

Acknowledgments
Inspired by yfinance examples and FastAPI tutorials. Built for Docker/K8s showcase.
