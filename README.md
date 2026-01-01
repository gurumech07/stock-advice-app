# Stock Advice App

A FastAPI-based stock analysis application that performs fundamental analysis on user-provided ticker symbols. It fetches data using yfinance, computes key metrics (P/E, debt-to-equity, ROE, EPS growth), generates interactive Plotly charts, and provides a simple buy/sell/hold rating based on rules-based scoring.

**Tech Stack**: Python (FastAPI, yfinance, pandas, plotly), Docker, Kubernetes (kind for local dev).

[![Docker](https://img.shields.io/badge/Docker-Deployed-blue)](https://hub.docker.com/) [![Kubernetes](https://img.shields.io/badge/K8s-kind-green)](https://kind.sigs.k8s.io/)

## Features

- Input: POST `/analyze` with `{"symbol": "AAPL"}`
- Output: Metrics, score (0-100), rating (Buy/Hold/Sell), Plotly chart JSON
- Local K8s deployment via kind
- Educational demo for DevSecOps portfolios

## Quick Start

### Prerequisites
- Docker Desktop
- kubectl
- kind (`brew install kind` on macOS)

### Local Development
1. Clone & install:
   ```bash
   git clone <repo-url>
   cd stock-advice-app
   pip install -r app/requirements.txt
   uvicorn app.main:app --reload --port 8000
Test:

bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
Docker Build & Run
bash
docker build -t stock-advice .
docker run -p 8000:8000 stock-advice
Kubernetes (kind Local Cluster)
Create cluster:

bash
kind create cluster --config k8s/kind-config.yaml
kind load docker-image stock-advice:latest
Deploy:

bash
kubectl apply -f k8s/
Access Swagger UI: http://localhost:8000/docs

Project Structure
text
stock-advice-app/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── models.py        # Pydantic schemas
│   ├── analysis.py      # yfinance + scoring logic
│   └── charts.py        # Plotly generation
├── requirements.txt
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kind-config.yaml
├── .dockerignore
├── .gitignore
└── README.md
Core Logic Example (analysis.py snippet)
python
import yfinance as yf
import plotly.graph_objects as go

def analyze_stock(symbol: str):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    score = 0
    if info.get('forwardPE', 999) < 15: score += 25
    if info.get('debtToEquity', 999) < 50: score += 25
    if info.get('returnOnEquity', 0) > 0.15: score += 25
    if info.get('earningsGrowth', 0) > 0.10: score += 25
    rating = "Buy" if score > 75 else "Hold" if score > 50 else "Sell"
    
    # Sample Plotly chart
    fig = go.Figure(data=[go.Bar(x=['P/E', 'D/E', 'ROE'], y=[info.get('forwardPE'), info.get('debtToEquity'), info.get('returnOnEquity')])])
    return {"score": score, "rating": rating, "metrics": info, "chart": fig.to_json()}
🚨 Critical Disclaimers (Required Reading)
This project is for EDUCATIONAL and PORTFOLIO PURPOSES ONLY.

1. Not Investment Advice
Ratings are simplistic algorithms, not professional analysis.

Never trade based on this app alone. Seek advice from certified financial professionals.

No guarantees on accuracy or performance.

2. Data Usage Restrictions
Relies on yfinance (Yahoo Finance scraper).

Non-commercial, personal use only per Yahoo TOS. No redistribution or production apps.

Data may be delayed/inaccurate.

3. Risk Warning
Investing involves loss risk. Past results ≠ future gains.

Developers not liable for any financial harm.

4. Regulatory Note (EU/DE Users)
Not registered investment advice (MiFID II/BaFin non-compliant).

Every API response includes: "disclaimer": "Educational only. Not advice."

License
MIT License. See LICENSE.

Contributing & Contact
PRs for improvements (e.g., ML scoring, CI/CD) welcome. Issues for bugs.

Built by a DevSecOps engineer prepping for Docker roles. Questions? Open an issue.

Updated: January 2026
