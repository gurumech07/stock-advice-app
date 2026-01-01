# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .analysis import analyze_stock
import json
from typing import Dict, Any

app = FastAPI()

class StockInput(BaseModel):
    symbol: str

@app.get("/get-top-10-stocks")
def get_top_10_stocks():
    # Placeholder for a function that would return top 10 stocks to buy
    from .analysis import get_top_10_stocks_to_buy
    return {"top_10_stocks": get_top_10_stocks_to_buy()}

@app.post("/analyze")
def get_analysis(stock: StockInput):
    return analyze_stock(stock.symbol)

@app.get("/charts-html", response_class=HTMLResponse)
def charts_html(symbol: str = "AAPL"):
    """Render all available charts returned by `generate_all_charts` dynamically.

    This iterates over whatever charts are present (metrics_bar, price_trend,
    financials_pie, shap_explain, etc.) so new charts show up automatically.
    """
    result = analyze_stock(symbol)
    charts = result.get("charts", {})

    # build container divs for each chart
    chart_divs = []
    chart_scripts = []
    for key, chart_json in charts.items():
        div_id = f"chart_{key}"
        chart_divs.append(f'<div id="{div_id}" style="width:100%;height:420px;margin-bottom:24px"></div>')
        # escape the JSON string into JS by double-encoding
        chart_scripts.append(
            f"const {div_id} = JSON.parse({json.dumps(chart_json)}); Plotly.newPlot('{div_id}', {div_id}.data, {div_id}.layout);"
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>All Charts: {symbol}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body style="font-family: Arial; margin: 24px;">
        <h1>All Charts: {symbol}</h1>
        <p>{result.get('disclaimer')}</p>
        {''.join(chart_divs)}
        <script>
            {"\n".join(chart_scripts)}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/analyze-charts-html", response_class=HTMLResponse)
def analyze_charts_html(symbol: str = "AAPL"):
    """Render analysis summary plus charts: fundamentals table, technicals chart,
    model probabilities, and the existing chart set (price, financials, SHAP).
    """
    result = analyze_stock(symbol)
    charts = result.get("charts", {})
    fundamentals = result.get('fundamentals', {})
    technical = result.get('technical', {})
    model = result.get('model', {})

    # Prepare HTML pieces
    # fundamentals table
    fund_rows = []
    for k, v in fundamentals.items():
        fund_rows.append(f"<tr><th style='text-align:left;padding:6px'>{k}</th><td style='padding:6px'>{v}</td></tr>")
    fund_table = f"<table style='border-collapse:collapse'>{''.join(fund_rows)}</table>"

    # technicals chart data (labels + numeric values)
    tech_labels = []
    tech_values = []
    for key in ['sma_50', 'sma_200', 'rsi_14', 'volatility_30d', 'return_30d', 'return_1y']:
        if key in technical and technical.get(key) is not None:
            tech_labels.append(key)
            tech_values.append(technical.get(key))

    # model probs (if available)
    model_probs = model.get('probs') or {}

    # Build chart divs and scripts
    chart_divs = []
    chart_scripts = []

    # include existing charts first (price_trend, financials, metrics, shap)
    for key, chart_json in charts.items():
        div_id = f"chart_{key}"
        chart_divs.append(f"<div id=\"{div_id}\" style=\"width:100%;height:420px;margin-bottom:18px\"></div>")
        chart_scripts.append(
            f"const {div_id} = JSON.parse({json.dumps(chart_json)}); Plotly.newPlot('{div_id}', {div_id}.data, {div_id}.layout);"
        )

    # technicals chart div
    chart_divs.append('<div id="chart_technical" style="width:100%;height:360px;margin-bottom:18px"></div>')
    chart_scripts.append(
        "Plotly.newPlot('chart_technical', [{"
        + "x: " + json.dumps(tech_labels) + ", y: " + json.dumps(tech_values) + ", type: 'bar', marker:{color:'#1f77b4'} }], {title:'Technical Indicators'} );"
    )

    # model probs chart div
    chart_divs.append('<div id="chart_model" style="width:100%;height:320px;margin-bottom:18px"></div>')
    chart_scripts.append(
        "Plotly.newPlot('chart_model', [{"
        + "x: " + json.dumps(list(model_probs.keys())) + ", y: " + json.dumps(list(model_probs.values())) + ", type: 'bar', marker:{color:['#2ca02c','#9467bd','#d62728']} }], {title:'Model class probabilities'} );"
    )

    # friendly header values
    display_name = result.get('name') or symbol
    display_price = result.get('price')
    price_src = result.get('price_source')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analyze Charts: {symbol}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body style="font-family: Arial; margin: 24px;">
        <h1>{display_name} ({symbol})</h1>
        <h2>Price: {display_price} &nbsp;&nbsp; <small>source: {price_src}</small></h2>
        <p>{result.get('disclaimer')}</p>

        <h2>Fundamentals</h2>
        {fund_table}

        <h2>Model Summary</h2>
        <p>Present: {model.get('present')} &nbsp; Top class: {model.get('top_class')} &nbsp; Score: {model.get('score')}</p>

        {''.join(chart_divs)}

        <script>
            {"\n".join(chart_scripts)}
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html)

@app.get("/healthcheck")
def healthcheck():
    return {"status": "healthy"}    