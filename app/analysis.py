# app/analysis.py
import yfinance as yf
import pandas as pd
from .charts import generate_all_charts
from pathlib import Path
import joblib
import numpy as np


# Attempt to load a fine-tuned model bundle if present. The bundle is a joblib
# dump with {'model': estimator, 'le': label_encoder, 'features': feature_names}
_MODEL_BUNDLE_PATH = Path(__file__).parent / 'finetuned_model_bundle.pkl'
_FINETUNE_BUNDLE = None
if _MODEL_BUNDLE_PATH.exists():
    try:
        _FINETUNE_BUNDLE = joblib.load(_MODEL_BUNDLE_PATH)
    except Exception:
        _FINETUNE_BUNDLE = None


def _build_price_features_from_ticker(ticker_obj):
    # returns 1D numpy array matching training order: ['ret_30','ret_90','ret_180','vol_30','vol_90']
    hist = ticker_obj.history(period='5y')
    closes = hist['Close']
    if len(closes) < 200:
        return None
    ret_30 = closes.pct_change(30).iloc[-1]
    ret_90 = closes.pct_change(90).iloc[-1]
    ret_180 = closes.pct_change(180).iloc[-1]
    vol_30 = closes.pct_change().rolling(30).std().iloc[-1]
    vol_90 = closes.pct_change().rolling(90).std().iloc[-1]
    feats = np.array([ret_30, ret_90, ret_180, vol_30, vol_90], dtype=float)
    if not np.isfinite(feats).all():
        return None
    return feats


def analyze_stock(symbol: str):
    ticker = yf.Ticker(symbol)
    info = ticker.info

    # Baseline simple scoring logic (fallback / interpretable)
    score = 0
    if info.get('forwardPE', 999) < 15: score += 25
    if info.get('debtToEquity', 999) < 50: score += 25
    if info.get('returnOnEquity', 0) > 0.15: score += 25
    if info.get('earningsGrowth', 0) > 0.10: score += 25

    rating = "Buy" if score > 75 else "Hold" if score > 50 else "Sell"

    # If a fine-tuned model exists, use it to produce a data-driven score+rating
    if _FINETUNE_BUNDLE is not None:
        try:
            feats = _build_price_features_from_ticker(ticker)
            if feats is not None:
                model = _FINETUNE_BUNDLE.get('model')
                le = _FINETUNE_BUNDLE.get('le')
                classes = list(le.classes_)
                proba = model.predict_proba(feats.reshape(1, -1))[0]
                # prefer probability of 'Buy' if present
                buy_idx = None
                if 'Buy' in classes:
                    buy_idx = classes.index('Buy')
                else:
                    # fallback: choose class with max prob
                    buy_idx = int(np.argmax(proba == np.max(proba)))
                data_score = float(proba[buy_idx]) * 100.0
                data_class = le.inverse_transform([int(np.argmax(proba))])[0]
                # combine model score with baseline (simple average)
                score = int((score + data_score) / 2)
                rating = data_class
        except Exception:
            # if model inference fails, fall back to baseline
            pass

    charts = generate_all_charts(symbol, info)

    # --- Market & recent price ---
    price = None
    price_source = None
    price_discrepancy = None
    last_price_timestamp = None
    last_close = None
    try:
        hist_short = ticker.history(period='5d')
        if 'Close' in hist_short.columns and len(hist_short['Close']) > 0:
            last_close = float(hist_short['Close'].iloc[-1])
            last_price_timestamp = str(hist_short.index[-1])
            price = last_close
            price_source = 'history.Close'
    except Exception:
        last_close = None

    if price is None:
        price_info = info.get('currentPrice', None)
        if price_info is not None:
            price = price_info
            price_source = 'info.currentPrice'
        else:
            price = 0
            price_source = 'unknown'

    try:
        if last_close is not None and info.get('currentPrice', None) is not None:
            price_discrepancy = float(info.get('currentPrice')) - float(last_close)
    except Exception:
        price_discrepancy = None

    # --- Historical series and technicals ---
    tech = {}
    try:
        hist_1y = ticker.history(period='1y')
        closes = hist_1y['Close'] if 'Close' in hist_1y.columns else None
        if closes is not None and len(closes) > 0:
            tech['last_close'] = float(closes.iloc[-1])
            tech['open'] = float(hist_1y['Open'].iloc[-1]) if 'Open' in hist_1y.columns else None
            tech['day_low'] = float(hist_1y['Low'].iloc[-1]) if 'Low' in hist_1y.columns else None
            tech['day_high'] = float(hist_1y['High'].iloc[-1]) if 'High' in hist_1y.columns else None
            tech['volume'] = int(hist_1y['Volume'].iloc[-1]) if 'Volume' in hist_1y.columns else None
            # SMA/EMA
            if len(closes) >= 50:
                tech['sma_50'] = float(closes.rolling(window=50).mean().iloc[-1])
            else:
                tech['sma_50'] = None
            if len(closes) >= 200:
                tech['sma_200'] = float(closes.rolling(window=200).mean().iloc[-1])
            else:
                tech['sma_200'] = None
            # RSI 14
            try:
                delta = closes.diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / (avg_loss.replace(0, np.nan))
                rsi = 100 - (100 / (1 + rs))
                tech['rsi_14'] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None
            except Exception:
                tech['rsi_14'] = None
            # Volatility (30d std of daily returns)
            try:
                tech['volatility_30d'] = float(closes.pct_change().rolling(30).std().iloc[-1])
            except Exception:
                tech['volatility_30d'] = None
            # Recent returns
            def _ret(days):
                try:
                    if len(closes) > days:
                        return float(closes.iloc[-1] / closes.iloc[-1-days] - 1.0)
                except Exception:
                    return None
            tech['return_1d'] = _ret(1)
            tech['return_7d'] = _ret(7)
            tech['return_30d'] = _ret(30)
            tech['return_90d'] = _ret(90)
            tech['return_1y'] = _ret(252)
            # Max drawdown over period
            try:
                cummax = closes.cummax()
                drawdown = (closes / cummax - 1.0)
                tech['max_drawdown_1y'] = float(drawdown.min())
            except Exception:
                tech['max_drawdown_1y'] = None
        else:
            tech = {}
    except Exception:
        tech = {}

    # --- Fundamental & metadata fields (best-effort from info) ---
    fundamentals = {
        'market_cap': info.get('marketCap'),
        'currency': info.get('currency'),
        'exchange': info.get('exchange'),
        'beta': info.get('beta'),
        'dividend_yield': info.get('dividendYield') or info.get('dividendYield'),
        'trailingPE': info.get('trailingPE'),
        'forwardPE': info.get('forwardPE'),
        'pegRatio': info.get('pegRatio'),
        'trailingEps': info.get('trailingEps'),
        'totalRevenue': info.get('totalRevenue'),
        'profitMargins': info.get('profitMargins'),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'country': info.get('country')
    }

    # --- Analyst / earnings info ---
    analyst = {
        'recommendationKey': info.get('recommendationKey'),
        'recommendationMean': info.get('recommendationMean'),
        'earnings_date': None
    }
    try:
        cal = ticker.calendar
        if cal is not None and not cal.empty:
            # calendar often has 'Earnings Date' column or index
            try:
                analyst['earnings_date'] = str(cal.loc['Earnings Date'][0]) if 'Earnings Date' in cal.index else None
            except Exception:
                analyst['earnings_date'] = None
    except Exception:
        pass

    # --- Model diagnostics ---
    model_present = False
    model_probs = None
    model_top_class = None
    model_score = None
    try:
        if _FINETUNE_BUNDLE is not None:
            model_present = True
            feats = _build_price_features_from_ticker(ticker)
            if feats is not None:
                model = _FINETUNE_BUNDLE.get('model')
                le = _FINETUNE_BUNDLE.get('le')
                classes = list(le.classes_)
                proba = model.predict_proba(feats.reshape(1, -1))[0]
                model_probs = {c: float(p) for c, p in zip(classes, proba)}
                # top class and score
                top_idx = int(np.argmax(proba))
                model_top_class = le.inverse_transform([top_idx])[0]
                # prefer Buy probability as model_score when available
                if 'Buy' in classes:
                    buy_idx = classes.index('Buy')
                    model_score = float(proba[buy_idx]) * 100.0
                else:
                    model_score = float(proba[top_idx]) * 100.0
    except Exception:
        model_present = model_present

    # --- Chart numeric diagnostics (last/min/max, adj close, currency, info price ratio) ---
    chart_stats = {}
    try:
        hist_check = ticker.history(period='1y')
        closes_chk = pd.to_numeric(hist_check['Close'], errors='coerce').dropna()
        if len(closes_chk) > 0:
            chart_stats['last'] = float(closes_chk.iloc[-1])
            chart_stats['min'] = float(closes_chk.min())
            chart_stats['max'] = float(closes_chk.max())
            # adj close if present
            if 'Adj Close' in hist_check.columns:
                try:
                    adj_chk = pd.to_numeric(hist_check['Adj Close'], errors='coerce').dropna()
                    if len(adj_chk) > 0:
                        # align indices
                        common = closes_chk.index.intersection(adj_chk.index)
                        if len(common) > 0:
                            chart_stats['adj_last'] = float(adj_chk.reindex(common).dropna().iloc[-1])
                        else:
                            chart_stats['adj_last'] = float(adj_chk.iloc[-1])
                except Exception:
                    pass
    except Exception:
        chart_stats = {}

    try:
        chart_stats['currency'] = info.get('currency')
        chart_stats['info_price'] = info.get('currentPrice')
        if chart_stats.get('info_price') and chart_stats.get('last'):
            try:
                ratio = float(chart_stats['last']) / float(chart_stats['info_price'])
                chart_stats['info_ratio'] = float(ratio)
                chart_stats['warning'] = True if (ratio > 10 or ratio < 0.1) else False
            except Exception:
                chart_stats['info_ratio'] = None
                chart_stats['warning'] = False
    except Exception:
        pass

    return {
        "symbol": symbol,
        "name": info.get('shortName', 'N/A'),
        "price": price,
        "price_source": price_source,
        "price_discrepancy": price_discrepancy,
        "last_price_timestamp": last_price_timestamp,
        "score": score,
        "rating": rating,
        "metrics": {k: v for k, v in info.items()
                    if k in ['forwardPE', 'debtToEquity', 'returnOnEquity', 'earningsGrowth', 'currentRatio']},
        "fundamentals": fundamentals,
        "technical": tech,
        "analyst": analyst,
        "model": {
            'present': model_present,
            'probs': model_probs,
            'top_class': model_top_class,
            'score': model_score
        },
        "chart_stats": chart_stats,
        "charts": charts,
        "disclaimer": "Educational tool only. Not financial advice."
    }

def get_top_10_stocks_to_buy():
    # Placeholder for a function that would return top 10 stocks to buy
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']