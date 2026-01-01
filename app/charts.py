"""
charts.py - Plotly chart generation for stock fundamental analysis.
Serializes figures to JSON for FastAPI responses.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any
import yfinance as yf
from pathlib import Path
import joblib
import numpy as np

try:
    import shap
except Exception:
    shap = None

# Defensive fallback: if numpy in this environment lacks `trapz` (some broken installs),
# provide a lightweight implementation so SHAP's usage of np.trapz won't fail.
if not hasattr(np, 'trapz'):
    def _np_trapz(y, x=None, dx=1.0):
        y = np.asarray(y)
        if x is None:
            if y.size < 2:
                return 0.0
            return (np.sum(y) - 0.5 * (y[0] + y[-1])) * dx
        x = np.asarray(x)
        if x.shape != y.shape:
            # try to broadcast or fallback
            x = np.asarray(x)
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)
    np.trapz = _np_trapz

# Polyfill for numpy.in1d (some minimal numpy builds may be missing it)
if not hasattr(np, 'in1d'):
    try:
        np.in1d = np.isin
    except Exception:
        def _np_in1d(ar1, ar2, assume_unique=False, invert=False):
            a1 = np.asarray(ar1)
            a2 = np.asarray(ar2)
            a2set = set(a2.tolist())
            res = np.array([x in a2set for x in a1], dtype=bool)
            if invert:
                return ~res
            return res
        np.in1d = _np_in1d


def create_metrics_bar_chart(metrics: Dict[str, Any]) -> str:
    """
    Bar chart for key fundamental ratios.
    """
    fig = go.Figure()

    # Key metrics with fallback values
    data = {
        'Forward P/E': metrics.get('forwardPE', 0),
        'Debt/Equity': metrics.get('debtToEquity', 0),
        'ROE (%)': metrics.get('returnOnEquity', 0) * 100 if metrics.get('returnOnEquity') else 0,
        'EPS Growth (%)': metrics.get('earningsGrowth', 0) * 100 if metrics.get('earningsGrowth') else 0,
        'Current Ratio': metrics.get('currentRatio', 0),
    }

    fig.add_trace(
        go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=[f'{v:.2f}' if isinstance(v, (int, float)) else str(v) for v in data.values()],
            textposition='auto',
        )
    )

    fig.update_layout(
        title="Key Fundamental Metrics",
        xaxis_title="Metrics",
        yaxis_title="Value",
        height=400,
        showlegend=False,
        template="plotly_white"
    )

    return fig.to_json()


def create_historical_trend_chart(symbol: str) -> str:
    """
    Line chart for historical price + volume trends (1Y).
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")
    # try to capture currency and reported current price for diagnostics
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    currency = info.get('currency')
    info_price = info.get('currentPrice')

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price', 'Volume'),
        row_width=[0.7, 0.3]
    )

    # Price line
    # Ensure Close is numeric and drop NaNs before plotting
    try:
        closes_series = pd.to_numeric(hist['Close'], errors='coerce').dropna()
    except Exception:
        closes_series = hist['Close']
    # Ensure Adj Close is numeric and drop NaNs before plotting
    try:
        adj_series = pd.to_numeric(hist['Adj Close'], errors='coerce').dropna() if 'Adj Close' in hist.columns else None
    except Exception:
        adj_series = None

    fig.add_trace(
        go.Scatter(
            x=closes_series.index,
            y=closes_series.values,
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )

    # Add SMA overlays when data long enough
    try:
        if 'Close' in hist.columns and len(hist['Close']) >= 50:
            sma50 = hist['Close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=sma50,
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#ff7f0e', width=1.5, dash='dash')
                ),
                row=1, col=1
            )
        if 'Close' in hist.columns and len(hist['Close']) >= 200:
            sma200 = hist['Close'].rolling(window=200).mean()
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=sma200,
                    mode='lines',
                    name='SMA 200',
                    line=dict(color='#2ca02c', width=1.5, dash='dot')
                ),
                row=1, col=1
            )
    except Exception:
        # If SMA computation fails, continue without overlays
        pass

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.5)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f"{symbol} - 1 Year Price & Volume Trends",
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    # Make sure y-axis auto-ranges to data (avoid forcing 0 baseline)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, autorange=True, rangemode='normal', automargin=True)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Add a small annotation with numeric diagnostics so users can spot mismatches
    try:
        if len(closes_series) > 0:
            last_val = float(closes_series.iloc[-1])
            min_val = float(closes_series.min())
            max_val = float(closes_series.max())
            ann_text = f"Last: ${last_val:,.2f}"
            if adj_series is not None and len(adj_series) > 0:
                try:
                    adj_last = float(adj_series.reindex(closes_series.index).dropna().iloc[-1])
                    ann_text += f" — Adj: ${adj_last:,.2f}"
                except Exception:
                    pass
            if currency:
                ann_text += f" ({currency})"
            ann_text += f" — Min: ${min_val:,.2f} — Max: ${max_val:,.2f}"

            # add annotation
            fig.add_annotation(xref='paper', yref='paper', x=0.01, y=0.98,
                               text=ann_text, showarrow=False, align='left', bgcolor='rgba(255,255,255,0.9)')

            # warn if information price differs greatly from last close
            try:
                if info_price is not None and info_price > 0:
                    ratio = last_val / float(info_price) if float(info_price) != 0 else 0
                    if ratio > 10 or ratio < 0.1:
                        warn = f"Warning: chart last_close ${last_val:,.2f} differs from info.currentPrice ${float(info_price):,.2f} (ratio {ratio:.2f}). Check currency/adj-close/data source."
                        fig.add_annotation(xref='paper', yref='paper', x=0.01, y=0.92,
                                           text=warn, showarrow=False, align='left', bgcolor='rgba(255,230,230,0.9)', font={'color':'#a00'})
            except Exception:
                pass
    except Exception:
        pass

    return fig.to_json()


def create_financials_pie_chart(symbol: str) -> str:
    """
    Pie chart for latest income statement breakdown.
    """
    ticker = yf.Ticker(symbol)
    # Try latest annual financials, fallback to quarterly or return explanatory fig
    try:
        fin_df = ticker.financials
        if fin_df is None or fin_df.shape[1] == 0:
            fin_df = ticker.quarterly_financials
        if fin_df is None or fin_df.shape[1] == 0:
            raise ValueError('No financials available')
        # pick latest column (most recent year)
        col = fin_df.columns[0]
        financials = fin_df[col]
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text='No financials available', showarrow=False)
        fig.update_layout(title='Income Statement Breakdown (Latest Year)', template='plotly_white', height=300)
        return fig.to_json()

    # Clean series: drop NaN and zero/near-zero values
    try:
        s = financials.dropna().astype(float)
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text='Financials data not in expected format', showarrow=False)
        fig.update_layout(title='Income Statement Breakdown (Latest Year)', template='plotly_white', height=300)
        return fig.to_json()

    # remove very small values and zeros
    s = s[~(s.abs() <= 1e-8)]
    if s.empty:
        fig = go.Figure()
        fig.add_annotation(text='No meaningful financial items to display', showarrow=False)
        fig.update_layout(title='Income Statement Breakdown (Latest Year)', template='plotly_white', height=300)
        return fig.to_json()

    # Focus on top 5 by absolute value
    key_items = s.abs().nlargest(5)
    # Preserve original signed values for labeling
    orig = s.reindex(key_items.index)
    labels = [str(idx).strip() for idx in key_items.index]
    values = key_items.values

    # If values sum to zero (unlikely after filtering), return explanatory fig
    if abs(values.sum()) < 1e-8:
        fig = go.Figure()
        fig.add_annotation(text='Financial values sum to zero', showarrow=False)
        fig.update_layout(title='Income Statement Breakdown (Latest Year)', template='plotly_white', height=300)
        return fig.to_json()

    # Build pie chart using absolute values but show sign in hover/labels
    hover_text = []
    for lab, v in zip(labels, orig.values):
        sign = '+' if v >= 0 else '-'
        hover_text.append(f"{lab}: {sign}{abs(v):,.0f}")

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            hoverinfo='label+percent+text',
            text=hover_text,
            marker_colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        )
    ])

    fig.update_layout(
        title="Income Statement Breakdown (Latest Year)",
        height=400,
        showlegend=True,
        template="plotly_white"
    )

    return fig.to_json()


def _build_price_features_for_shap(symbol: str):
    """Build the same 5 price features used by the finetune trainer.

    Returns (feature_array, feature_names) or (None, None) on failure.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='5y')
    if 'Close' not in hist.columns or len(hist['Close']) < 200:
        return None, None
    closes = hist['Close']
    ret_30 = closes.pct_change(30).iloc[-1]
    ret_90 = closes.pct_change(90).iloc[-1]
    ret_180 = closes.pct_change(180).iloc[-1]
    vol_30 = closes.pct_change().rolling(30).std().iloc[-1]
    vol_90 = closes.pct_change().rolling(90).std().iloc[-1]
    feats = np.array([ret_30, ret_90, ret_180, vol_30, vol_90], dtype=float)
    feature_names = ['ret_30', 'ret_90', 'ret_180', 'vol_30', 'vol_90']
    if not np.isfinite(feats).all():
        return None, None
    return feats.reshape(1, -1), feature_names


def create_shap_explain_chart(symbol: str) -> str:
    """Generate a SHAP bar chart (instance-level) if a finetuned model bundle exists.

    If SHAP or the model bundle is not available, returns a small text figure explaining why.
    """
    bundle_path = Path(__file__).parent / 'finetuned_model_bundle.pkl'
    if not bundle_path.exists():
        fig = go.Figure()
        fig.add_annotation(text='No finetuned model bundle found', showarrow=False)
        fig.update_layout(title='SHAP Explainability', template='plotly_white', height=300)
        return fig.to_json()

    if shap is None:
        fig = go.Figure()
        fig.add_annotation(text='SHAP not installed', showarrow=False)
        fig.update_layout(title='SHAP Explainability', template='plotly_white', height=300)
        return fig.to_json()

    try:
        bundle = joblib.load(bundle_path)
        model = bundle.get('model')
        feature_names = bundle.get('features', ['f0', 'f1', 'f2', 'f3', 'f4'])
        background = bundle.get('background', None)
        scaler = bundle.get('scaler', None)
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text='Failed to load model bundle', showarrow=False)
        fig.update_layout(title='SHAP Explainability', template='plotly_white', height=300)
        return fig.to_json()

    feats, feature_names_check = _build_price_features_for_shap(symbol)
    if feats is None:
        fig = go.Figure()
        fig.add_annotation(text='Insufficient historical data for features', showarrow=False)
        fig.update_layout(title='SHAP Explainability', template='plotly_white', height=300)
        return fig.to_json()

    # Ensure we use the saved feature names if available
    names = feature_names if feature_names else feature_names_check

    # Apply scaler if available (model was trained on scaled features)
    try:
        if scaler is not None:
            feats_scaled = scaler.transform(feats)
            bg_scaled = scaler.transform(background) if (background is not None) else None
        else:
            feats_scaled = feats
            bg_scaled = background
    except Exception:
        feats_scaled = feats
        bg_scaled = background

    # Build DataFrame for SHAP with proper column names
    try:
        X_df = pd.DataFrame(feats_scaled, columns=names)
    except Exception:
        X_df = pd.DataFrame(feats_scaled)

    # Align instance features to the model's expected input if possible.
    target_n = None
    target_names = None
    try:
        # sklearn models often have `n_features_in_` and `feature_names_in_` attributes
        if hasattr(model, 'n_features_in_'):
            target_n = int(model.n_features_in_)
        if hasattr(model, 'feature_names_in_'):
            target_names = list(model.feature_names_in_)
    except Exception:
        target_n = None
        target_names = None

    # If bundle saved feature list matches model expectation, prefer that
    try:
        bundle_feature_list = bundle.get('features', None)
        if bundle_feature_list and target_n is None:
            target_n = len(bundle_feature_list)
            target_names = bundle_feature_list
    except Exception:
        pass

    if target_n is not None and X_df.shape[1] != target_n:
        print(f"Model expects {target_n} features but instance has {X_df.shape[1]}; attempting to align.")
        # Build a new DataFrame with target columns in the right order.
        new_cols = target_names if target_names is not None else (bundle_feature_list if bundle_feature_list is not None else None)
        if new_cols is not None and len(new_cols) == target_n:
            aligned = {}
            # if background exists, compute means to fill missing features
            bg_means = None
            if bg_df is not None:
                try:
                    bg_means = bg_df.mean().to_dict()
                except Exception:
                    bg_means = None
            for col in new_cols:
                if col in X_df.columns:
                    aligned[col] = X_df.loc[0, col]
                else:
                    # fill from background mean if available, else 0
                    if bg_means is not None and col in bg_means:
                        aligned[col] = float(bg_means[col])
                    else:
                        aligned[col] = 0.0
            X_df = pd.DataFrame([aligned], columns=new_cols)
            print(f"Aligned instance to model features: {len(X_df.columns)} cols")
            names = list(X_df.columns)
        else:
            # As a last-resort, pad/truncate numeric array to match target_n
            arr = np.asarray(feats_scaled).ravel()
            if arr.size < target_n:
                pad = np.zeros(target_n - arr.size, dtype=arr.dtype)
                arr = np.concatenate([arr, pad])
            else:
                arr = arr[:target_n]
            X_df = pd.DataFrame([arr], columns=[f'f{i}' for i in range(len(arr))])
            names = list(X_df.columns)
            print(f"Padded/truncated instance to {len(arr)} features to match model.")

    bg_df = None
    if bg_scaled is not None:
        try:
            bg_df = pd.DataFrame(bg_scaled, columns=names)
        except Exception:
            bg_df = pd.DataFrame(bg_scaled)

    # Defensive check: ensure background and instance have same feature count.
    # If not, try to align by slicing the background to the instance columns.
    try:
        if bg_df is not None and X_df.shape[1] != bg_df.shape[1]:
            print(f"SHAP background shape mismatch: instance cols={X_df.shape[1]} bg cols={bg_df.shape[1]}; attempting to align background")
            try:
                # If background has extra columns, slice to match instance
                if bg_df.shape[1] >= X_df.shape[1]:
                    bg_df = bg_df.iloc[:, : X_df.shape[1]]
                    bg_df.columns = names[: bg_df.shape[1]]
                    print(f"Sliced background to {bg_df.shape}")
                else:
                    # background has fewer columns; disable and fallback to instance
                    print("Background has fewer columns than instance; falling back to instance as background")
                    bg_df = None
            except Exception as e:
                print(f"Failed to align background: {e}; disabling background")
                bg_df = None
    except Exception:
        bg_df = None

    # Ensure we always pass something to the explainer as the background/masker.
    # If bg_df is None, use the instance DataFrame so SHAP can build a masker.
    if bg_df is None:
        try:
            print("Using instance data as background for SHAP (bg_df was None)")
            bg_df = X_df.copy()
        except Exception:
            bg_df = None

    try:
        # For tree models (RandomForest/LightGBM/XGBoost) prefer TreeExplainer
        # and request probability outputs to match predict_proba. Set
        # feature_perturbation='interventional' and disable the strict
        # additivity check if needed to avoid numeric mismatches.
        if hasattr(model, 'predict_proba'):
            expl = shap.TreeExplainer(
                model,
                bg_df,
                model_output='probability',
                feature_perturbation='interventional',
                check_additivity=False,
            )
        else:
            # Fallback to model-agnostic explainer using predict function
            expl = shap.Explainer(model.predict if hasattr(model, 'predict') else model, bg_df)
        shap_exp = expl(X_df)
    except Exception as e:
        try:
            # Secondary fallback: use model.predict_proba directly with a generic explainer
            if hasattr(model, 'predict_proba'):
                expl = shap.Explainer(model.predict_proba, bg_df)
                shap_exp = expl(X_df)
            else:
                raise
        except Exception as e2:
            # Return a figure that contains the exception messages to help debugging
            msg = str(e2) or str(e)
            short = msg if len(msg) < 300 else msg[:300] + '...'
            fig = go.Figure()
            fig.add_annotation(text=f'SHAP explain failed: {short}', showarrow=False)
            fig.update_layout(title='SHAP Explainability', template='plotly_white', height=300)
            print('SHAP explain error:', msg)
            return fig.to_json()

    vals = shap_exp.values
    # vals may be shape (n_samples, n_features) or (n_samples, n_outputs, n_features)
    try:
        if vals.ndim == 2:
            inst_shap = vals[0]
        elif vals.ndim == 3:
            # choose class index by model prediction
            probs = model.predict_proba(feats_scaled)[0]
            cls_idx = int(np.argmax(probs))
            inst_shap = vals[0, cls_idx, :]
        else:
            inst_shap = np.ravel(vals[0])
    except Exception:
        inst_shap = np.ravel(vals)[0:len(names)]

    # Build horizontal bar chart of signed SHAP values
    vals_arr = np.array(inst_shap, dtype=float)
    order = np.argsort(np.abs(vals_arr))[::-1]
    sorted_names = [names[i] for i in order]
    sorted_vals = vals_arr[order]
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in sorted_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in sorted_vals],
        textposition='auto'
    ))
    fig.update_layout(
        title='SHAP: Feature contributions (instance)',
        xaxis_title='SHAP value',
        height=350,
        template='plotly_white'
    )
    return fig.to_json()


def generate_all_charts(symbol: str, metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate complete chart set for API response.
    """
    charts = {
        "metrics_bar": create_metrics_bar_chart(metrics),
        "price_trend": create_historical_trend_chart(symbol),
        "financials_pie": create_financials_pie_chart(symbol)
    }
    # Add SHAP explainability chart when possible
    try:
        charts['shap_explain'] = create_shap_explain_chart(symbol)
    except Exception:
        # non-critical: ignore SHAP failures
        pass
    return charts
