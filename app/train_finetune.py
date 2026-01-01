"""
train_finetune.py

Simple training script that creates a price-only supervised dataset
from multiple tickers' historical prices and trains a RandomForest
classifier to predict Buy/Hold/Sell over a 1-year horizon.

Usage:
    python -m app.train_finetune

Outputs:
    - app/finetuned_model_bundle.pkl  (joblib dump with {'model', 'le', 'features'})

This is a lightweight demo to enable a fine-tuned model inside the app.
"""
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def extract_price_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'close': series}).copy()
    df['ret_30'] = df['close'].pct_change(30)
    df['ret_90'] = df['close'].pct_change(90)
    df['ret_180'] = df['close'].pct_change(180)
    df['vol_30'] = df['close'].pct_change().rolling(30).std()
    df['vol_90'] = df['close'].pct_change().rolling(90).std()
    df = df.dropna()
    return df


def build_dataset(tickers: List[str], history_years: int = 5, horizon_days: int = 252):
    X_rows = []
    y = []
    for symbol in tickers:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period=f"{history_years}y").sort_index()
            if len(hist) < horizon_days + 200:
                continue
            feat_df = extract_price_features(hist['Close'])
            closes = hist['Close']
            for i in range(len(feat_df)):
                idx = feat_df.index[i]
                # ensure future point exists
                try:
                    future_idx = closes.index.get_loc(idx) + horizon_days
                except Exception:
                    continue
                if future_idx >= len(closes):
                    break
                future_ret = closes.iloc[future_idx] / closes.loc[idx] - 1.0
                # label mapping
                if future_ret > 0.15:
                    label = 'Buy'
                elif future_ret < -0.05:
                    label = 'Sell'
                else:
                    label = 'Hold'
                row = feat_df.iloc[i].values.astype(float)
                if np.isfinite(row).all():
                    X_rows.append(row)
                    y.append(label)
        except Exception:
            continue

    feature_names = ['ret_30', 'ret_90', 'ret_180', 'vol_30', 'vol_90']
    X = np.array(X_rows)
    return X, np.array(y), feature_names


def train_and_save(tickers: List[str], out_path: Path):
    X, y, feature_names = build_dataset(tickers)
    if len(X) == 0:
        raise RuntimeError('No training samples collected; expand ticker list or history length.')
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # Optional scaler to normalize features; saved in bundle for consistent preprocessing at inference
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=le.classes_))

    # Save a small background sample (for SHAP) and the scaler + feature names
    bg_count = min(100, len(X_train))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=bg_count, replace=False)
    background = X_train[bg_idx]

    bundle = {
        'model': clf,
        'le': le,
        'features': feature_names,
        'background': background,
        'scaler': scaler
    }
    joblib.dump(bundle, out_path)
    print(f"Saved model bundle to {out_path}")


def main():
    # small seed list; users should replace with larger universe for production
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ',
        'PG', 'MA', 'DIS', 'ADBE', 'CRM', 'PYPL', 'INTC', 'CSCO', 'ORCL', 'NFLX'
    ]
    out = Path(__file__).parent / 'finetuned_model_bundle.pkl'
    train_and_save(tickers, out)


if __name__ == '__main__':
    main()
