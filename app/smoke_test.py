"""Smoke test for finetuned model bundle and analysis flow.

Run with:
    python -m app.smoke_test^

This script will:
- report whether `app/finetuned_model_bundle.pkl` exists
- call `analyze_stock("GOOG")` and validate the response shape
"""
import sys
from pathlib import Path
import json

from .analysis import analyze_stock


def validate_response(res: dict) -> None:
    assert isinstance(res, dict), "analyze_stock did not return a dict"
    for k in ("symbol", "score", "rating", "metrics", "charts"):
        assert k in res, f"missing key in response: {k}"
    # new expected keys
    assert 'price' in res, 'missing price in response'
    assert 'price_source' in res, 'missing price_source in response'
    assert isinstance(res.get('charts'), dict), 'charts should be a dict of chart JSON'
    # expect price trend chart present
    assert 'price_trend' in res.get('charts', {}), 'price_trend chart missing'
    score = res["score"]
    assert isinstance(score, (int, float)), "score must be numeric"
    assert 0 <= score <= 100, f"score out of range: {score}"
    rating = res["rating"]
    assert rating in ("Buy", "Hold", "Sell"), f"unexpected rating: {rating}"
    # model key present (may indicate no model present inside)
    assert 'model' in res, 'missing model diagnostics key'
    # chart diagnostics
    assert 'chart_stats' in res, 'missing chart_stats diagnostics in response'


def main():
    base = Path(__file__).parent
    bundle = base / 'finetuned_model_bundle.pkl'
    if bundle.exists():
        print(f"Found finetuned model bundle at: {bundle}")
    else:
        print("No finetuned model bundle found. Proceeding with rules-based fallback.")

    print("Calling analyze_stock('GOOG')...")
    try:
        res = analyze_stock('GOOG')
    except Exception as e:
        print(f"analyze_stock raised an exception: {e}")
        sys.exit(2)

    try:
        validate_response(res)
    except AssertionError as ae:
        print(f"SMOKE TEST FAILED: {ae}")
        print("Response was:")
        print(json.dumps(res, indent=2, default=str))
        sys.exit(3)

    print("SMOKE TEST PASSED: analyze_stock returned a valid response.")
    print(json.dumps({"symbol": res.get('symbol'), "score": res.get('score'), "rating": res.get('rating')}, indent=2))


if __name__ == '__main__':
    main()
