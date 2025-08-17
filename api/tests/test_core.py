import pandas as pd
from app import normalize_symbol, compute_features

def test_normalize_symbol():
    assert normalize_symbol("BTCUSDT") == "BTC/USDT"
    assert normalize_symbol("btc/usdt") == "BTC/USDT"

def test_compute_features_output():
    # df mínimo: 50 filas simuladas
    df = pd.DataFrame({
        "ts":   range(50),
        "open": [100.0 + i for i in range(50)],
        "high": [100.5 + i for i in range(50)],
        "low":  [99.5  + i for i in range(50)],
        "close":[100.2 + i for i in range(50)],
        "volume":[1.0]*50
    })
    out = compute_features(df)
    assert "nowcast_ret" in out
    assert "vol" in out
    assert "vol_regime" in out
    assert "risk_score" in out
    assert len(out["features_tail"]) <= 5

