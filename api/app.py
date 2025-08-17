# api/app.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
import json

from pathlib import Path  # usar Path de forma consistente
import ccxt
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# App & métricas en memoria
# -----------------------------------------------------------------------------
app = FastAPI(title="Crypto MLOps MVP", version="0.1.0")

app.state.metrics = {
    "start_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "requests_total": 0,
    "signals_total": 0,
    "last_signal_at": None,
}

# -----------------------------------------------------------------------------
# Persistencia: /app/data (montado desde ./data en docker-compose)
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"  # <- /app/data
DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_JSONL = DATA_DIR / "signals.jsonl"


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Middleware simple para contar requests
# -----------------------------------------------------------------------------
@app.middleware("http")
async def count_requests(request, call_next):
    app.state.metrics["requests_total"] += 1
    response = await call_next(request)
    return response


# -----------------------------------------------------------------------------
# Modelos I/O
# -----------------------------------------------------------------------------
class MetricsOut(BaseModel):
    start_time: str
    requests_total: int
    signals_total: int
    last_signal_at: Optional[str] = None


class OHLCVBar(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    time: str  # ISO string


class OHLCVOut(BaseModel):
    symbol: str
    exchange: str
    timeframe: str
    limit: int
    rows: int
    data: List[OHLCVBar]


class SignalIn(BaseModel):
    symbol: str = "BTCUSDT"       # admite 'BTCUSDT' o 'BTC/USDT'
    horizon_min: int = 60
    explain: bool = False
    exchange: str = "binance"
    timeframe: str = "1h"
    limit: int = 200


class SignalOut(BaseModel):
    symbol: str
    horizon_min: int
    risk_score: float
    nowcast_ret: float
    vol_regime: str
    explain: Optional[dict] = None


# -----------------------------------------------------------------------------
# Utilidades de mercado y features
# -----------------------------------------------------------------------------
def normalize_symbol(s: str) -> str:
    s = s.upper().replace(" ", "")
    if "/" in s:
        return s
    if len(s) >= 6:
        return s[:-4] + "/" + s[-4:]  # BTCUSDT -> BTC/USDT
    return s


def fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex_cls = getattr(ccxt, exchange_name)
    ex = ex_cls({"enableRateLimit": True})
    mkt_symbol = normalize_symbol(symbol)
    ohlcv = ex.fetch_ohlcv(mkt_symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    # convertir a string ISO para ser 100% JSON-serializable de forma estable
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def compute_features(df: pd.DataFrame) -> dict:
    df = df.copy()

    # Para cálculos, necesitamos tiempo como datetime nuevamente (sin afectar el string en respuesta)
    t = pd.to_datetime(df["time"], utc=True)

    df["ret"] = df["close"].pct_change()
    df["vol24"] = df["ret"].rolling(24, min_periods=8).std()  # ~1 día en 1h
    df["sma12"] = df["close"].rolling(12, min_periods=4).mean()
    df["sma48"] = df["close"].rolling(48, min_periods=12).mean()

    last = df.iloc[-1]
    nowcast_ret = float(last["ret"]) if not np.isnan(last["ret"]) else 0.0
    vol = float(last["vol24"]) if not np.isnan(last["vol24"]) else 0.0

    # régimen de volatilidad (heurístico)
    if vol < 0.005:
        vol_regime = "calm"
    elif vol < 0.015:
        vol_regime = "normal"
    else:
        vol_regime = "turbulent"

    # risk_score 0–1: mezcla de volatilidad y momentum (cruce de medias)
    mom = 1.0 if (last["sma12"] > last["sma48"]) else 0.0
    vol_norm = float(np.tanh(vol * 50.0))  # aplastar
    # Menor riesgo si momentum a favor y vol baja
    risk_score = float(np.clip(0.6 * (1.0 - vol_norm) + 0.4 * (1.0 - mom), 0.0, 1.0))

    features_tail = df.tail(5)[["time", "close", "ret", "vol24", "sma12", "sma48"]].copy()
    # asegurar tipos JSON-friendly
    features_tail = features_tail.replace({np.nan: None})
    features_tail["close"] = features_tail["close"].astype(float)
    if "ret" in features_tail:
        features_tail["ret"] = features_tail["ret"].astype(float, errors="ignore")
    if "vol24" in features_tail:
        features_tail["vol24"] = features_tail["vol24"].astype(float, errors="ignore")
    if "sma12" in features_tail:
        features_tail["sma12"] = features_tail["sma12"].astype(float, errors="ignore")
    if "sma48" in features_tail:
        features_tail["sma48"] = features_tail["sma48"].astype(float, errors="ignore")

    return {
        "nowcast_ret": nowcast_ret,
        "vol": vol,
        "vol_regime": vol_regime,
        "risk_score": risk_score,
        "features_tail": features_tail.to_dict(orient="records"),
    }


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsOut)
def metrics() -> MetricsOut:
    return MetricsOut(**app.state.metrics)


@app.get("/v1/crypto/ohlcv", response_model=OHLCVOut)
def ohlcv(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    limit: int = 200,
) -> OHLCVOut:
    df = fetch_ohlcv(exchange, symbol, timeframe, limit)
    return OHLCVOut(
        symbol=normalize_symbol(symbol),
        exchange=exchange,
        timeframe=timeframe,
        limit=limit,
        rows=len(df),
        data=[OHLCVBar(**row) for row in df.tail(50).to_dict(orient="records")],  # cap de respuesta
    )


@app.post("/v1/crypto/signal", response_model=SignalOut)
def signal(inp: SignalIn = Body(...)) -> SignalOut:
    df = fetch_ohlcv(inp.exchange, inp.symbol, inp.timeframe, inp.limit)
    feats = compute_features(df)

    out = SignalOut(
        symbol=normalize_symbol(inp.symbol),
        horizon_min=inp.horizon_min,
        risk_score=round(feats["risk_score"], 4),
        nowcast_ret=round(feats["nowcast_ret"], 6),
        vol_regime=feats["vol_regime"],
        explain=feats if inp.explain else None,
    )

    # actualizar métricas y persistir señal
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    app.state.metrics["signals_total"] += 1
    app.state.metrics["last_signal_at"] = now_iso

    append_jsonl(
        SIGNALS_JSONL,
        {
            "ts": now_iso,
            "symbol": out.symbol,
            "horizon_min": out.horizon_min,
            "risk_score": out.risk_score,
            "nowcast_ret": out.nowcast_ret,
            "vol_regime": out.vol_regime,
            "exchange": inp.exchange,
            "timeframe": inp.timeframe,
            "limit": inp.limit,
        },
    )

    return out

  


