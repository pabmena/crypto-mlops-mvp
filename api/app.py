from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI(title="MVP de Crypto MLOps (mínimo)", version="0.1.0")

# ── Modelos ──────────────────────────────────────────────────────────────────
class SignalIn(BaseModel):
    symbol: str = "BTCUSDT"
    horizon_min: int = 60
    explain: Optional[bool] = False

class SignalOut(BaseModel):
    symbol: str
    horizon_min: int
    risk_score: float
    nowcast_ret: float
    vol_regime: str

# ── Métricas simples en memoria ──────────────────────────────────────────────
METRICS = {
    "start_time": datetime.utcnow().isoformat() + "Z",
    "requests_total": 0,
    "signals_total": 0,
    "last_signal_at": None,
}

@app.middleware("http")
async def _count_requests(request, call_next):
    METRICS["requests_total"] += 1
    response = await call_next(request)
    return response

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return METRICS

@app.post("/v1/crypto/signal", response_model=SignalOut)
def signal(inp: SignalIn):
    # Stub temporal (luego lo reemplazamos por el modelo real)
    METRICS["signals_total"] += 1
    METRICS["last_signal_at"] = datetime.utcnow().isoformat() + "Z"
    return {
        "symbol": inp.symbol,
        "horizon_min": inp.horizon_min,
        "risk_score": 0.42,
        "nowcast_ret": 0.0015,
        "vol_regime": "calm",
    }

