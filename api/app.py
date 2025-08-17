from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Crypto MLOps MVP (mínimo)")

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/crypto/signal", response_model=SignalOut)
def signal(inp: SignalIn):
    # Stub temporal
    return {
        "symbol": inp.symbol,
        "horizon_min": inp.horizon_min,
        "risk_score": 0.42,
        "nowcast_ret": 0.0015,
        "vol_regime": "calm",
    }
