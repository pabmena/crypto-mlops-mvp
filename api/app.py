"""
Crypto MLOps MVP - Integración completa UI original + ML capabilities
Combina la interfaz visual de la app original con los servicios ML avanzados
"""
import os
import sys
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, Body, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Configurar paths para ML
sys.path.append('/app')
sys.path.append('/app/ml/inference')

# Importar servicios ML (opcional)
try:
    from ml_service import ml_service
    ML_AVAILABLE = True
    print("✅ ML service loaded successfully")
except ImportError as e:
    print(f"⚠️ ML service not available: {e}")
    ML_AVAILABLE = False

# -----------------------------------------------------------------------------
# App & métricas en memoria (ORIGINAL)
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Crypto MLOps MVP - Integrated", 
    version="2.1.0",
    description="Original UI + ML capabilities integrated"
)

app.state.metrics = {
    "start_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "requests_total": 0,
    "signals_total": 0,
    "ml_predictions_total": 0,
    "last_signal_at": None,
    "last_ml_prediction_at": None,
    "ml_available": ML_AVAILABLE,
}

# -----------------------------------------------------------------------------
# Persistencia: /app/data (ORIGINAL + ML)
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_JSONL = DATA_DIR / "signals.jsonl"
ML_PREDICTIONS_JSONL = DATA_DIR / "ml_predictions.jsonl"

MAX_BYTES = int(os.getenv("SIGNALS_MAX_BYTES", "2000000"))  # ~2 MB

def append_jsonl(path: Path, obj: dict) -> None:
    """Función original de persistencia con rotación"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > MAX_BYTES:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        rotated = path.with_name(f"{path.stem}-{ts}.jsonl")
        path.rename(rotated)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -----------------------------------------------------------------------------
# Middleware simple para contar requests (ORIGINAL)
# -----------------------------------------------------------------------------
@app.middleware("http")
async def count_requests(request, call_next):
    app.state.metrics["requests_total"] += 1
    response = await call_next(request)
    return response

# -----------------------------------------------------------------------------
# Modelos I/O (EXPANDIDOS)
# -----------------------------------------------------------------------------
class MetricsOut(BaseModel):
    start_time: str
    requests_total: int
    signals_total: int
    ml_predictions_total: int = 0
    last_signal_at: Optional[str] = None
    last_ml_prediction_at: Optional[str] = None
    ml_available: bool = False

class OHLCVBar(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    time: str

class OHLCVOut(BaseModel):
    symbol: str
    exchange: str
    timeframe: str
    limit: int
    rows: int
    data: List[OHLCVBar]

class SignalIn(BaseModel):
    symbol: str = "BTCUSDT"
    horizon_min: int = 60
    explain: bool = False
    exchange: str = "binance"
    timeframe: str = "1h"
    limit: int = 200

class MLSignalIn(BaseModel):
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    limit: int = 200
    include_heuristic: bool = True

class SignalOut(BaseModel):
    symbol: str
    horizon_min: int
    risk_score: float
    nowcast_ret: float
    vol_regime: str
    method: str = "heuristic"
    explain: Optional[dict] = None

class MLSignalOut(BaseModel):
    symbol: str
    method: str = "ml"
    timestamp: str
    model_version: Optional[str] = None
    ml_prediction: Optional[dict] = None
    heuristic_comparison: Optional[dict] = None

# -----------------------------------------------------------------------------
# Utilidades de mercado y features (ORIGINALES)
# -----------------------------------------------------------------------------
def normalize_symbol(s: str) -> str:
    """Función original de normalización"""
    s = s.upper().replace(" ", "")
    if "/" in s:
        return s
    if len(s) >= 6:
        return s[:-4] + "/" + s[-4:]
    return s

def fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Función original de fetch OHLCV"""
    ex_cls = getattr(ccxt, exchange_name)
    ex = ex_cls({"enableRateLimit": True})
    mkt_symbol = normalize_symbol(symbol)
    ohlcv = ex.fetch_ohlcv(mkt_symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df

def compute_features(df: pd.DataFrame) -> dict:
    """Función original de cálculo de features"""
    df = df.copy()

    if "time" not in df.columns:
        if "ts" in df.columns:
            t = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
            if t.isna().all():
                t = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
            df["time"] = t
        else:
            df["time"] = pd.to_datetime(pd.RangeIndex(len(df)), unit="s", utc=True)

    close = pd.to_numeric(df["close"], errors="coerce")
    df["ret"] = close.pct_change()
    df["vol24"] = df["ret"].rolling(24, min_periods=8).std()
    df["sma12"] = close.rolling(12, min_periods=4).mean()
    df["sma48"] = close.rolling(48, min_periods=12).mean()

    last = df.iloc[-1]
    nowcast_ret = float(last.get("ret", np.nan))
    vol = float(last.get("vol24", np.nan))

    if np.isnan(nowcast_ret):
        nowcast_ret = 0.0
    if np.isnan(vol):
        vol = 0.0

    if vol < 0.005:
        vol_regime = "calm"
    elif vol < 0.015:
        vol_regime = "normal"
    else:
        vol_regime = "turbulent"

    mom = 1.0 if (last.get("sma12", np.nan) > last.get("sma48", np.nan)) else 0.0
    vol_norm = float(np.tanh(vol * 50.0))
    risk_score = float(np.clip(0.6 * (1.0 - vol_norm) + 0.4 * (1.0 - mom), 0.0, 1.0))

    return {
        "nowcast_ret": nowcast_ret,
        "vol": vol,
        "vol_regime": vol_regime,
        "risk_score": risk_score,
        "features_tail": df.tail(5)[["time", "close", "ret", "vol24", "sma12", "sma48"]]
            .to_dict(orient="records"),
    }

# Nueva función para obtener datos (compatible con ML service)
def get_crypto_data(symbol: str, exchange: str = "binance", 
                   timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
    """Wrapper que convierte fetch_ohlcv al formato esperado por ML service"""
    df_original = fetch_ohlcv(exchange, symbol, timeframe, limit)
    
    # Convertir al formato esperado por ML service
    df_ml = df_original.copy()
    df_ml['timestamp'] = pd.to_datetime(df_ml['ts'], unit='ms')
    df_ml = df_ml[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    return df_ml

# -----------------------------------------------------------------------------
# Endpoints ORIGINALES
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ml_available": ML_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }

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
        data=[OHLCVBar(**row) for row in df.tail(50).to_dict(orient="records")],
    )

@app.post("/v1/crypto/signal", response_model=SignalOut)
def signal(inp: SignalIn = Body(...)):
    """Endpoint original para señales heurísticas"""
    try:
        df = fetch_ohlcv(inp.exchange, inp.symbol, inp.timeframe, inp.limit)
        if df is None or df.empty or len(df) < 10:
            raise HTTPException(status_code=503, detail="Insufficient OHLCV data from exchange")

        feats = compute_features(df)

        out = {
            "symbol": normalize_symbol(inp.symbol),
            "horizon_min": int(inp.horizon_min),
            "risk_score": float(round(feats["risk_score"], 4)),
            "nowcast_ret": float(round(feats["nowcast_ret"], 6)),
            "vol_regime": str(feats["vol_regime"]),
            "method": "heuristic",
            "explain": feats if inp.explain else None,
        }

        app.state.metrics["signals_total"] += 1
        app.state.metrics["last_signal_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        try:
            append_jsonl(SIGNALS_JSONL, {
                "ts": app.state.metrics["last_signal_at"],
                "symbol": out["symbol"],
                "horizon_min": out["horizon_min"],
                "risk_score": out["risk_score"],
                "nowcast_ret": out["nowcast_ret"],
                "vol_regime": out["vol_regime"],
                "method": "heuristic",
                "exchange": inp.exchange,
                "timeframe": inp.timeframe,
                "limit": inp.limit,
            })
        except Exception as e:
            print(f"[persist][WARN] {type(e).__name__}: {e}")

        return out

    except HTTPException:
        raise
    except ccxt.BaseError as e:
        print(f"[ccxt][ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="Exchange error via ccxt")
    except Exception as e:
        print(f"[signal][ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal error in /v1/crypto/signal")

@app.get("/v1/crypto/signals/tail")
def signals_tail(n: int = 5) -> list[dict[str, Any]]:
    """Función original para obtener últimas señales"""
    if not SIGNALS_JSONL.exists():
        return []
    lines = SIGNALS_JSONL.read_text(encoding="utf-8").splitlines()
    tail = lines[-n:] if len(lines) >= n else lines
    return [json.loads(x) for x in tail]

# -----------------------------------------------------------------------------
# Nuevos Endpoints ML
# -----------------------------------------------------------------------------
@app.post("/v1/crypto/ml-signal", response_model=MLSignalOut)
def ml_signal(inp: MLSignalIn = Body(...)):
    """Nuevo endpoint para señales ML"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Obtener datos en formato ML
        df = get_crypto_data(inp.symbol, inp.exchange, inp.timeframe, inp.limit)
        
        # Predicción ML
        ml_prediction = ml_service.predict_volatility(df)
        
        # Respuesta base
        response_data = {
            "symbol": normalize_symbol(inp.symbol),
            "method": "ml",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_version": ml_prediction.get("model_version"),
            "ml_prediction": ml_prediction
        }
        
        # Comparación heurística si se solicita
        if inp.include_heuristic:
            df_heuristic = fetch_ohlcv(inp.exchange, inp.symbol, inp.timeframe, inp.limit)
            heuristic_features = compute_features(df_heuristic)
            
            response_data["heuristic_comparison"] = {
                "heuristic_risk_score": heuristic_features["risk_score"],
                "heuristic_vol_regime": heuristic_features["vol_regime"],
                "ml_vs_heuristic": {
                    "volatility_diff": ml_prediction.get("prediction", 0) - heuristic_features["vol"],
                    "regime_match": ml_prediction.get("volatility_regime") == heuristic_features["vol_regime"]
                }
            }
        
        # Actualizar métricas
        app.state.metrics["ml_predictions_total"] += 1
        app.state.metrics["last_ml_prediction_at"] = response_data["timestamp"]
        
        # Persistir
        try:
            append_jsonl(ML_PREDICTIONS_JSONL, response_data)
        except Exception as e:
            print(f"[persist][WARN] ML prediction: {type(e).__name__}: {e}")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ml-signal][ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal error in ML prediction")

@app.get("/v1/crypto/ml-predictions/tail")
def ml_predictions_tail(n: int = 5) -> list[dict[str, Any]]:
    """Obtener últimas predicciones ML"""
    if not ML_PREDICTIONS_JSONL.exists():
        return []
    lines = ML_PREDICTIONS_JSONL.read_text(encoding="utf-8").splitlines()
    tail = lines[-n:] if len(lines) >= n else lines
    return [json.loads(x) for x in tail]

@app.get("/v1/ml/model/info")
def model_info():
    """Información del modelo ML"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    return ml_service.get_model_info()

@app.post("/v1/ml/model/reload")
def model_reload():
    """Recargar modelo ML"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    result = ml_service.reload_model()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/v1/crypto/signals/compare")
def signals_compare(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    limit: int = 200
):
    """Comparar señales heurística vs ML"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    # Obtener datos para ambos métodos
    df_heuristic = fetch_ohlcv(exchange, symbol, timeframe, limit)
    df_ml = get_crypto_data(symbol, exchange, timeframe, limit)
    
    # Generar predicciones
    heuristic_features = compute_features(df_heuristic)
    ml_prediction = ml_service.predict_volatility(df_ml)
    
    return {
        "symbol": normalize_symbol(symbol),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "heuristic": {
            "risk_score": heuristic_features["risk_score"],
            "vol_regime": heuristic_features["vol_regime"],
            "volatility": heuristic_features["vol"]
        },
        "ml": {
            "predicted_volatility": ml_prediction.get("prediction"),
            "vol_regime": ml_prediction.get("volatility_regime"),
            "confidence": ml_prediction.get("confidence"),
            "risk_level": ml_prediction.get("risk_level")
        },
        "comparison": {
            "volatility_diff": ml_prediction.get("prediction", 0) - heuristic_features["vol"],
            "regime_agreement": ml_prediction.get("volatility_regime") == heuristic_features["vol_regime"],
            "ml_confidence": ml_prediction.get("confidence", 0)
        }
    }

# -----------------------------------------------------------------------------
# UI INTEGRADA (Original + ML capabilities)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Crypto MLOps MVP - Integrated</title>
<style>
  :root{
    --bg:#f3f4f6; --panel:#ffffff; --ink:#0f172a; --muted:#475569; --line:#e5e7eb;
    --green:#22c55e; --amber:#f59e0b; --red:#ef4444; --gray:#94a3b8; --blue:#3b82f6; --purple:#8b5cf6;
    --green-bg:#dcfce7; --green-fg:#065f46;
    --amber-bg:#fef3c7; --amber-fg:#7c2d12;
    --red-bg:#fee2e2;   --red-fg:#7f1d1d;
    --gray-bg:#e5e7eb;  --gray-fg:#111827;
    --blue-bg:#dbeafe;  --blue-fg:#1e3a8a;
    --purple-bg:#ede9fe; --purple-fg:#581c87;
  }
  body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
  .wrap{max-width:1400px;margin:32px auto;padding:0 16px}
  h1{font-weight:800;letter-spacing:.2px;margin:0 0 8px}
  h2{font-weight:700;margin:16px 0 8px}
  h3{font-weight:600;margin:12px 0 6px}
  
  /* Tabs */
  .tabs{display:flex;gap:4px;margin:16px 0;border-bottom:2px solid var(--line)}
  .tab{padding:8px 16px;background:transparent;border:none;border-bottom:2px solid transparent;cursor:pointer;font-weight:600;color:var(--muted)}
  .tab.active{color:var(--blue);border-bottom-color:var(--blue)}
  .tab:hover{background:var(--line)}
  
  .tab-content{display:none}
  .tab-content.active{display:block}
  
  .row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .triple{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
  .muted{color:var(--muted)}
  label{display:block;font-weight:600;margin:10px 0 6px}
  input,select{width:220px;max-width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:#fff}
  .btn{border:1px solid var(--line);background:#fff;border-radius:10px;padding:8px 12px;cursor:pointer;margin:2px}
  .btn:hover{filter:brightness(.97)}
  .btn.primary{background:#0f172a;color:#fff;border-color:#0f172a}
  .btn.ml{background:var(--purple);color:#fff;border-color:var(--purple)}
  .btn.compare{background:var(--amber);color:#fff;border-color:var(--amber)}
  
  .pill{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;font-weight:700;border:1px solid transparent}
  .dot{width:10px;height:10px;border-radius:50%}
  .pill-green{background:var(--green-bg);color:var(--green-fg);border-color:var(--green)}
  .pill-amber{background:var(--amber-bg);color:var(--amber-fg);border-color:var(--amber)}
  .pill-red{background:var(--red-bg);color:var(--red-fg);border-color:var(--red)}
  .pill-gray{background:var(--gray-bg);color:var(--gray-fg);border-color:var(--gray)}
  .pill-blue{background:var(--blue-bg);color:var(--blue-fg);border-color:var(--blue)}
  .pill-purple{background:var(--purple-bg);color:var(--purple-fg);border-color:var(--purple)}
  
  pre{background:#0f172a;color:#e5e7eb;border-radius:8px;padding:10px;overflow:auto;max-height:200px}
  table{width:100%;border-collapse:collapse;border:1px solid var(--line);border-radius:12px;overflow:hidden}
  th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  
  .legend{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:4px 8px;background:#fff}
  .pills-row{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 12px 0}
  .w200{width:200px}
  
  .ml-status{display:flex;align-items:center;gap:8px;margin:8px 0}
  .status-dot{width:8px;height:8px;border-radius:50%}
  .status-online{background:var(--green)}
  .status-offline{background:var(--red)}
  
  .comparison-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
  .comparison-item{padding:8px;border:1px solid var(--line);border-radius:8px;background:#fafafa}
</style>
</head>
<body>
<div class="wrap">
  <h1>🚀 Crypto MLOps MVP - Integrated</h1>
  <div class="muted">
    Heuristic + ML predictions | MLflow · Airflow · MinIO integration
    <div class="ml-status">
      <span class="status-dot" id="mlStatusDot"></span>
      <span id="mlStatusText">Checking ML status...</span>
    </div>
  </div>

  <!-- Tabs Navigation -->
  <div class="tabs">
    <button class="tab active" onclick="showTab('signals')">📊 Signals</button>
    <button class="tab" onclick="showTab('ml')">🤖 ML Predictions</button>
    <button class="tab" onclick="showTab('compare')">⚖️ Compare</button>
    <button class="tab" onclick="showTab('data')">📈 OHLCV Data</button>
    <button class="tab" onclick="showTab('history')">📋 History</button>
  </div>

  <!-- Tab: Signals (Original) -->
  <div id="tab-signals" class="tab-content active">
    <div class="row">
      <div class="card">
        <h3>🎯 Signal Parameters</h3>
        <label>Symbol</label>
        <input id="symbol" value="BTCUSDT" class="w200"/>
        <label>Exchange</label>
        <select id="exchange" class="w200">
          <option value="binance" selected>binance</option>
        </select>
        <label>Timeframe</label>
        <select id="timeframe" class="w200">
          <option>1h</option><option>15m</option><option>4h</option><option>1d</option>
        </select>
        <label>Limit</label>
        <input id="limit" type="number" value="200" class="w200"/>
        <label>Horizon (min)</label>
        <input id="horizon" type="number" value="60" class="w200"/>

        <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn primary" id="btnHeuristic">🧮 Heuristic Signal</button>
          <button class="btn ml" id="btnML">🤖 ML Signal</button>
          <button class="btn compare" id="btnCompare">⚖️ Compare Both</button>
        </div>

        <div style="margin-top:12px" class="muted">
          <b>Heuristic:</b> Traditional indicators (SMA, volatility)<br/>
          <b>ML:</b> Machine learning predictions via MLflow
        </div>
        
        <div style="margin-top:14px" class="legend">
          <span class="badge"><span class="dot" style="background:var(--green)"></span> Risk: Low &lt; 0.40</span>
          <span class="badge"><span class="dot" style="background:var(--amber)"></span> Risk: Medium 0.40–0.69</span>
          <span class="badge"><span class="dot" style="background:var(--red)"></span> Risk: High ≥ 0.70</span>
        </div>
      </div>

      <div class="card">
        <h3>📡 Current Signal</h3>
        <div class="pills-row">
          <div id="riskPill" class="pill pill-gray">
            <span class="dot" id="riskDot" style="background:var(--gray)"></span>
            Risk —
          </div>
          <div id="volPill" class="pill pill-gray">
            <span class="dot" id="volDot" style="background:var(--gray)"></span>
            Vol —
          </div>
          <div id="methodPill" class="pill pill-gray">
            <span class="dot" style="background:var(--gray)"></span>
            Method —
          </div>
        </div>
        <div class="muted" id="sigMeta">—</div>
        <div style="margin-top:10px" id="sigExplain"><pre>—</pre></div>
      </div>
    </div>
  </div>

  <!-- Tab: ML Predictions -->
  <div id="tab-ml" class="tab-content">
    <div class="row">
      <div class="card">
        <h3>🤖 ML Model Control</h3>
        <div class="ml-status">
          <span class="status-dot" id="mlModelStatusDot"></span>
          <span id="mlModelStatus">Loading...</span>
        </div>
        
        <div style="margin:12px 0">
          <button class="btn ml" id="btnReloadModel">🔄 Reload Model</button>
          <button class="btn" id="btnModelInfo">ℹ️ Model Info</button>
        </div>
        
        <div style="margin-top:10px" id="modelInfo"><pre>—</pre></div>
      </div>
      
      <div class="card">
        <h3>🎯 ML Prediction Result</h3>
        <div class="pills-row">
          <div id="mlConfidencePill" class="pill pill-gray">
            <span class="dot" style="background:var(--gray)"></span>
            Confidence —
          </div>
          <div id="mlRiskPill" class="pill pill-gray">
            <span class="dot" style="background:var(--gray)"></span>
            ML Risk —
          </div>
        </div>
        <div class="muted" id="mlPredMeta">—</div>
        <div style="margin-top:10px" id="mlPredExplain"><pre>—</pre></div>
      </div>
    </div>
  </div>

  <!-- Tab: Compare -->
  <div id="tab-compare" class="tab-content">
    <div class="card">
      <h3>⚖️ Heuristic vs ML Comparison</h3>
      <div style="margin:12px 0">
        <button class="btn compare" id="btnRunComparison">🔄 Run Comparison</button>
      </div>
      
      <div class="comparison-grid" id="comparisonResults">
        <div class="comparison-item">
          <h4>📊 Heuristic Method</h4>
          <div id="heuristicResults">
            <div class="muted">No data yet</div>
          </div>
        </div>
        <div class="comparison-item">
          <h4>🤖 ML Method</h4>
          <div id="mlResults">
            <div class="muted">No data yet</div>
          </div>
        </div>
      </div>
      
      <div style="margin-top:12px">
        <h4>📈 Analysis</h4>
        <div id="comparisonAnalysis">
          <div class="muted">Run comparison to see analysis</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab: OHLCV Data -->
  <div id="tab-data" class="tab-content">
    <div class="card">
      <h3>📈 OHLCV Data (last 50 rows)</h3>
      <div style="margin:12px 0">
        <button class="btn" id="btnRefreshOHLCV">🔄 Refresh Data</button>
      </div>
      <div style="overflow:auto;max-height:500px">
        <table id="tblOHLCV"><thead></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- Tab: History -->
  <div id="tab-history" class="tab-content">
    <div class="row">
      <div class="card">
        <h3>📊 Recent Heuristic Signals</h3>
        <div style="margin:12px 0">
          <button class="btn" id="btnRefreshHeuristicHistory">🔄 Refresh</button>
        </div>
        <div style="overflow:auto;max-height:400px">
          <table id="tblHeuristicHistory"><thead></thead><tbody></tbody></table>
        </div>
      </div>
      
      <div class="card">
        <h3>🤖 Recent ML Predictions</h3>
        <div style="margin:12px 0">
          <button class="btn ml" id="btnRefreshMLHistory">🔄 Refresh</button>
        </div>
        <div style="overflow:auto;max-height:400px">
          <table id="tblMLHistory"><thead></thead><tbody></tbody></table>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  const $ = (id) => document.getElementById(id);

  // === TAB MANAGEMENT ===
  function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(`tab-${tabName}`).classList.add('active');
    event.target.classList.add('active');
  }

  // === UTILITY FUNCTIONS ===
  function riskBand(r) {
    if (r == null || isNaN(r)) return { cls: "pill-gray", dot: "var(--gray)", label: "—" };
    if (r >= 0.70) return { cls: "pill-red", dot: "var(--red)", label: "HIGH" };
    if (r >= 0.40) return { cls: "pill-amber", dot: "var(--amber)", label: "MED" };
    return { cls: "pill-green", dot: "var(--green)", label: "LOW" };
  }

  function volBand(regime) {
    if (!regime) return { cls: "pill-gray", dot: "var(--gray)", text: "—" };
    const s = String(regime).toLowerCase();
    if (s === "calm") return { cls: "pill-blue", dot: "var(--blue)", text: "✓ calm" };
    if (s === "normal") return { cls: "pill-amber", dot: "var(--amber)", text: "≈ normal" };
    if (s === "turbulent") return { cls: "pill-red", dot: "var(--red)", text: "⚠ turbulent" };
    return { cls: "pill-gray", dot: "var(--gray)", text: s };
  }

  function methodBand(method) {
    if (!method) return { cls: "pill-gray", dot: "var(--gray)", text: "—" };
    if (method === "heuristic") return { cls: "pill-blue", dot: "var(--blue)", text: "🧮 Heuristic" };
    if (method === "ml") return { cls: "pill-purple", dot: "var(--purple)", text: "🤖 ML" };
    return { cls: "pill-gray", dot: "var(--gray)", text: method };
  }

  function confidenceBand(confidence) {
    if (confidence == null || isNaN(confidence)) return { cls: "pill-gray", dot: "var(--gray)", text: "—" };
    const c = Number(confidence);
    if (c >= 0.8) return { cls: "pill-green", dot: "var(--green)", text: `✓ High (${c.toFixed(2)})` };
    if (c >= 0.6) return { cls: "pill-amber", dot: "var(--amber)", text: `≈ Med (${c.toFixed(2)})` };
    return { cls: "pill-red", dot: "var(--red)", text: `⚠ Low (${c.toFixed(2)})` };
  }

  async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  // === API FUNCTIONS ===
  async function checkMLStatus() {
    try {
      const health = await fetchJSON('/health');
      const statusDot = $('mlStatusDot');
      const statusText = $('mlStatusText');
      
      if (health.ml_available) {
        statusDot.className = 'status-dot status-online';
        statusText.textContent = '✅ ML service online';
      } else {
        statusDot.className = 'status-dot status-offline';
        statusText.textContent = '❌ ML service offline';
      }
    } catch (e) {
      const statusDot = $('mlStatusDot');
      const statusText = $('mlStatusText');
      statusDot.className = 'status-dot status-offline';
      statusText.textContent = '❌ Error checking ML status';
      console.error('ML status check failed:', e);
    }
  }

  async function getModelInfo() {
    try {
      const info = await fetchJSON('/v1/ml/model/info');
      $('modelInfo').innerHTML = `<pre>${JSON.stringify(info, null, 2)}</pre>`;
      
      const modelStatusDot = $('mlModelStatusDot');
      const modelStatus = $('mlModelStatus');
      
      if (info.model_loaded) {
        modelStatusDot.className = 'status-dot status-online';
        modelStatus.textContent = `✅ Model loaded: ${info.model_version || 'unknown'}`;
      } else {
        modelStatusDot.className = 'status-dot status-offline';
        modelStatus.textContent = '❌ No model loaded';
      }
    } catch (e) {
      $('modelInfo').innerHTML = `<pre>Error: ${e.message}</pre>`;
      $('mlModelStatusDot').className = 'status-dot status-offline';
      $('mlModelStatus').textContent = '❌ Error getting model info';
    }
  }

  async function reloadModel() {
    try {
      $('btnReloadModel').textContent = '🔄 Reloading...';
      $('btnReloadModel').disabled = true;
      
      const result = await fetchJSON('/v1/ml/model/reload', { method: 'POST' });
      alert(`Model reload: ${result.success ? 'SUCCESS' : 'FAILED'}\n${result.message}`);
      
      await getModelInfo(); // Refresh model info
    } catch (e) {
      alert(`Model reload failed: ${e.message}`);
    } finally {
      $('btnReloadModel').textContent = '🔄 Reload Model';
      $('btnReloadModel').disabled = false;
    }
  }

  async function runOHLCV() {
    const q = new URLSearchParams({
      symbol: $('symbol').value || 'BTCUSDT',
      exchange: $('exchange').value || 'binance',
      timeframe: $('timeframe').value || '1h',
      limit: String($('limit').value || 200)
    }).toString();
    
    const data = await fetchJSON(`/v1/crypto/ohlcv?${q}`);
    renderOHLCV(data);
  }

  async function runHeuristicSignal() {
    const body = {
      symbol: $('symbol').value || 'BTCUSDT',
      horizon_min: Number($('horizon').value || 60),
      explain: true,
      exchange: $('exchange').value || 'binance',
      timeframe: $('timeframe').value || '1h',
      limit: Number($('limit').value || 200),
    };
    
    const sig = await fetchJSON('/v1/crypto/signal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    renderHeuristicSignal(sig);
  }

  async function runMLSignal() {
    const body = {
      symbol: $('symbol').value || 'BTCUSDT',
      exchange: $('exchange').value || 'binance',
      timeframe: $('timeframe').value || '1h',
      limit: Number($('limit').value || 200),
      include_heuristic: true
    };
    
    const sig = await fetchJSON('/v1/crypto/ml-signal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    renderMLSignal(sig);
  }

  async function runComparison() {
    const q = new URLSearchParams({
      symbol: $('symbol').value || 'BTCUSDT',
      exchange: $('exchange').value || 'binance',
      timeframe: $('timeframe').value || '1h',
      limit: String($('limit').value || 200)
    }).toString();
    
    const comparison = await fetchJSON(`/v1/crypto/signals/compare?${q}`);
    renderComparison(comparison);
  }

  // === RENDER FUNCTIONS ===
  function renderHeuristicSignal(sig) {
    $('sigMeta').innerHTML = `
      <b>Symbol:</b> ${sig.symbol} &nbsp;&nbsp; 
      <b>Horizon:</b> ${sig.horizon_min} min<br/>
      <b>Risk (0–1):</b> ${Number(sig.risk_score).toFixed(6)} &nbsp;&nbsp; 
      <b>Nowcast ret:</b> ${Number(sig.nowcast_ret).toFixed(6)}
    `;

    // Risk pill
    const rb = riskBand(Number(sig.risk_score));
    const rP = $('riskPill');
    rP.className = `pill ${rb.cls}`;
    rP.innerHTML = `<span class="dot" style="background:${rb.dot}"></span>
                    Risk ${rb.label} (${Number(sig.risk_score).toFixed(2)})`;

    // Vol pill
    const vb = volBand(sig.vol_regime);
    const vP = $('volPill');
    vP.className = `pill ${vb.cls}`;
    vP.innerHTML = `<span class="dot" style="background:${vb.dot}"></span>
                    Vol ${vb.text}`;

    // Method pill
    const mb = methodBand(sig.method);
    const mP = $('methodPill');
    mP.className = `pill ${mb.cls}`;
    mP.innerHTML = `<span class="dot" style="background:${mb.dot}"></span>
                    ${mb.text}`;

    // Features table
    if (sig.explain && sig.explain.features_tail) {
      const rows = sig.explain.features_tail.map(x =>
        `${x.time}  close=${(+x.close).toFixed(6)}  ret=${Number(x.ret || 0).toFixed(6)}  vol24=${Number(x.vol24 || 0).toFixed(6)}`
      ).join("\n");
      $('sigExplain').innerHTML = `<pre>${rows}</pre>`;
    } else {
      $('sigExplain').innerHTML = `<pre>—</pre>`;
    }
  }

  function renderMLSignal(sig) {
    const pred = sig.ml_prediction || {};
    
    $('mlPredMeta').innerHTML = `
      <b>Symbol:</b> ${sig.symbol} &nbsp;&nbsp; 
      <b>Model:</b> ${sig.model_version || 'unknown'}<br/>
      <b>Prediction:</b> ${Number(pred.prediction || 0).toFixed(6)} &nbsp;&nbsp; 
      <b>Regime:</b> ${pred.volatility_regime || '—'}
    `;

    // Confidence pill
    const cb = confidenceBand(pred.confidence);
    const cP = $('mlConfidencePill');
    cP.className = `pill ${cb.cls}`;
    cP.innerHTML = `<span class="dot" style="background:${cb.dot}"></span>
                    ${cb.text}`;

    // ML Risk pill
    const risk = pred.risk_level;
    const rb = riskBand(risk === 'high' ? 0.8 : risk === 'medium' ? 0.5 : 0.2);
    const rP = $('mlRiskPill');
    rP.className = `pill ${rb.cls}`;
    rP.innerHTML = `<span class="dot" style="background:${rb.dot}"></span>
                    ${pred.risk_level || '—'} Risk`;

    // Show full prediction
    $('mlPredExplain').innerHTML = `<pre>${JSON.stringify(pred, null, 2)}</pre>`;
  }

  function renderComparison(comp) {
    // Heuristic results
    const heur = comp.heuristic;
    const heurHTML = `
      <div><b>Risk Score:</b> ${Number(heur.risk_score).toFixed(4)}</div>
      <div><b>Vol Regime:</b> ${heur.vol_regime}</div>
      <div><b>Volatility:</b> ${Number(heur.volatility).toFixed(6)}</div>
    `;
    $('heuristicResults').innerHTML = heurHTML;

    // ML results
    const ml = comp.ml;
    const mlHTML = `
      <div><b>Predicted Vol:</b> ${Number(ml.predicted_volatility || 0).toFixed(6)}</div>
      <div><b>Vol Regime:</b> ${ml.vol_regime || '—'}</div>
      <div><b>Confidence:</b> ${Number(ml.confidence || 0).toFixed(4)}</div>
      <div><b>Risk Level:</b> ${ml.risk_level || '—'}</div>
    `;
    $('mlResults').innerHTML = mlHTML;

    // Analysis
    const analysis = comp.comparison;
    const analysisHTML = `
      <div><b>Volatility Diff:</b> ${Number(analysis.volatility_diff).toFixed(6)}</div>
      <div><b>Regime Agreement:</b> ${analysis.regime_agreement ? '✅ Yes' : '❌ No'}</div>
      <div><b>ML Confidence:</b> ${Number(analysis.ml_confidence).toFixed(4)}</div>
    `;
    $('comparisonAnalysis').innerHTML = analysisHTML;
  }

  function renderOHLCV(data) {
    const head = ['time', 'open', 'high', 'low', 'close', 'volume'];
    $('tblOHLCV').querySelector('thead').innerHTML =
      `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>`;
    const rows = (data.data || []).map(r =>
      `<tr><td>${r.time}</td><td>${r.open}</td><td>${r.high}</td><td>${r.low}</td><td>${r.close}</td><td>${r.volume}</td></tr>`
    ).join('');
    $('tblOHLCV').querySelector('tbody').innerHTML = rows;
  }

  async function refreshHeuristicHistory() {
    try {
      const signals = await fetchJSON('/v1/crypto/signals/tail?n=10');
      const head = ['timestamp', 'symbol', 'method', 'risk_score', 'vol_regime', 'nowcast_ret'];
      $('tblHeuristicHistory').querySelector('thead').innerHTML =
        `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>`;
      
      const rows = signals.map(s => 
        `<tr>
          <td>${s.ts || '—'}</td>
          <td>${s.symbol || '—'}</td>
          <td>${s.method || 'heuristic'}</td>
          <td>${Number(s.risk_score || 0).toFixed(4)}</td>
          <td>${s.vol_regime || '—'}</td>
          <td>${Number(s.nowcast_ret || 0).toFixed(6)}</td>
        </tr>`
      ).join('');
      $('tblHeuristicHistory').querySelector('tbody').innerHTML = rows;
    } catch (e) {
      console.error('Error loading heuristic history:', e);
    }
  }

  async function refreshMLHistory() {
    try {
      const predictions = await fetchJSON('/v1/crypto/ml-predictions/tail?n=10');
      const head = ['timestamp', 'symbol', 'prediction', 'confidence', 'risk_level', 'regime'];
      $('tblMLHistory').querySelector('thead').innerHTML =
        `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>`;
      
      const rows = predictions.map(p => {
        const pred = p.ml_prediction || {};
        return `<tr>
          <td>${p.timestamp || '—'}</td>
          <td>${p.symbol || '—'}</td>
          <td>${Number(pred.prediction || 0).toFixed(6)}</td>
          <td>${Number(pred.confidence || 0).toFixed(4)}</td>
          <td>${pred.risk_level || '—'}</td>
          <td>${pred.volatility_regime || '—'}</td>
        </tr>`;
      }).join('');
      $('tblMLHistory').querySelector('tbody').innerHTML = rows;
    } catch (e) {
      console.error('Error loading ML history:', e);
    }
  }

  // === EVENT LISTENERS ===
  $('btnHeuristic').addEventListener('click', async () => {
    try {
      await runHeuristicSignal();
      await runOHLCV();
    } catch (e) {
      alert(`Error: ${e.message}`);
    }
  });

  $('btnML').addEventListener('click', async () => {
    try {
      await runMLSignal();
    } catch (e) {
      alert(`ML Error: ${e.message}`);
    }
  });

  $('btnCompare').addEventListener('click', async () => {
    try {
      await runComparison();
      showTab('compare'); // Switch to compare tab
    } catch (e) {
      alert(`Comparison Error: ${e.message}`);
    }
  });

  $('btnRefreshOHLCV').addEventListener('click', runOHLCV);
  $('btnRunComparison').addEventListener('click', runComparison);
  $('btnReloadModel').addEventListener('click', reloadModel);
  $('btnModelInfo').addEventListener('click', getModelInfo);
  $('btnRefreshHeuristicHistory').addEventListener('click', refreshHeuristicHistory);
  $('btnRefreshMLHistory').addEventListener('click', refreshMLHistory);

  // === INITIALIZATION ===
  (async () => {
    try {
      // Check ML status first
      await checkMLStatus();
      
      // Load initial data
      await runOHLCV();
      await runHeuristicSignal();
      
      // Try to get model info if ML is available
      try {
        await getModelInfo();
      } catch (e) {
        console.log('ML model info not available:', e.message);
      }
      
      // Load history
      await refreshHeuristicHistory();
      try {
        await refreshMLHistory();
      } catch (e) {
        console.log('ML history not available:', e.message);
      }
      
    } catch (e) {
      console.error('Initialization error:', e);
    }
  })();
</script>
</body>
</html>
"""
    return HTMLResponse(html)

# -----------------------------------------------------------------------------
# STARTUP EVENT
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Inicialización al startup"""
    print("🚀 Crypto MLOps MVP - Integrated Starting...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🤖 ML Available: {ML_AVAILABLE}")
    
    if ML_AVAILABLE:
        try:
            model_info = ml_service.get_model_info()
            print(f"🧠 ML Model loaded: {model_info['model_loaded']}")
            print(f"📊 Model version: {model_info.get('model_version', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Error getting ML model info: {e}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)