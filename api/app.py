"""
Crypto MLOps MVP - API FastAPI extendida con ML capabilities
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import pandas as pd
import numpy as np
import ccxt
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Configurar paths
sys.path.append('/app')
sys.path.append('/app/ml/inference')

# Importar servicios ML
try:
    from ml_service import ml_service
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML service not available: {e}")
    ML_AVAILABLE = False

# === MODELOS DE DATOS ===
class OHLCVRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    exchange: str = Field(default="binance", description="Exchange name")
    timeframe: str = Field(default="1h", description="Timeframe")
    limit: int = Field(default=200, description="Number of candles")

class SignalRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    horizon_min: int = Field(default=60, description="Prediction horizon in minutes")
    explain: bool = Field(default=True, description="Include explanation")
    exchange: str = Field(default="binance", description="Exchange name") 
    timeframe: str = Field(default="1h", description="Timeframe")
    limit: int = Field(default=200, description="Number of candles")

class MLSignalRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    exchange: str = Field(default="binance", description="Exchange name")
    timeframe: str = Field(default="1h", description="Timeframe")
    limit: int = Field(default=200, description="Number of candles for context")
    include_heuristic: bool = Field(default=True, description="Include heuristic comparison")

# === INICIALIZACIÓN ===
app = FastAPI(
    title="Crypto MLOps MVP API",
    description="Extended crypto signals API with ML capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Métricas in-memory (existente)
metrics = {
    "start_time": datetime.now().isoformat() + "Z",
    "requests_total": 0,
    "signals_total": 0,
    "ml_predictions_total": 0,
    "last_signal_at": None,
    "last_ml_prediction_at": None
}

# Directorio de datos
DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(exist_ok=True)
SIGNALS_FILE = DATA_DIR / "signals.jsonl"
ML_PREDICTIONS_FILE = DATA_DIR / "ml_predictions.jsonl"

# === FUNCIONES EXISTENTES ===
def get_crypto_data(symbol: str, exchange: str = "binance", 
                   timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
    """Obtener datos OHLCV de la exchange"""
    try:
        ex = getattr(ccxt, exchange)()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")

def calculate_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcular features técnicos (función existente)"""
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['vol24'] = df['ret'].rolling(window=24).std()
    df['sma12'] = df['close'].rolling(window=12).mean()
    df['sma48'] = df['close'].rolling(window=48).mean()
    
    # Última fila con features
    last_row = df.iloc[-1]
    
    # Nowcast return (predicción simple)
    recent_ret = df['ret'].tail(12).mean()
    vol = last_row['vol24'] if pd.notna(last_row['vol24']) else 0.02
    
    # Clasificar régimen de volatilidad
    if vol < 0.015:
        vol_regime = "calm"
    elif vol < 0.035:
        vol_regime = "normal" 
    else:
        vol_regime = "turbulent"
    
    # Risk score heurístico
    risk_score = min(0.99, max(0.01, vol * 50 + abs(recent_ret) * 10))
    
    return {
        "nowcast_ret": float(recent_ret) if pd.notna(recent_ret) else 0.0,
        "vol": float(vol),
        "vol_regime": vol_regime,
        "risk_score": float(risk_score),
        "features_tail": [{
            "time": last_row['timestamp'].isoformat() if 'timestamp' in df.columns else datetime.now().isoformat(),
            "close": float(last_row['close']),
            "ret": float(last_row['ret']) if pd.notna(last_row['ret']) else 0.0,
            "vol24": float(last_row['vol24']) if pd.notna(last_row['vol24']) else 0.0,
            "sma12": float(last_row['sma12']) if pd.notna(last_row['sma12']) else float(last_row['close']),
            "sma48": float(last_row['sma48']) if pd.notna(last_row['sma48']) else float(last_row['close'])
        }]
    }

def persist_signal(signal_data: Dict[str, Any], filename: Path):
    """Persistir señal en archivo JSONL"""
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal_data, default=str) + "\n")
    except Exception as e:
        print(f"Error persisting signal: {e}")

# === ENDPOINTS EXISTENTES ===
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global metrics
    metrics["requests_total"] += 1
    
    return {
        "status": "ok",
        "ml_available": ML_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Métricas del servicio"""
    global metrics
    metrics["requests_total"] += 1
    return metrics

@app.get("/v1/crypto/ohlcv")
async def get_ohlcv_data(
    symbol: str = "BTCUSDT",
    exchange: str = "binance", 
    timeframe: str = "1h",
    limit: int = 200
):
    """Obtener datos OHLCV"""
    global metrics
    metrics["requests_total"] += 1
    
    df = get_crypto_data(symbol, exchange, timeframe, limit)
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe, 
        "limit": len(df),
        "rows": len(df),
        "data": df.to_dict('records')
    }

@app.post("/v1/crypto/signal")
async def generate_signal(request: SignalRequest):
    """Generar señal heurística (endpoint existente)"""
    global metrics
    metrics["requests_total"] += 1
    metrics["signals_total"] += 1
    metrics["last_signal_at"] = datetime.now().isoformat() + "Z"
    
    # Obtener datos
    df = get_crypto_data(request.symbol, request.exchange, request.timeframe, request.limit)
    
    # Calcular features
    features = calculate_features(df)
    
    # Preparar respuesta
    response = {
        "symbol": request.symbol,
        "horizon_min": request.horizon_min,
        "risk_score": features["risk_score"],
        "nowcast_ret": features["nowcast_ret"],
        "vol_regime": features["vol_regime"],
        "timestamp": datetime.now().isoformat(),
        "method": "heuristic"
    }
    
    if request.explain:
        response["explain"] = features
    
    # Persistir
    persist_signal(response, SIGNALS_FILE)
    
    return response

@app.get("/v1/crypto/signals/tail")
async def get_recent_signals(n: int = 5):
    """Obtener últimas n señales"""
    global metrics
    metrics["requests_total"] += 1
    
    if not SIGNALS_FILE.exists():
        return []
    
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        recent_lines = lines[-n:] if len(lines) >= n else lines
        signals = [json.loads(line.strip()) for line in recent_lines if line.strip()]
        
        return signals[::-1]  # Más reciente primero
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading signals: {str(e)}")

# === NUEVOS ENDPOINTS ML ===
@app.post("/v1/crypto/ml-signal")
async def generate_ml_signal(request: MLSignalRequest):
    """Generar señal usando modelo ML"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    global metrics
    metrics["requests_total"] += 1
    metrics["ml_predictions_total"] += 1
    metrics["last_ml_prediction_at"] = datetime.now().isoformat() + "Z"
    
    # Obtener datos
    df = get_crypto_data(request.symbol, request.exchange, request.timeframe, request.limit)
    
    # Predicción ML
    ml_prediction = ml_service.predict_volatility(df)
    
    # Preparar respuesta base
    response = {
        "symbol": request.symbol,
        "timestamp": datetime.now().isoformat(),
        "method": "ml",
        "model_version": ml_prediction.get("model_version"),
        "ml_prediction": ml_prediction
    }
    
    # Incluir comparación heurística si se solicita
    if request.include_heuristic:
        heuristic_features = calculate_features(df)
        response["heuristic_comparison"] = {
            "heuristic_risk_score": heuristic_features["risk_score"],
            "heuristic_vol_regime": heuristic_features["vol_regime"],
            "ml_vs_heuristic": {
                "volatility_diff": ml_prediction.get("prediction", 0) - heuristic_features["vol"],
                "regime_match": ml_prediction.get("volatility_regime") == heuristic_features["vol_regime"]
            }
        }
    
    # Persistir predicción ML
    persist_signal(response, ML_PREDICTIONS_FILE)
    
    return response

@app.get("/v1/crypto/ml-predictions/tail")
async def get_recent_ml_predictions(n: int = 5):
    """Obtener últimas n predicciones ML"""
    global metrics
    metrics["requests_total"] += 1
    
    if not ML_PREDICTIONS_FILE.exists():
        return []
    
    try:
        with open(ML_PREDICTIONS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        recent_lines = lines[-n:] if len(lines) >= n else lines
        predictions = [json.loads(line.strip()) for line in recent_lines if line.strip()]
        
        return predictions[::-1]  # Más reciente primero
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading ML predictions: {str(e)}")

@app.get("/v1/ml/model/info")
async def get_model_info():
    """Información del modelo ML cargado"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    global metrics
    metrics["requests_total"] += 1
    
    return ml_service.get_model_info()

@app.post("/v1/ml/model/reload")
async def reload_model():
    """Recargar modelo ML desde MLflow"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    global metrics  
    metrics["requests_total"] += 1
    
    result = ml_service.reload_model()
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result

@app.get("/v1/crypto/signals/compare")
async def compare_signals(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h", 
    limit: int = 200
):
    """Comparar señal heurística vs ML"""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    global metrics
    metrics["requests_total"] += 1
    
    # Obtener datos
    df = get_crypto_data(symbol, exchange, timeframe, limit)
    
    # Generar ambas señales
    heuristic_features = calculate_features(df)
    ml_prediction = ml_service.predict_volatility(df)
    
    # Comparación
    comparison = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
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
    
    return comparison

# === STARTUP ===
@app.on_event("startup")
async def startup_event():
    """Inicialización al startup"""
    print("🚀 Crypto MLOps MVP API Starting...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🤖 ML Available: {ML_AVAILABLE}")
    
    if ML_AVAILABLE:
        model_info = ml_service.get_model_info()
        print(f"🧠 ML Model loaded: {model_info['model_loaded']}")
        print(f"📊 Model version: {model_info['model_version']}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)