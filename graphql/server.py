"""
GraphQL Server para Crypto MLOps API
"""
import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
from typing import List, Optional
import requests
from datetime import datetime
import json

# === TIPOS GRAPHQL ===
@strawberry.type
class OHLCVData:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@strawberry.type
class OHLCVResponse:
    symbol: str
    exchange: str
    timeframe: str
    rows: int
    data: List[OHLCVData]
    error: Optional[str] = None

@strawberry.type
class FeatureData:
    time: str
    close: float
    ret: float
    vol24: float
    sma12: float
    sma48: float

@strawberry.type
class SignalExplanation:
    nowcast_ret: float
    vol: float
    vol_regime: str
    risk_score: float
    features_tail: List[FeatureData]

@strawberry.type
class Signal:
    symbol: str
    horizon_min: int
    risk_score: float
    nowcast_ret: float
    vol_regime: str
    timestamp: str
    method: str
    explanation: Optional[SignalExplanation] = None
    error: Optional[str] = None

@strawberry.type
class MLPrediction:
    prediction: Optional[float]
    current_volatility: Optional[float]
    volatility_regime: Optional[str]
    risk_level: Optional[str]
    confidence: float
    prediction_timestamp: Optional[str]
    error: Optional[str]

@strawberry.type
class HeuristicComparison:
    heuristic_risk_score: float
    heuristic_vol_regime: str
    volatility_diff: float
    regime_match: bool

@strawberry.type
class MLSignal:
    symbol: str
    timestamp: str
    method: str
    model_version: Optional[str]
    ml_prediction: MLPrediction
    heuristic_comparison: Optional[HeuristicComparison] = None
    error: Optional[str] = None

@strawberry.type
class ModelInfo:
    model_loaded: bool
    model_version: Optional[str]
    model_name: str
    mlflow_uri: str
    sequence_length: Optional[int]

@strawberry.type
class HealthStatus:
    status: str
    ml_available: bool
    timestamp: str

@strawberry.type
class SignalComparison:
    symbol: str
    timestamp: str
    heuristic_risk_score: float
    heuristic_vol_regime: str
    heuristic_volatility: float
    ml_predicted_volatility: Optional[float]
    ml_vol_regime: Optional[str]
    ml_confidence: float
    ml_risk_level: Optional[str]
    volatility_diff: Optional[float]
    regime_agreement: bool
    error: Optional[str] = None

# === INPUTS ===
@strawberry.input
class OHLCVInput:
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    limit: int = 200

@strawberry.input
class SignalInput:
    symbol: str = "BTCUSDT"
    horizon_min: int = 60
    explain: bool = True
    exchange: str = "binance"
    timeframe: str = "1h"
    limit: int = 200

@strawberry.input
class MLSignalInput:
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    limit: int = 200
    include_heuristic: bool = True

# === RESOLVERS ===
class CryptoAPI:
    def __init__(self):
        self.api_endpoint = "http://api:8000"
    
    def get_ohlcv_data(self, input: OHLCVInput) -> OHLCVResponse:
        """Obtener datos OHLCV"""
        try:
            params = {
                'symbol': input.symbol,
                'exchange': input.exchange,
                'timeframe': input.timeframe,
                'limit': input.limit
            }
            
            response = requests.get(f"{self.api_endpoint}/v1/crypto/ohlcv", params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                ohlcv_data = [
                    OHLCVData(
                        timestamp=int(item.get('timestamp', 0)),
                        open=float(item.get('open', 0)),
                        high=float(item.get('high', 0)),
                        low=float(item.get('low', 0)),
                        close=float(item.get('close', 0)),
                        volume=float(item.get('volume', 0))
                    )
                    for item in data.get('data', [])
                ]
                
                return OHLCVResponse(
                    symbol=data.get('symbol', ''),
                    exchange=data.get('exchange', ''),
                    timeframe=data.get('timeframe', ''),
                    rows=data.get('rows', 0),
                    data=ohlcv_data
                )
            else:
                return OHLCVResponse(
                    symbol=input.symbol,
                    exchange=input.exchange,
                    timeframe=input.timeframe,
                    rows=0,
                    data=[],
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return OHLCVResponse(
                symbol=input.symbol,
                exchange=input.exchange,
                timeframe=input.timeframe,
                rows=0,
                data=[],
                error=f"Request error: {str(e)}"
            )
    
    def generate_signal(self, input: SignalInput) -> Signal:
        """Generar señal heurística"""
        try:
            payload = {
                'symbol': input.symbol,
                'horizon_min': input.horizon_min,
                'explain': input.explain,
                'exchange': input.exchange,
                'timeframe': input.timeframe,
                'limit': input.limit
            }
            
            response = requests.post(
                f"{self.api_endpoint}/v1/crypto/signal",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                explanation = None
                if data.get('explain'):
                    explain_data = data['explain']
                    features_tail = [
                        FeatureData(
                            time=feature.get('time', ''),
                            close=float(feature.get('close', 0)),
                            ret=float(feature.get('ret', 0)),
                            vol24=float(feature.get('vol24', 0)),
                            sma12=float(feature.get('sma12', 0)),
                            sma48=float(feature.get('sma48', 0))
                        )
                        for feature in explain_data.get('features_tail', [])
                    ]
                    
                    explanation = SignalExplanation(
                        nowcast_ret=float(explain_data.get('nowcast_ret', 0)),
                        vol=float(explain_data.get('vol', 0)),
                        vol_regime=explain_data.get('vol_regime', ''),
                        risk_score=float(explain_data.get('risk_score', 0)),
                        features_tail=features_tail
                    )
                
                return Signal(
                    symbol=data.get('symbol', ''),
                    horizon_min=data.get('horizon_min', 0),
                    risk_score=float(data.get('risk_score', 0)),
                    nowcast_ret=float(data.get('nowcast_ret', 0)),
                    vol_regime=data.get('vol_regime', ''),
                    timestamp=data.get('timestamp', ''),
                    method=data.get('method', ''),
                    explanation=explanation
                )
            else:
                return Signal(
                    symbol=input.symbol,
                    horizon_min=input.horizon_min,
                    risk_score=0,
                    nowcast_ret=0,
                    vol_regime='unknown',
                    timestamp=datetime.now().isoformat(),
                    method='heuristic',
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return Signal(
                symbol=input.symbol,
                horizon_min=input.horizon_min,
                risk_score=0,
                nowcast_ret=0,
                vol_regime='unknown',
                timestamp=datetime.now().isoformat(),
                method='heuristic',
                error=f"Request error: {str(e)}"
            )
    
    def generate_ml_signal(self, input: MLSignalInput) -> MLSignal:
        """Generar señal ML"""
        try:
            payload = {
                'symbol': input.symbol,
                'exchange': input.exchange,
                'timeframe': input.timeframe,
                'limit': input.limit,
                'include_heuristic': input.include_heuristic
            }
            
            response = requests.post(
                f"{self.api_endpoint}/v1/crypto/ml-signal",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # ML Prediction
                ml_pred_data = data.get('ml_prediction', {})
                ml_prediction = MLPrediction(
                    prediction=ml_pred_data.get('prediction'),
                    current_volatility=ml_pred_data.get('current_volatility'),
                    volatility_regime=ml_pred_data.get('volatility_regime'),
                    risk_level=ml_pred_data.get('risk_level'),
                    confidence=float(ml_pred_data.get('confidence', 0)),
                    prediction_timestamp=ml_pred_data.get('prediction_timestamp'),
                    error=ml_pred_data.get('error')
                )
                
                # Heuristic Comparison
                heuristic_comparison = None
                if data.get('heuristic_comparison'):
                    heur_comp = data['heuristic_comparison']
                    ml_vs_heur = heur_comp.get('ml_vs_heuristic', {})
                    
                    heuristic_comparison = HeuristicComparison(
                        heuristic_risk_score=float(heur_comp.get('heuristic_risk_score', 0)),
                        heuristic_vol_regime=heur_comp.get('heuristic_vol_regime', ''),
                        volatility_diff=float(ml_vs_heur.get('volatility_diff', 0)),
                        regime_match=bool(ml_vs_heur.get('regime_match', False))
                    )
                
                return MLSignal(
                    symbol=data.get('symbol', ''),
                    timestamp=data.get('timestamp', ''),
                    method=data.get('method', ''),
                    model_version=data.get('model_version'),
                    ml_prediction=ml_prediction,
                    heuristic_comparison=heuristic_comparison
                )
            else:
                return MLSignal(
                    symbol=input.symbol,
                    timestamp=datetime.now().isoformat(),
                    method='ml',
                    model_version=None,
                    ml_prediction=MLPrediction(
                        prediction=None,
                        current_volatility=None,
                        volatility_regime=None,
                        risk_level=None,
                        confidence=0,
                        prediction_timestamp=None,
                        error=f"API error: {response.status_code}"
                    ),
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return MLSignal(
                symbol=input.symbol,
                timestamp=datetime.now().isoformat(),
                method='ml',
                model_version=None,
                ml_prediction=MLPrediction(
                    prediction=None,
                    current_volatility=None,
                    volatility_regime=None,
                    risk_level=None,
                    confidence=0,
                    prediction_timestamp=None,
                    error=f"Request error: {str(e)}"
                ),
                error=f"Request error: {str(e)}"
            )
    
    def get_model_info(self) -> ModelInfo:
        """Obtener información del modelo ML"""
        try:
            response = requests.get(f"{self.api_endpoint}/v1/ml/model/info")
            
            if response.status_code == 200:
                data = response.json()
                return ModelInfo(
                    model_loaded=bool(data.get('model_loaded', False)),
                    model_version=data.get('model_version'),
                    model_name=data.get('model_name', ''),
                    mlflow_uri=data.get('mlflow_uri', ''),
                    sequence_length=data.get('sequence_length')
                )
            else:
                return ModelInfo(
                    model_loaded=False,
                    model_version=None,
                    model_name='unknown',
                    mlflow_uri='unknown',
                    sequence_length=None
                )
                
        except Exception as e:
            return ModelInfo(
                model_loaded=False,
                model_version=None,
                model_name='error',
                mlflow_uri='error',
                sequence_length=None
            )
    
    def get_health_status(self) -> HealthStatus:
        """Health check"""
        try:
            response = requests.get(f"{self.api_endpoint}/health")
            
            if response.status_code == 200:
                data = response.json()
                return HealthStatus(
                    status=data.get('status', 'unknown'),
                    ml_available=bool(data.get('ml_available', False)),
                    timestamp=data.get('timestamp', datetime.now().isoformat())
                )
            else:
                return HealthStatus(
                    status='error',
                    ml_available=False,
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            return HealthStatus(
                status='error',
                ml_available=False,
                timestamp=datetime.now().isoformat()
            )

# Instancia global del API
crypto_api = CryptoAPI()

# === QUERIES ===
@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> HealthStatus:
        """Health check del sistema"""
        return crypto_api.get_health_status()
    
    @strawberry.field
    def model_info(self) -> ModelInfo:
        """Información del modelo ML cargado"""
        return crypto_api.get_model_info()
    
    @strawberry.field
    def ohlcv_data(self, input: OHLCVInput) -> OHLCVResponse:
        """Obtener datos OHLCV de criptomonedas"""
        return crypto_api.get_ohlcv_data(input)

# === MUTATIONS ===
@strawberry.type
class Mutation:
    @strawberry.field
    def generate_signal(self, input: SignalInput) -> Signal:
        """Generar señal heurística"""
        return crypto_api.generate_signal(input)
    
    @strawberry.field
    def generate_ml_signal(self, input: MLSignalInput) -> MLSignal:
        """Generar señal usando modelo ML"""
        return crypto_api.generate_ml_signal(input)

# === ESQUEMA ===
schema = strawberry.Schema(query=Query, mutation=Mutation)

# === APLICACION FASTAPI ===
app = FastAPI(title="Crypto MLOps GraphQL API", version="1.0.0")

graphql_app = GraphQLRouter(schema, graphiql=True)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
async def root():
    return {
        "message": "Crypto MLOps GraphQL API",
        "version": "1.0.0",
        "graphql_endpoint": "/graphql",
        "playground": "/graphql (GraphiQL)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)