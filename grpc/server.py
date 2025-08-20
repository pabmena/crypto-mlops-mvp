"""
gRPC Server para Crypto MLOps API
"""
import grpc
from concurrent import futures
import time
import requests
import json
from datetime import datetime
import asyncio
import threading

# Importar protos generados (se generarán con protoc)
import crypto_service_pb2
import crypto_service_pb2_grpc

class CryptoServicer(crypto_service_pb2_grpc.CryptoServiceServicer):
    def __init__(self):
        self.api_endpoint = "http://api:8000"
        
    def GetOHLCV(self, request, context):
        """Obtener datos OHLCV"""
        try:
            params = {
                'symbol': request.symbol or 'BTCUSDT',
                'exchange': request.exchange or 'binance',
                'timeframe': request.timeframe or '1h',
                'limit': request.limit or 200
            }
            
            response = requests.get(f"{self.api_endpoint}/v1/crypto/ohlcv", params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                ohlcv_data = []
                for item in data.get('data', []):
                    ohlcv_data.append(crypto_service_pb2.OHLCVData(
                        timestamp=int(item.get('timestamp', 0)),
                        open=float(item.get('open', 0)),
                        high=float(item.get('high', 0)),
                        low=float(item.get('low', 0)),
                        close=float(item.get('close', 0)),
                        volume=float(item.get('volume', 0))
                    ))
                
                return crypto_service_pb2.OHLCVResponse(
                    symbol=data.get('symbol', ''),
                    exchange=data.get('exchange', ''),
                    timeframe=data.get('timeframe', ''),
                    rows=data.get('rows', 0),
                    data=ohlcv_data
                )
            else:
                return crypto_service_pb2.OHLCVResponse(
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return crypto_service_pb2.OHLCVResponse(
                error=f"gRPC error: {str(e)}"
            )
    
    def GenerateSignal(self, request, context):
        """Generar señal heurística"""
        try:
            payload = {
                'symbol': request.symbol or 'BTCUSDT',
                'horizon_min': request.horizon_min or 60,
                'explain': request.explain,
                'exchange': request.exchange or 'binance',
                'timeframe': request.timeframe or '1h',
                'limit': request.limit or 200
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
                    features_tail = []
                    
                    for feature in explain_data.get('features_tail', []):
                        features_tail.append(crypto_service_pb2.FeatureData(
                            time=feature.get('time', ''),
                            close=float(feature.get('close', 0)),
                            ret=float(feature.get('ret', 0)),
                            vol24=float(feature.get('vol24', 0)),
                            sma12=float(feature.get('sma12', 0)),
                            sma48=float(feature.get('sma48', 0))
                        ))
                    
                    explanation = crypto_service_pb2.SignalExplanation(
                        nowcast_ret=float(explain_data.get('nowcast_ret', 0)),
                        vol=float(explain_data.get('vol', 0)),
                        vol_regime=explain_data.get('vol_regime', ''),
                        risk_score=float(explain_data.get('risk_score', 0)),
                        features_tail=features_tail
                    )
                
                return crypto_service_pb2.SignalResponse(
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
                return crypto_service_pb2.SignalResponse(
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return crypto_service_pb2.SignalResponse(
                error=f"gRPC error: {str(e)}"
            )
    
    def GenerateMLPrediction(self, request, context):
        """Generar predicción ML"""
        try:
            payload = {
                'symbol': request.symbol or 'BTCUSDT',
                'exchange': request.exchange or 'binance',
                'timeframe': request.timeframe or '1h',
                'limit': request.limit or 200,
                'include_heuristic': request.include_heuristic
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
                ml_prediction = crypto_service_pb2.MLPrediction(
                    prediction=float(ml_pred_data.get('prediction', 0)),
                    current_volatility=float(ml_pred_data.get('current_volatility', 0)),
                    volatility_regime=ml_pred_data.get('volatility_regime', ''),
                    risk_level=ml_pred_data.get('risk_level', ''),
                    confidence=float(ml_pred_data.get('confidence', 0)),
                    prediction_timestamp=ml_pred_data.get('prediction_timestamp', ''),
                    error=ml_pred_data.get('error', '')
                )
                
                # Heuristic Comparison (si disponible)
                heuristic_comparison = None
                if data.get('heuristic_comparison'):
                    heur_comp = data['heuristic_comparison']
                    ml_vs_heur = heur_comp.get('ml_vs_heuristic', {})
                    
                    heuristic_comparison = crypto_service_pb2.HeuristicComparison(
                        heuristic_risk_score=float(heur_comp.get('heuristic_risk_score', 0)),
                        heuristic_vol_regime=heur_comp.get('heuristic_vol_regime', ''),
                        ml_vs_heuristic=crypto_service_pb2.MLVsHeuristic(
                            volatility_diff=float(ml_vs_heur.get('volatility_diff', 0)),
                            regime_match=bool(ml_vs_heur.get('regime_match', False))
                        )
                    )
                
                return crypto_service_pb2.MLPredictionResponse(
                    symbol=data.get('symbol', ''),
                    timestamp=data.get('timestamp', ''),
                    method=data.get('method', ''),
                    model_version=data.get('model_version', ''),
                    ml_prediction=ml_prediction,
                    heuristic_comparison=heuristic_comparison
                )
            else:
                return crypto_service_pb2.MLPredictionResponse(
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return crypto_service_pb2.MLPredictionResponse(
                error=f"gRPC error: {str(e)}"
            )
    
    def CompareSignals(self, request, context):
        """Comparar señales heurística vs ML"""
        try:
            params = {
                'symbol': request.symbol or 'BTCUSDT',
                'exchange': request.exchange or 'binance',
                'timeframe': request.timeframe or '1h',
                'limit': request.limit or 200
            }
            
            response = requests.get(
                f"{self.api_endpoint}/v1/crypto/signals/compare",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Heuristic signal
                heur_data = data.get('heuristic', {})
                heuristic_signal = crypto_service_pb2.HeuristicSignal(
                    risk_score=float(heur_data.get('risk_score', 0)),
                    vol_regime=heur_data.get('vol_regime', ''),
                    volatility=float(heur_data.get('volatility', 0))
                )
                
                # ML signal
                ml_data = data.get('ml', {})
                ml_signal = crypto_service_pb2.MLSignal(
                    predicted_volatility=float(ml_data.get('predicted_volatility', 0)),
                    vol_regime=ml_data.get('vol_regime', ''),
                    confidence=float(ml_data.get('confidence', 0)),
                    risk_level=ml_data.get('risk_level', '')
                )
                
                # Comparison
                comp_data = data.get('comparison', {})
                comparison = crypto_service_pb2.SignalComparison(
                    volatility_diff=float(comp_data.get('volatility_diff', 0)),
                    regime_agreement=bool(comp_data.get('regime_agreement', False)),
                    ml_confidence=float(comp_data.get('ml_confidence', 0))
                )
                
                return crypto_service_pb2.CompareSignalsResponse(
                    symbol=data.get('symbol', ''),
                    timestamp=data.get('timestamp', ''),
                    heuristic=heuristic_signal,
                    ml=ml_signal,
                    comparison=comparison
                )
            else:
                return crypto_service_pb2.CompareSignalsResponse(
                    error=f"API error: {response.status_code}"
                )
                
        except Exception as e:
            return crypto_service_pb2.CompareSignalsResponse(
                error=f"gRPC error: {str(e)}"
            )
    
    def HealthCheck(self, request, context):
        """Health check del servicio"""
        try:
            response = requests.get(f"{self.api_endpoint}/health")
            
            if response.status_code == 200:
                data = response.json()
                return crypto_service_pb2.HealthCheckResponse(
                    status=data.get('status', 'unknown'),
                    ml_available=bool(data.get('ml_available', False)),
                    timestamp=data.get('timestamp', datetime.now().isoformat()),
                    version="1.0.0"
                )
            else:
                return crypto_service_pb2.HealthCheckResponse(
                    status="error",
                    ml_available=False,
                    timestamp=datetime.now().isoformat(),
                    version="1.0.0"
                )
                
        except Exception as e:
            return crypto_service_pb2.HealthCheckResponse(
                status="error",
                ml_available=False,
                timestamp=datetime.now().isoformat(),
                version="1.0.0"
            )
    
    def StreamPrices(self, request, context):
        """Stream de precios en tiempo real"""
        symbols = request.symbols if request.symbols else ['BTCUSDT']
        update_interval = max(request.update_interval_ms, 1000) / 1000.0  # mínimo 1 segundo
        
        print(f"Starting price stream for symbols: {symbols}, interval: {update_interval}s")
        
        try:
            while True:
                for symbol in symbols:
                    try:
                        # Obtener datos actuales
                        params = {'symbol': symbol, 'limit': 24}  # últimas 24 horas
                        response = requests.get(f"{self.api_endpoint}/v1/crypto/ohlcv", params=params)
                        
                        if response.status_code == 200:
                            data = response.json()
                            ohlcv_data = data.get('data', [])
                            
                            if ohlcv_data:
                                latest = ohlcv_data[-1]
                                first = ohlcv_data[0] if len(ohlcv_data) > 1 else latest
                                
                                # Calcular cambio 24h
                                current_price = float(latest.get('close', 0))
                                old_price = float(first.get('close', current_price))
                                change_24h = ((current_price - old_price) / old_price) * 100 if old_price > 0 else 0
                                
                                # Calcular volatilidad simple
                                prices = [float(item.get('close', 0)) for item in ohlcv_data[-24:]]
                                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
                                volatility = (sum(r*r for r in returns) / len(returns)) ** 0.5 if returns else 0
                                
                                # Determinar régimen
                                if volatility < 0.01:
                                    regime = "calm"
                                elif volatility < 0.03:
                                    regime = "normal"
                                else:
                                    regime = "turbulent"
                                
                                price_update = crypto_service_pb2.PriceUpdate(
                                    symbol=symbol,
                                    price=current_price,
                                    change_24h=change_24h,
                                    volume_24h=float(latest.get('volume', 0)),
                                    timestamp=int(time.time() * 1000),
                                    volatility=volatility,
                                    regime=regime
                                )
                                
                                yield price_update
                                
                    except Exception as e:
                        print(f"Error streaming {symbol}: {e}")
                        continue
                
                time.sleep(update_interval)
                
        except Exception as e:
            print(f"Stream error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, f"Stream failed: {str(e)}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    crypto_service_pb2_grpc.add_CryptoServiceServicer_to_server(CryptoServicer(), server)
    listen_addr = '0.0.0.0:50051'
    server.add_insecure_port(listen_addr)
    
    print(f"🚀 gRPC Server starting on {listen_addr}")
    server.start()
    
    try:
        while True:
            time.sleep(86400)  # Keep running
    except KeyboardInterrupt:
        print("🛑 Shutting down gRPC server...")
        server.stop(0)

if __name__ == '__main__':
    serve()