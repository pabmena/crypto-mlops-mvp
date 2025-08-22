"""
Kafka Consumer para procesar datos crypto y generar predicciones
"""
import json
import time
import os
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaProducer
import logging
import pandas as pd
from collections import defaultdict, deque
import requests
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoConsumer:
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
        self.input_topic = os.getenv('KAFKA_TOPIC', 'crypto-prices')
        self.prediction_topic = 'predictions'
        self.alert_topic = 'alerts'
        
        # Buffer para almacenar datos históricos por símbolo
        self.price_buffer = defaultdict(lambda: deque(maxlen=100))  # Últimas 100 observaciones
        self.prediction_cache = {}
        
        # APIs
        self.api_endpoint = os.getenv('API_ENDPOINT', 'http://api:8000')
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        
        # Configurar consumer
        self.consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=[self.kafka_servers],
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='crypto-ml-processor',
            auto_offset_reset='latest',
            enable_auto_commit=True,
            consumer_timeout_ms=10000  # 10 segundos timeout
        )
        
        # Configurar producer para enviar predicciones
        self.producer = KafkaProducer(
            bootstrap_servers=[self.kafka_servers],
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
        )
        
        logger.info(f"🚀 CryptoConsumer initialized")
        logger.info(f"📥 Consuming from: {self.input_topic}")
        logger.info(f"📤 Publishing to: {self.prediction_topic}, {self.alert_topic}")
    
    def add_to_buffer(self, symbol: str, price_data: Dict[str, Any]):
        """Agregar datos al buffer histórico"""
        buffer_entry = {
            'timestamp': price_data.get('timestamp'),
            'price': price_data.get('price'),
            'volume': price_data.get('volume_24h', 0),
            'change_24h': price_data.get('change_24h', 0),
            'volatility': price_data.get('volatility', 0),
            'source': price_data.get('source', 'unknown')
        }
        
        self.price_buffer[symbol].append(buffer_entry)
        logger.debug(f"📊 Buffer updated for {symbol}: {len(self.price_buffer[symbol])} entries")
    
    def calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calcular indicadores técnicos en tiempo real"""
        buffer = self.price_buffer[symbol]
        
        if len(buffer) < 12:  # Necesitamos al menos 12 puntos
            return None
        
        # Convertir a DataFrame para cálculos
        df = pd.DataFrame(list(buffer))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calcular indicadores
        df['returns'] = df['price'].pct_change()
        df['sma_12'] = df['price'].rolling(window=min(12, len(df))).mean()
        df['sma_48'] = df['price'].rolling(window=min(48, len(df))).mean()
        df['volatility_rolling'] = df['returns'].rolling(window=min(24, len(df))).std()
        
        # RSI
        if len(df) >= 14:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            df['rsi'] = 50  # Neutral
        
        # Obtener última fila
        latest = df.iloc[-1]
        
        return {
            'symbol': symbol,
            'current_price': float(latest['price']),
            'sma_12': float(latest['sma_12']) if pd.notna(latest['sma_12']) else float(latest['price']),
            'sma_48': float(latest['sma_48']) if pd.notna(latest['sma_48']) else float(latest['price']),
            'rsi': float(latest['rsi']) if pd.notna(latest['rsi']) else 50.0,
            'volatility': float(latest['volatility_rolling']) if pd.notna(latest['volatility_rolling']) else 0.02,
            'trend': 'bullish' if latest['sma_12'] > latest['sma_48'] else 'bearish',
            'timestamp': latest['timestamp'].isoformat(),
            'buffer_size': len(buffer)
        }
    
    def generate_heuristic_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generar señal heurística en tiempo real"""
        if not indicators:
            return None
        
        symbol = indicators['symbol']
        volatility = indicators['volatility']
        rsi = indicators['rsi']
        
        # Clasificar régimen de volatilidad
        if volatility < 0.015:
            vol_regime = "calm"
            risk_level = "low"
        elif volatility < 0.035:
            vol_regime = "normal"
            risk_level = "medium"
        else:
            vol_regime = "turbulent"
            risk_level = "high"
        
        # Score de riesgo combinado
        vol_score = min(0.5, volatility * 25)  # Máximo 0.5
        rsi_score = abs(rsi - 50) / 100  # 0 = neutral, 0.5 = extremo
        risk_score = min(0.99, vol_score + rsi_score * 0.3)
        
        # Señal de trading simple
        if rsi > 70 and indicators['trend'] == 'bullish':
            signal = "overbought"
        elif rsi < 30 and indicators['trend'] == 'bearish':
            signal = "oversold"
        elif indicators['trend'] == 'bullish' and rsi > 50:
            signal = "bullish"
        elif indicators['trend'] == 'bearish' and rsi < 50:
            signal = "bearish"
        else:
            signal = "neutral"
        
        return {
            'symbol': symbol,
            'method': 'heuristic_realtime',
            'signal': signal,
            'risk_score': round(risk_score, 4),
            'risk_level': risk_level,
            'vol_regime': vol_regime,
            'volatility': round(volatility, 6),
            'rsi': round(rsi, 2),
            'trend': indicators['trend'],
            'confidence': 0.7,  # Confianza fija para heurística
            'timestamp': datetime.now().isoformat(),
            'indicators': indicators
        }
    
    def request_ml_prediction(self, symbol: str) -> Dict[str, Any]:
        """Solicitar predicción ML a la API"""
        try:
            # Verificar si hay predicción reciente en cache (últimos 5 minutos)
            cache_key = f"{symbol}_ml"
            if cache_key in self.prediction_cache:
                cached_time = datetime.fromisoformat(self.prediction_cache[cache_key]['timestamp'])
                if datetime.now() - cached_time < timedelta(minutes=5):
                    logger.debug(f"🎯 Using cached ML prediction for {symbol}")
                    return self.prediction_cache[cache_key]
            
            # Solicitar nueva predicción
            payload = {
                'symbol': symbol.replace('/', ''),  # BTC/USDT -> BTCUSDT
                'exchange': 'binance',
                'timeframe': '1h',
                'limit': 200,
                'include_heuristic': False
            }
            
            response = requests.post(
                f"{self.api_endpoint}/v1/crypto/ml-signal",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                ml_pred = data.get('ml_prediction', {})
                
                prediction = {
                    'symbol': symbol,
                    'method': 'ml_lstm',
                    'predicted_volatility': ml_pred.get('prediction'),
                    'volatility_regime': ml_pred.get('volatility_regime'),
                    'risk_level': ml_pred.get('risk_level'),
                    'confidence': ml_pred.get('confidence', 0),
                    'model_version': data.get('model_version'),
                    'timestamp': datetime.now().isoformat(),
                    'error': ml_pred.get('error')
                }
                
                # Cachear resultado
                self.prediction_cache[cache_key] = prediction
                logger.debug(f"🤖 ML prediction generated for {symbol}")
                
                return prediction
            else:
                logger.warning(f"⚠️ ML API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error requesting ML prediction for {symbol}: {e}")
            return None
    
    def detect_anomalies(self, price_data: Dict[str, Any], indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detectar anomalías en tiempo real"""
        anomalies = []
        
        symbol = price_data['symbol']
        price = price_data['price']
        change_24h = price_data.get('change_24h', 0)
        volatility = indicators.get('volatility', 0) if indicators else 0
        
        # Anomalía: Cambio de precio extremo
        if abs(change_24h) > 10:  # >10% en 24h
            anomalies.append({
                'type': 'extreme_price_change',
                'symbol': symbol,
                'value': change_24h,
                'threshold': 10,
                'severity': 'high' if abs(change_24h) > 20 else 'medium',
                'message': f"{symbol} moved {change_24h:.2f}% in 24h"
            })
        
        # Anomalía: Volatilidad extrema
        if volatility > 0.05:  # >5% volatilidad
            anomalies.append({
                'type': 'high_volatility',
                'symbol': symbol,
                'value': volatility,
                'threshold': 0.05,
                'severity': 'high' if volatility > 0.1 else 'medium',
                'message': f"{symbol} volatility at {volatility:.4f}"
            })
        
        # Anomalía: RSI extremo
        if indicators and 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 80 or rsi < 20:
                anomalies.append({
                    'type': 'extreme_rsi',
                    'symbol': symbol,
                    'value': rsi,
                    'threshold': '80/20',
                    'severity': 'medium',
                    'message': f"{symbol} RSI at extreme level: {rsi:.1f}"
                })
        
        return anomalies
    
    def process_message(self, message):
        """Procesar mensaje de Kafka"""
        try:
            price_data = message.value
            symbol = price_data.get('symbol')
            
            if not symbol:
                logger.warning("⚠️ Message without symbol")
                return
            
            logger.debug(f"📨 Processing {symbol}: ${price_data.get('price', 0):.4f}")
            
            # Agregar al buffer
            self.add_to_buffer(symbol, price_data)
            
            # Calcular indicadores técnicos
            indicators = self.calculate_technical_indicators(symbol)
            
            if not indicators:
                logger.debug(f"⏳ Not enough data for {symbol} indicators yet")
                return
            
            # Generar señal heurística
            heuristic_signal = self.generate_heuristic_signal(indicators)
            
            # Solicitar predicción ML (cada 10 mensajes para no sobrecargar)
            ml_prediction = None
            if len(self.price_buffer[symbol]) % 10 == 0:
                ml_prediction = self.request_ml_prediction(symbol)
            
            # Detectar anomalías
            anomalies = self.detect_anomalies(price_data, indicators)
            
            # Preparar resultado combinado
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price_data': price_data,
                'heuristic_signal': heuristic_signal,
                'ml_prediction': ml_prediction,
                'anomalies': anomalies,
                'processing_latency_ms': 0  # Se calculará después
            }
            
            # Publicar resultado
            self.publish_prediction(result)
            
            # Publicar alertas si hay anomalías
            if anomalies:
                self.publish_alerts(symbol, anomalies)
            
            # Log periódico
            if len(self.price_buffer[symbol]) % 50 == 0:
                logger.info(f"📊 Processed 50 messages for {symbol}. Buffer size: {len(self.price_buffer[symbol])}")
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def publish_prediction(self, result: Dict[str, Any]):
        """Publicar resultado de predicción"""
        try:
            future = self.producer.send(
                self.prediction_topic,
                key=result['symbol'],
                value=result
            )
            
            logger.debug(f"📤 Published prediction for {result['symbol']}")
            
        except Exception as e:
            logger.error(f"❌ Error publishing prediction: {e}")
    
    def publish_alerts(self, symbol: str, anomalies: List[Dict[str, Any]]):
        """Publicar alertas de anomalías"""
        try:
            for anomaly in anomalies:
                alert = {
                    'type': 'anomaly_alert',
                    'symbol': symbol,
                    'anomaly': anomaly,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.producer.send(
                    self.alert_topic,
                    key=symbol,
                    value=alert
                )
                
                logger.info(f"🚨 ALERT {symbol}: {anomaly['message']}")
            
        except Exception as e:
            logger.error(f"❌ Error publishing alerts: {e}")
    
    def run(self):
        """Ejecutar consumer principal"""
        logger.info("🔄 Starting consumer loop...")
        
        try:
            for message in self.consumer:
                self.process_message(message)
                
                # Flush producer periódicamente
                if hasattr(self, '_message_count'):
                    self._message_count += 1
                else:
                    self._message_count = 1
                
                if self._message_count % 10 == 0:
                    self.producer.flush()
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping consumer...")
        except Exception as e:
            logger.error(f"❌ Consumer error: {e}")
        finally:
            self.close()
    
    def close(self):
        """Cerrar consumer y producer"""
        logger.info("🔒 Closing Kafka consumer and producer...")
        self.consumer.close()
        self.producer.close()

def main():
    """Función principal"""
    consumer = CryptoConsumer()
    
    try:
        consumer.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down consumer...")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()