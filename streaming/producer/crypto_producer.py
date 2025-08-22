"""
Kafka Producer para simular datos crypto en tiempo real
"""
import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
import ccxt
import logging
import os
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoProducer:
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
        self.topic = os.getenv('KAFKA_TOPIC', 'crypto-prices')
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        
        # Configurar producer
        self.producer = KafkaProducer(
            bootstrap_servers=[self.kafka_servers],
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
            batch_size=16384,
            linger_ms=10,
            acks='all'
        )
        
        # Cache de precios para simular variaciones realistas
        self.price_cache = {}
        self.exchange = ccxt.binance()
        
        # Inicializar cache de precios
        self.initialize_price_cache()
        
        logger.info(f"🚀 CryptoProducer initialized for topic: {self.topic}")
        logger.info(f"📊 Tracking symbols: {self.symbols}")
    
    def initialize_price_cache(self):
        """Inicializar cache con precios reales"""
        for symbol in self.symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                self.price_cache[symbol] = {
                    'last_price': float(ticker['last']),
                    'volume': float(ticker['baseVolume']),
                    'change_24h': float(ticker['percentage']) if ticker['percentage'] else 0,
                    'high_24h': float(ticker['high']),
                    'low_24h': float(ticker['low']),
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"✅ Initialized {symbol}: ${ticker['last']}")
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize {symbol}: {e}")
                # Precio base ficticio
                base_prices = {
                    'BTC/USDT': 50000, 'ETH/USDT': 3000, 'BNB/USDT': 300,
                    'ADA/USDT': 0.5, 'DOT/USDT': 15
                }
                price = base_prices.get(symbol, 100)
                self.price_cache[symbol] = {
                    'last_price': price,
                    'volume': random.uniform(1000, 10000),
                    'change_24h': random.uniform(-5, 5),
                    'high_24h': price * 1.05,
                    'low_24h': price * 0.95,
                    'timestamp': datetime.now().isoformat()
                }
    
    def simulate_price_movement(self, symbol: str) -> Dict[str, Any]:
        """Simular movimiento de precio realista"""
        if symbol not in self.price_cache:
            self.initialize_price_cache()
        
        current = self.price_cache[symbol]
        
        # Generar variación aleatoria (más realista)
        # Volatilidad base según el símbolo
        volatilities = {
            'BTC/USDT': 0.02, 'ETH/USDT': 0.025, 'BNB/USDT': 0.03,
            'ADA/USDT': 0.04, 'DOT/USDT': 0.035
        }
        
        base_vol = volatilities.get(symbol, 0.03)
        
        # Factor de tendencia (simulamos ciclos de mercado)
        trend_factor = random.uniform(-0.001, 0.001)
        
        # Variación de precio
        price_change = random.normalvariate(trend_factor, base_vol)
        new_price = current['last_price'] * (1 + price_change)
        
        # Variación de volumen
        volume_change = random.uniform(0.8, 1.2)
        new_volume = current['volume'] * volume_change
        
        # Actualizar cache
        price_data = {
            'symbol': symbol,
            'price': round(new_price, 8),
            'volume_24h': round(new_volume, 2),
            'change_24h': round(((new_price - current['last_price']) / current['last_price']) * 100, 4),
            'high_24h': max(current['high_24h'], new_price),
            'low_24h': min(current['low_24h'], new_price),
            'timestamp': datetime.now().isoformat(),
            'source': 'simulated',
            'volatility': abs(price_change),
            'trend': 'up' if price_change > 0 else 'down' if price_change < 0 else 'stable'
        }
        
        # Actualizar cache
        self.price_cache[symbol]['last_price'] = new_price
        self.price_cache[symbol]['volume'] = new_volume
        self.price_cache[symbol]['high_24h'] = price_data['high_24h']
        self.price_cache[symbol]['low_24h'] = price_data['low_24h']
        
        return price_data
    
    def fetch_real_price(self, symbol: str) -> Dict[str, Any]:
        """Obtener precio real de Binance"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            price_data = {
                'symbol': symbol,
                'price': float(ticker['last']),
                'volume_24h': float(ticker['baseVolume']),
                'change_24h': float(ticker['percentage']) if ticker['percentage'] else 0,
                'high_24h': float(ticker['high']),
                'low_24h': float(ticker['low']),
                'timestamp': datetime.now().isoformat(),
                'source': 'binance',
                'volatility': abs(float(ticker['percentage']) / 100) if ticker['percentage'] else 0,
                'trend': 'up' if ticker['percentage'] and ticker['percentage'] > 0 else 'down' if ticker['percentage'] and ticker['percentage'] < 0 else 'stable'
            }
            
            return price_data
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch real price for {symbol}: {e}")
            return self.simulate_price_movement(symbol)
    
    def send_price_update(self, price_data: Dict[str, Any]):
        """Enviar actualización de precio a Kafka"""
        try:
            # Enviar a tópico principal
            future = self.producer.send(
                self.topic,
                key=price_data['symbol'],
                value=price_data
            )
            
            # Callback para confirmar envío
            def on_send_success(record_metadata):
                logger.debug(f"✅ Sent {price_data['symbol']}: ${price_data['price']} to {record_metadata.topic}[{record_metadata.partition}]")
            
            def on_send_error(excp):
                logger.error(f"❌ Failed to send {price_data['symbol']}: {excp}")
            
            future.add_callback(on_send_success)
            future.add_errback(on_send_error)
            
            # Enviar también a tópico de alertas si hay cambios significativos
            if abs(price_data['change_24h']) > 5:  # Cambio > 5%
                alert_data = {
                    'type': 'price_alert',
                    'symbol': price_data['symbol'],
                    'price': price_data['price'],
                    'change_24h': price_data['change_24h'],
                    'timestamp': price_data['timestamp'],
                    'message': f"{price_data['symbol']} moved {price_data['change_24h']:.2f}% in 24h"
                }
                
                self.producer.send('alerts', key=price_data['symbol'], value=alert_data)
                logger.info(f"🚨 ALERT: {alert_data['message']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending price update: {e}")
    
    def run_real_data_mode(self, interval_seconds: int = 5):
        """Ejecutar producer con datos reales"""
        logger.info(f"🔄 Starting real data mode (interval: {interval_seconds}s)")
        
        while True:
            try:
                for symbol in self.symbols:
                    price_data = self.fetch_real_price(symbol)
                    self.send_price_update(price_data)
                    
                    # Log cada 10 actualizaciones
                    if random.random() < 0.1:
                        logger.info(f"📊 {symbol}: ${price_data['price']:.4f} ({price_data['change_24h']:+.2f}%)")
                
                # Flush producer
                self.producer.flush()
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping real data mode...")
                break
            except Exception as e:
                logger.error(f"❌ Error in real data mode: {e}")
                time.sleep(interval_seconds)
    
    def run_simulation_mode(self, interval_seconds: int = 1):
        """Ejecutar producer con datos simulados (más rápido)"""
        logger.info(f"🎮 Starting simulation mode (interval: {interval_seconds}s)")
        
        while True:
            try:
                for symbol in self.symbols:
                    # 70% simulado, 30% datos reales mezclados
                    if random.random() < 0.7:
                        price_data = self.simulate_price_movement(symbol)
                    else:
                        price_data = self.fetch_real_price(symbol)
                    
                    self.send_price_update(price_data)
                
                # Log periódico
                if random.random() < 0.05:  # 5% de probabilidad
                    symbol = random.choice(self.symbols)
                    cached = self.price_cache[symbol]
                    logger.info(f"💹 {symbol}: ${cached['last_price']:.4f}")
                
                self.producer.flush()
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping simulation mode...")
                break
            except Exception as e:
                logger.error(f"❌ Error in simulation mode: {e}")
                time.sleep(interval_seconds)
    
    def run_mixed_mode(self, real_interval: int = 30, sim_interval: int = 2):
        """Modo mixto: datos reales cada 30s, simulados cada 2s"""
        logger.info(f"🔀 Starting mixed mode (real: {real_interval}s, sim: {sim_interval}s)")
        
        last_real_update = 0
        
        while True:
            try:
                current_time = time.time()
                
                # Actualización con datos reales cada X segundos
                if current_time - last_real_update >= real_interval:
                    logger.info("📡 Fetching real data...")
                    for symbol in self.symbols:
                        price_data = self.fetch_real_price(symbol)
                        self.send_price_update(price_data)
                    last_real_update = current_time
                else:
                    # Datos simulados entre actualizaciones reales
                    for symbol in self.symbols:
                        price_data = self.simulate_price_movement(symbol)
                        self.send_price_update(price_data)
                
                self.producer.flush()
                time.sleep(sim_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping mixed mode...")
                break
            except Exception as e:
                logger.error(f"❌ Error in mixed mode: {e}")
                time.sleep(sim_interval)
    
    def close(self):
        """Cerrar producer"""
        logger.info("🔒 Closing Kafka producer...")
        self.producer.close()

def main():
    """Función principal"""
    # Configuración desde variables de entorno
    mode = os.getenv('PRODUCER_MODE', 'mixed')  # real, simulation, mixed
    interval = int(os.getenv('PRODUCER_INTERVAL', '2'))
    
    producer = CryptoProducer()
    
    try:
        if mode == 'real':
            producer.run_real_data_mode(interval)
        elif mode == 'simulation':
            producer.run_simulation_mode(interval)
        else:  # mixed
            producer.run_mixed_mode()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down producer...")
    finally:
        producer.close()

if __name__ == "__main__":
    main()