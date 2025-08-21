"""
Modelo LSTM para predicción de volatilidad de criptomonedas
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class VolatilityLSTM:
    def __init__(self, sequence_length=60, features=['close', 'volume'], 
                 lstm_units=50, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.features = features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Preparar features técnicos para el modelo"""
        data = df.copy()
        
        # Calcular features técnicos
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=24).std()
        data['sma_12'] = data['close'].rolling(window=12).mean()
        data['sma_48'] = data['close'].rolling(window=48).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        data['bollinger_upper'], data['bollinger_lower'] = self._calculate_bollinger(data['close'])
        
        # Logaritmo de volumen para normalizar
        data['log_volume'] = np.log1p(data['volume'])
        
        # Features de momentum
        data['momentum_12'] = data['close'] / data['close'].shift(12) - 1
        data['momentum_24'] = data['close'] / data['close'].shift(24) - 1
        
        # Target: volatilidad futura (próximas 24h)
        data['target_volatility'] = data['volatility'].shift(-24)
        
        # Eliminar NaN
        data = data.dropna()
        
        return data
    
    def _calculate_rsi(self, close_prices, window=14):
        """Calcular RSI (Relative Strength Index)"""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger(self, close_prices, window=20, num_std=2):
        """Calcular Bandas de Bollinger"""
        sma = close_prices.rolling(window=window).mean()
        std = close_prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def create_sequences(self, X, y):
        """Crear secuencias temporales para LSTM"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, n_features):
        """Construir el modelo LSTM"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, n_features)),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(25, activation='relu'),
            Dropout(self.dropout_rate / 2),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df, test_size=0.2, epochs=50, batch_size=32, experiment_name="volatility_prediction"):
        """Entrenar el modelo con MLflow tracking"""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("sequence_length", self.sequence_length)
            mlflow.log_param("lstm_units", self.lstm_units)
            mlflow.log_param("dropout_rate", self.dropout_rate)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            
            # Preparar data
            data = self.prepare_features(df)
            
            # Seleccionar features
            feature_cols = ['close', 'log_volume', 'returns', 'volatility', 
                          'sma_12', 'sma_48', 'rsi', 'bollinger_upper', 
                          'bollinger_lower', 'momentum_12', 'momentum_24']
            
            self.feature_columns = feature_cols
            
            X = data[feature_cols].values
            y = data['target_volatility'].values
            
            # Split train/test
            split_index = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Normalizar
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
            
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            # Crear secuencias
            X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
            X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
            
            mlflow.log_param("n_features", X_train_seq.shape[2])
            mlflow.log_param("train_samples", len(X_train_seq))
            mlflow.log_param("test_samples", len(X_test_seq))
            
            # Construir modelo
            self.model = self.build_model(X_train_seq.shape[2])
            
            # Entrenar
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_seq, y_test_seq),
                verbose=1
            )
            
            # Predicciones
            y_pred_scaled = self.model.predict(X_test_seq)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_original = self.scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
            
            # Métricas
            mse = mean_squared_error(y_test_original, y_pred)
            mae = mean_absolute_error(y_test_original, y_pred)
            rmse = np.sqrt(mse)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("final_train_loss", history.history['loss'][-1])
            mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
            
            # Log model
            mlflow.tensorflow.log_model(
                self.model, 
                "model",
                signature=mlflow.models.signature.infer_signature(X_train_seq, y_pred_scaled),
                keras_model_kwargs={"save_format": "h5"}
            )
            
            # Log artifacts (scalers)
            import joblib
            joblib.dump(self.scaler_X, "/tmp/scaler_X.pkl")
            joblib.dump(self.scaler_y, "/tmp/scaler_y.pkl")
            mlflow.log_artifact("/tmp/scaler_X.pkl")
            mlflow.log_artifact("/tmp/scaler_y.pkl")
            
            print(f"Model trained successfully!")
            print(f"Test MSE: {mse:.6f}")
            print(f"Test MAE: {mae:.6f}")
            print(f"Test RMSE: {rmse:.6f}")
            
            return history
    
    def predict(self, df, last_n_points=None):
        """Realizar predicción de volatilidad"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preparar data
        data = self.prepare_features(df)
        
        if last_n_points:
            data = data.tail(last_n_points + self.sequence_length)
        
        # Usar las mismas columnas del entrenamiento
        X = data[self.feature_columns].values
        
        # Normalizar
        X_scaled = self.scaler_X.transform(X)
        
        # Crear secuencia (solo necesitamos la última)
        if len(X_scaled) >= self.sequence_length:
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Predicción
            pred_scaled = self.model.predict(X_seq, verbose=0)
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            
            return max(0, pred)  # La volatilidad no puede ser negativa
        else:
            return None
    
    def load_model(self, model_uri):
        """Cargar modelo desde MLflow"""
        import joblib
        
        # Cargar modelo
        self.model = mlflow.tensorflow.load_model(model_uri)
        
        # Cargar scalers (asumir que están en el mismo directorio)
        import os
        model_path = model_uri.replace('file://', '')
        scaler_x_path = os.path.join(os.path.dirname(model_path), 'scaler_X.pkl')
        scaler_y_path = os.path.join(os.path.dirname(model_path), 'scaler_y.pkl')
        
        if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
            self.scaler_X = joblib.load(scaler_x_path)
            self.scaler_y = joblib.load(scaler_y_path)


def setup_mlflow_environment():
    """Configurar el entorno de MLflow para usar MinIO local"""
    
    # Configurar variables de entorno para MinIO (S3 compatible)
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"
    
    # Usar las variables de entorno del docker-compose si existen
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Silenciar warning de Git
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"S3 Endpoint: {s3_endpoint}")
    print("Using MinIO S3 local storage for artifacts")


def train_model_pipeline(symbol="BTCUSDT", days_back=30):
    """Pipeline completo de entrenamiento"""
    import ccxt
    from datetime import datetime, timedelta
    
    print(f"Starting training pipeline for {symbol}...")
    
    # Obtener datos
    exchange = ccxt.binance()
    
    # Calcular timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    # Obtener datos OHLCV
    ohlcv = exchange.fetch_ohlcv(
        symbol, 
        timeframe='1h',
        since=int(start_time.timestamp() * 1000),
        limit=1000
    )
    
    # Convertir a DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Entrenar modelo
    model = VolatilityLSTM(sequence_length=48, lstm_units=64)
    history = model.train(df, epochs=30, batch_size=16)
    
    return model, history


if __name__ == "__main__":
    # Configurar entorno MLflow
    setup_mlflow_environment()
    
    # Configurar experimento
    mlflow.set_experiment("crypto_volatility_prediction")
    
    # Entrenar modelo
    model, history = train_model_pipeline("BTC/USDT", days_back=60)
    
    print("Training completed!")