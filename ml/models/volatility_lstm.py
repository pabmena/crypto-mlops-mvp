"""
VolatilityLSTM - LSTM para predicción de volatilidad de criptomonedas
Versión lista para producción con registro automático en MLflow
"""
import os
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging más específica
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolatilityLSTM:
    def __init__(self, sequence_length=60, features=['close', 'volume'], lstm_units=50, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.features = features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = None

    # --------------------- Feature Engineering ---------------------
    def prepare_features(self, df):
        """Prepara las features para el modelo"""
        try:
            data = df.copy()
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=24).std()
            data['sma_12'] = data['close'].rolling(window=12).mean()
            data['sma_48'] = data['close'].rolling(window=48).mean()
            data['rsi'] = self._calculate_rsi(data['close'])
            data['bollinger_upper'], data['bollinger_lower'] = self._calculate_bollinger(data['close'])
            data['log_volume'] = np.log1p(data['volume'])
            data['momentum_12'] = data['close'] / data['close'].shift(12) - 1
            data['momentum_24'] = data['close'] / data['close'].shift(24) - 1
            data['target_volatility'] = data['volatility'].shift(-24)
            data = data.dropna()
            logger.info(f"Features preparadas. Filas: {len(data)}")
            return data
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            raise

    def _calculate_rsi(self, close_prices, window=14):
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger(self, close_prices, window=20, num_std=2):
        sma = close_prices.rolling(window=window).mean()
        std = close_prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    # --------------------- Model Building ---------------------
    def build_model(self, n_features):
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25, activation='relu'),
            Dropout(self.dropout_rate / 2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        logger.info(f"Modelo construido con {n_features} features")
        return model

    # --------------------- Training ---------------------
    def train(self, df, test_size=0.2, epochs=50, batch_size=32, registered_model_name="crypto-predictor"):
        """
        Entrena y registra el modelo en MLflow.
        """
        logger.info(f"Iniciando entrenamiento del modelo: {registered_model_name}")
        
        try:
            with mlflow.start_run():
                # Log parámetros
                mlflow.log_param("sequence_length", self.sequence_length)
                mlflow.log_param("lstm_units", self.lstm_units)
                mlflow.log_param("dropout_rate", self.dropout_rate)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)

                # Preparar data
                data = self.prepare_features(df)
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

                logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

                # Normalizar
                X_train_scaled = self.scaler_X.fit_transform(X_train)
                X_test_scaled = self.scaler_X.transform(X_test)
                y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

                # Crear secuencias
                X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
                X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)

                logger.info(f"Secuencias creadas - Train: {len(X_train_seq)}, Test: {len(X_test_seq)}")

                mlflow.log_param("n_features", X_train_seq.shape[2])
                mlflow.log_param("train_samples", len(X_train_seq))
                mlflow.log_param("test_samples", len(X_test_seq))

                # Construir modelo
                self.model = self.build_model(X_train_seq.shape[2])

                # Entrenar
                logger.info("Iniciando entrenamiento...")
                history = self.model.fit(X_train_seq, y_train_seq,
                                         epochs=epochs, batch_size=batch_size,
                                         validation_data=(X_test_seq, y_test_seq), 
                                         verbose=1)

                # Métricas
                logger.info("Calculando métricas...")
                y_pred_scaled = self.model.predict(X_test_seq)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
                y_test_original = self.scaler_y.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
                mse = mean_squared_error(y_test_original, y_pred)
                mae = mean_absolute_error(y_test_original, y_pred)
                rmse = np.sqrt(mse)
                
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("final_train_loss", history.history['loss'][-1])
                mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

                logger.info(f"Métricas - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

                # Log training history por epoch
                for epoch, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

                # Guardar scalers
                os.makedirs("/tmp/model_artifacts", exist_ok=True)
                joblib.dump(self.scaler_X, "/tmp/model_artifacts/scaler_X.pkl")
                joblib.dump(self.scaler_y, "/tmp/model_artifacts/scaler_y.pkl")
                joblib.dump(self.feature_columns, "/tmp/model_artifacts/feature_columns.pkl")
                
                mlflow.log_artifact("/tmp/model_artifacts/scaler_X.pkl", artifact_path="model_artifacts")
                mlflow.log_artifact("/tmp/model_artifacts/scaler_y.pkl", artifact_path="model_artifacts")
                mlflow.log_artifact("/tmp/model_artifacts/feature_columns.pkl", artifact_path="model_artifacts")

                # Crear signature
                from mlflow.models.signature import infer_signature
                signature = infer_signature(X_train_seq, y_pred_scaled)

                # ----------------- Registrar modelo -----------------
                logger.info(f"Registrando modelo en MLflow: {registered_model_name}")
                client = MlflowClient()
                
                # Verificar/crear registered model
                try:
                    registered_model = client.get_registered_model(registered_model_name)
                    logger.info(f"Modelo registrado existente: {registered_model_name}")
                except Exception:
                    client.create_registered_model(registered_model_name)
                    logger.info(f"Nuevo modelo registrado creado: {registered_model_name}")

                # Log del modelo
                model_info = mlflow.tensorflow.log_model(
                    model=self.model,
                    artifact_path="model",
                    signature=signature,
                    keras_model_kwargs={"save_format": "h5"},
                    registered_model_name=registered_model_name
                )

                # Obtener la versión registrada automáticamente
                versions = client.get_latest_versions(registered_model_name)
                version_number = versions[-1].version

                # Transicionar a Production
                try:
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=version_number,
                        stage="Production"
                    )
                    logger.info(f"Modelo v{version_number} transicionado a Production")
                except Exception as e:
                    logger.warning(f"No se pudo transicionar a Production: {e}")

                logger.info(f"✅ Modelo entrenado y registrado exitosamente! Versión {version_number}")

                return self.model, mse, mae

        except Exception as e:
            logger.error(f"❌ Error durante el entrenamiento: {e}")
            if mlflow.active_run():
                mlflow.log_param("error", str(e))
            raise e

    # --------------------- Predicción ---------------------
    def predict(self, df, last_n_points=None):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        try:
            data = self.prepare_features(df)
            if last_n_points:
                data = data.tail(last_n_points + self.sequence_length)
            X = data[self.feature_columns].values
            X_scaled = self.scaler_X.transform(X)
            if len(X_scaled) >= self.sequence_length:
                X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                pred_scaled = self.model.predict(X_seq, verbose=0)
                pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
                logger.info(f"Predicción realizada: {pred}")
                return max(0, pred)
            else:
                logger.warning(f"Datos insuficientes para predicción. Necesario: {self.sequence_length}, Disponible: {len(X_scaled)}")
            return None
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise

    # --------------------- Cargar modelo ---------------------
    def load_model_from_mlflow(self, model_uri: str):
        """Carga modelo desde MLflow con manejo de errores mejorado"""
        try:
            logger.info(f"Cargando modelo desde: {model_uri}")
            self.model = mlflow.tensorflow.load_model(model_uri)
            
            # Intentar cargar artifacts
            try:
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Descargar artifacts
                    artifacts_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir)
                    
                    # Cargar scalers si existen
                    scaler_x_path = os.path.join(artifacts_path, "model_artifacts", "scaler_X.pkl")
                    scaler_y_path = os.path.join(artifacts_path, "model_artifacts", "scaler_y.pkl")
                    features_path = os.path.join(artifacts_path, "model_artifacts", "feature_columns.pkl")
                    
                    if os.path.exists(scaler_x_path):
                        self.scaler_X = joblib.load(scaler_x_path)
                        logger.info("Scaler X cargado")
                    
                    if os.path.exists(scaler_y_path):
                        self.scaler_y = joblib.load(scaler_y_path)
                        logger.info("Scaler Y cargado")
                    
                    if os.path.exists(features_path):
                        self.feature_columns = joblib.load(features_path)
                        logger.info(f"Feature columns cargadas: {self.feature_columns}")
                        
            except Exception as artifact_error:
                logger.warning(f"No se pudieron cargar algunos artifacts: {artifact_error}")
            
            logger.info(f"✅ Modelo cargado correctamente desde MLflow")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo desde MLflow: {e}")
            return False

    def get_model_info(self):
        """Obtiene información del modelo cargado"""
        if self.model is None:
            return {"status": "not_loaded", "message": "Modelo no cargado"}
        
        try:
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            
            return {
                "status": "loaded",
                "sequence_length": self.sequence_length,
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "feature_columns": self.feature_columns,
                "model_summary": "\n".join(model_summary) if model_summary else "No disponible"
            }
        except Exception as e:
            return {"status": "error", "message": f"Error obteniendo info del modelo: {e}"}

# --------------------- Script principal ---------------------
if __name__ == "__main__":
    import ccxt
    from datetime import datetime, timedelta

    # Configuración MLflow
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = "crypto_volatility_prediction"
    client = MlflowClient()
    
    try:
        if client.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
            logger.info(f"Experimento creado: {experiment_name}")
        else:
            logger.info(f"Experimento existente: {experiment_name}")
    except Exception as e:
        logger.error(f"Error configurando experimento: {e}")

    mlflow.set_experiment(experiment_name)

    # Obtener datos OHLCV
    symbol = "BTC/USDT"
    days_back = 60
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    logger.info(f"Obteniendo datos {symbol} desde {start_time} hasta {end_time}...")
    
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', since=int(start_time.timestamp()*1000), limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        if df.empty or len(df) < 100:
            raise ValueError("Datos insuficientes para entrenamiento.")

        logger.info(f"Datos obtenidos: {len(df)} filas")

        # Entrenar modelo - CAMBIO AQUÍ: usar "crypto-predictor" para consistencia
        model = VolatilityLSTM(sequence_length=48, lstm_units=64, dropout_rate=0.2)
        logger.info("Iniciando entrenamiento y registro en MLflow...")
        
        trained_model, mse, mae = model.train(
            df, 
            epochs=30, 
            batch_size=16, 
            registered_model_name="crypto-predictor"  # Nombre consistente con la API
        )

        logger.info(f"✅ Entrenamiento finalizado! MSE: {mse:.6f}, MAE: {mae:.6f}")
        logger.info("✅ Modelo registrado en MLflow con éxito en el Model Registry como 'crypto-predictor'")

    except Exception as e:
        logger.error(f"❌ Error en el script principal: {e}")
        raise
