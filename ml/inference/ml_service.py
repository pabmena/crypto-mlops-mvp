import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import warnings
import joblib

# Ignorar advertencias de pydantic y MLflow
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
REGISTERED_MODEL_NAME = "crypto-predictor"

# Columnas de features esperadas por el LSTM
LSTM_FEATURE_COLUMNS = ['close', 'log_volume', 'returns', 'volatility', 
                        'sma_12', 'sma_48', 'rsi', 'bollinger_upper', 
                        'bollinger_lower', 'momentum_12', 'momentum_24']

# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------
def _calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _calculate_bollinger(close_prices, window=20, num_std=2):
    sma = close_prices.rolling(window=window).mean()
    std = close_prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def prepare_features_for_prediction(df):
    """Prepara las features para predicción replicando el entrenamiento"""
    data = df.copy()
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=24).std()
    data['sma_12'] = data['close'].rolling(window=12).mean()
    data['sma_48'] = data['close'].rolling(window=48).mean()
    data['rsi'] = _calculate_rsi(data['close'])
    data['bollinger_upper'], data['bollinger_lower'] = _calculate_bollinger(data['close'])
    data['log_volume'] = np.log1p(data['volume'])
    data['momentum_12'] = data['close'] / data['close'].shift(12) - 1
    data['momentum_24'] = data['close'] / data['close'].shift(24) - 1
    return data.dropna()

# -----------------------------------------------------------------------------
# Servicio de ML
# -----------------------------------------------------------------------------
class MLService:
    SEQUENCE_LENGTH = 48
    def __init__(self, model_name: str = REGISTERED_MODEL_NAME):
        self.model = None
        self.model_name = model_name
        self.model_version = None
        self.model_loaded = False
        self.load_time = None
        self.metrics = None
        self.feature_columns = LSTM_FEATURE_COLUMNS  # columnas que el modelo espera

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    # --------------------- Cargar modelo ---------------------
    def load_model(self) -> bool:
        """Carga el modelo LSTM desde MLflow"""
        print("🔄 Cargando modelo de producción desde MLflow...")
        try:
            # Cargar modelo
            model_uri = f"models:/{self.model_name}/Production"
            self.model = mlflow.tensorflow.load_model(model_uri)

            # Obtener versión y metrics
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(self.model_name, stages=["Production"])
            if not model_versions:
                print(f"❌ No se encontró modelo '{self.model_name}' en Production")
                self.model = None
                self.model_loaded = False
                return False

            self.model_version = model_versions[0].version
            run_id = model_versions[0].run_id
            run = client.get_run(run_id)
            self.metrics = run.data.metrics
            self.model_loaded = True
            self.load_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            # Descargar artifacts y cargar scalers
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                artifacts_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir)
                scaler_x_path = Path(artifacts_path) / "model_artifacts" / "scaler_X.pkl"
                scaler_y_path = Path(artifacts_path) / "model_artifacts" / "scaler_y.pkl"
                features_path = Path(artifacts_path) / "model_artifacts" / "feature_columns.pkl"

                if scaler_x_path.exists():
                    self.scaler_X = joblib.load(scaler_x_path)
                if scaler_y_path.exists():
                    self.scaler_y = joblib.load(scaler_y_path)
                if features_path.exists():
                    self.feature_columns = joblib.load(features_path)

            # Ajustar SEQUENCE_LENGTH si el modelo lo define
            if hasattr(self.model, 'sequence_length'):
                self.SEQUENCE_LENGTH = self.model.sequence_length

            print(f"✅ Modelo '{self.model_name}' versión {self.model_version} cargado correctamente.")
            return True

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model = None
            self.model_loaded = False
            return False

    def reload_model(self) -> dict:
        print("🔁 Solicitud de recarga de modelo recibida.")
        success = self.load_model()
        return {"success": success, "message": "Modelo recargado exitosamente" if success else "Fallo al recargar"}

    # --------------------- Predicción ---------------------
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predicción usando LSTM sin scalers (datos crudos).
        """
        if not self.model_loaded or self.model is None:
            return {"error": "Modelo no cargado"}

        try:
            df_processed = prepare_features_for_prediction(df)

            # Verificar columnas faltantes
            missing_cols = [col for col in self.feature_columns if col not in df_processed.columns]
            if missing_cols:
                return {"error": f"Faltan columnas: {missing_cols}"}

            # Seleccionar últimas SEQUENCE_LENGTH filas
            df_seq = df_processed[self.feature_columns].tail(self.SEQUENCE_LENGTH)

            # Rellenar con ceros si hay menos filas
            if len(df_seq) < self.SEQUENCE_LENGTH:
                pad = pd.DataFrame(
                    0,
                    index=range(self.SEQUENCE_LENGTH - len(df_seq)),
                    columns=self.feature_columns
                )
                df_seq = pd.concat([pad, df_seq], ignore_index=True)

            # Convertir a 3D para LSTM: (1, SEQUENCE_LENGTH, n_features)
            seq_input = df_seq.values.reshape(1, self.SEQUENCE_LENGTH, len(self.feature_columns))

            # Realizar predicción directamente
            predicted_volatility = float(self.model.predict(seq_input, verbose=0)[0][0])

            # Determinar régimen de volatilidad
            if predicted_volatility < 0.005:
                vol_regime = "calm"
            elif predicted_volatility < 0.015:
                vol_regime = "normal"
            else:
                vol_regime = "turbulent"

            return {
                "prediction": predicted_volatility,
                "volatility_regime": vol_regime,
                "model_version": self.model_version
            }

        except Exception as e:
            return {"error": f"Error en la predicción: {str(e)}"}

    # --------------------- Info del modelo ---------------------
    def get_model_info(self) -> dict:
        info = {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "load_time": self.load_time,
            "metrics": self.metrics,
            "sequence_length": self.SEQUENCE_LENGTH,
            "feature_columns": self.feature_columns
        }
        return info

# Instancia global
ml_service = MLService()

