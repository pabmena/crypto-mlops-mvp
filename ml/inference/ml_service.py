import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------

# Nombres de modelos y características
REGISTERED_MODEL_NAME = "crypto-predictor"
# Se asume que estos son los nombres de las columnas que tu modelo espera
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility', 
                   'sma_12', 'sma_48', 'volume_sma', 'rsi', 'bb_upper', 'bb_lower']

# -----------------------------------------------------------------------------
# Servicio de ML
# -----------------------------------------------------------------------------
class MLService:
    def __init__(self, model_name: str = REGISTERED_MODEL_NAME):
        """
        Inicializa el servicio de ML.
        No carga el modelo aquí. La carga se hará explícitamente.
        """
        self.model = None
        self.model_name = model_name
        self.model_version = None
        self.model_loaded = False
        self.load_time = None
        self.metrics = None
        
        # Conectar al servidor de MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    def load_model(self) -> bool:
        """
        Carga la última versión del modelo en producción desde MLflow.
        """
        print("🔄 Cargando modelo de producción desde MLflow...")
        try:
            # Uri para cargar la última versión del modelo en etapa 'Production'
            model_uri = f"models:/{self.model_name}/Production"
            
            # Cargar el modelo
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Obtener información de la versión del modelo
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(self.model_name, stages=["Production"])
            
            if not model_versions:
                print("❌ No se encontró ninguna versión del modelo en la etapa 'Production'.")
                self.model = None
                self.model_loaded = False
                return False
            
            self.model_version = model_versions[0].version
            self.model_loaded = True
            self.load_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            # Obtener métricas y parámetros del modelo si están disponibles
            run_id = model_versions[0].run_id
            run = client.get_run(run_id)
            self.metrics = run.data.metrics
            
            print(f"✅ Modelo '{self.model_name}' versión {self.model_version} cargado exitosamente.")
            return True
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            self.model = None
            self.model_loaded = False
            return False

    def reload_model(self) -> dict:
        """
        Recarga el modelo. Útil para ser llamado por el DAG después del entrenamiento.
        """
        print("🔁 Solicitud de recarga de modelo recibida.")
        if self.load_model():
            return {"success": True, "message": "Modelo recargado exitosamente."}
        else:
            return {"success": False, "message": "Fallo al recargar el modelo."}

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Realiza una predicción utilizando el modelo cargado.
        """
        if not self.model_loaded or self.model is None:
            print("[MLService] Modelo no está cargado para predicción")
            return {"error": "Modelo no cargado"}
        
        try:
            # Asegurar que el DataFrame tenga las columnas correctas
            df_features = df[FEATURE_COLUMNS]
            
            # El modelo MLflow espera un DataFrame
            prediction_df = pd.DataFrame(df_features.tail(1), columns=FEATURE_COLUMNS)
            
            # Realizar la predicción
            prediction = self.model.predict(prediction_df)
            predicted_volatility = float(prediction[0])
            
            # Calcular régimen de volatilidad basado en la predicción
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
            print(f"❌ Error durante la predicción: {e}")
            return {"error": f"Error en la predicción: {str(e)}"}

    def get_model_info(self) -> dict:
        """
        Proporciona metadatos del modelo cargado.
        """
        info = {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "load_time": self.load_time,
            "metrics": self.metrics
        }
        return info

# Crea una instancia global del servicio, pero el modelo NO se carga de inmediato.
ml_service = MLService()
