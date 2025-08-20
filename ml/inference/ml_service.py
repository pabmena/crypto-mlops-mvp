"""
Servicio de inferencia ML para integrar con FastAPI
"""
import os
import sys
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Agregar path para importar el modelo
sys.path.append('/app/ml/models')
from volatility_lstm import VolatilityLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLInferenceService:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_name = "volatility_lstm"
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        
        # Configurar MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Cargar modelo al inicializar
        self.load_latest_model()
    
    def load_latest_model(self) -> bool:
        """Cargar el modelo más reciente desde MLflow"""
        try:
            # Buscar el modelo más reciente
            client = mlflow.tracking.MlflowClient()
            
            # Intentar obtener el modelo de producción
            try:
                model_version = client.get_latest_versions(
                    self.model_name, 
                    stages=["Production"]
                )[0]
                logger.info(f"Loading production model version: {model_version.version}")
            except:
                # Si no hay modelo en producción, usar el más reciente
                try:
                    model_version = client.get_latest_versions(self.model_name)[0]
                    logger.info(f"Loading latest model version: {model_version.version}")
                except:
                    logger.warning("No registered model found, will try to load from runs")
                    return self._load_from_latest_run()
            
            # Construir URI del modelo
            model_uri = f"models:/{self.model_name}/{model_version.version}"
            
            # Cargar modelo
            self.model = VolatilityLSTM()
            self.model.load_model(model_uri)
            self.model_version = model_version.version
            
            logger.info(f"Successfully loaded model version {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_from_latest_run(self) -> bool:
        """Cargar modelo desde el run más reciente"""
        try:
            # Buscar experimento
            experiment = mlflow.get_experiment_by_name("crypto_volatility_prediction")
            if not experiment:
                logger.error("Experiment 'crypto_volatility_prediction' not found")
                return False
            
            # Buscar runs del experimento
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                logger.error("No runs found in experiment")
                return False
            
            # Cargar modelo del run más reciente
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/model"
            
            self.model = VolatilityLSTM()
            self.model.load_model(model_uri)
            self.model_version = f"run-{run_id[:8]}"
            
            logger.info(f"Successfully loaded model from run: {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from run: {e}")
            return False
    
    def predict_volatility(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir volatilidad usando el modelo ML
        
        Args:
            ohlcv_data: DataFrame con columnas [open, high, low, close, volume]
        
        Returns:
            Diccionario con predicción y metadata
        """
        if self.model is None:
            return {
                "error": "Model not loaded",
                "prediction": None,
                "confidence": 0.0,
                "model_version": None
            }
        
        try:
            # Realizar predicción
            predicted_vol = self.model.predict(ohlcv_data)
            
            if predicted_vol is None:
                return {
                    "error": "Insufficient data for prediction",
                    "prediction": None,
                    "confidence": 0.0,
                    "model_version": self.model_version
                }
            
            # Calcular volatilidad actual para comparación
            returns = ohlcv_data['close'].pct_change()
            current_vol = returns.rolling(window=24).std().iloc[-1]
            
            # Estimar confidence basado en la diferencia con volatilidad histórica
            vol_diff = abs(predicted_vol - current_vol) if pd.notna(current_vol) else 0
            confidence = max(0.1, 1.0 - min(vol_diff * 10, 0.9))  # Heurística simple
            
            # Clasificar régimen de volatilidad predicho
            if predicted_vol < 0.01:
                vol_regime = "calm"
                risk_level = "low"
            elif predicted_vol < 0.03:
                vol_regime = "normal"
                risk_level = "medium"
            else:
                vol_regime = "turbulent"
                risk_level = "high"
            
            return {
                "prediction": float(predicted_vol),
                "current_volatility": float(current_vol) if pd.notna(current_vol) else None,
                "volatility_regime": vol_regime,
                "risk_level": risk_level,
                "confidence": float(confidence),
                "model_version": self.model_version,
                "prediction_timestamp": datetime.now().isoformat(),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "prediction": None,
                "confidence": 0.0,
                "model_version": self.model_version
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo cargado"""
        return {
            "model_loaded": self.model is not None,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "mlflow_uri": self.mlflow_tracking_uri,
            "sequence_length": getattr(self.model, 'sequence_length', None) if self.model else None
        }
    
    def reload_model(self) -> Dict[str, Any]:
        """Recargar el modelo desde MLflow"""
        old_version = self.model_version
        success = self.load_latest_model()
        
        return {
            "success": success,
            "old_version": old_version,
            "new_version": self.model_version,
            "message": "Model reloaded successfully" if success else "Failed to reload model"
        }

# Instancia global del servicio
ml_service = MLInferenceService()