# ml/inference/ml_service.py
"""
ML Service para carga y uso de modelos de MLflow
Ubicación: ml/inference/ml_service.py
"""
import sys
import os
import logging
from typing import Optional, Dict, Any
import pandas as pd

# Añadir el directorio raíz del proyecto al Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ahora importar el modelo LSTM
try:
    from ml.models.volatility_lstm import VolatilityLSTM
    LSTM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ VolatilityLSTM importado correctamente")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Error importando VolatilityLSTM: {e}")
    LSTM_AVAILABLE = False
    # Fallback a implementación standalone
    VolatilityLSTM = None

class MLService:
    def __init__(self):
        self.model: Optional[VolatilityLSTM] = None
        self.model_name = "crypto-predictor"
        self.model_stage = "Production"
        self.is_loaded = False
        self.lstm_available = LSTM_AVAILABLE
        self.load_model()
        
    def load_model(self) -> bool:
        """
        Carga el modelo desde MLflow
        """
        if not self.lstm_available:
            logger.error("[MLService] VolatilityLSTM no disponible")
            return False
            
        try:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            logger.info(f"[MLService] Cargando modelo desde {model_uri}")
            
            self.model = VolatilityLSTM()
            success = self.model.load_model_from_mlflow(model_uri)
            
            if success:
                self.is_loaded = True
                logger.info("[MLService] ✅ Modelo cargado exitosamente")
                return True
            else:
                logger.error("[MLService] ❌ Error cargando modelo")
                return False
                
        except Exception as e:
            logger.error(f"[MLService] ❌ Error cargando modelo: {e}")
            self.is_loaded = False
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo actual
        """
        if not self.lstm_available:
            return {
                "status": "error",
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "is_loaded": False,
                "lstm_available": False,
                "message": "VolatilityLSTM no disponible - error de importación"
            }
            
        if not self.is_loaded or self.model is None:
            return {
                "status": "not_loaded",
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "is_loaded": False,
                "lstm_available": True,
                "message": "Modelo no cargado"
            }
        
        try:
            model_info = self.model.get_model_info()
            return {
                "status": "loaded",
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "is_loaded": True,
                "lstm_available": True,
                **model_info
            }
        except Exception as e:
            logger.error(f"[MLService] Error obteniendo info del modelo: {e}")
            return {
                "status": "error",
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "is_loaded": self.is_loaded,
                "lstm_available": True,
                "message": f"Error obteniendo información: {e}"
            }
    
    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """
        Realiza predicción usando el modelo cargado
        """
        if not self.lstm_available:
            logger.error("[MLService] VolatilityLSTM no disponible para predicción")
            return None
            
        if not self.is_loaded or self.model is None:
            logger.error("[MLService] Modelo no está cargado para predicción")
            return None
        
        try:
            prediction = self.model.predict(df)
            logger.info(f"[MLService] Predicción realizada: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"[MLService] Error en predicción: {e}")
            return None

    def predict_volatility(self, df: pd.DataFrame, last_n_points: Optional[int] = None) -> Optional[float]:
        """
        Método específico para predicción de volatilidad.
        df: pd.DataFrame con las columnas esperadas por VolatilityLSTM.
        last_n_points: cantidad de puntos recientes a usar en la predicción (opcional)
        """
        if not self.lstm_available:
            logger.error("[MLService] VolatilityLSTM no disponible para predict_volatility")
            return None

        if not self.is_loaded or self.model is None:
            logger.error("[MLService] Modelo no cargado para predict_volatility")
            return None

        if not isinstance(df, pd.DataFrame):
            logger.error("[MLService] Input debe ser un pd.DataFrame")
            return None

        try:
            pred = self.model.predict(df, last_n_points=last_n_points)
            logger.info(f"[MLService] Predicción de volatilidad realizada: {pred}")
            return pred
        except Exception as e:
            logger.error(f"[MLService] Error en predict_volatility: {e}")
            return None
    
    def is_model_available(self) -> bool:
        """
        Verifica si el modelo está disponible y listo
        """
        return self.lstm_available and self.is_loaded and self.model is not None
    
    def reload_model(self) -> bool:
        """
        Recarga el modelo desde MLflow
        """
        logger.info("[MLService] Recargando modelo...")
        self.is_loaded = False
        self.model = None
        return self.load_model()

# Instancia global del servicio ML
ml_service = MLService()
