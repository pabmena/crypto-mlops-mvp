"""
ML Service para carga y uso de modelos de MLflow
"""
import logging
import os
from typing import Optional, Dict, Any
import pandas as pd
from volatility_lstm import VolatilityLSTM

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.model: Optional[VolatilityLSTM] = None
        self.model_name = "crypto-predictor"
        self.model_stage = "Production"
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Carga el modelo desde MLflow
        """
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
        if not self.is_loaded or self.model is None:
            return {
                "status": "not_loaded",
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "is_loaded": False,
                "message": "Modelo no cargado"
            }
        
        try:
            model_info = self.model.get_model_info()
            return {
                "status": "loaded",
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "is_loaded": True,
                **model_info
            }
        except Exception as e:
            logger.error(f"[MLService] Error obteniendo info del modelo: {e}")
            return {
                "status": "error",
                "model_name": self.model_name,
                "model_stage": self.model_stage,
                "is_loaded": self.is_loaded,
                "message": f"Error obteniendo información: {e}"
            }
    
    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """
        Realiza predicción usando el modelo cargado
        """
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
    
    def is_model_available(self) -> bool:
        """
        Verifica si el modelo está disponible y listo
        """
        return self.is_loaded and self.model is not None
    
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

