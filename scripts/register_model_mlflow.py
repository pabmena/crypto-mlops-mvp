"""
Script para registrar volatility_lstm.pkl en MLflow
Ubicación sugerida: scripts/register_model_mlflow.py
"""

import os
import pickle
import mlflow
import mlflow.pyfunc
from pathlib import Path

# Configuración
MLFLOW_URI = "http://localhost:5000"  # Cambiar si tu MLflow está en otra URI
EXPERIMENT_NAME = "crypto-predictor"
REGISTERED_MODEL_NAME = "crypto-predictor"
PKL_PATH = "volatility_lstm.pkl"  # Debe estar en la raíz del proyecto

# Wrapper para MLflow PyFunc
class LSTMWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(PKL_PATH, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def main():
    # Revisar que exista el pkl
    if not Path(PKL_PATH).exists():
        raise FileNotFoundError(f"No se encontró {PKL_PATH}")

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LSTMWrapper(),
            registered_model_name=REGISTERED_MODEL_NAME
        )
        print(f"✅ Modelo {REGISTERED_MODEL_NAME} registrado en MLflow")

if __name__ == "__main__":
    main()
