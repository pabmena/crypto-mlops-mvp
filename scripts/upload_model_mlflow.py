import os
import pickle
import logging
import mlflow
import mlflow.pytorch

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de paths
LOCAL_MODEL_PATH = "ml/models/volatility_lstm.pth"  # Path relativo dentro del contenedor
PICKLE_MODEL_PATH = "ml/models/volatility_lstm.pkl"

# Nombre del experimento
EXPERIMENT_NAME = "crypto-predictor"

# Configurar tracking URI de MLflow desde variable de entorno
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

# Crear o usar el experimento
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    logger.info(f"✅ Experimento '{EXPERIMENT_NAME}' creado con ID {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    logger.info(f"Usando experimento existente '{EXPERIMENT_NAME}' con ID {experiment_id}")

# Definición simple del modelo placeholder (si no tenés pesos)
import torch
import torch.nn as nn

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Inicializar modelo
model = VolatilityLSTM()

# Intentar cargar pesos si existen
if os.path.exists(LOCAL_MODEL_PATH):
    model.load_state_dict(torch.load(LOCAL_MODEL_PATH))
    logger.info(f"✅ Pesos cargados desde {LOCAL_MODEL_PATH}")
else:
    logger.warning(f"[WARN] No se encontró el archivo {LOCAL_MODEL_PATH}, subiendo modelo sin pesos")

# Serializar modelo con pickle
with open(PICKLE_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
logger.info(f"Modelo serializado en {PICKLE_MODEL_PATH}")

# Subir a MLflow
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_artifact(PICKLE_MODEL_PATH, artifact_path="volatility_lstm")
    logger.info(f"✅ Modelo subido a MLflow bajo el experimento '{EXPERIMENT_NAME}'")

logger.info("Script finalizado")


