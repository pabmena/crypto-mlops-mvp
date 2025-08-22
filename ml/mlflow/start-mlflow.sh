#!/bin/bash
echo "Starting MLflow tracking server..."
sleep 30  # Espera simple de 30 segundos
exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root s3://mlflow \
    --host 0.0.0.0 \
    --port 5000