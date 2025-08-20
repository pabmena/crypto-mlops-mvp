#!/bin/bash

# Esperar a que PostgreSQL esté disponible
echo "Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Esperar a que MinIO esté disponible
echo "Waiting for MinIO..."
while ! nc -z minio 9000; do
  sleep 1
done
echo "MinIO is ready!"

# Iniciar MLflow tracking server
echo "Starting MLflow tracking server..."
mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root s3://mlflow \
    --host 0.0.0.0 \
    --port 5000