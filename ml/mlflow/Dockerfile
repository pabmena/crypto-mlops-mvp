FROM python:3.11-slim

WORKDIR /mlflow

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLFlow y dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para artefactos
RUN mkdir -p /mlflow/mlruns

# Exponer puerto
EXPOSE 5000

# Script de inicio
COPY start-mlflow.sh /mlflow/
RUN chmod +x /mlflow/start-mlflow.sh

CMD ["/mlflow/start-mlflow.sh"]