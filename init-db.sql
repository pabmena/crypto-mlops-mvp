-- Crear base de datos para MLFlow
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlops;

-- Crear base de datos para Airflow
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO mlops;