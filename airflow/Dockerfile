FROM apache/airflow:2.7.1

USER root

# Dependencias del sistema necesarias para compilar paquetes de Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    pkg-config \
    python3-dev \
    gfortran \
    libblas-dev \
    liblapack-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiamos el requirements.txt
COPY requirements.txt /tmp/requirements.txt

# Instalamos las dependencias de Python **globalmente**
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Volvemos al usuario airflow
USER airflow

