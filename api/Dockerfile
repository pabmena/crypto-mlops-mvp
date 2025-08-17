FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# 👇 clave para que 'import app' funcione siempre
ENV PYTHONPATH=/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código + tests
COPY app.py ./
COPY tests ./tests

# Levantar API (para modo servicio normal)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800", "--reload"]

