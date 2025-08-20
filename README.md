# 🚀 Crypto MLOps MVP

> **Infra mínima viva para señales de riesgo y volatilidad cripto**

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📚 Información Académica

**🎓 Trabajo Final** de la materia **Operaciones de Aprendizaje de Máquina 2**  
**🏛️ Curso de Especialización en Inteligencia Artificial**

### 👨‍🎓 Alumnos:
- **Pablo Ariel Menardi** - `a1814`
- **Ezequiel Alejandro Caamaño** - `a1802`

---

## 🎯 Proyecto

**Objetivo:** Exponer señales simples de riesgo y volatilidad para cripto (hoy: BTC/USDT) a través de una API FastAPI corriendo en Docker, con persistencia local y utilidades básicas de operación (métricas, reporte diario y una UI mínima).

**Enfoque:** **MVP** - Todo en local, sin nubes ni servicios pagos.

---

# Crypto MLOps MVP - Extended Edition

Infraestructura completa de MLOps para señales de riesgo y volatilidad de criptomonedas con capacidades avanzadas de ML, orquestación, APIs modernas y streaming en tiempo real.

## 🎯 Características Principales

- **🤖 Machine Learning**: Modelo LSTM para predicción de volatilidad
- **📊 MLFlow**: Tracking de experimentos y model registry
- **🔄 Orquestación**: Pipelines automatizados con Airflow + MinIO
- **🌐 APIs Modernas**: REST, GraphQL y gRPC
- **📡 Streaming**: Kafka para datos en tiempo real
- **📈 Dashboards**: Interfaces web para monitoreo y análisis
- **🐳 Docker**: Todo containerizado y fácil de deployar

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CRYPTO MLOPS ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────┤
│  Client Layer                                                       │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐  │
│  │   REST API  │   GraphQL   │    gRPC     │    Web Dashboards   │  │
│  │ :8800/docs  │ :4000/gql   │   :50051    │   Multiple UIs      │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘  │
│                                                                     │
│  Processing Layer                                                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐  │
│  │   FastAPI   │  ML Service │  Streaming  │      Airflow        │  │
│  │ (Main API)  │   (LSTM)    │   (Kafka)   │   (Pipelines)       │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘  │
│                                                                     │
│  Data & ML Layer                                                    │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐  │
│  │   MLFlow    │    MinIO    │ PostgreSQL  │    Kafka Topics     │  │
│  │ :5000/ui    │ :9001/ui    │    (DB)     │  (prices/alerts)    │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## ⚡ Instalación Rápida

### Prerequisitos
- Docker Desktop (con al menos 8GB RAM disponibles)
- Git
- 10GB+ espacio libre en disco

### Setup Automático
```bash
# Clonar repositorio
git clone https://github.com/pabmena/crypto-mlops-mvp.git
cd crypto-mlops-mvp

# Ejecutar setup completo (recomendado)
chmod +x scripts/setup.sh
./scripts/setup.sh

# O usar make si prefieres
make setup
```

### Setup Manual
```bash
# 1. Crear archivos de configuración
cp .env.example .env

# 2. Levantar servicios
docker-compose up -d --build

# 3. Esperar inicialización (2-3 minutos)
make check-health

# 4. Configurar buckets de MinIO
make setup-buckets
```

## 📊 Dashboards y Servicios

Después de la instalación, tendrás acceso a:

| Servicio | URL | Credenciales | Descripción |
|----------|-----|-------------|-------------|
| **FastAPI** | http://localhost:8800 | - | API REST principal |
| **FastAPI Docs** | http://localhost:8800/docs | - | Swagger UI |
| **MLFlow** | http://localhost:5000 | - | Experiments & Models |
| **Airflow** | http://localhost:8080 | admin/admin123 | Pipeline orchestration |
| **MinIO Console** | http://localhost:9001 | minioadmin/minioadmin123 | Data lake storage |
| **GraphQL** | http://localhost:4000/graphql | - | GraphQL playground |
| **gRPC Server** | localhost:50051 | - | gRPC API |
| **Kafka UI** | http://localhost:8088 | - | Stream monitoring |
| **Original UI** | file://./ui/index.html | - | Simple web interface |

## 🚀 Uso Rápido

### 1. Generar Señal Heurística
```bash
curl -X POST "http://localhost:8800/v1/crypto/signal" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","explain":true}'
```

### 2. Generar Predicción ML
```bash
curl -X POST "http://localhost:8800/v1/crypto/ml-signal" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","include_heuristic":true}'
```

### 3. Comparar Métodos
```bash
curl "http://localhost:8800/v1/crypto/signals/compare?symbol=BTCUSDT"
```

### 4. GraphQL Query
```graphql
query {
  health {
    status
    mlAvailable
  }
  modelInfo {
    modelLoaded
    modelVersion
  }
}
```

### 5. Ver Streaming en Tiempo Real
```bash
# Monitorear tópicos de Kafka
make show-kafka-topics

# Ver logs del streaming
make logs-kafka
```

## 🔧 Comandos Útiles

```bash
# Estado de servicios
make check-health
make dashboard-urls

# Logs y monitoreo
make logs                    # Todos los logs
make logs-api               # Solo API
make logs-kafka             # Solo Kafka streaming
make monitor                # Monitoreo en tiempo real

# Gestión de datos
make train-model            # Entrenar modelo ML
make test-apis              # Probar todos los endpoints
make generate-test-data     # Generar datos de prueba

# Mantenimiento
make clean                  # Limpiar recursos Docker
make backup-data            # Backup de datos
make dev-reset              # Reset completo del entorno
```

## 📡 APIs Disponibles

### REST API (FastAPI)
```bash
# Endpoints principales
GET  /health                           # Health check
GET  /metrics                          # Métricas del sistema
GET  /v1/crypto/ohlcv                  # Datos OHLCV
POST /v1/crypto/signal                 # Señal heurística
POST /v1/crypto/ml-signal              # Predicción ML
GET  /v1/crypto/signals/compare        # Comparar métodos
GET  /v1/ml/model/info                 # Info del modelo
POST /v1/ml/model/reload               # Recargar modelo
```

### GraphQL API
```bash
# Endpoint: http://localhost:4000/graphql
# Queries disponibles:
- health(): HealthStatus
- modelInfo(): ModelInfo  
- ohlcvData(input): OHLCVResponse

# Mutations disponibles:
- generateSignal(input): Signal
- generateMlSignal(input): MLSignal
```

### gRPC API
```bash
# Puerto: 50051
# Servicios disponibles:
- GetOHLCV(OHLCVRequest) -> OHLCVResponse
- GenerateSignal(SignalRequest) -> SignalResponse
- GenerateMLPrediction(MLPredictionRequest) -> MLPredictionResponse
- CompareSignals(CompareSignalsRequest) -> CompareSignalsResponse
- HealthCheck(HealthCheckRequest) -> HealthCheckResponse
- StreamPrices(StreamRequest) -> stream PriceUpdate
```

## 🤖 Machine Learning

### Modelo LSTM
- **Arquitectura**: LSTM bidireccional para predicción de volatilidad
- **Features**: Precio, volumen, indicadores técnicos (RSI, SMA, Bollinger)
- **Target**: Volatilidad futura (24h)
- **Framework**: TensorFlow/Keras

### Entrenamiento
```bash
# Entrenar modelo manualmente
make train-model

# Ver experimentos en MLFlow
open http://localhost:5000

# Recargar modelo en producción
curl -X POST http://localhost:8800/v1/ml/model/reload
```

## 🔄 Orquestación con Airflow

### DAGs Disponibles
- **crypto_ml_pipeline**: Pipeline completo de ML
  - Extracción de datos crypto
  - Procesamiento y feature engineering
  - Validación de calidad
  - Reentrenamiento de modelo
  - Deploy automático

### Monitoreo
```bash
# Ver Airflow UI
open http://localhost:8080

# Ejecutar DAG manualmente
# Desde la UI de Airflow, triggear "crypto_ml_pipeline"
```

## 📊 Data Lake (MinIO)

### Buckets Creados
- **raw-data**: Datos crudos de exchanges
- **processed-data**: Datos procesados con features
- **models**: Modelos ML entrenados
- **mlflow**: Artefactos de MLFlow
- **quality-reports**: Reportes de calidad

### Acceso
```bash
# UI de MinIO
open http://localhost:9001

# CLI (dentro del container)
docker-compose exec minio mc ls local/
```

## 📡 Streaming con Kafka

### Tópicos
- **crypto-prices**: Precios en tiempo real
- **predictions**: Predicciones generadas
- **alerts**: Alertas de anomalías

### Monitoreo
```bash
# UI de Kafka
open http://localhost:8088

# Ver mensajes en tiempo real
make show-kafka-topics
```

## 📈 Métricas y Monitoreo

### Métricas Disponibles
```json
{
  "start_time": "2025-01-20T10:00:00Z",
  "requests_total": 1542,
  "signals_total": 234,
  "ml_predictions_total": 89,
  "last_signal_at": "2025-01-20T10:30:00Z",
  "last_ml_prediction_at": "2025-01-20T10:25:00Z"
}
```

### Dashboard en Tiempo Real
```bash
make monitor  # Monitoreo interactivo
```

## 🧪 Testing

```bash
# Tests unitarios
make test

# Test de APIs
make test-apis

# Generación de datos de prueba
make generate-test-data
```

## 🛠 Desarrollo

### Estructura de Carpetas
```
crypto-mlops-mvp/
├── api/                    # FastAPI application
├── ml/                     # ML models and services
│   ├── models/            # Model definitions
│   ├── inference/         # Inference service
│   └── mlflow/           # MLFlow configuration
├── airflow/               # Airflow DAGs
├── streaming/             # Kafka producer/consumer
├── grpc/                  # gRPC server
├── graphql/               # GraphQL server
├── scripts/               # Setup and utility scripts
└── data/                  # Persistent data
```

### Variables de Entorno
Todas las configuraciones están en `.env`:
```bash
# Database
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops123

# MLFlow
MLFLOW_TRACKING_URI=http://localhost:5000

# MinIO
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

## 🔧 Troubleshooting

### Problemas Comunes

1. **Servicios no responden**
```bash
make check-health          # Verificar estado
docker-compose restart api # Reiniciar servicio específico
```

2. **Falta de memoria**
```bash
docker system prune -f     # Limpiar recursos
make clean                 # Reset completo
```

3. **Puerto ocupado**
```bash
sudo netstat -tlnp | grep :8800  # Ver qué usa el puerto
```

4. **MLFlow no conecta**
```bash
make logs-mlflow           # Ver logs de MLFlow
docker-compose restart mlflow postgres
```

5. **Kafka no produce/consume**
```bash
make logs-kafka            # Ver logs de Kafka
docker-compose restart kafka zookeeper
```

### Logs Útiles
```bash
# Ver todos los logs
make logs

# Logs específicos por servicio
docker-compose logs -f api
docker-compose logs -f mlflow  
docker-compose logs -f airflow-webserver
docker-compose logs -f crypto-producer
```

## 🚢 Deployment

### Producción
Para un entorno de producción, considera:

1. **Seguridad**
   - Cambiar credenciales por defecto
   - Configurar HTTPS/TLS
   - Implementar autenticación

2. **Escalabilidad**
   - Usar Kubernetes en lugar de Docker Compose
   - Configurar auto-scaling
   - Load balancers

3. **Monitoreo**
   - Integrar con Prometheus/Grafana
   - Configurar alertas
   - Logging centralizado

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

## 🤝 Contribución

1. Fork del proyecto
2. Crear feature branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la branch (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📚 Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [MinIO Documentation](https://docs.min.io/)
