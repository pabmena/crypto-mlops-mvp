# 🚀 Crypto MLOps MVP

> Infraestructura completa de MLOps para señales de riesgo y volatilidad de criptomonedas con capacidades avanzadas de ML, orquestación, APIs modernas y streaming en tiempo real.

## 🎯 Trabajo Final - Operaciones de Aprendizaje de Máquina 2
**🏛️ Curso de Especialización en Inteligencia Artificial**

**Autores:**
- Pablo Ariel Menardi (a1814)  
- Ezequiel Alejandro Caamaño (a1802)

---

## 📋 Resumen Ejecutivo

**Objetivo:** Exponer señales simples de riesgo y volatilidad para criptomonedas (BTC/USDT) a través de una infraestructura MLOps completa con APIs modernas, streaming en tiempo real y capacidades de ML avanzadas.

**Enfoque:** MVP local sin dependencias de servicios cloud pagos, pero con arquitectura enterprise-ready.

**Aclaración Importante:** Los resultados obtenidos con el presente trabajo, no constituyen recomendación de operaciones en mercados reales. Su desarrollo tiene SOLO FINES ACADÉMICOS

## Flujos y tecnologías

- *Ingesta y orquestación*: Airflow ejecuta el DAG crypto_ml_pipeline.py para extraer y procesar OHLCV y (cuando corresponde) reentrenar y desplegar el modelo. Los datos intermedios y artefactos de jobs se manejan como archivos en data/ y via S3 (MinIO). Las ejecuciones quedan registradas en la BD de Airflow (PostgreSQL) configurada en docker-compose.yml.

- *Tracking y artefactos*: MLflow corre con backend en PostgreSQL (MLFLOW_BACKEND_STORE_URI=postgresql://.../mlflow) y almacena artefactos en MinIO (compatible S3) bajo el bucket mlflow. Los scripts scripts/* registran y promueven modelos; la API los carga desde MLflow en producción.

- *Serving y UI*: La API FastAPI (api/app.py) expone endpoints para señales heurísticas y predicciones ML, y una UI integrada que consume esos endpoints. La API lee el modelo de MLflow en startup, y persiste históricos ligeros en archivos JSONL (api/data/*) para la vista de “History”.

- *Streaming en tiempo real (Kafka)*: El producer publica ticks de precios en el tópico crypto-prices (reales vía CCXT o simulados). El consumer lee ese stream, calcula indicadores, genera señal heurística, consulta la API para predicción ML y publica resultados en predictions y alertas en alerts. Estos servicios demuestran el pipeline streaming y su integración con la API de ML; no están conectados directamente a la UI por simplicidad, pero podrían integrarse fácilmente exponiendo en FastAPI un WebSocket/Server-Sent Events que consuma predictions o agregando un endpoint que lea del stream/cache para que la UI lo consulte.

- *Bases de datos usadas*:
  - *PostgreSQL*: backend de MLflow (runs/metrics/params) y base de Airflow.
  - *MinIO (S3)*: almacenamiento de artefactos de MLflow (modelos, scalers, etc.).
  - Archivos locales JSONL para historiales simples de señales/predicciones en la API.
 
En conjunto, Airflow coordina los workflows batch, MLFlow versiona y sirve modelos con artefactos en MinIO, FastApi sirve predicciones y la UI, y Kafka muestra la variante streaming de ingesta y scoring en tiempo real, con la opción de conectarla a la UI via FastApi si se quiere visualización live.

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CRYPTO MLOPS ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────┤
│ Client Layer                                                        │
│ ┌─────────────┬─────────────┬─────────────┬─────────────────────┐   │
│ │ REST API    │ GraphQL     │ gRPC        │ Web Dashboards      │   │
│ │ :8800/docs  │ :4000/gql   │ :50051      │ Multiple UIs        │   │
│ └─────────────┴─────────────┴─────────────┴─────────────────────┘   │
│                                                                     │
│ Processing Layer                                                    │
│ ┌─────────────┬─────────────┬─────────────┬─────────────────────┐   │
│ │ FastAPI     │ ML Service  │ Streaming   │ Airflow             │   │
│ │ (Main API)  │ (LSTM)      │ (Kafka)     │ (Pipelines)         │   │
│ └─────────────┴─────────────┴─────────────┴─────────────────────┘   │
│                                                                     │
│ Data & ML Layer                                                     │
│ ┌─────────────┬─────────────┬─────────────┬─────────────────────┐   │
│ │ MLFlow      │ MinIO       │ PostgreSQL  │ Kafka Topics        │   │
│ │ :5000/ui    │ :9001/ui    │ (DB)        │ (prices/alerts)     │   │
│ └─────────────┴─────────────┴─────────────┴─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## ✨ Características Principales

- **🤖 Machine Learning:** Modelo LSTM bidireccional para predicción de volatilidad
- **📊 MLFlow:** Tracking completo de experimentos y model registry
- **🔄 Orquestación:** Pipelines automatizados con Airflow + MinIO
- **🌐 APIs Modernas:** REST, GraphQL y gRPC para máxima flexibilidad
- **📡 Streaming:** Kafka para datos en tiempo real
- **📈 Dashboards:** Interfaces web para monitoreo y análisis
- **🐳 Docker:** Todo containerizado y production-ready

---

## 🛠️ Prerrequisitos

**Sistema Operativo:** Linux, macOS, o Windows con WSL2

**Requisitos:**
- Docker Desktop con **mínimo 8GB RAM** disponibles
- Git
- **10GB+** de espacio libre en disco
- Puertos disponibles: 8800, 5000, 8080, 9001, 4000, 8088, 50051, 9092

---

## 🚀 Instalación Rápida

### Opción 1: Setup Automatizado (Recomendado)

```bash
# 1. Clonar repositorio
git clone https://github.com/pabmena/crypto-mlops-mvp.git
cd crypto-mlops-mvp

# 2. Cambiar a la branch correcta
git checkout feature/mlflow-implementation

# 3. Ejecutar setup completo
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Opción 2: Setup Manual

```bash
# 1. Crear archivos de configuración
cp .env.example .env

# 2. Levantar servicios
docker-compose up -d --build

# 3. Esperar inicialización (2-3 minutos)
make check-health

# 4. Configurar buckets de MinIO
make setup-buckets

# 5. Entrenar modelo inicial (opcional)
make train-model
```

### Opción 3: Usando Make

```bash
# Todo en un comando
make setup
```

---

## 🌐 Servicios Disponibles

Una vez iniciado el sistema, tendrás acceso a:

| Servicio | URL | Credenciales | Descripción |
|----------|-----|--------------|-------------|
| **FastAPI** | [http://localhost:8800/docs](http://localhost:8800/docs) | - | API principal con Swagger UI |
| **MLFlow** | [http://localhost:5000](http://localhost:5000) | - | Experimentos ML y model registry |
| **Airflow** | [http://localhost:8080](http://localhost:8080) | `admin/admin` | Orquestación de pipelines |
| **MinIO** | [http://localhost:9001](http://localhost:9001) | `minioadmin/minioadmin123` | Object storage UI |
| **GraphQL** | [http://localhost:4000/graphql](http://localhost:4000/graphql) | - | Playground GraphQL |
| **Kafka UI** | [http://localhost:8088](http://localhost:8088) | - | Monitoreo de topics Kafka |

---

## 🧪 Testing y Verificación

### Verificar Estado del Sistema

```bash
# Estado de todos los servicios
make check-health

# URLs de todos los dashboards
make dashboard-urls

# Monitoreo en tiempo real
make monitor
```

### Tests de API REST

```bash
# Señal heurística básica
curl -X POST "http://localhost:8800/v1/crypto/signal" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","explain":true}'

# Predicción con ML
curl -X POST "http://localhost:8800/v1/crypto/ml-signal" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","include_heuristic":true}'

# Comparación de métodos
curl "http://localhost:8800/v1/crypto/signals/compare?symbol=BTCUSDT"
```

### Test GraphQL

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

### Tests Automatizados

```bash
# Ejecutar suite completa de tests
make test-apis

# Generar datos de prueba
make generate-test-data

# Tests unitarios
make test
```

---

## 📊 Machine Learning

### Modelo LSTM

**Arquitectura:** LSTM bidireccional para predicción de volatilidad  
**Features:** Precio, volumen, RSI, SMA, Bollinger Bands  
**Target:** Volatilidad futura (24h)  
**Framework:** TensorFlow/Keras

### Gestión del Modelo

```bash
# Entrenar modelo desde cero
make train-model

# Ver experimentos en MLFlow
open http://localhost:5000

# Recargar modelo en producción
curl -X POST http://localhost:8800/v1/ml/model/reload
```

### Endpoints ML

| Endpoint | Método | Descripción |
|----------|---------|------------|
| `/v1/ml/model/info` | GET | Información del modelo actual |
| `/v1/ml/model/reload` | POST | Recargar modelo desde MLFlow |
| `/v1/crypto/ml-signal` | POST | Generar predicción ML |
| `/v1/crypto/signals/compare` | GET | Comparar métodos heurístico vs ML |

---

## 🔄 Orquestación con Airflow

### DAG Principal: `crypto_ml_pipeline`

**Tareas:**
1. Extracción de datos crypto
2. Procesamiento y feature engineering  
3. Validación de calidad de datos
4. Reentrenamiento de modelo
5. Deploy automático a producción

### Gestión de Pipelines

```bash
# Acceder a Airflow UI
open http://localhost:8080

# Ver logs de Airflow
make logs-airflow

# Triggear pipeline manualmente desde UI o:
# En Airflow UI -> DAGs -> crypto_ml_pipeline -> Trigger DAG
```

---

## 📡 Streaming con Kafka

### Topics Disponibles

- **crypto-prices:** Precios en tiempo real
- **predictions:** Predicciones generadas  
- **alerts:** Alertas de anomalías

### Monitoreo de Streaming

```bash
# Ver topics activos
make show-kafka-topics

# Logs del streaming
make logs-kafka

# UI de Kafka
open http://localhost:8088
```

### Ejemplo de Mensaje

```json
{
  "symbol": "BTCUSDT",
  "price": 43250.00,
  "volume": 1234.56,
  "timestamp": "2025-08-21T10:30:00Z",
  "volatility_prediction": 0.0234
}
```

---

## 💾 Gestión de Datos con MinIO

### Buckets Automáticos

- **raw-data:** Datos crudos de exchanges
- **processed-data:** Features procesadas
- **models:** Modelos ML entrenados
- **mlflow:** Artefactos de MLFlow
- **quality-reports:** Reportes de calidad

### Comandos Útiles

```bash
# Acceder a MinIO UI
open http://localhost:9001

# CLI dentro del container
docker-compose exec minio mc ls local/

# Backup de datos
make backup-data
```

---

## 🔌 APIs Disponibles

### REST API (FastAPI)

**Base URL:** `http://localhost:8800`

#### Endpoints Principales

```bash
GET    /health              # Health check
GET    /metrics             # Métricas del sistema  
GET    /v1/crypto/ohlcv     # Datos OHLCV
POST   /v1/crypto/signal    # Señal heurística
POST   /v1/crypto/ml-signal # Predicción ML
GET    /v1/crypto/signals/compare # Comparar métodos
```

### GraphQL API

**URL:** `http://localhost:4000/graphql`

#### Queries Disponibles

- `health()`: Estado del sistema
- `modelInfo()`: Información del modelo ML  
- `ohlcvData(input)`: Datos históricos

#### Mutations Disponibles

- `generateSignal(input)`: Generar señal heurística
- `generateMlSignal(input)`: Generar predicción ML

### gRPC API

**Puerto:** `50051`

#### Servicios Disponibles

- `GetOHLCV`: Obtener datos históricos
- `GenerateSignal`: Generar señal heurística  
- `GenerateMLPrediction`: Predicción ML
- `CompareSignals`: Comparar métodos
- `HealthCheck`: Verificar estado
- `StreamPrices`: Stream de precios en tiempo real

---

## 📈 Monitoreo y Métricas

### Métricas del Sistema

```bash
# Ver métricas en tiempo real
curl http://localhost:8800/metrics
```

### Ejemplo de Respuesta

```json
{
  "start_time": "2025-08-21T10:00:00Z",
  "requests_total": 1542,
  "signals_total": 234, 
  "ml_predictions_total": 89,
  "last_signal_at": "2025-08-21T10:30:00Z",
  "last_ml_prediction_at": "2025-08-21T10:25:00Z"
}
```

### Comandos de Monitoreo

```bash
# Monitoreo interactivo
make monitor

# Logs por servicio
make logs-api      # Solo API
make logs-mlflow   # Solo MLFlow  
make logs-kafka    # Solo Kafka
make logs-airflow  # Solo Airflow

# Todos los logs
make logs
```

---

## 🛠️ Comandos Make Disponibles

### Setup y Configuración
```bash
make setup          # Setup completo automático
make check-health    # Verificar estado de servicios  
make setup-buckets   # Configurar buckets de MinIO
```

### Desarrollo y Testing  
```bash
make test           # Tests unitarios
make test-apis      # Tests de endpoints
make train-model    # Entrenar modelo ML
make generate-test-data # Generar datos de prueba
```

### Monitoreo y Logs
```bash
make monitor        # Monitoreo en tiempo real
make dashboard-urls # URLs de dashboards
make logs          # Ver todos los logs
make logs-api      # Logs específicos del API
```

### Mantenimiento
```bash
make clean         # Limpiar recursos Docker
make backup-data   # Backup de datos
make dev-reset     # Reset completo del entorno
```

### Kafka y Streaming
```bash
make show-kafka-topics # Ver topics de Kafka
make logs-kafka       # Logs del streaming
```

---

## ⚙️ Configuración

### Variables de Entorno (.env)

```bash
# Database
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops123
POSTGRES_DB=crypto_mlops

# MLFlow  
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlops:mlops123@postgres:5432/crypto_mlops

# MinIO
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_AUTO_CREATE_TOPICS_ENABLE=true

# API Keys (opcional para datos reales)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

---

## 🐛 Troubleshooting

### Problemas Comunes

#### Servicios no responden
```bash
make check-health
docker-compose restart api
```

#### Falta de memoria
```bash
docker system prune -f
make clean
```

#### Puerto ocupado
```bash
# Verificar qué proceso usa el puerto
sudo netstat -tlnp | grep :8800

# Liberar puerto si es necesario
sudo kill -9 <PID>
```

#### MLFlow no conecta
```bash
make logs-mlflow
docker-compose restart mlflow postgres
```

#### Kafka no produce/consume  
```bash
make logs-kafka
docker-compose restart kafka zookeeper
```

### Logs Detallados

```bash
# Ver todos los logs
make logs

# Logs específicos por servicio  
docker-compose logs -f api
docker-compose logs -f mlflow
docker-compose logs -f airflow-webserver
docker-compose logs -f crypto-producer
```

### Reset Completo

```bash
# Si nada funciona, reset completo
make dev-reset
```

---

## 🏭 Consideraciones para Producción

### Seguridad

- **Cambiar credenciales por defecto** en `.env`
- **Configurar HTTPS/TLS** para todos los servicios
- **Implementar autenticación** y autorización
- **Configurar firewall** y network policies

### Escalabilidad

- **Migrar a Kubernetes** en lugar de Docker Compose
- **Configurar auto-scaling** para componentes críticos
- **Implementar load balancers**
- **Usar base de datos gestionada**

### Monitoreo

- **Integrar con Prometheus/Grafana**
- **Configurar alertas** proactivas
- **Logging centralizado** con ELK Stack
- **APM** para performance monitoring

### Backup y Recuperación

- **Backup automático** de datos críticos
- **Disaster recovery plan**
- **Testing de backups** regular

---

## 📁 Estructura del Proyecto

```
crypto-mlops-mvp/
├── api/                    # FastAPI application
│   ├── models/            # Pydantic models
│   ├── routes/            # API routes  
│   └── services/          # Business logic
├── ml/                     # ML models and services
│   ├── models/            # Model definitions
│   ├── inference/         # Inference service
│   ├── training/          # Training scripts
│   └── mlflow/            # MLFlow configuration
├── airflow/               # Airflow DAGs
│   ├── dags/             # DAG definitions
│   └── plugins/          # Custom plugins
├── streaming/             # Kafka producer/consumer
│   ├── producer/         # Data producers
│   └── consumer/         # Data consumers
├── grpc/                  # gRPC server
├── graphql/               # GraphQL server  
├── scripts/               # Setup and utility scripts
├── data/                  # Persistent data
├── docker-compose.yml     # Services orchestration
├── .env.example          # Environment variables template
├── Makefile              # Automation commands
└── README.md             # This file
```

---

## 🤝 Contribuir

1. **Fork** del proyecto
2. Crear **feature branch**: `git checkout -b feature/nueva-funcionalidad`  
3. **Commit** cambios: `git commit -am 'Agregar nueva funcionalidad'`
4. **Push** a la branch: `git push origin feature/nueva-funcionalidad`
5. Crear **Pull Request**

---

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

---

## 📞 Soporte

Para reportar bugs o solicitar features:
- **Issues:** [GitHub Issues](https://github.com/pabmena/crypto-mlops-mvp/issues)
- **Documentación:** Este README
- **Contacto:** Pablo Menardi & Ezequiel Caamaño

---

## 🙏 Agradecimientos

Proyecto desarrollado como Trabajo Final para la materia **Operaciones de Aprendizaje de Máquina 2** del **Curso de Especialización en Inteligencia Artificial**.

**Universidad:** Universidad de Buenos Aires  
**Año:** 2025

---

> 💡 **Tip:** Para una experiencia óptima, inicia con `make setup` y luego accede a http://localhost:8800/docs para explorar la API interactiva.
