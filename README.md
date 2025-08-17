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

## 📋 Tabla de Contenido

- [🎯 Estado Actual](#-estado-actual)
- [🏗️ Arquitectura](#️-arquitectura)
- [📋 Requisitos](#-requisitos)
- [⚡ Instalación y Arranque](#-instalación-y-arranque)
- [📂 Estructura del Repositorio](#-estructura-del-repositorio)
- [⚙️ Configuración](#️-configuración)
- [🔌 Endpoints de la API](#-endpoints-de-la-api)
- [💾 Persistencia de Datos](#-persistencia-de-datos)
- [📊 Métricas y Observabilidad](#-métricas-y-observabilidad)
- [🖥️ Interfaz Local (UI)](#️-interfaz-local-ui)
- [📈 Reporte Diario](#-reporte-diario)
- [🧪 Tests](#-tests)
- [🔧 Troubleshooting](#-troubleshooting)
- [✅ Criterios de Aceptación](#-criterios-de-aceptación)
- [🗺️ Roadmap](#️-roadmap)
- [💰 Costos y Herramientas](#-costos-y-herramientas)
- [📄 Licencia](#-licencia)
- [📚 Comandos Útiles](#-comandos-útiles)

---

## 🎯 Estado Actual

### ✅ Lo que ya funciona:

- **🐳 Contenedor FastAPI** (Python 3.11) sirviendo en `http://localhost:8800`
- **📡 Endpoints completos:**
  - `GET /health` — healthcheck
  - `GET /metrics` — métricas de servicio en memoria
  - `GET /v1/crypto/ohlcv` — OHLCV real vía ccxt (Binance; timeframe configurable)
  - `POST /v1/crypto/signal` — señal heurística con features (retornos, volatilidad, SMA12/48, régimen de volatilidad)
  - `GET /v1/crypto/signals/tail?n=5` — últimas n señales persistidas (JSONL)
- **💾 Persistencia local:** `./data/signals.jsonl` (mapeado al contenedor como `/app/data`)
- **🖥️ UI mínima:** página estática que llama a la API y muestra la señal con indicador de riesgo por color
- **📈 Reporte diario:** script `tools/daily_report.ps1` que consulta la API y genera `report.md`
- **🧪 Tests:** pytest para funciones core
- **📖 Documentación:** OpenAPI/Swagger en `http://localhost:8800/docs`

### 🎯 Decisiones Clave:

- **🔒 Sin claves privadas:** datos públicos (ccxt sin auth)
- **📁 Docker Compose** mapea `./data` → `/app/data` para que los archivos queden en tu PC
- **🪟 Comandos pensados** para Windows/PowerShell (funciona también con Git Bash)

---

## 🏗️ Arquitectura

```
Cliente (curl / PowerShell / UI estática)
               │
               ▼
        FastAPI (app.py)
  ┌───────────┬───────────┐
  │ ccxt      │ Features  │
  │ (Binance) │ (ret/vol) │
  └───────────┴───────────┘
               │
     ./data/signals.jsonl (host)
```

---

## 📋 Requisitos

- **🐳 Docker Desktop** actualizado (con Docker Compose)
- **🪟 Windows 10/11** (PowerShell) o Git Bash
- **⚡ (Opcional)** `make` en PATH para atajos

---

## ⚡ Instalación y Arranque

### 1️⃣ Obtener el Repositorio

```powershell
cd C:\Dev
# Si aún no lo tienes:
# git clone https://github.com/<tu-usuario>/crypto-mlops-mvp.git
cd .\crypto-mlops-mvp
```

### 2️⃣ Preparar Configuración

```powershell
if (-not (Test-Path .\.env) -and (Test-Path .\.env.example)) {
  Copy-Item .\.env.example .\.env
}
```

> **💡 Nota:** Hoy el `.env` es opcional; quedará para configuraciones futuras.

### 3️⃣ Build & Up

**Con make:**
```bash
make up
```

**Sin make:**
```bash
docker compose up -d --build
```

### 4️⃣ Verificar

- **📖 Docs:** http://localhost:8800/docs
- **❤️ Health check:**
```powershell
Invoke-RestMethod http://localhost:8800/health
```

---

## 📂 Estructura del Repositorio

```
crypto-mlops-mvp/
├─ 📁 api/
│  ├─ 📄 app.py              # FastAPI + lógica de features/señales
│  ├─ 📄 requirements.txt    # fastapi, uvicorn, pydantic, ccxt, pandas, numpy, pytest...
│  ├─ 🐳 Dockerfile          # ENV PYTHONPATH=/app para que 'import app' funcione
│  └─ 📁 tests/
│     └─ 📄 test_core.py     # tests unitarios básicos
├─ 📁 data/                  # persistencia local (montado como /app/data)
│  └─ 📄 signals.jsonl       # (lo genera la API al llamar /v1/crypto/signal)
├─ 📁 tools/
│  └─ 📄 daily_report.ps1    # genera report.md con métricas + última señal
├─ 📁 ui/
│  └─ 📄 index.html          # interfaz mínima (estática)
├─ 🐳 docker-compose.yml
├─ ⚙️ .env.example
├─ 🛠️ Makefile               # (opcional; atajos up/down/logs/test/report)
└─ 📖 README.md
```

---

## ⚙️ Configuración

### Variables de Entorno (`.env`)

Variables reservadas para futuras integraciones:

```env
# Ejemplos (no usados hoy)
# BINANCE_API_KEY=
# BINANCE_API_SECRET=
# LOG_LEVEL=INFO
```

> **🔒 Importante:** Todo corre con datos públicos.

---

## 🔌 Endpoints de la API

**Base URL:** `http://localhost:8800`

### 1️⃣ `GET /health`

**Respuesta:**
```json
{
  "status": "ok"
}
```

### 2️⃣ `GET /metrics`

**Respuesta:**
```json
{
  "start_time": "2025-08-17T05:23:31.686555Z",
  "requests_total": 3,
  "signals_total": 1,
  "last_signal_at": "2025-08-17T05:24:00.229495Z"
}
```

### 3️⃣ `GET /v1/crypto/ohlcv`

**Query Parameters:**
- `symbol` (default: `BTCUSDT` o `BTC/USDT`)
- `exchange` (default: `binance`)
- `timeframe` (default: `1h`)
- `limit` (default: `200`)

**Ejemplo:**
```powershell
Invoke-RestMethod "http://localhost:8800/v1/crypto/ohlcv?symbol=BTCUSDT&exchange=binance&timeframe=1h&limit=50" `
| ConvertTo-Json -Depth 4
```

**Respuesta:**
```json
{
  "symbol": "BTC/USDT",
  "exchange": "binance",
  "timeframe": "1h",
  "limit": 50,
  "rows": 50,
  "data": [
    {
      "ts": 1755392400000,
      "open": 117255.18,
      "high": 118076.11,
      "low": 117001.50,
      "close": 118076.11,
      "volume": 1234.56
    }
  ]
}
```

### 4️⃣ `POST /v1/crypto/signal`

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "horizon_min": 60,
  "explain": true,
  "exchange": "binance",
  "timeframe": "1h",
  "limit": 200
}
```

**Ejemplo:**
```powershell
$body = @{ 
  symbol="BTCUSDT"; 
  horizon_min=60; 
  explain=$true; 
  exchange="binance"; 
  timeframe="1h"; 
  limit=200 
} | ConvertTo-Json

Invoke-RestMethod http://localhost:8800/v1/crypto/signal `
  -Method POST `
  -ContentType 'application/json' `
  -Body $body `
| ConvertTo-Json -Depth 6
```

**Respuesta:**
```json
{
  "symbol": "BTC/USDT",
  "horizon_min": 60,
  "risk_score": 0.95,
  "nowcast_ret": 0.0007,
  "vol_regime": "calm",
  "explain": {
    "nowcast_ret": 0.0007,
    "vol": 0.0014,
    "vol_regime": "calm",
    "risk_score": 0.9582,
    "features_tail": [
      {
        "time": "2025-08-17T04:00:00Z",
        "close": 118076.11,
        "ret": 0.0033,
        "vol24": 0.0015,
        "sma12": 117500.25,
        "sma48": 116800.75
      }
    ]
  }
}
```

**📋 Efectos Colaterales:**
- ➕ Incrementa `signals_total` y `requests_total`
- 🕐 Actualiza `last_signal_at`
- 💾 Persiste un renglón en `./data/signals.jsonl`

### 5️⃣ `GET /v1/crypto/signals/tail?n=5`

Devuelve las últimas `n` señales persistidas (JSON por línea).

**Ejemplo:**
```powershell
Invoke-RestMethod "http://localhost:8800/v1/crypto/signals/tail?n=5" | ConvertTo-Json -Depth 3
```

---

## 💾 Persistencia de Datos

### 📁 Volumen Mapeado

```yaml
# docker-compose.yml
volumes:
  - ./data:/app/data
```

### 📄 Archivo Principal

- **Ubicación:** `./data/signals.jsonl`
- **Formato:** Una señal por línea (JSONL)

### 👀 Chequeo Rápido

```powershell
Get-Content .\data\signals.jsonl -Tail 5
```

---

## 📊 Métricas y Observabilidad

### 📈 Endpoint de Métricas

`GET /metrics` entrega contadores in-memory:
- `requests_total`
- `signals_total` 
- `last_signal_at`
- `start_time`

### 📝 Logs

```bash
docker compose logs -f api
```

> **🚀 Próximamente:** Integración con Prometheus/Grafana en el Roadmap.

---

## 🖥️ Interfaz Local (UI)

### 📄 Archivo

`ui/index.html` (estático)

### 🌐 Cómo Abrir

**Opción A (rápida):**
Doble clic en `ui/index.html` (si el navegador permite CORS local)

**Opción B (segura):**
```bash
cd ui
python -m http.server 8088
# luego abrir http://localhost:8088
```

### 🎨 Características

- **📊 Última señal:** symbol, risk_score, vol_regime, hora
- **🎯 Indicador de riesgo por color:**
  - `calm` → 🟢 verde / ✅
  - `normal` → 🟡 amarillo / ⚠️
  - `turbulent` → 🔴 rojo / 🔴
- **🔄 Botón** para refrescar datos

> **💡 Uso recomendado:** Tener la API levantada y, en otra pestaña, esta UI para monitoreo manual rápido.

---

## 📈 Reporte Diario

### 📄 Script

`tools/daily_report.ps1`

### 📋 Contenido del Reporte

- Estado de `/metrics`
- Última señal de `/v1/crypto/signal`
- TODOs de 24h (plantilla simple)

### ▶️ Ejecutar

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\daily_report.ps1
Get-Content .\report.md -TotalCount 60
```

> **⚠️ Importante:** Confirmar que la API esté arriba (`docker compose up -d --build`) y que PowerShell permite ejecutar scripts.

---

## 🧪 Tests

### 📦 Configuración

- **Framework:** pytest (incluido en `api/requirements.txt`)
- **Environment:** `ENV PYTHONPATH=/app` en Dockerfile para que `pytest` pueda hacer `import app`

### ▶️ Ejecutar Tests

```bash
docker compose run --rm api pytest -q
```

### 🔧 Troubleshooting

Si aparece `ModuleNotFoundError: No module named 'app'`, reconstruir:

```bash
docker compose build --no-cache api
```

---

## 🔧 Troubleshooting

### 1️⃣ `NameError: name 'pathlib' is not defined`

**✅ Solucionado** unificando a:

```python
from pathlib import Path
```

y usando `Path(...)` en todo el código.

### 2️⃣ No se escribe `signals.jsonl`

**🔍 Verificar volumen:**
```powershell
docker compose config | Select-String -Pattern '/app/data'
```

**🔄 Prueba ida y vuelta:**
```bash
docker compose exec api sh -lc 'mkdir -p /app/data && date > /app/data/roundtrip.txt && ls -la /app/data && cat /app/data/roundtrip.txt'
Get-Content .\data\roundtrip.txt
```

### 3️⃣ ccxt / datos de exchange

La API usa datos públicos. Si falla:
- Probá un `limit` menor
- Revisá tu conexión a internet

### 4️⃣ Swagger no muestra endpoints

1. Reabrí `http://localhost:8800/docs`
2. Si faltan, reconstruí:
```bash
docker compose up -d --build
```

---

## ✅ Criterios de Aceptación

### 🎯 Nivel Local

- ✅ **Servicio ML en local:** FastAPI + Docker con endpoints claros
- ✅ **Datos reales:** Binance (OHLCV) vía ccxt
- ✅ **Ciclo mínimo:** features → señal → persistencia → métricas → UI/reportes
- ✅ **Automación:** script diario en `tools/`
- ✅ **Buenas prácticas:** tests básicos, README, Makefile, .env, logs y métricas

---

## 🗺️ Roadmap

### 🚀 Prioridad 1

- ✅ OHLCV real con ccxt
- ✅ Persistencia JSONL
- ✅ Tests básicos
- 🔄 **Batch:** `POST /v1/crypto/signal/batch` (múltiples símbolos/timeframes)
- 🔄 **Persistir métricas** en SQLite (`./data`)

### ⚡ Prioridad 2

- 📊 Backtesting simple (rolling window)
- 📖 Endpoint de `explain` más detallado
- 🐳 Docker `HEALTHCHECK` + `/version`

### 🌟 Prioridad 3

- 📈 Prometheus/Grafana
- 🔄 Airflow/MLflow/MinIO
- 🚢 Canary deploy / rollback
- 🔒 Seguridad: API-key y rate-limit

---

## 💰 Costos y Herramientas

### 🛠️ Desarrollo Principal

- **GitHub Copilot + ChatGPT** (bajo costo/flat)
- **Claude Code / Flow:** opcional, para auditorías/ediciones puntuales

> **💡 Recomendación:** Cerrar sesiones/terminales al terminar para evitar procesos colgados.

### 🏃‍♂️ Runtime del MVP

- **Sin APIs pagas** (ccxt usa datos públicos)
- **Costo total:** $0 💸

---

## 📄 Licencia

**MIT** (o la que se defina para el repo). Agregar `LICENSE` si corresponde.

---

## 📚 Comandos Útiles

### 🐳 Docker Operations

```bash
# Levantar servicios
docker compose up -d --build

# Ver estado
docker compose ps

# Ver logs
docker compose logs -f api

# Parar servicios
docker compose down
```

### 🧪 Probar Endpoints (PowerShell)

```powershell
# Health check
Invoke-RestMethod http://localhost:8800/health

# OHLCV data
Invoke-RestMethod "http://localhost:8800/v1/crypto/ohlcv?symbol=BTCUSDT&exchange=binance&timeframe=1h&limit=50" `
| ConvertTo-Json -Depth 4

# Generate signal
$body = @{ 
  symbol="BTCUSDT"; 
  horizon_min=60; 
  explain=$true; 
  exchange="binance"; 
  timeframe="1h"; 
  limit=200 
} | ConvertTo-Json

Invoke-RestMethod http://localhost:8800/v1/crypto/signal `
  -Method POST `
  -ContentType 'application/json' `
  -Body $body `
| ConvertTo-Json -Depth 6
```

### 💾 Ver Persistencia

```powershell
Get-Content .\data\signals.jsonl -Tail 5
```

---

> **📝 Nota del alumno:** Este MVP prioriza claridad y reproducibilidad local. La idea es cerrar un circuito pequeño pero completo (datos reales → features → señal → persistencia → UI/reporte), y dejar el terreno preparado para escalar con Airflow/MLflow/MinIO cuando el tiempo lo permita.

---

<div align="center">

**🚀 ¡Listo para ser usado!**

[![⭐ Star this repo](https://img.shields.io/github/stars/tu-usuario/crypto-mlops-mvp?style=social)](https://github.com/tu-usuario/crypto-mlops-mvp)

</div>