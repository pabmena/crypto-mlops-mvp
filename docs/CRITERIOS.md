# Criterios de Aprobación — Nivel Local

**Objetivo del curso (resumen)**  
Disponibilizar herramientas de ML en un entorno productivo local usando prácticas de MLOps, con orquestación y documentación.

## Qué entregamos (y cómo mapea)

1) **Servicio ML local en contenedor** ✅  
   - FastAPI + Docker Compose sirviendo en `http://localhost:8800`
   - Endpoints: `/health`, `/metrics`, `/v1/crypto/ohlcv`, `/v1/crypto/signal`, `/v1/crypto/signals/tail`
   - UI embebida accesible en `/`

2) **Datos reales** ✅  
   - OHLCV de Binance vía `ccxt` (público, sin keys).

3) **Ciclo mínimo ML** ✅  
   - Ingesta OHLCV → features (ret, vol24, SMA12/48) → **señal** (risk_score + vol_regime)  
   - Persistencia de cada señal en `./data/signals.jsonl`  
   - Métricas de servicio (`requests_total`, `signals_total`, `last_signal_at`)  
   - Script de reporte diario (`tools/daily_report.ps1`)

4) **Prácticas de desarrollo y doc** ✅  
   - `pytest` con test de features  
   - README detallado + Manual de UI + Runbook  
   - Makefile (atajos) y docker-compose  
   - Estilo coherente de imports/rutas (Path)

5) **Orquestación (nivel local)** ✅ *enfoque liviano*  
   - **Docker Compose** (orquesta servicio/entorno)  
   - **Makefile + script diario** (`tools/daily_report.ps1`) como orquestación mínima de tareas recurrentes  
   - *Nota*: Airflow/MLflow quedan en Roadmap del repo (no requeridos para local).

## Cobertura de requisitos de la cátedra (Nivel Local)
- Implementación en local del ciclo del modelo **hasta artefacto de señal** ✅  
- Uso de orquestación **ligera** (Compose + scripts) y buenas prácticas + documentación ✅  
- Instrucciones claras de instalación/ejecución Docker ✅

## Alcance y límites
- **NO** incluye entrenamientos/registro de modelos (queda para el Nivel Contenedores/Roadmap).
- **NO** incluye GraphQL/gRPC/Streaming ni Federado (opcional para futuras iteraciones).
