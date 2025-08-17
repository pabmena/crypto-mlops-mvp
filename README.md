Crypto MLOps MVP  Infra mínima viva

Objetivo: exponer señales simples de riesgo y volatilidad para cripto (por ahora BTC/USDT) a través de una API FastAPI en Docker, con persistencia local y utilidades de operación (métricas y reporte diario).
Enfoque: MVP presentable, simple y barato de operar. Sin dependencias de nubes ni servicios pagos para correr local.

## Tabla de contenido
- [Arquitectura y estado actual](#arquitectura-y-estado-actual)
- [Requisitos](#requisitos)
- [Instalación y arranque](#instalación-y-arranque)
- [Estructura del repo](#estructura-del-repo)
- [Configuración (.env)](#configuración-env)
- [Endpoints de la API](#endpoints-de-la-api)
- [Persistencia y datos](#persistencia-y-datos)
- [Métricas / Observabilidad básica](#métricas--observabilidad-básica)
- [Reporte diario (PowerShell)](#reporte-diario-powershell)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Roadmap próximo](#roadmap-próximo)
- [Cost control (IA / herramientas)](#cost-control-ia--herramientas)
- [Licencia](#licencia)
- [Apéndice: comandos útiles](#apéndice-comandos-útiles)

Arquitectura y estado actual

Lo que ya hace:

Contenedor FastAPI (Python 3.11) sirviendo en http://localhost:8800.

Endpoints:

GET /health  healthcheck.

GET /metrics  métricas de servicio en memoria.

GET /v1/crypto/ohlcv  OHLCV real vía ccxt (Binance, timeframe configurable).

POST /v1/crypto/signal  señal heurística con features (retornos, volatilidad, SMA12/48, régimen de vol).

Persistencia local: ./data/signals.jsonl (mapeado al contenedor como /app/data).

Reporte diario: script tools/daily_report.ps1 que consulta API y genera report.md.

Tests (pytest) para funciones core.

OpenAPI/Swagger en http://localhost:8800/docs.

Decisiones clave:

Sin claves/API privadas: solo data pública (ccxt/Exchange sin auth).

Docker Compose mapea ./data  /app/data para que los archivos queden en tu PC.

Código pensado para Windows/PowerShell y también Git Bash.

Requisitos

Docker Desktop actualizado (con Docker Compose).

Windows 10/11 (PowerShell) o Git Bash.

(Opcional) make en PATH para atajos (de lo contrario, usar docker compose).

Instalación y arranque
1) Obtener el repo

Si ya lo tienes en C:\Dev\crypto-mlops-mvp, salta a 2).

cd C:\Dev
# Clona tu repo aquí si aún no lo tienes
# git clone <URL> crypto-mlops-mvp
cd .\crypto-mlops-mvp

2) Preparar .env
if (-not (Test-Path .\.env) -and (Test-Path .\.env.example)) {
  Copy-Item .\.env.example .\.env
}


Hoy el .env no es crítico; se reserva para configuraciones futuras (keys de exchange privadas, etc.).

3) Build & Up

Con make (si lo tienes):

make up


Sin make:

docker compose up -d --build

4) Verificar

Docs: http://localhost:8800/docs

Health:

Invoke-RestMethod http://localhost:8800/health

Estructura del repo
crypto-mlops-mvp/
 api/
   app.py                # FastAPI + lógica de features/señales
   requirements.txt      # fastapi, uvicorn, pydantic, ccxt, pandas, numpy, pytest...
   Dockerfile            # ENV PYTHONPATH=/app para que 'import app' funcione
   tests/
      test_core.py       # tests unitarios básicos
 data/                    # (persistencia local, montado en contenedor como /app/data)
   signals.jsonl         # (lo genera la API al solicitar /v1/crypto/signal)
 tools/
   daily_report.ps1      # genera report.md con métricas + última señal
 docker-compose.yml
 .env.example
 Makefile                 # (opcional; atajos para up/down/logs/test/report)
 README.md

Configuración (.env)

Variables reservadas (placeholder para futuras integraciones):

# Ejemplos futuros (no usados hoy)
# BINANCE_API_KEY=
# BINANCE_API_SECRET=
# LOG_LEVEL=INFO


Hoy no se requieren claves. Todo corre con datos públicos.

Endpoints de la API

Base: http://localhost:8800

GET /health

200: {"status": "ok"}

GET /metrics

Estructura:

{
  "start_time": "2025-08-17T05:23:31.686555Z",
  "requests_total": 3,
  "signals_total": 1,
  "last_signal_at": "2025-08-17T05:24:00.229495Z"
}

GET /v1/crypto/ohlcv

Query params:

symbol (default BTCUSDT o BTC/USDT)

exchange (default binance)

timeframe (default 1h)

limit (default 200)

Ejemplo (PowerShell):

Invoke-RestMethod "http://localhost:8800/v1/crypto/ohlcv?symbol=BTCUSDT&exchange=binance&timeframe=1h&limit=50" `
| ConvertTo-Json -Depth 4


Respuesta (resumen):

{
  "symbol": "BTC/USDT",
  "exchange": "binance",
  "timeframe": "1h",
  "limit": 50,
  "rows": 50,
  "data": [ { "ts": 1755392400000, "open": 117255.18, ... } ]
}

POST /v1/crypto/signal

Body:

{
  "symbol": "BTCUSDT",
  "horizon_min": 60,
  "explain": true,
  "exchange": "binance",
  "timeframe": "1h",
  "limit": 200
}


Ejemplo (PowerShell):

$body = @{ symbol="BTCUSDT"; horizon_min=60; explain=$true; exchange="binance"; timeframe="1h"; limit=200 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8800/v1/crypto/signal -Method POST -ContentType 'application/json' -Body $body `
| ConvertTo-Json -Depth 6


Respuesta (resumen):

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
      { "time": "2025-08-17T04:00:00Z", "close": 118076.11, "ret": 0.0033, "vol24": 0.0015, "sma12": ..., "sma48": ... },
      ...
    ]
  }
}


Cada llamada a /v1/crypto/signal:

Incrementa signals_total y requests_total.

Actualiza last_signal_at.

Persiste un renglón en ./data/signals.jsonl.

Persistencia y datos

Volumen mapeado en docker-compose.yml:

volumes:
  - ./data:/app/data


Archivo principal: ./data/signals.jsonl (una señal por línea, JSONL).

Chequeo rápido:

Get-Content .\data\signals.jsonl -Tail 5

Métricas / Observabilidad básica

GET /metrics entrega contadores in-memory:

requests_total, signals_total, last_signal_at, start_time.

Para métricas serias (Prometheus/Grafana) queda en Roadmap.

Logs: usar docker compose logs -f api.

Reporte diario (PowerShell)

Script: tools/daily_report.ps1
Genera report.md con:

estado de /metrics

última señal de /v1/crypto/signal

TODOs de 24h

Ejecutar:

powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\daily_report.ps1
Get-Content .\report.md -TotalCount 60


Si marca error, asegúrate de que la API esté arriba (docker compose up -d --build) y que PowerShell no bloquee scripts.

Tests

Requisitos dentro del contenedor:

pytest está listado en api/requirements.txt.

Nota sobre imports (import app)
El Dockerfile establece ENV PYTHONPATH=/app para que pytest pueda importar app.py.

Correr tests:

docker compose run --rm api pytest -q


Si alguna vez ves ModuleNotFoundError: No module named 'app', reconstruye la imagen:

docker compose build --no-cache api

Troubleshooting

1) NameError: name 'pathlib' is not defined
Solucionado unificando import a:

from pathlib import Path


y usando Path(...) en todo el código.

2) Persistencia no escribe signals.jsonl

Verifica volumen con:

docker compose config | Select-String -Pattern '/app/data'


Prueba de ida y vuelta:

docker compose exec api sh -lc 'mkdir -p /app/data && date > /app/data/roundtrip.txt && ls -la /app/data && cat /app/data/roundtrip.txt'
Get-Content .\data\roundtrip.txt


3) ccxt / datos de exchange

La API usa datos públicos. Si falla, intenta menor limit o revisa tu conexión.

4) Swagger no muestra endpoints

Abre de nuevo http://localhost:8800/docs.

Si faltan, reconstruye: docker compose up -d --build.


Criterios de aceptación (Nivel Local)

Servicio ML en local: ✅ FastAPI + Docker con endpoints claros.

Datos reales: ✅ Binance (OHLCV) vía ccxt.

Ciclo mínimo: ✅ features → señal → persistencia → métricas → reporte.

Automación: ✅ script diario en tools/.

Buenas prácticas: ✅ tests básicos, README, Makefile, .env, logs y métricas.

Roadmap futuro (no requerido en local): Airflow/MLflow/MinIO, GraphQL/gRPC/Streaming, Backtesting, CI/CD, Postgres, seguridad API-key y rate-limit.


Cost control (IA / herramientas)

Desarrollo principal: GitHub Copilot + ChatGPT (bajo costo/flat).

Claude Code / Claude-Flow: opcional y acotado a tareas puntuales (auditorías/ediciones masivas).

Evita sesiones largas o swarm innecesario.

Cierra CLI/terminal cuando termines para no dejar procesos colgados.

No hay consumo de APIs pagas en runtime del MVP (datos de ccxt públicos).

Licencia (MIT).

Apéndice: comandos útiles

Levantar/Parar/Logs

docker compose up -d --build
docker compose ps
docker compose logs -f api
docker compose down


Probar endpoints rápido (PowerShell)

Invoke-RestMethod http://localhost:8800/health

Invoke-RestMethod "http://localhost:8800/v1/crypto/ohlcv?symbol=BTCUSDT&exchange=binance&timeframe=1h&limit=50" `
| ConvertTo-Json -Depth 4

$body = @{ symbol="BTCUSDT"; horizon_min=60; explain=$true; exchange="binance"; timeframe="1h"; limit=200 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8800/v1/crypto/signal -Method POST -ContentType 'application/json' -Body $body `
| ConvertTo-Json -Depth 6


Ver persistencia

Get-Content .\data\signals.jsonl -Tail 5
