Crypto MLOps MVP · Infra mínima viva

Objetivo. Exponer señales simples de riesgo y volatilidad para cripto (hoy: BTC/USDT) a través de una API FastAPI corriendo en Docker, con persistencia local y utilidades básicas de operación (métricas, reporte diario y una UI mínima).

Enfoque. -MVP- Todo en local, sin nubes ni servicios pagos.

Tabla de contenido

1. Qué es y estado actual

2. Arquitectura mínima

3. Requisitos

4. Instalación y arranque

5. Estructura del repo

6. Configuración (.env)

7. Endpoints de la API

8. Persistencia de datos

9. Métricas / Observabilidad

10. Interfaz local (UI)

11. Reporte diario

12. Tests

13. Troubleshooting

14. Criterios de aceptación (Nivel Local)

15. Roadmap

16. Costos / Herramientas

17. Licencia

Apéndice · Comandos útiles

1) Qué es y estado actual

Lo que ya hace:

Contenedor FastAPI (Python 3.11) sirviendo en http://localhost:8800.

Endpoints:

GET /health — healthcheck.

GET /metrics — métricas de servicio en memoria.

GET /v1/crypto/ohlcv — OHLCV real vía ccxt (Binance; timeframe configurable).

POST /v1/crypto/signal — señal heurística con features (retornos, volatilidad, SMA12/48, régimen de volatilidad).

GET /v1/crypto/signals/tail?n=5 — últimas n señales persistidas (JSONL).

Persistencia local: ./data/signals.jsonl (mapeado al contenedor como /app/data).

UI mínima: página estática que llama a la API y muestra la señal con indicador de riesgo por color.

Reporte diario: script tools/daily_report.ps1 que consulta la API y genera report.md.

Tests (pytest) para funciones core.

OpenAPI/Swagger: http://localhost:8800/docs.

Decisiones clave:

Sin claves privadas: datos públicos (ccxt sin auth).

Docker Compose mapea ./data -> /app/data para que los archivos queden en tu PC.

Comandos pensados para Windows/PowerShell (funciona también con Git Bash).

2) Arquitectura mínima
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

3) Requisitos

Docker Desktop actualizado (con Docker Compose).

Windows 10/11 (PowerShell) o Git Bash.

(Opcional) make en PATH para atajos.

4) Instalación y arranque
4.1 Obtener el repo
cd C:\Dev
# Si aún no lo tienes:
# git clone https://github.com/<tu-usuario>/crypto-mlops-mvp.git
cd .\crypto-mlops-mvp

4.2 Preparar .env
if (-not (Test-Path .\.env) -and (Test-Path .\.env.example)) {
  Copy-Item .\.env.example .\.env
}


Hoy el .env es opcional; quedará para configuraciones futuras.

4.3 Build & Up

Con make:

make up


Sin make:

docker compose up -d --build

4.4 Verificar

Docs: http://localhost:8800/docs

Health:

Invoke-RestMethod http://localhost:8800/health

5) Estructura del repo
crypto-mlops-mvp/
├─ api/
│  ├─ app.py              # FastAPI + lógica de features/señales
│  ├─ requirements.txt    # fastapi, uvicorn, pydantic, ccxt, pandas, numpy, pytest...
│  ├─ Dockerfile          # ENV PYTHONPATH=/app para que 'import app' funcione
│  └─ tests/
│     └─ test_core.py     # tests unitarios básicos
├─ data/                  # persistencia local (montado como /app/data)
│  └─ signals.jsonl       # (lo genera la API al llamar /v1/crypto/signal)
├─ tools/
│  └─ daily_report.ps1    # genera report.md con métricas + última señal
├─ ui/
│  └─ index.html          # interfaz mínima (estática)
├─ docker-compose.yml
├─ .env.example
├─ Makefile               # (opcional; atajos up/down/logs/test/report)
└─ README.md

6) Configuración (.env)

Variables reservadas (para futuras integraciones):

# Ejemplos (no usados hoy)
# BINANCE_API_KEY=
# BINANCE_API_SECRET=
# LOG_LEVEL=INFO


Todo corre con datos públicos.

7) Endpoints de la API

Base: http://localhost:8800

7.1 GET /health

200 OK

{"status": "ok"}

7.2 GET /metrics
{
  "start_time": "2025-08-17T05:23:31.686555Z",
  "requests_total": 3,
  "signals_total": 1,
  "last_signal_at": "2025-08-17T05:24:00.229495Z"
}

7.3 GET /v1/crypto/ohlcv

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
  "data": [ { "ts": 1755392400000, "open": 117255.18, "...": "..." } ]
}

7.4 POST /v1/crypto/signal

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
      { "time": "2025-08-17T04:00:00Z", "close": 118076.11, "ret": 0.0033, "vol24": 0.0015, "sma12":  "...", "sma48": "..." }
    ]
  }
}


Efectos colaterales de cada llamada:

Incrementa signals_total y requests_total.

Actualiza last_signal_at.

Persiste un renglón en ./data/signals.jsonl.

7.5 GET /v1/crypto/signals/tail?n=5

Devuelve las últimas n señales persistidas (JSON por línea).

Invoke-RestMethod "http://localhost:8800/v1/crypto/signals/tail?n=5" | ConvertTo-Json -Depth 3

8) Persistencia de datos

Volumen (docker-compose.yml):

volumes:
  - ./data:/app/data


Archivo principal: ./data/signals.jsonl (una señal por línea — JSONL).

Chequeo rápido:

Get-Content .\data\signals.jsonl -Tail 5

9) Métricas / Observabilidad

GET /metrics entrega contadores in-memory:

requests_total, signals_total, last_signal_at, start_time.

Logs:

docker compose logs -f api


Integración con Prometheus/Grafana queda en Roadmap.

10) Interfaz local (UI)

Archivo: ui/index.html (estático).

Cómo abrir:

Opción A (rápida): doble clic en ui/index.html (si el navegador permite CORS local).

Opción B (segura): servir estático, por ejemplo:

cd ui
python -m http.server 8088
# luego abrir http://localhost:8088


Qué muestra:

Última señal (symbol, risk_score, vol_regime, hora).

Indicador de riesgo por color:

calm → verde / ✅

normal → amarillo / ⚠️

turbulent → rojo / 🔴

Botón para refrescar datos.

Uso recomendado: tener la API levantada y, en otra pestaña, esta UI para monitoreo manual rápido.

11) Reporte diario

Script: tools/daily_report.ps1
Genera report.md con:

Estado de /metrics

Última señal de /v1/crypto/signal

TODOs de 24 h (plantilla simple)

Ejecutar:

powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\daily_report.ps1
Get-Content .\report.md -TotalCount 60


Si falla, confirmar que la API esté arriba (docker compose up -d --build) y que PowerShell permite ejecutar scripts.

12) Tests

Pytest está en api/requirements.txt.

El Dockerfile establece ENV PYTHONPATH=/app para que pytest pueda import app.

Correr tests:

docker compose run --rm api pytest -q


Si aparece ModuleNotFoundError: No module named 'app', reconstruir:

docker compose build --no-cache api

13) Troubleshooting

1) NameError: name 'pathlib' is not defined
Solucionado unificando a:

from pathlib import Path


y usando Path(...) en todo el código.

2) No se escribe signals.jsonl
Ver volumen:

docker compose config | Select-String -Pattern '/app/data'


Prueba ida y vuelta:

docker compose exec api sh -lc 'mkdir -p /app/data && date > /app/data/roundtrip.txt && ls -la /app/data && cat /app/data/roundtrip.txt'
Get-Content .\data\roundtrip.txt


3) ccxt / datos de exchange
La API usa datos públicos. Si falla, probá un limit menor o revisá tu conexión.

4) Swagger no muestra endpoints
Reabrí http://localhost:8800/docs.
Si faltan, reconstruí:

docker compose up -d --build

14) Criterios de aceptación (Nivel Local)

Servicio ML en local: ✅ FastAPI + Docker con endpoints claros.

Datos reales: ✅ Binance (OHLCV) vía ccxt.

Ciclo mínimo: ✅ features → señal → persistencia → métricas → UI/reportes.

Automación: ✅ script diario en tools/.

Buenas prácticas: ✅ tests básicos, README, Makefile, .env, logs y métricas.

15) Roadmap

Prioridad 1

(Hecho) OHLCV real con ccxt.

(Hecho) Persistencia JSONL.

(Hecho) Tests básicos.

Batch: POST /v1/crypto/signal/batch (múltiples símbolos/timeframes).

Persistir métricas en SQLite (./data).

Prioridad 2

Backtesting simple (rolling window).

Endpoint de explain más detallado.

Docker HEALTHCHECK + /version.

Prioridad 3

Prometheus/Grafana.

Airflow/MLflow/MinIO.

Canary deploy / rollback.

Seguridad: API-key y rate-limit.

16) Costos / Herramientas

Desarrollo principal: GitHub Copilot + ChatGPT (bajo costo/flat).

Claude Code / Flow: opcional, para auditorías/ediciones puntuales.

Recomiendo cerrar sesiones/terminales al terminar para evitar procesos colgados.

Runtime del MVP: sin APIs pagas (ccxt usa datos públicos).

17) Licencia

MIT (o la que se defina para el repo). Agregar LICENSE si corresponde.

Apéndice · Comandos útiles

Levantar / Parar / Logs

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

Nota del alumno. Este MVP prioriza claridad y reproducibilidad local. La idea es cerrar un circuito pequeño pero completo (datos reales → features → señal → persistencia → UI/reporte), y dejar el terreno preparado para escalar con Airflow/MLflow/MinIO cuando el tiempo lo permita.