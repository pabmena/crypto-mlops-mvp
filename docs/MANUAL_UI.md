Manual de Usuario (UI) — Crypto MLOps MVP

Este manual explica cómo instalar, usar e interpretar la interfaz web del MVP para señales de riesgo/volatilidad en cripto (hoy: BTC/USDT).

1) ¿Qué es y para qué sirve?

La UI es una página web embebida en la API (FastAPI) que te permite:

Consultar OHLCV reales (vía ccxt) para un símbolo/timeframe.

Generar una “señal” heurística con:

risk_score en rango 0–1 (bajo/medio/alto).

vol_regime (calm / normal / turbulent).

Un snapshot de 5 features recientes (retornos, vol24, SMA12/48).

Ver indicadores visuales (color + icono) para entender de un vistazo si hay riesgo elevado o volatilidad adversa.

Persistir cada señal en ./data/signals.jsonl para auditoría y reportes.

Beneficios: insight inmediato, reproducible y persistente; no usa claves privadas ni servicios pagos; corre local en Docker.

2) Requisitos

Docker Desktop (con Docker Compose) actualizado.

Windows 10/11 (PowerShell) o Linux/Mac (bash).

(Opcional) make en el PATH para atajos.

3) Instalación y arranque

Clonar / ubicarse en el repo

cd C:\Dev
# git clone <tu-repo> crypto-mlops-mvp
cd .\crypto-mlops-mvp


Copiar .env de ejemplo (opcional hoy)

if (-not (Test-Path .\.env) -and (Test-Path .\.env.example)) { Copy-Item .\.env.example .\.env }


Levantar la API+UI

Con make:

make up


Sin make:

docker compose up -d --build


Abrir la UI

Navegá a: http://localhost:8800/
(Swagger: http://localhost:8800/docs)

4) Mapa de la pantalla

La UI tiene tres áreas:

Parámetros (panel izquierdo)

Símbolo: por defecto BTCUSDT (también acepta BTC/USDT).

Exchange: binance.

Timeframe: 1h (también 15m/4h/1d).

Limit: cantidad de velas a pedir (ej. 200).

Horizonte (min): horizonte de señal (ej. 60).

Botones:

Refrescar señal: hace POST /v1/crypto/signal, actualiza indicadores y persiste la señal.

OHLCV: vuelve a cargar tabla GET /v1/crypto/ohlcv.

Debajo verás leyendas:

Riesgo: verde (bajo), ámbar (medio), rojo (alto).

Volatilidad: azul (calm ✓), ámbar (normal ≈), rojo (turbulent ⚠).

Señal actual (panel derecho)

Píldora “Riesgo”

Bajo (verde): risk_score < 0.40

Medio (ámbar): 0.40–0.69

Alto (rojo): ≥ 0.70

Píldora “Vol”

calm (azul, ✓)

normal (ámbar, ≈)

turbulent (rojo, ⚠)

Meta: símbolo, horizonte, risk_score, nowcast_ret.

Features (últimas 5 filas): time, close, ret, vol24, sma12, sma48.

Tabla OHLCV (últimas 50 filas)

time, open, high, low, close, volume

Se actualiza al abrir y cuando presionás OHLCV o Refrescar señal.

5) Interpretación de los indicadores
5.1 Riesgo (0–1)

< 0.40 (verde): contexto benigno; riesgo bajo.

0.40–0.69 (ámbar): precaución, señales mixtas.

≥ 0.70 (rojo): riesgo alto; preferible reducir exposición / no abrir.

Heurística actual: combina una normalización de volatilidad reciente con “momentum” (SMA12 vs. SMA48).

5.2 Régimen de volatilidad

calm (✓ azul): movimientos acotados; menor probabilidad de “whipsaws”.

normal (≈ ámbar): condiciones promedio.

turbulent (⚠ rojo): alta inestabilidad; spreads y slippage tienden a empeorar.

5.3 Features (cola de 5)

ret: retorno instantáneo (último cierre vs. anterior).

vol24: desviación std de ret en ventana ~24 (en 1h equivale a ~1 día).

sma12 / sma48: promedios móviles; su cruce indica sesgo de corto vs. medio plazo.

6) Operación paso a paso

Abrí http://localhost:8800/.
La UI carga automáticamente OHLCV y la señal con los parámetros por defecto.

Ajustá Símbolo, Timeframe, Limit y Horizonte si lo necesitás.

Presioná Refrescar señal:

Se recalcula la señal y se persiste en ./data/signals.jsonl.

Se actualizan:

Píldora Riesgo (color + valor).

Píldora Vol (estado calm/normal/turbulent con icono).

Features y tabla OHLCV.

Opcional: mirá /metrics para contadores (requests_total, signals_total, last_signal_at).

Opcional: inspeccioná las últimas señales guardadas:

UI/PowerShell:
Get-Content .\data\signals.jsonl -Tail 5

API (si tenés el endpoint habilitado):
GET /v1/crypto/signals/tail?n=5

7) Casos de uso rápidos

Monitoreo exprés:
Dejá la UI abierta; al cambiar de hora (o timeframe) presioná Refrescar señal para actualizar la lectura.

Comparar timeframes:
Mirá 1h vs 4h: si ambos marcan rojo y turbulent, la evidencia es más fuerte.

Reporte:
Ejecutá tools/daily_report.ps1 (si lo tenés) para generar un report.md con la última señal y métricas.

8) Endpoints útiles (para validar lo que ves)

Swagger: http://localhost:8800/docs

Salud: GET /health

Métricas: GET /metrics

OHLCV: GET /v1/crypto/ohlcv?symbol=BTCUSDT&exchange=binance&timeframe=1h&limit=50

Señal: POST /v1/crypto/signal (body JSON con symbol, horizon_min, explain, exchange, timeframe, limit)

Señales recientes (si lo agregaste): GET /v1/crypto/signals/tail?n=5

9) Troubleshooting (UI)

La página carga pero las píldoras quedan en “—”

Presioná Refrescar señal.

Revisá que la API responda: GET /health y GET /v1/crypto/ohlcv.

500 Internal Server Error al refrescar señal

Mirá logs: docker compose logs -f api --tail=200.

Suele ser fallo transitorio de ccxt/exchange. Probá bajar limit o volver a intentar.

No se escribe signals.jsonl

Verificá el volumen: docker compose config | Select-String '/app/data'.

Prueba de ida/vuelta:

docker compose exec api sh -lc 'mkdir -p /app/data && date > /app/data/roundtrip.txt && ls -la /app/data && cat /app/data/roundtrip.txt'


Luego en host: Get-Content .\data\roundtrip.txt

Swagger no lista endpoints

Reconstruí: docker compose up -d --build.

10) Beneficios para el usuario final

Lectura inmediata del riesgo con codificación de color + icono.

Transparencia: ves OHLCV y las features usadas.

Trazabilidad: cada señal queda guardada en JSONL.

Costo cero en ejecución: datos públicos, local, sin nubes ni claves.

Extensible: permite sumar endpoints (batch, backtesting, reportes) y observabilidad.

11) Limitaciones (del MVP actual)

Señal heurística (no es un modelo entrenado con backtesting formal).

Orientado a BTC/USDT y binance (fácil de extender).

Sin autenticación / control de acceso (para uso local).

No es consejo financiero.

12) Mantenimiento y pruebas

Tests (dentro del contenedor):

docker compose run --rm api pytest -q


Reconstruir imagen ante cambios en dependencias:

docker compose build --no-cache api
docker compose up -d

13) Preguntas frecuentes

¿Cada clic “Refrescar señal” guarda un renglón?
Sí, en ./data/signals.jsonl (mapeado a /app/data en el contenedor).

¿Puedo cambiar de símbolo?
Sí, usando el campo “Símbolo” (p. ej. ETHUSDT). Si el exchange no lo soporta, la API devuelve error.

¿Por qué a veces falla OHLCV?
ccxt depende de la disponibilidad del exchange; reintentá con menos limit.

Anexo: Comandos rápidos
# Levantar / bajar
docker compose up -d --build
docker compose logs -f api
docker compose down

# Probar endpoints
Invoke-RestMethod http://localhost:8800/health
Invoke-RestMethod "http://localhost:8800/v1/crypto/ohlcv?symbol=BTCUSDT&exchange=binance&timeframe=1h&limit=50" | ConvertTo-Json -Depth 4

$body = @{ symbol="BTCUSDT"; horizon_min=60; explain=$true; exchange="binance"; timeframe="1h"; limit=200 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8800/v1/crypto/signal -Method POST -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6

# Ver persistencia
Get-Content .\data\signals.jsonl -Tail 5