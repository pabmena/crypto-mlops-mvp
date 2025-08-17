# api/app.py
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime, timezone
import json
from pathlib import Path  # usar Path de forma consistente
import ccxt
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# App & métricas en memoria
# -----------------------------------------------------------------------------
app = FastAPI(title="Crypto MLOps MVP", version="0.1.0")

app.state.metrics = {
    "start_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "requests_total": 0,
    "signals_total": 0,
    "last_signal_at": None,
}

# -----------------------------------------------------------------------------
# Persistencia: /app/data (montado desde ./data en docker-compose)
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"  # <- /app/data
DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_JSONL = DATA_DIR / "signals.jsonl"

import os
from datetime import datetime, timezone

MAX_BYTES = int(os.getenv("SIGNALS_MAX_BYTES", "2000000"))  # ~2 MB

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > MAX_BYTES:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        rotated = path.with_name(f"{path.stem}-{ts}.jsonl")
        path.rename(rotated)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Middleware simple para contar requests
# -----------------------------------------------------------------------------
@app.middleware("http")
async def count_requests(request, call_next):
    app.state.metrics["requests_total"] += 1
    response = await call_next(request)
    return response


# -----------------------------------------------------------------------------
# Modelos I/O
# -----------------------------------------------------------------------------
class MetricsOut(BaseModel):
    start_time: str
    requests_total: int
    signals_total: int
    last_signal_at: Optional[str] = None


class OHLCVBar(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    time: str  # ISO string


class OHLCVOut(BaseModel):
    symbol: str
    exchange: str
    timeframe: str
    limit: int
    rows: int
    data: List[OHLCVBar]


class SignalIn(BaseModel):
    symbol: str = "BTCUSDT"       # admite 'BTCUSDT' o 'BTC/USDT'
    horizon_min: int = 60
    explain: bool = False
    exchange: str = "binance"
    timeframe: str = "1h"
    limit: int = 200


class SignalOut(BaseModel):
    symbol: str
    horizon_min: int
    risk_score: float
    nowcast_ret: float
    vol_regime: str
    explain: Optional[dict] = None


# -----------------------------------------------------------------------------
# Utilidades de mercado y features
# -----------------------------------------------------------------------------
def normalize_symbol(s: str) -> str:
    s = s.upper().replace(" ", "")
    if "/" in s:
        return s
    if len(s) >= 6:
        return s[:-4] + "/" + s[-4:]  # BTCUSDT -> BTC/USDT
    return s


def fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex_cls = getattr(ccxt, exchange_name)
    ex = ex_cls({"enableRateLimit": True})
    mkt_symbol = normalize_symbol(symbol)
    ohlcv = ex.fetch_ohlcv(mkt_symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    # convertir a string ISO para ser 100% JSON-serializable de forma estable
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df

def compute_features(df: pd.DataFrame) -> dict:
    df = df.copy()

    # --- asegurar columna de tiempo ---
    if "time" not in df.columns:
        if "ts" in df.columns:
            t = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
            # si todo NaT (p.ej. ts en segundos), reintenta en segundos
            if t.isna().all():
                t = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
            df["time"] = t
        else:
            # último recurso: timeline sintética
            df["time"] = pd.to_datetime(pd.RangeIndex(len(df)), unit="s", utc=True)

    # --- features básicas ---
    close = pd.to_numeric(df["close"], errors="coerce")
    df["ret"] = close.pct_change()
    df["vol24"] = df["ret"].rolling(24, min_periods=8).std()      # ~1d en 1h
    df["sma12"] = close.rolling(12, min_periods=4).mean()
    df["sma48"] = close.rolling(48, min_periods=12).mean()

    last = df.iloc[-1]
    nowcast_ret = float(last.get("ret", np.nan))
    vol = float(last.get("vol24", np.nan))

    if np.isnan(nowcast_ret):
        nowcast_ret = 0.0
    if np.isnan(vol):
        vol = 0.0

    # régimen de volatilidad
    if vol < 0.005:
        vol_regime = "calm"
    elif vol < 0.015:
        vol_regime = "normal"
    else:
        vol_regime = "turbulent"

    # score de riesgo (0..1)
    mom = 1.0 if (last.get("sma12", np.nan) > last.get("sma48", np.nan)) else 0.0
    vol_norm = float(np.tanh(vol * 50.0))
    risk_score = float(np.clip(0.6 * (1.0 - vol_norm) + 0.4 * (1.0 - mom), 0.0, 1.0))

    return {
        "nowcast_ret": nowcast_ret,
        "vol": vol,
        "vol_regime": vol_regime,
        "risk_score": risk_score,
        "features_tail": df.tail(5)[["time", "close", "ret", "vol24", "sma12", "sma48"]]
            .to_dict(orient="records"),
    }

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsOut)
def metrics() -> MetricsOut:
    return MetricsOut(**app.state.metrics)


@app.get("/v1/crypto/ohlcv", response_model=OHLCVOut)
def ohlcv(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    limit: int = 200,
) -> OHLCVOut:
    df = fetch_ohlcv(exchange, symbol, timeframe, limit)
    return OHLCVOut(
        symbol=normalize_symbol(symbol),
        exchange=exchange,
        timeframe=timeframe,
        limit=limit,
        rows=len(df),
        data=[OHLCVBar(**row) for row in df.tail(50).to_dict(orient="records")],  # cap de respuesta
    )


@app.post("/v1/crypto/signal", response_model=SignalOut)
def signal(inp: SignalIn = Body(...)):
    try:
        # 1) Datos
        df = fetch_ohlcv(inp.exchange, inp.symbol, inp.timeframe, inp.limit)
        if df is None or df.empty or len(df) < 10:
            # Evita IndexError en compute_features y devuelve 503 “temporal”
            raise HTTPException(status_code=503, detail="Insufficient OHLCV data from exchange")

        # 2) Features
        feats = compute_features(df)

        # 3) Salida
        out = {
            "symbol": normalize_symbol(inp.symbol),
            "horizon_min": int(inp.horizon_min),
            "risk_score": float(round(feats["risk_score"], 4)),
            "nowcast_ret": float(round(feats["nowcast_ret"], 6)),
            "vol_regime": str(feats["vol_regime"]),
            "explain": feats if inp.explain else None,
        }

        # 4) Métricas
        app.state.metrics["signals_total"] += 1
        app.state.metrics["last_signal_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # 5) Persistencia (no romper si falla)
        try:
            append_jsonl(SIGNALS_JSONL, {
                "ts": app.state.metrics["last_signal_at"],
                "symbol": out["symbol"],
                "horizon_min": out["horizon_min"],
                "risk_score": out["risk_score"],
                "nowcast_ret": out["nowcast_ret"],
                "vol_regime": out["vol_regime"],
                "exchange": inp.exchange,
                "timeframe": inp.timeframe,
                "limit": inp.limit,
            })
        except Exception as e:
            # log suave, pero no afectes la respuesta
            print(f"[persist][WARN] {type(e).__name__}: {e}")

        # 6) ¡SIEMPRE retornar!
        return out

    except HTTPException:
        # re-levanta HTTPs ya controlados (503, 4xx)
        raise
    except ccxt.BaseError as e:
        print(f"[ccxt][ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="Exchange error via ccxt")
    except Exception as e:
        print(f"[signal][ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal error in /v1/crypto/signal")

@app.get("/v1/crypto/signals/tail")
def signals_tail(n: int = 5) -> list[dict[str, Any]]:
    """Devuelve las últimas n señales persistidas en ./data/signals.jsonl"""
    if not SIGNALS_JSONL.exists():
        return []
    lines = SIGNALS_JSONL.read_text(encoding="utf-8").splitlines()
    tail = lines[-n:] if len(lines) >= n else lines
    return [json.loads(x) for x in tail]

@app.get("/", response_class=HTMLResponse)
def home():
    html = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Crypto MLOps MVP</title>
<style>
  :root{
    --bg:#f3f4f6; --panel:#ffffff; --ink:#0f172a; --muted:#475569; --line:#e5e7eb;
    --green:#22c55e; --amber:#f59e0b; --red:#ef4444; --gray:#94a3b8; --blue:#3b82f6;
    /* fondos (compatibles sin color-mix) */
    --green-bg:#dcfce7; --green-fg:#065f46;
    --amber-bg:#fef3c7; --amber-fg:#7c2d12;
    --red-bg:#fee2e2;   --red-fg:#7f1d1d;
    --gray-bg:#e5e7eb;  --gray-fg:#111827;
    --blue-bg:#dbeafe;  --blue-fg:#1e3a8a;
  }
  body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
  .wrap{max-width:1200px;margin:32px auto;padding:0 16px}
  h1{font-weight:800;letter-spacing:.2px;margin:0 0 8px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
  .muted{color:var(--muted)}
  label{display:block;font-weight:600;margin:10px 0 6px}
  input,select{width:220px;max-width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:#fff}
  .btn{border:1px solid var(--line);background:#fff;border-radius:10px;padding:8px 12px;cursor:pointer}
  .btn:hover{filter:brightness(.97)}
  .btn.primary{background:#0f172a;color:#fff;border-color:#0f172a}
  .pill{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;font-weight:700;border:1px solid transparent}
  .dot{width:10px;height:10px;border-radius:50%}
  .pill-green{background:var(--green-bg);color:var(--green-fg);border-color:var(--green)}
  .pill-amber{background:var(--amber-bg);color:var(--amber-fg);border-color:var(--amber)}
  .pill-red{background:var(--red-bg);color:var(--red-fg);border-color:var(--red)}
  .pill-gray{background:var(--gray-bg);color:var(--gray-fg);border-color:var(--gray)}
  .pill-blue{background:var(--blue-bg);color:var(--blue-fg);border-color:var(--blue)}
  pre{background:#0f172a;color:#e5e7eb;border-radius:8px;padding:10px;overflow:auto;max-height:160px}
  table{width:100%;border-collapse:collapse;border:1px solid var(--line);border-radius:12px;overflow:hidden}
  th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  .legend{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:4px 8px;background:#fff}
  .pills-row{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 12px 0}
  .w200{width:200px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Crypto MLOps MVP</h1>
  <div class="muted">API: <code>/v1/crypto/ohlcv</code> · <code>/v1/crypto/signal</code> · Métricas: <code>/metrics</code></div>

  <div class="row" style="margin-top:16px">
    <!-- Panel de parámetros -->
    <div class="card">
      <h3>Parámetros</h3>
      <label>Símbolo</label>
      <input id="symbol" value="BTCUSDT" class="w200"/>
      <label>Exchange</label>
      <select id="exchange" class="w200">
        <option value="binance" selected>binance</option>
      </select>
      <label>Timeframe</label>
      <select id="timeframe" class="w200">
        <option>1h</option><option>15m</option><option>4h</option><option>1d</option>
      </select>
      <label>Limit</label>
      <input id="limit" type="number" value="200" class="w200"/>
      <label>Horizonte (min)</label>
      <input id="horizon" type="number" value="60" class="w200"/>

      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn primary" id="btnSignal">Refrescar señal</button>
        <button class="btn" id="btnOHLCV">OHLCV</button>
      </div>

      <div style="margin-top:12px" class="muted">
        Consejo: “Refrescar señal” hace un POST a <code>/v1/crypto/signal</code> (y persiste en <code>/app/data/signals.jsonl</code>).
      </div>
      <div style="margin-top:14px" class="legend">
        <span class="badge"><span class="dot" style="background:var(--green)"></span> Riesgo bajo &lt; 0.40</span>
        <span class="badge"><span class="dot" style="background:var(--amber)"></span> Riesgo medio 0.40–0.69</span>
        <span class="badge"><span class="dot" style="background:var(--red)"></span> Riesgo alto ≥ 0.70</span>
      </div>
      <div style="margin-top:6px" class="legend">
        <span class="badge"><span class="dot" style="background:var(--blue)"></span> Vol: <b>calm</b> (✓)</span>
        <span class="badge"><span class="dot" style="background:var(--amber)"></span> Vol: <b>normal</b> (≈)</span>
        <span class="badge"><span class="dot" style="background:var(--red)"></span> Vol: <b>turbulent</b> (⚠)</span>
      </div>
    </div>

    <!-- Panel de señal -->
    <div class="card">
      <h3>Señal actual</h3>

      <!-- Indicadores -->
      <div class="pills-row">
        <div id="riskPill" class="pill pill-gray">
          <span class="dot" id="riskDot" style="background:var(--gray)"></span>
          Riesgo —
        </div>
        <div id="volPill" class="pill pill-gray">
          <span class="dot" id="volDot" style="background:var(--gray)"></span>
          Vol —
        </div>
      </div>

      <div class="muted" id="sigMeta">—</div>
      <div style="margin-top:10px" id="sigExplain"><pre>—</pre></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>OHLCV (últimas 50 filas)</h3>
    <div style="overflow:auto;max-height:420px">
      <table id="tbl"><thead></thead><tbody></tbody></table>
    </div>
  </div>
</div>

<script>
  const $ = (id)=>document.getElementById(id);

  function riskBand(r){
    if(r==null || isNaN(r)) return {cls:"pill-gray", dot:"var(--gray)", label:"—"};
    if(r>=0.70) return {cls:"pill-red",   dot:"var(--red)",   label:"ALTO"};
    if(r>=0.40) return {cls:"pill-amber", dot:"var(--amber)", label:"MEDIO"};
    return {cls:"pill-green", dot:"var(--green)", label:"BAJO"};
  }

  function volBand(regime){
    if(!regime) return {cls:"pill-gray", dot:"var(--gray)", text:"—"};
    const s = String(regime).toLowerCase();
    if(s==="calm")      return {cls:"pill-blue",  dot:"var(--blue)",  text:"✓ calm"};
    if(s==="normal")    return {cls:"pill-amber", dot:"var(--amber)", text:"≈ normal"};
    if(s==="turbulent") return {cls:"pill-red",   dot:"var(--red)",   text:"⚠ turbulent"};
    return {cls:"pill-gray", dot:"var(--gray)", text:s};
  }

  async function fetchJSON(url, opts){ const r = await fetch(url, opts); if(!r.ok) throw new Error(await r.text()); return r.json(); }

  async function runOHLCV(){
    const q = new URLSearchParams({
      symbol: $('symbol').value || 'BTCUSDT',
      exchange: $('exchange').value || 'binance',
      timeframe: $('timeframe').value || '1h',
      limit: String($('limit').value || 200)
    }).toString();
    const data = await fetchJSON(`/v1/crypto/ohlcv?${q}`);
    renderOHLCV(data);
  }

  async function runSignal(){
    const body = {
      symbol: $('symbol').value || 'BTCUSDT',
      horizon_min: Number($('horizon').value || 60),
      explain: true,
      exchange: $('exchange').value || 'binance',
      timeframe: $('timeframe').value || '1h',
      limit: Number($('limit').value || 200),
    };
    const sig = await fetchJSON('/v1/crypto/signal', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    renderSignal(sig);
  }

  function renderSignal(sig){
    $('sigMeta').innerHTML = `
      <b>Símbolo:</b> ${sig.symbol} &nbsp;&nbsp; 
      <b>Horizonte:</b> ${sig.horizon_min} min<br/>
      <b>Riesgo (0–1):</b> ${Number(sig.risk_score).toFixed(6)} &nbsp;&nbsp; 
      <b>Nowcast ret:</b> ${Number(sig.nowcast_ret).toFixed(6)}
    `;

    // Pill de riesgo
    const rb = riskBand(Number(sig.risk_score));
    const rP = $('riskPill');
    rP.className = `pill ${rb.cls}`;
    rP.innerHTML = `<span class="dot" style="background:${rb.dot}"></span>
                    Riesgo ${rb.label} (${Number(sig.risk_score).toFixed(2)})`;

    // Pill de volatilidad
    const vb = volBand(sig.vol_regime);
    const vP = $('volPill');
    vP.className = `pill ${vb.cls}`;
    vP.innerHTML = `<span class="dot" style="background:${vb.dot}"></span>
                    Vol ${vb.text}`;

    // Tabla/trace de features
    if(sig.explain && sig.explain.features_tail){
      const rows = sig.explain.features_tail.map(x =>
        `${x.time}    close=${(+x.close).toFixed(6)}    ret=${Number(x.ret||0).toFixed(6)}    vol24=${Number(x.vol24||0).toFixed(6)}    sma12=${Number(x.sma12||0).toFixed(6)}    sma48=${Number(x.sma48||0).toFixed(6)}`
      ).join("\n");
      $('sigExplain').innerHTML = `<pre>${rows}</pre>`;
    }else{
      $('sigExplain').innerHTML = `<pre>—</pre>`;
    }
  }

  function renderOHLCV(data){
    const head = ['time','open','high','low','close','volume'];
    $('tbl').querySelector('thead').innerHTML =
      `<tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr>`;
    const rows = (data.data||[]).map(r =>
      `<tr><td>${r.time}</td><td>${r.open}</td><td>${r.high}</td><td>${r.low}</td><td>${r.close}</td><td>${r.volume}</td></tr>`
    ).join('');
    $('tbl').querySelector('tbody').innerHTML = rows;
  }

  // Botones
  $('btnOHLCV').addEventListener('click', runOHLCV);
  $('btnSignal').addEventListener('click', async ()=>{ await runSignal(); await runOHLCV(); });

  // Autoload: carga OHLCV + Señal al abrir
  (async ()=>{ try{ await runOHLCV(); await runSignal(); }catch(e){ console.error(e); }})();
</script>
</body>
</html>
"""
    return HTMLResponse(html)

