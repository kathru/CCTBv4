"""
Walk-Forward Validation Framework — V4
=======================================
Valida estatisticamente o Signal Engine usando 8 anos de dados reais OKX.

Metodologia:
  - Walk-forward: janela deslizante de 6m treino / 2m teste
  - Out-of-sample: últimos 6 meses completamente isolados
  - Calibração: logistic regression sobre features → outcome
  - Métricas: win rate, PF, Sharpe, drawdown, por regime e por par
  - Benchmark: buy-and-hold BTC

Dados: OKX autenticado — 1H desde Jan 2018 (8+ anos)
Pares: BTC-USDT, ETH-USDT, SOL-USDT

Uso:
  python3 validate.py [--fetch] [--walkforward] [--calibrate] [--report]
  python3 validate.py  # roda tudo
"""

import os, sys, time, math, json, hmac, hashlib, base64, argparse, statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import requests
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("[AVISO] sklearn não instalado — calibração desabilitada")

# ── Configuração ──────────────────────────────────────────────────────────────

API_KEY    = os.getenv("OKX_API_KEY",    "")
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

if not API_KEY or not SECRET_KEY or not PASSPHRASE:
    # Tenta carregar de code.env
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code.env")
    if os.path.exists(_env_path):
        for _line in open(_env_path):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
        API_KEY    = os.getenv("OKX_API_KEY",    "")
        SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
        PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
BASE_URL   = "https://www.okx.com"

PAIRS      = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
TIMEFRAME  = "1H"
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")

INITIAL_CAPITAL = 1000.0   # USD por walk-forward window
# Custos via FeeModel canônico (ver strategies/fee_model.py)
try:
    from strategies.fee_model import FEE as _FEE
    FEE_RATE = _FEE.maker              # 0.001 — entrada limit (maker)
    SLIPPAGE = _FEE.slippage_base      # 0.0002 — slippage base conservador
except ImportError:
    FEE_RATE = 0.001
    SLIPPAGE = 0.0002

# Walk-forward params
TRAIN_MONTHS = 6
TEST_MONTHS  = 2
STEP_MONTHS  = 1
OOS_MONTHS   = 6           # últimos 6 meses completamente fora

# ── Auth OKX ──────────────────────────────────────────────────────────────────

_session = requests.Session() if 'requests' in sys.modules else None

def _sign(ts: str, method: str, path: str, body: str = "") -> str:
    msg = f"{ts}{method.upper()}{path}{body}"
    return base64.b64encode(
        hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()

def _auth_headers(method: str, path_with_qs: str) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {
        "OK-ACCESS-KEY":        API_KEY,
        "OK-ACCESS-SIGN":       _sign(ts, method, path_with_qs),
        "OK-ACCESS-TIMESTAMP":  ts,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type":         "application/json",
    }

def fetch_candles_auth(inst_id: str, bar: str = "1H") -> list:
    """Busca todos os candles históricos disponíveis via OKX autenticado + paginação."""
    path  = "/api/v5/market/history-candles"
    all_c, after, pages = [], None, 0

    print(f"  Buscando {inst_id} {bar}...", end="", flush=True)
    while True:
        params = {"instId": inst_id, "bar": bar, "limit": "300"}
        if after:
            params["after"] = after
        qs      = "&".join(f"{k}={v}" for k, v in params.items())
        full_path = f"{path}?{qs}"
        try:
            r    = _session.get(BASE_URL + path, params=params,
                                headers=_auth_headers("GET", full_path), timeout=12)
            data = r.json().get("data", [])
        except Exception as e:
            print(f" ERRO: {e}")
            break

        if not data:
            break

        all_c.extend(data)
        after  = data[-1][0]
        pages += 1
        time.sleep(0.15)

        if pages % 20 == 0:
            oldest = datetime.utcfromtimestamp(int(data[-1][0]) // 1000).strftime("%Y-%m")
            print(f" {pages}p/{oldest}", end="", flush=True)

    # Ordena ASC
    all_c.sort(key=lambda x: int(x[0]))
    candles = [{
        "ts":     int(c[0]) // 1000,
        "open":   float(c[1]), "high": float(c[2]),
        "low":    float(c[3]), "close": float(c[4]),
        "volume": float(c[5]),
    } for c in all_c]

    oldest = datetime.utcfromtimestamp(candles[0]["ts"]).strftime("%Y-%m-%d") if candles else "?"
    print(f" → {len(candles)} candles desde {oldest}")
    return candles


# ── Data Layer ────────────────────────────────────────────────────────────────

def save_candles(pair: str, candles: list):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{pair}_{TIMEFRAME}.json")
    with open(path, "w") as f:
        json.dump(candles, f)
    print(f"  Salvo: {path} ({len(candles)} candles)")

def load_candles(pair: str) -> list:
    path = os.path.join(DATA_DIR, f"{pair}_{TIMEFRAME}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def fetch_all():
    print("\n[FASE 1] Buscando dados históricos OKX autenticado...")
    for pair in PAIRS:
        candles = fetch_candles_auth(pair, TIMEFRAME)
        if candles:
            save_candles(pair, candles)
    print("Dados salvos.")


# ── Indicadores ───────────────────────────────────────────────────────────────

def ema(vals: list, span: int) -> list:
    k, r = 2 / (span + 1), [vals[0]]
    for v in vals[1:]:
        r.append(v * k + r[-1] * (1 - k))
    return r

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return 0.0
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    return sum(trs[-period:]) / period

def calc_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2: return 20.0
    dm_p, dm_m, trs = [], [], []
    for i in range(1, len(closes)):
        h_d = highs[i]-highs[i-1]; l_d = lows[i-1]-lows[i]
        dm_p.append(max(h_d, 0) if h_d > l_d else 0)
        dm_m.append(max(l_d, 0) if l_d > h_d else 0)
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    def smooth(a, p):
        s = sum(a[:p]); r = [s]
        for v in a[p:]: r.append(r[-1] - r[-1]/p + v)
        return r
    atr_s = smooth(trs, period)
    dmp_s = smooth(dm_p, period)
    dmm_s = smooth(dm_m, period)
    dx = []
    for i in range(len(atr_s)):
        if atr_s[i] == 0: continue
        dip = dmp_s[i]/atr_s[i]*100; dim = dmm_s[i]/atr_s[i]*100
        dx.append(abs(dip-dim)/(dip+dim)*100 if (dip+dim) > 0 else 0)
    return sum(dx[-period:]) / period if len(dx) >= period else 20.0

def bb_width_pct(closes, period=20, hist=60):
    if len(closes) < period + hist: return 0.5
    widths = []
    for i in range(hist):
        sub = closes[-(period+hist)+i:-(hist)+i+period]
        if len(sub) < period: continue
        m = sum(sub)/period
        s = math.sqrt(sum((v-m)**2 for v in sub)/period)
        widths.append(s*4/m if m > 0 else 0)
    sub = closes[-period:]
    m = sum(sub)/period
    s = math.sqrt(sum((v-m)**2 for v in sub)/period)
    cur = s*4/m if m > 0 else 0
    below = sum(1 for w in widths if w <= cur)
    return below / len(widths) if widths else 0.5


# ── Feature Engineering ───────────────────────────────────────────────────────

def extract_features(candles: list, i: int) -> dict:
    """
    Extrai features no índice i usando apenas dados passados (no lookahead).
    Retorna dict de features para o Signal Engine.
    """
    if i < 100:
        return None

    window = candles[max(0, i-200):i+1]
    closes  = [c["close"]  for c in window]
    highs   = [c["high"]   for c in window]
    lows    = [c["low"]    for c in window]
    volumes = [c["volume"] for c in window]

    if len(closes) < 50:
        return None

    # EMAs
    e9  = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)
    e200= ema(closes, 200) if len(closes) >= 200 else ema(closes, len(closes)//2)

    # ADX e ATR
    adx_val  = calc_adx(highs, lows, closes)
    atr_val  = calc_atr(highs, lows, closes)
    atr_pct  = atr_val / closes[-1] if closes[-1] > 0 else 0

    # BB width percentile
    bb_pct   = bb_width_pct(closes)
    atr_prev = calc_atr(highs[:-14], lows[:-14], closes[:-14], 14)
    atr_exp  = atr_val / atr_prev if atr_prev > 0 else 1.0

    # Volume delta (proxy orderflow via candle direction)
    recent = window[-10:]
    bv = sum(c["volume"] for c in recent if c["close"] >= c["open"])
    sv = sum(c["volume"] for c in recent if c["close"] <  c["open"])
    tv = bv + sv
    taker_imbal = (bv - sv) / tv if tv > 0 else 0.0

    # Momentum
    ret_1h  = (closes[-1] - closes[-2])  / closes[-2]  if len(closes) >= 2  else 0
    ret_4h  = (closes[-1] - closes[-5])  / closes[-5]  if len(closes) >= 5  else 0
    ret_24h = (closes[-1] - closes[-25]) / closes[-25] if len(closes) >= 25 else 0

    # Market Structure
    pivots_h = [max(highs[j-3:j+3]) for j in range(3, min(len(highs)-3, 30), 3)]
    pivots_l = [min(lows[j-3:j+3])  for j in range(3, min(len(lows)-3, 30), 3)]
    hh_score = (sum(1 for k in range(1,len(pivots_h)) if pivots_h[k]>pivots_h[k-1]) / max(len(pivots_h)-1,1)) if len(pivots_h) >= 2 else 0.5
    hl_score = (sum(1 for k in range(1,len(pivots_l))  if pivots_l[k]>pivots_l[k-1])  / max(len(pivots_l)-1,1))  if len(pivots_l) >= 2 else 0.5

    # Regime simples
    regime_code = 0  # chop
    if adx_val > 30 and e9[-1] > e21[-1] > e50[-1]: regime_code = 1   # trend up
    elif adx_val > 30 and e9[-1] < e21[-1] < e50[-1]: regime_code = -1 # trend down
    elif bb_pct < 0.15 and adx_val < 18: regime_code = 2               # compression

    return {
        # Trend features
        "ema_alignment":   1 if e9[-1]>e21[-1]>e50[-1] else (-1 if e9[-1]<e21[-1]<e50[-1] else 0),
        "price_vs_ema200": (closes[-1] - e200[-1]) / e200[-1] if e200[-1] > 0 else 0,
        "adx":             adx_val / 50,          # normalizado
        # Volatility features
        "bb_percentile":   bb_pct,
        "atr_pct":         min(atr_pct, 0.10),
        "atr_expansion":   min(atr_exp, 3.0) / 3.0,
        # Momentum features
        "ret_1h":          ret_1h,
        "ret_4h":          ret_4h,
        "ret_24h":         ret_24h,
        # Market structure
        "hh_score":        hh_score,
        "hl_score":        hl_score,
        # Orderflow proxy
        "taker_imbal":     taker_imbal,
        # Volume
        "vol_ratio":       volumes[-1] / (sum(volumes[-20:]) / 20) if sum(volumes[-20:]) > 0 else 1.0,
        # Regime
        "regime_code":     regime_code,
        # Metadata (não entra no modelo)
        "_ts":    candles[i]["ts"],
        "_close": closes[-1],
        "_atr":   atr_val,
    }


def label_outcome(candles: list, i: int, horizon: int = 4,
                  rr: float = 2.0) -> int:
    """
    Label 1 = trade vencedor, 0 = perdedor.
    Simula: entra no candle i+1, SL = ATR×1.5, TP = SL×rr.
    Sem lookahead: usa candles i+1..i+horizon.
    """
    if i + horizon >= len(candles):
        return -1  # sem dados suficientes

    feats = extract_features(candles, i)
    if feats is None:
        return -1

    entry  = candles[i+1]["open"]
    atr    = feats["_atr"]
    sl_pct = min(0.06, max(0.01, atr * 1.5 / entry))
    sl     = entry * (1 - sl_pct)
    tp     = entry * (1 + sl_pct * rr)

    for j in range(i+1, min(i+1+horizon, len(candles))):
        low  = candles[j]["low"]
        high = candles[j]["high"]
        if high >= tp: return 1   # TP hit
        if low  <= sl: return 0   # SL hit

    # EOH: retorno parcial vs custo
    exit_price = candles[min(i+horizon, len(candles)-1)]["close"]
    net_return = (exit_price - entry) / entry - (FEE_RATE + SLIPPAGE) * 2
    return 1 if net_return > 0 else 0


# ── Walk-Forward Engine ───────────────────────────────────────────────────────

def months_to_bars(n_months: int) -> int:
    return n_months * 30 * 24  # ~30 dias × 24h

def run_walkforward(candles_map: dict):
    """
    Walk-forward: para cada janela deslizante:
      - Coleta features + labels no período de treino
      - Testa estratégia no período de teste
      - Reporta métricas por janela
    """
    print("\n[FASE 2] Walk-Forward Validation...")

    # Usa BTC como âncora temporal
    btc = candles_map.get("BTC-USDT", [])
    if not btc:
        print("  ERRO: sem dados BTC")
        return []

    total_bars  = len(btc)
    oos_bars    = months_to_bars(OOS_MONTHS)
    train_bars  = months_to_bars(TRAIN_MONTHS)
    test_bars   = months_to_bars(TEST_MONTHS)
    step_bars   = months_to_bars(STEP_MONTHS)

    # Define janela disponível (exclui OOS)
    available   = total_bars - oos_bars
    print(f"  Total: {total_bars} barras | OOS reservado: {oos_bars} | Disponível: {available}")

    # OOS period
    oos_start_ts = btc[available]["ts"]
    oos_end_ts   = btc[-1]["ts"]
    print(f"  OOS: {datetime.utcfromtimestamp(oos_start_ts).strftime('%Y-%m-%d')} → {datetime.utcfromtimestamp(oos_end_ts).strftime('%Y-%m-%d')}")

    windows = []
    start   = 100  # warmup mínimo

    while start + train_bars + test_bars <= available:
        train_end = start + train_bars
        test_end  = train_end + test_bars

        train_slice = {p: c[start:train_end]    for p, c in candles_map.items()}
        test_slice  = {p: c[train_end:test_end]  for p, c in candles_map.items()}

        train_start_ts = btc[start]["ts"]
        train_end_ts   = btc[train_end-1]["ts"]
        test_start_ts  = btc[train_end]["ts"]
        test_end_ts    = btc[min(test_end-1, len(btc)-1)]["ts"]

        result = evaluate_window(
            train_slice=train_slice,
            test_slice=test_slice,
            window_id=len(windows)+1,
            train_period=(train_start_ts, train_end_ts),
            test_period=(test_start_ts, test_end_ts),
        )
        windows.append(result)
        start += step_bars

    return windows, oos_start_ts, candles_map


def evaluate_window(train_slice, test_slice, window_id,
                    train_period, test_period) -> dict:
    """Avalia uma janela: calibra no treino, mede no teste."""

    train_start = datetime.utcfromtimestamp(train_period[0]).strftime("%Y-%m")
    test_start  = datetime.utcfromtimestamp(test_period[0]).strftime("%Y-%m")

    # ── Coleta features + labels no treino ───────────────────────────────
    X_train, y_train = [], []
    for pair, candles in train_slice.items():
        horizon = 4 if "1H" in TIMEFRAME else 2
        step    = 3  # não amostrar cada candle (autocorrelação)
        for i in range(100, len(candles) - horizon, step):
            feats = extract_features(candles, i)
            if feats is None: continue
            label = label_outcome(candles, i, horizon)
            if label == -1: continue
            feature_vec = [
                feats["ema_alignment"], feats["price_vs_ema200"],
                feats["adx"], feats["bb_percentile"], feats["atr_pct"],
                feats["atr_expansion"], feats["ret_1h"], feats["ret_4h"],
                feats["ret_24h"], feats["hh_score"], feats["hl_score"],
                feats["taker_imbal"], feats["vol_ratio"], feats["regime_code"],
            ]
            X_train.append(feature_vec)
            y_train.append(label)

    # ── Calibra modelo no treino ──────────────────────────────────────────
    model = None
    auc_train = 0.5
    base_wr   = sum(y_train) / len(y_train) if y_train else 0.5

    if HAS_ML and len(X_train) >= 50 and len(set(y_train)) == 2:
        try:
            scaler = StandardScaler()
            Xs     = scaler.fit_transform(X_train)
            model  = LogisticRegression(max_iter=500, C=0.5)
            model.fit(Xs, y_train)
            probs  = model.predict_proba(Xs)[:, 1]
            auc_train = roc_auc_score(y_train, probs)
        except Exception as e:
            model = None

    # ── Simula trades no teste ────────────────────────────────────────────
    trades = []
    capital = INITIAL_CAPITAL
    equity_curve = [capital]

    for pair, candles in test_slice.items():
        # Precisa do contexto histórico — concatena treino + teste
        full = train_slice[pair] + candles
        offset = len(train_slice[pair])

        for i in range(offset + 100, len(full) - 4, 3):
            feats = extract_features(full, i)
            if feats is None: continue

            # Score: usa modelo calibrado ou heurística
            if model is not None:
                vec  = [[feats["ema_alignment"], feats["price_vs_ema200"],
                         feats["adx"], feats["bb_percentile"], feats["atr_pct"],
                         feats["atr_expansion"], feats["ret_1h"], feats["ret_4h"],
                         feats["ret_24h"], feats["hh_score"], feats["hl_score"],
                         feats["taker_imbal"], feats["vol_ratio"], feats["regime_code"]]]
                vec_s = scaler.transform(vec)
                score = model.predict_proba(vec_s)[0][1]
            else:
                # Heurística simples
                score = (
                    max(0, feats["ema_alignment"] * 0.3) +
                    (1 - feats["bb_percentile"]) * 0.2 +
                    max(0, feats["ret_4h"] * 5) * 0.2 +
                    feats["hh_score"] * 0.15 +
                    max(0, feats["taker_imbal"]) * 0.15
                )
                score = max(0.0, min(1.0, score + 0.3))

            # Entrada: threshold 0.55
            if score < 0.55: continue
            if feats["ema_alignment"] < 0: continue  # só long

            label = label_outcome(full, i, horizon=4)
            if label == -1: continue

            # Simula o trade
            entry     = full[i+1]["open"]
            atr       = feats["_atr"]
            sl_pct    = min(0.06, max(0.01, atr * 1.5 / entry))
            tp_pct    = sl_pct * 2.0
            size_usd  = min(capital * 0.08, capital * 0.08 * score)
            fee       = size_usd * (FEE_RATE + SLIPPAGE)
            pnl       = size_usd * (tp_pct if label == 1 else -sl_pct) - fee * 2
            capital  += pnl

            trades.append({
                "pair":    pair, "ts": feats["_ts"],
                "score":   round(score, 3), "label": label,
                "pnl":     round(pnl, 4), "sl_pct": sl_pct,
                "regime":  feats["regime_code"],
            })
            equity_curve.append(capital)

            if capital <= INITIAL_CAPITAL * 0.50:  # circuit breaker
                break

    # ── Métricas do teste ─────────────────────────────────────────────────
    n      = len(trades)
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate     = len(wins) / n if n > 0 else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses)) or 1e-6
    pf           = gross_profit / gross_loss
    total_ret    = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Drawdown
    peak = INITIAL_CAPITAL; max_dd = 0.0
    for v in equity_curve:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd

    # Sharpe
    sharpe = 0.0
    if len(equity_curve) > 10:
        rets = [(equity_curve[i]-equity_curve[i-1])/equity_curve[i-1]
                for i in range(1, len(equity_curve))]
        if len(rets) > 2:
            m = sum(rets)/len(rets); s = statistics.stdev(rets)
            sharpe = (m/s * math.sqrt(252*24)) if s > 0 else 0

    # Por regime
    by_regime = {}
    for rc, name in {1:"TREND_UP", -1:"TREND_DOWN", 0:"CHOP", 2:"COMPRESS"}.items():
        rt = [t for t in trades if t["regime"] == rc]
        if rt:
            rw = sum(1 for t in rt if t["pnl"] > 0)
            by_regime[name] = {
                "n": len(rt), "wr": round(rw/len(rt), 3),
                "pnl": round(sum(t["pnl"] for t in rt), 2)
            }

    print(f"  W{window_id:02d} treino={train_start} teste={test_start} | "
          f"n={n} WR={win_rate*100:.1f}% PF={pf:.2f} ret={total_ret*100:+.1f}% "
          f"DD={max_dd*100:.1f}% AUC={auc_train:.3f}")

    return {
        "window":     window_id,
        "train_start": train_start, "test_start": test_start,
        "n_trades":   n, "n_train_samples": len(X_train),
        "win_rate":   round(win_rate, 4),
        "profit_factor": round(pf, 3),
        "total_return":  round(total_ret, 4),
        "max_drawdown":  round(max_dd, 4),
        "sharpe":        round(sharpe, 3),
        "auc_train":     round(auc_train, 3),
        "base_wr":       round(base_wr, 3),
        "by_regime":     by_regime,
        "capital_final": round(capital, 2),
        "trades":        trades,   # trade-level para report.py
    }


# ── Out-of-Sample ─────────────────────────────────────────────────────────────

def run_oos(candles_map: dict, oos_start_ts: int, all_windows: list) -> dict:
    """
    Avalia no período OOS usando modelo treinado em TODOS os dados anteriores.
    Este é o teste definitivo — nunca visto durante desenvolvimento.
    """
    print("\n[FASE 3] Out-of-Sample (período nunca visto)...")

    btc = candles_map.get("BTC-USDT", [])
    oos_idx = next((i for i, c in enumerate(btc) if c["ts"] >= oos_start_ts), len(btc))

    train_slice = {p: c[:oos_idx]    for p, c in candles_map.items()}
    test_slice  = {p: c[oos_idx:]    for p, c in candles_map.items()}

    oos_start_str = datetime.utcfromtimestamp(oos_start_ts).strftime("%Y-%m-%d")
    oos_end_str   = datetime.utcfromtimestamp(btc[-1]["ts"]).strftime("%Y-%m-%d")
    print(f"  OOS: {oos_start_str} → {oos_end_str}")

    result = evaluate_window(
        train_slice=train_slice,
        test_slice=test_slice,
        window_id=0,
        train_period=(btc[0]["ts"], btc[oos_idx-1]["ts"]),
        test_period=(oos_start_ts, btc[-1]["ts"]),
    )
    result["window"] = "OOS"
    result["oos_start"] = oos_start_str
    result["oos_end"]   = oos_end_str
    return result


# ── Benchmark: Buy & Hold ─────────────────────────────────────────────────────

def calc_benchmark(candles_map: dict, oos_start_ts: int) -> dict:
    """Retorno buy-and-hold BTC no período OOS."""
    btc = candles_map.get("BTC-USDT", [])
    oos = [c for c in btc if c["ts"] >= oos_start_ts]
    if not oos: return {}
    ret = (oos[-1]["close"] - oos[0]["open"]) / oos[0]["open"]
    return {"bh_return": round(ret, 4), "bh_pct": round(ret*100, 2)}


# ── Relatório Final ───────────────────────────────────────────────────────────

def print_report(windows: list, oos: dict, benchmark: dict):
    print("\n" + "="*70)
    print("  RELATÓRIO DE VALIDAÇÃO — V4 SIGNAL ENGINE")
    print("="*70)

    if windows:
        # Estatísticas agregadas walk-forward
        wrs  = [w["win_rate"]      for w in windows if w["n_trades"] > 5]
        pfs  = [w["profit_factor"] for w in windows if w["n_trades"] > 5]
        rets = [w["total_return"]  for w in windows if w["n_trades"] > 5]
        dds  = [w["max_drawdown"]  for w in windows if w["n_trades"] > 5]
        aucs = [w["auc_train"]     for w in windows if w["n_trades"] > 5]
        ns   = [w["n_trades"]      for w in windows]

        print(f"\n  WALK-FORWARD ({len(windows)} janelas de {TRAIN_MONTHS}m treino / {TEST_MONTHS}m teste)")
        print(f"  {'Métrica':<22} {'Média':>8} {'Mediana':>8} {'Min':>8} {'Max':>8} {'Estável?':>10}")
        print("  " + "-"*66)

        def row(name, vals):
            if not vals: return
            m  = sum(vals)/len(vals)
            md = sorted(vals)[len(vals)//2]
            mn = min(vals); mx = max(vals)
            cv = (statistics.stdev(vals)/abs(m)) if len(vals) > 1 and m != 0 else 0
            stable = "✅ Sim" if cv < 0.40 else "⚠️ Não"
            print(f"  {name:<22} {m:>8.3f} {md:>8.3f} {mn:>8.3f} {mx:>8.3f} {stable:>10}")

        row("Win Rate",      wrs)
        row("Profit Factor", pfs)
        row("Retorno",       rets)
        row("Max Drawdown",  dds)
        row("AUC (treino)",  aucs)

        # Frequência
        avg_n = sum(ns)/len(ns) if ns else 0
        print(f"\n  Trades médios/janela: {avg_n:.1f}")

        # Consistência: % janelas com WR > 50%
        consistent = sum(1 for w in wrs if w > 0.50) / len(wrs) if wrs else 0
        profitable  = sum(1 for r in rets if r > 0)  / len(rets) if rets else 0
        print(f"  Janelas WR > 50%:     {consistent*100:.0f}%")
        print(f"  Janelas com lucro:    {profitable*100:.0f}%")

        # Edge summary
        avg_wr = sum(wrs)/len(wrs) if wrs else 0
        avg_pf = sum(pfs)/len(pfs) if pfs else 0
        print(f"\n  VEREDICTO WALK-FORWARD:")
        if avg_wr >= 0.55 and avg_pf >= 1.3 and profitable >= 0.65:
            print("  ✅ EDGE DETECTADO — Win rate e PF consistentes acima do breakeven")
        elif avg_wr >= 0.50 and avg_pf >= 1.1:
            print("  ⚠️  EDGE MARGINAL — Existe sinal, mas fraco. Requer refinamento")
        else:
            print("  ❌ SEM EDGE COMPROVADO — Heurística sem vantagem estatística clara")

    # OOS
    if oos:
        print(f"\n  OUT-OF-SAMPLE ({oos.get('oos_start')} → {oos.get('oos_end')}) — RESULTADO DEFINITIVO")
        print(f"  Trades:        {oos['n_trades']}")
        print(f"  Win Rate:      {oos['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {oos['profit_factor']:.2f}")
        print(f"  Retorno:       {oos['total_return']*100:+.2f}%")
        print(f"  Max Drawdown:  {oos['max_drawdown']*100:.2f}%")
        print(f"  Sharpe:        {oos['sharpe']:.2f}")
        print(f"  AUC treino:    {oos['auc_train']:.3f}")

        if benchmark:
            print(f"\n  BENCHMARK Buy&Hold BTC OOS: {benchmark['bh_pct']:+.2f}%")
            outperf = oos['total_return'] - benchmark['bh_return']
            print(f"  Alpha vs B&H: {outperf*100:+.2f}%")

        if oos.get("by_regime"):
            print(f"\n  POR REGIME (OOS):")
            for regime, stats in oos["by_regime"].items():
                print(f"    {regime:<16} n={stats['n']:>4} WR={stats['wr']*100:.0f}% P&L=${stats['pnl']:+.2f}")

        print(f"\n  VEREDICTO FINAL (OOS):")
        if oos['win_rate'] >= 0.55 and oos['profit_factor'] >= 1.3:
            print("  ✅ EDGE CONFIRMADO no out-of-sample")
            print("  → Sistema pode ser promovido de heurística para modelo semi-validado")
        elif oos['win_rate'] >= 0.50 and oos['profit_factor'] >= 1.0:
            print("  ⚠️  EDGE MARGINAL no OOS — existe sinal fraco")
            print("  → Refinar parâmetros e retestsar antes de escalar")
        else:
            print("  ❌ SEM EDGE no OOS")
            print("  → Rever arquitetura do Signal Engine — heurística sem vantagem")

    # Salva resultado
    result_path = os.path.join(DATA_DIR, "validation_result.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({"windows": windows, "oos": oos, "benchmark": benchmark}, f, indent=2)
    print(f"\n  Resultado salvo: {result_path}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation V4")
    parser.add_argument("--fetch",       action="store_true", help="Busca dados históricos OKX")
    parser.add_argument("--walkforward", action="store_true", help="Roda walk-forward")
    parser.add_argument("--report",      action="store_true", help="Apenas relatório (dados já calculados)")
    args = parser.parse_args()

    run_all = not any([args.fetch, args.walkforward, args.report])

    # ── Fetch ──────────────────────────────────────────────────────────
    if args.fetch or run_all:
        fetch_all()

    # ── Load ───────────────────────────────────────────────────────────
    print("\n[CARREGANDO] Dados históricos...")
    candles_map = {}
    for pair in PAIRS:
        c = load_candles(pair)
        if c:
            candles_map[pair] = c
            oldest = datetime.utcfromtimestamp(c[0]["ts"]).strftime("%Y-%m-%d")
            print(f"  {pair}: {len(c)} candles desde {oldest}")
        else:
            print(f"  {pair}: SEM DADOS — rode com --fetch primeiro")

    if not candles_map:
        print("Sem dados. Rode: python3 validate.py --fetch")
        sys.exit(1)

    # ── Walk-Forward ────────────────────────────────────────────────────
    if args.walkforward or run_all:
        if not HAS_ML:
            print("[AVISO] sklearn não disponível — instale: pip install scikit-learn")

        windows_result = run_walkforward(candles_map)
        if isinstance(windows_result, tuple):
            windows, oos_start_ts, _ = windows_result
        else:
            windows, oos_start_ts = windows_result, None

        # OOS
        oos       = run_oos(candles_map, oos_start_ts, windows) if oos_start_ts else {}
        benchmark = calc_benchmark(candles_map, oos_start_ts) if oos_start_ts else {}

        # Relatório
        print_report(windows, oos, benchmark)

    elif args.report:
        result_path = os.path.join(DATA_DIR, "validation_result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            print_report(r["windows"], r["oos"], r["benchmark"])
        else:
            print("Sem resultado salvo. Rode sem --report primeiro.")


if __name__ == "__main__":
    main()
