"""
Signal Score Calibration — CCTBv4
==================================
Responde: score=0.65 realmente significa 65% de chance de win?

Metodologia:
  1. Carrega candles históricos OKX 1H (8 anos, BTC/ETH/SOL)
  2. Amostra a cada 4 barras (evita correlação — horizon do label é 4h)
  3. Para cada ponto, computa compute_signal_score() com market_context derivado dos candles
  4. Simula o outcome real: SL/TP nos próximos 4 candles (mesmo label_outcome do validate.py)
  5. Agrupa por score bucket e computa: win_rate, avg_win, avg_loss, EV, profit_factor
  6. Brier Score: mede calibração (0 = perfeito, 0.25 = aleatório)
  7. Platt Scaling: regressão logística score → P(win) real
     Produz coeficientes (a, b): P_calibrada = sigmoid(a × score + b)

Limitação importante (documentada):
  Orderflow real (taker_ratio, OI, funding) não existe em OHLCV histórico.
  O modelo usa proxy por volume delta (candle bullish/bearish).
  Isso torna a calibração CONSERVADORA — live performance pode ser melhor.

Saída:
  Tabela formatada por: bucket global, par, regime, condição de mercado
  data/calibration_result.json — métricas completas
  data/calibration_coef.json  — coeficientes Platt Scaling para o signal engine

Uso:
  python3 calibrate.py [--pairs BTC-USDT ETH-USDT] [--sample-every 4] [--min-score 0.48]
  python3 calibrate.py --report-only   # lê resultado salvo e re-imprime
"""

import os, sys, json, math, time, argparse, statistics
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.signal_engine import compute_signal_score

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")
OUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration_result.json")
COEF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration_coef.json")

from strategies.fee_model import FEE as _FEE
FEE_RATE   = _FEE.maker              # 0.001 — entrada limit
SLIPPAGE   = _FEE.slippage_base      # 0.0002 — slippage conservador
TOTAL_COST = _FEE.backtest_round_trip()   # round-trip: maker+taker+2×slippage ≈ 0.0054

SCORE_BUCKETS = [
    (0.48, 0.52),
    (0.52, 0.55),
    (0.55, 0.58),
    (0.58, 0.62),
    (0.62, 0.66),
    (0.66, 0.70),
    (0.70, 1.01),
]

BUCKET_LABELS = [
    "0.48–0.52",
    "0.52–0.55",
    "0.55–0.58",
    "0.58–0.62",
    "0.62–0.66",
    "0.66–0.70",
    "0.70+   ",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(values, span):
    if not values: return []
    k = 2 / (span + 1)
    r = [values[0]]
    for v in values[1:]: r.append(v * k + r[-1] * (1 - k))
    return r

def _atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return closes[-1] * 0.02 if closes else 0
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    return sum(trs[-period:]) / period

def _bb_width_pct(closes, period=20):
    if len(closes) < period: return 0.5
    recent = closes[-period:]
    mid    = sum(recent) / period
    std    = statistics.stdev(recent)
    width  = 2 * std / mid if mid > 0 else 0
    # Percentil dentro dos últimos 100 períodos
    widths = []
    for j in range(period, min(len(closes), period + 100)):
        s = closes[j-period:j]
        m = sum(s)/period; sd = statistics.stdev(s)
        widths.append(2*sd/m if m > 0 else 0)
    if not widths: return 0.5
    below = sum(1 for w in widths if w <= width)
    return below / len(widths)


def build_market_context(candles: list, i: int) -> dict:
    """
    Reconstrói market_context a partir de candles OHLCV.
    Campos reais (OI, funding, orderbook) = 0 — efeito conservador.
    """
    window  = candles[max(0, i-200):i+1]
    closes  = [c["close"]  for c in window]
    highs   = [c["high"]   for c in window]
    lows    = [c["low"]    for c in window]
    volumes = [c["volume"] for c in window]

    if len(closes) < 25:
        return {}

    # Vol percentile (BB width)
    bb_pct = _bb_width_pct(closes)

    # ATR expansion
    atr_now  = _atr(highs, lows, closes)
    atr_prev = _atr(highs[:-14], lows[:-14], closes[:-14])
    atr_exp  = atr_now / atr_prev if atr_prev > 0 else 1.0

    # Volume delta proxy (candle direction como proxy de taker)
    recent_c = window[-10:]
    bv = sum(c["volume"] for c in recent_c if c["close"] >= c["open"])
    sv = sum(c["volume"] for c in recent_c if c["close"] <  c["open"])
    tv = bv + sv
    taker_imbal = (bv - sv) / tv if tv > 0 else 0.0

    # Volume delta % recente
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
    vol_delta_pct = (volumes[-1] - avg_vol) / avg_vol if avg_vol > 0 else 0.0

    # Realized vol (std de retornos log 24h)
    rets = []
    for j in range(1, min(25, len(closes))):
        if closes[j-1] > 0:
            rets.append(math.log(closes[j] / closes[j-1]))
    realized_vol = statistics.stdev(rets) * math.sqrt(24) if len(rets) > 2 else 0.02

    return {
        "vol_percentile": bb_pct,
        "atr_rate": {
            "expansion": round(atr_exp, 4),
            "atr_pct":   round(atr_now / closes[-1], 5) if closes[-1] > 0 else 0.02,
        },
        "taker_ratio": {
            "imbalance": round(taker_imbal, 4),
        },
        "volume_delta": {
            "delta_pct": round(vol_delta_pct, 4),
            "aggressive": vol_delta_pct > 0.5,
        },
        "realized_vol": round(realized_vol, 5),
        # Dados não disponíveis historicamente — neutros
        "orderbook":      {"imbalance": 0.0, "spread_pct": 0.0002},
        "open_interest":  {"expanding": False, "oi_change_pct": 0.0},
        "funding":        {"funding_rate": 0.0},
    }


def regime_from_candles(candles: list, i: int) -> str:
    """Determina regime V4 a partir de features computáveis de candles."""
    window  = candles[max(0, i-100):i+1]
    closes  = [c["close"] for c in window]
    highs   = [c["high"]  for c in window]
    lows    = [c["low"]   for c in window]

    if len(closes) < 30:
        return "MEAN_REVERTING_CHOP"

    e9  = _ema(closes, 9)
    e21 = _ema(closes, 21)
    e50 = _ema(closes, 50) if len(closes) >= 50 else _ema(closes, len(closes)//2)

    adx_val = _atr(highs, lows, closes, 14) / closes[-1] * 100 if closes[-1] > 0 else 0
    bb_pct  = _bb_width_pct(closes)

    # Volatility
    rets = [(closes[j] - closes[j-1]) / closes[j-1] for j in range(1, min(25, len(closes)))]
    vol_24h = statistics.stdev(rets) if len(rets) > 2 else 0.01

    if vol_24h > 0.04:
        return "PANIC_LIQUIDATION"
    if bb_pct < 0.15:
        return "VOLATILITY_COMPRESSION"
    if len(e50) > 0 and e9[-1] > e21[-1] > e50[-1] and adx_val > 1.5:
        return "TREND_EXPANSION"
    if len(e50) > 0 and e9[-1] < e21[-1] < e50[-1] and adx_val > 1.5:
        return "TREND_EXHAUSTION"
    return "MEAN_REVERTING_CHOP"


def market_condition(candles: list, i: int) -> str:
    """bull / bear / chop baseado no retorno 24h."""
    if i < 25: return "chop"
    ret_24h = (candles[i]["close"] - candles[i-24]["close"]) / candles[i-24]["close"]
    if ret_24h >  0.03: return "bull"
    if ret_24h < -0.03: return "bear"
    return "chop"


def simulate_trade_pnl(
    candles: list, i: int,
    horizon: int = 4,
    rr: float = 2.0,
    atr_override: float = 0.0,
) -> tuple:
    """
    Simula trade entrando no candle i+1.
    Retorna (won: int, pnl_pct: float, exit_reason: str).
    won = 1 (TP), 0 (SL), -1 (sem dados).
    pnl_pct inclui fees e slippage.
    """
    if i + horizon >= len(candles):
        return -1, 0.0, "no_data"

    entry = candles[i+1]["open"]
    if entry <= 0:
        return -1, 0.0, "bad_entry"

    # ATR do contexto ou calcula
    if atr_override > 0:
        atr = atr_override
    else:
        window = candles[max(0, i-30):i+1]
        highs  = [c["high"]  for c in window]
        lows   = [c["low"]   for c in window]
        closes = [c["close"] for c in window]
        atr = _atr(highs, lows, closes)

    sl_pct = min(0.06, max(0.01, atr * 1.5 / entry))
    sl     = entry * (1 - sl_pct)
    tp     = entry * (1 + sl_pct * rr)

    for j in range(i+1, min(i+1+horizon, len(candles))):
        high = candles[j]["high"]
        low  = candles[j]["low"]
        if high >= tp:
            pnl = sl_pct * rr - TOTAL_COST
            return 1, round(pnl, 6), "tp"
        if low <= sl:
            pnl = -sl_pct - TOTAL_COST
            return 0, round(pnl, 6), "sl"

    # EOH — sai no close
    exit_price = candles[min(i+horizon, len(candles)-1)]["close"]
    pnl = (exit_price - entry) / entry - TOTAL_COST
    won = 1 if pnl > 0 else 0
    return won, round(pnl, 6), "eoh"


# ── Coleta de amostras ────────────────────────────────────────────────────────

def collect_samples(
    candles: list,
    pair: str,
    closes_map: dict,
    sample_every: int = 4,
    min_score: float = 0.48,
) -> list:
    """
    Amostra o sinal a cada `sample_every` barras.
    Retorna lista de dicts com score, outcome e metadados.
    """
    samples  = []
    n        = len(candles)
    pair_okx = pair.replace("-USDT", "-USD")   # normaliza para signal engine

    print(f"  {pair}: {n} candles, amostrando a cada {sample_every} barras...")
    t0 = time.time()
    last_pct = -1

    for i in range(100, n - 5, sample_every):
        # Progress
        pct = int((i - 100) / (n - 105) * 100)
        if pct // 10 > last_pct // 10:
            last_pct = pct
            elapsed = time.time() - t0
            eta = elapsed / (pct / 100) * (1 - pct / 100) if pct > 0 else 0
            print(f"    {pct:3d}% ({i}/{n}) eta {eta:.0f}s", end="\r")

        # market_context derivado de candles
        mc      = build_market_context(candles, i)
        regime  = regime_from_candles(candles, i)
        cond    = market_condition(candles, i)

        # closes_map para relative_strength
        # Usa apenas closes até i para evitar lookahead
        cm = {k: [c["close"] for c in v[:i+1]] for k, v in closes_map.items()}

        try:
            sig = compute_signal_score(
                pair=pair_okx,
                candles_1h=candles[max(0,i-250):i+1],
                market_context=mc,
                regime=regime,
                closes_map=cm,
                fee_rate=FEE_RATE + SLIPPAGE,
                expected_rr=2.0,
            )
        except Exception:
            continue

        score = sig.get("score", 0.0)
        if score < min_score:
            continue

        # Só toma amostras de direção "long" (bot só compra)
        if sig.get("direction") not in ("long", "neutral"):
            continue

        won, pnl_pct, exit_reason = simulate_trade_pnl(candles, i)
        if won == -1:
            continue

        samples.append({
            "pair":        pair_okx,
            "ts":          candles[i].get("ts", candles[i].get("start", 0)),
            "score":       round(score, 4),
            "direction":   sig.get("direction"),
            "regime":      regime,
            "condition":   cond,
            "won":         won,
            "pnl_pct":     pnl_pct,
            "exit_reason": exit_reason,
            "ev_predicted": round(sig.get("expected_value", 0), 4),
            "factors": {k: round(v.get("score", 0.5), 3) if isinstance(v, dict) else v
                        for k, v in sig.get("factors", {}).items()},
        })

    print(f"\n  {pair}: {len(samples)} amostras coletadas em {time.time()-t0:.0f}s")
    return samples


# ── Análise e calibração ──────────────────────────────────────────────────────

def bucket_of(score: float) -> int:
    for idx, (lo, hi) in enumerate(SCORE_BUCKETS):
        if lo <= score < hi:
            return idx
    return len(SCORE_BUCKETS) - 1


def analyze_bucket(rows: list) -> dict:
    """Computa métricas para uma lista de amostras."""
    if not rows:
        return {"n": 0}

    n       = len(rows)
    wins    = [r for r in rows if r["won"] == 1]
    losses  = [r for r in rows if r["won"] == 0]
    wr      = len(wins) / n

    win_pnls  = [r["pnl_pct"] for r in wins]
    loss_pnls = [r["pnl_pct"] for r in losses]

    avg_win  = sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    ev_net   = wr * avg_win + (1 - wr) * avg_loss

    gross_wins   = sum(win_pnls)
    gross_losses = abs(sum(loss_pnls)) if loss_pnls else 1e-9
    pf           = gross_wins / gross_losses if gross_losses > 0 else 0.0

    # Brier Score: MSE entre score previsto e outcome real
    brier = sum((r["score"] - r["won"]) ** 2 for r in rows) / n

    return {
        "n":           n,
        "win_rate":    round(wr, 4),
        "avg_win_pct": round(avg_win * 100, 3),
        "avg_loss_pct":round(avg_loss * 100, 3),
        "ev_net_pct":  round(ev_net * 100, 4),
        "profit_factor": round(pf, 3),
        "brier_score": round(brier, 4),
        "n_tp":        sum(1 for r in rows if r["exit_reason"] == "tp"),
        "n_sl":        sum(1 for r in rows if r["exit_reason"] == "sl"),
        "n_eoh":       sum(1 for r in rows if r["exit_reason"] == "eoh"),
    }


def platt_scaling(samples: list) -> dict:
    """
    Regressão logística score → P(win) real.
    Produz (a, b) tal que P_calibrada = sigmoid(a × score + b).
    Minimiza Brier Score da distribuição.

    Se sklearn não disponível, usa estimativa por Newton-Raphson manual.
    """
    if not samples:
        return {"a": 1.0, "b": 0.0, "method": "identity", "brier_before": 0, "brier_after": 0}

    scores   = [r["score"] for r in samples]
    outcomes = [r["won"]   for r in samples]

    brier_before = sum((s - o) ** 2 for s, o in zip(scores, outcomes)) / len(scores)

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        X = np.array(scores).reshape(-1, 1)
        y = np.array(outcomes)
        lr = LogisticRegression(C=1.0)
        lr.fit(X, y)
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])

        # Brier depois
        def sigmoid(x): return 1 / (1 + math.exp(-x))
        p_cal = [sigmoid(a * s + b) for s in scores]
        brier_after = sum((p - o) ** 2 for p, o in zip(p_cal, outcomes)) / len(scores)

        return {
            "a": round(a, 4),
            "b": round(b, 4),
            "method": "platt_logistic",
            "brier_before": round(brier_before, 5),
            "brier_after":  round(brier_after, 5),
            "improvement_pct": round((brier_before - brier_after) / brier_before * 100, 1) if brier_before > 0 else 0,
        }
    except ImportError:
        pass

    # Fallback: estimativa por isotonic regression manual (bucket médias)
    bucket_means = {}
    for r in samples:
        b_idx = bucket_of(r["score"])
        if b_idx not in bucket_means:
            bucket_means[b_idx] = []
        bucket_means[b_idx].append(r["won"])
    calib = {b: sum(v)/len(v) for b, v in bucket_means.items()}

    return {
        "a": 1.0, "b": 0.0,
        "method": "bucket_isotonic",
        "bucket_calibration": calib,
        "brier_before": round(brier_before, 5),
        "note": "sklearn não disponível — instale para Platt Scaling completo",
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 1.0, width: int = 20, fill: str = "█") -> str:
    n = int((value / max_val) * width) if max_val > 0 else 0
    return fill * n + "░" * (width - n)


def print_calibration_report(result: dict):
    samples  = result["samples"]
    coef     = result["platt"]
    by_bkt   = result["by_bucket"]
    by_pair  = result["by_pair"]
    by_regime= result["by_regime"]
    by_cond  = result["by_condition"]

    sep = "─" * 100

    print(f"\n{'='*100}")
    print(f"  SIGNAL SCORE CALIBRATION REPORT — CCTBv4")
    print(f"  {len(samples)} amostras | {result['n_pairs']} pares | "
          f"{result.get('date_range','')} | gerado {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*100}\n")

    # ── Tabela principal ──────────────────────────────────────────────────────
    print("TABELA GLOBAL (todos os pares, todos os regimes)\n")
    print(f"{'Score Bucket':<13} {'Trades':>7} {'Win Rate':>9} {'WR bar':>22} "
          f"{'Avg Win':>8} {'Avg Loss':>9} {'EV net%':>8} {'Prof.Factor':>12} {'Brier':>7}")
    print(sep)

    for idx, label in enumerate(BUCKET_LABELS):
        bkt = by_bkt.get(str(idx))
        if not bkt or bkt["n"] == 0:
            print(f"{label:<13} {'0':>7}  {'—':>9}  {'':>22}  {'—':>8}  {'—':>9}  {'—':>8}  {'—':>12}  {'—':>7}")
            continue
        wr  = bkt["win_rate"]
        bar = _bar(wr, 1.0, 20)
        ev  = bkt["ev_net_pct"]
        ev_s = f"{ev:+.3f}%" if ev else "  —"
        pf  = bkt["profit_factor"]
        pf_s = f"{pf:.2f}" if pf else "—"
        print(f"{label:<13} {bkt['n']:>7}  {wr:>8.1%}  {bar}  "
              f"{bkt['avg_win_pct']:>+7.2f}%  {bkt['avg_loss_pct']:>+8.2f}%  "
              f"{ev_s:>8}  {pf_s:>11}  {bkt['brier_score']:>7.4f}")

    total_n  = sum(by_bkt[str(i)]["n"] for i in range(len(SCORE_BUCKETS)) if str(i) in by_bkt)
    total_wr = (sum(by_bkt[str(i)]["n"] * by_bkt[str(i)]["win_rate"]
                    for i in range(len(SCORE_BUCKETS))
                    if str(i) in by_bkt and by_bkt[str(i)]["n"] > 0) / total_n) if total_n > 0 else 0
    print(sep)
    print(f"{'TOTAL':<13} {total_n:>7}  {total_wr:>8.1%}\n")

    # ── Platt Scaling ─────────────────────────────────────────────────────────
    print(f"\nPLATT SCALING (Logistic calibration: P_real = sigmoid(a × score + b))")
    print(sep)
    if coef.get("method") == "platt_logistic":
        a, b = coef["a"], coef["b"]
        def sigmoid(x): return 1 / (1 + math.exp(-x))
        print(f"  a = {a:+.4f}   b = {b:+.4f}")
        print(f"  Brier antes:  {coef['brier_before']:.5f}")
        print(f"  Brier depois: {coef['brier_after']:.5f}  (melhoria: {coef['improvement_pct']:.1f}%)")
        print()
        print(f"  {'Score raw':<12} {'P calibrada':<14} {'Delta':<10}")
        for s_raw in [0.50, 0.52, 0.55, 0.58, 0.62, 0.65, 0.68, 0.72]:
            p_cal = sigmoid(a * s_raw + b)
            delta = p_cal - s_raw
            flag  = "← over-confident" if delta < -0.03 else ("← under-confident" if delta > 0.03 else "✓")
            print(f"  {s_raw:.2f}         →   {p_cal:.3f}         {delta:+.3f}  {flag}")
    else:
        print(f"  Método: {coef.get('method')}")
        print(f"  {coef.get('note', '')}")
    print()

    # ── Por par ───────────────────────────────────────────────────────────────
    print(f"\nPOR PAR\n")
    print(f"{'Par':<12} {'Trades':>7} {'Win Rate':>9} {'EV net%':>9} {'P.Factor':>10} {'Brier':>8}")
    print(sep)
    for pair_k, bkt in sorted(by_pair.items()):
        if bkt["n"] == 0: continue
        ev_s = f"{bkt['ev_net_pct']:+.3f}%" if bkt['ev_net_pct'] else "—"
        print(f"{pair_k:<12} {bkt['n']:>7}  {bkt['win_rate']:>8.1%}  {ev_s:>8}  "
              f"{bkt['profit_factor']:>9.2f}  {bkt['brier_score']:>7.4f}")
    print()

    # ── Por regime ────────────────────────────────────────────────────────────
    print(f"\nPOR REGIME\n")
    print(f"{'Regime':<28} {'Trades':>7} {'Win Rate':>9} {'EV net%':>9} {'P.Factor':>10}")
    print(sep)
    for regime_k, bkt in sorted(by_regime.items(), key=lambda x: -x[1]["n"]):
        if bkt["n"] < 5: continue
        ev_s = f"{bkt['ev_net_pct']:+.3f}%" if bkt['ev_net_pct'] else "—"
        print(f"{regime_k:<28} {bkt['n']:>7}  {bkt['win_rate']:>8.1%}  {ev_s:>8}  "
              f"{bkt['profit_factor']:>9.2f}")
    print()

    # ── Por condição de mercado ────────────────────────────────────────────────
    print(f"\nPOR CONDIÇÃO DE MERCADO (ret24h > +3% = bull, < -3% = bear)\n")
    print(f"{'Condição':<12} {'Trades':>7} {'Win Rate':>9} {'EV net%':>9} {'P.Factor':>10}")
    print(sep)
    for cond_k, bkt in [("bull", by_cond.get("bull",{})),
                         ("chop", by_cond.get("chop",{})),
                         ("bear", by_cond.get("bear",{}))]:
        if not bkt or bkt.get("n",0) == 0: continue
        ev_s = f"{bkt['ev_net_pct']:+.3f}%" if bkt['ev_net_pct'] else "—"
        print(f"{cond_k:<12} {bkt['n']:>7}  {bkt['win_rate']:>8.1%}  {ev_s:>8}  "
              f"{bkt['profit_factor']:>9.2f}")
    print()

    # ── Diagnóstico de edge ───────────────────────────────────────────────────
    print(f"\nDIAGNÓSTICO DE EDGE\n")
    print(sep)
    wrs = [(BUCKET_LABELS[i], by_bkt[str(i)]["win_rate"], by_bkt[str(i)]["n"])
           for i in range(len(SCORE_BUCKETS))
           if str(i) in by_bkt and by_bkt[str(i)]["n"] >= 10]

    if len(wrs) >= 3:
        monotonic = all(wrs[j][1] <= wrs[j+1][1] for j in range(len(wrs)-1))
        print(f"  Win rate monotonicamente crescente com score? {'✅ SIM' if monotonic else '❌ NÃO'}")
        if not monotonic:
            print(f"  → Score é ranking, não probabilidade. Platt Scaling essencial.")
        else:
            print(f"  → Score tem poder preditivo monotônico. Calibração leve.")

    best  = max(wrs, key=lambda x: x[1]) if wrs else None
    worst = min(wrs, key=lambda x: x[1]) if wrs else None
    if best:  print(f"  Melhor bucket:  {best[0]}  WR={best[1]:.1%} (n={best[2]})")
    if worst: print(f"  Pior bucket:    {worst[0]}  WR={worst[1]:.1%} (n={worst[2]})")

    # EV positivo mínimo
    positive_ev = [(BUCKET_LABELS[i], by_bkt[str(i)]["ev_net_pct"])
                   for i in range(len(SCORE_BUCKETS))
                   if str(i) in by_bkt and by_bkt[str(i)].get("ev_net_pct", -99) > 0
                   and by_bkt[str(i)]["n"] >= 10]
    if positive_ev:
        min_score_bucket = BUCKET_LABELS.index(positive_ev[0][0])
        lo, _ = SCORE_BUCKETS[min_score_bucket]
        print(f"  EV positivo (net fees): a partir do bucket '{positive_ev[0][0]}' → score mínimo recomendado: {lo:.2f}")
    else:
        print(f"  ⚠ Nenhum bucket com EV líquido positivo — score atual não tem edge após fees")

    print(f"\n{'='*100}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_candles(pair: str) -> list:
    """Carrega candles do cache de dados históricos."""
    path = os.path.join(DATA_DIR, f"{pair}_1H.json")
    if not os.path.exists(path):
        print(f"  [AVISO] {path} não encontrado. Rode validate.py --fetch primeiro.")
        return []
    with open(path) as f:
        return json.load(f)


def run(pairs=None, sample_every=4, min_score=0.48):
    pairs = pairs or ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

    print(f"\n{'='*60}")
    print(f"  Signal Score Calibration — CCTBv4")
    print(f"  Pares: {pairs}  |  sample_every={sample_every}  |  min_score={min_score}")
    print(f"{'='*60}\n")

    # Carrega candles de todos os pares (para closes_map no relative_strength)
    all_candles = {}
    for pair in pairs:
        c = load_candles(pair)
        if c:
            all_candles[pair] = c
            print(f"  {pair}: {len(c)} candles carregados")

    if not all_candles:
        print("Nenhum candle disponível. Rode: python3 validate.py --fetch")
        return

    # Coleta amostras por par
    all_samples = []
    for pair, candles in all_candles.items():
        # closes_map com candles do mesmo tamanho
        cm = {k: v for k, v in all_candles.items()}
        samples = collect_samples(candles, pair, cm, sample_every, min_score)
        all_samples.extend(samples)

    print(f"\nTotal: {len(all_samples)} amostras coletadas\n")

    if not all_samples:
        print("Sem amostras. Verifique min_score e dados disponíveis.")
        return

    # ── Agrupa por bucket ─────────────────────────────────────────────────────
    by_bucket   = defaultdict(list)
    by_pair     = defaultdict(list)
    by_regime   = defaultdict(list)
    by_condition= defaultdict(list)

    for s in all_samples:
        by_bucket[bucket_of(s["score"])].append(s)
        by_pair[s["pair"]].append(s)
        by_regime[s["regime"]].append(s)
        by_condition[s["condition"]].append(s)

    # ── Analisa cada grupo ────────────────────────────────────────────────────
    result = {
        "n_samples": len(all_samples),
        "n_pairs":   len(all_candles),
        "date_range": f"{datetime.now().year-8}–{datetime.now().year}",
        "by_bucket":    {str(k): analyze_bucket(v) for k, v in by_bucket.items()},
        "by_pair":      {k: analyze_bucket(v) for k, v in by_pair.items()},
        "by_regime":    {k: analyze_bucket(v) for k, v in by_regime.items()},
        "by_condition": {k: analyze_bucket(v) for k, v in by_condition.items()},
        "platt":        platt_scaling(all_samples),
        "samples":      all_samples[:5000],   # salva até 5k amostras para auditoria
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Salva coeficientes de calibração ─────────────────────────────────────
    coef_data = {
        "platt_a":   result["platt"]["a"],
        "platt_b":   result["platt"]["b"],
        "method":    result["platt"]["method"],
        "brier_raw": result["platt"].get("brier_before", 0),
        "brier_cal": result["platt"].get("brier_after", 0),
        "n_samples": len(all_samples),
        "generated_at": result["generated_at"],
        # Calibration lookup: score bucket → real win rate observada
        "bucket_win_rates": {
            BUCKET_LABELS[int(k)].strip(): v["win_rate"]
            for k, v in result["by_bucket"].items()
            if v.get("n", 0) >= 10
        },
    }
    os.makedirs(os.path.dirname(COEF_PATH), exist_ok=True)
    with open(COEF_PATH, "w") as f:
        json.dump(coef_data, f, indent=2)
    print(f"Coeficientes salvos em {COEF_PATH}")

    # ── Salva resultado completo ──────────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "samples"}, f, indent=2)
    print(f"Resultado salvo em {OUT_PATH}")

    print_calibration_report(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Score Calibration — CCTBv4")
    parser.add_argument("--pairs",        nargs="+", default=["BTC-USDT","ETH-USDT","SOL-USDT"])
    parser.add_argument("--sample-every", type=int,   default=4,
                        help="Amostra a cada N barras (default 4 = evita correlação no horizon de 4h)")
    parser.add_argument("--min-score",    type=float, default=0.48,
                        help="Score mínimo para incluir na calibração (default 0.48)")
    parser.add_argument("--report-only",  action="store_true",
                        help="Lê resultado salvo e re-imprime sem recalcular")
    args = parser.parse_args()

    if args.report_only:
        if not os.path.exists(OUT_PATH):
            print(f"Resultado não encontrado: {OUT_PATH}")
            print("Rode primeiro sem --report-only")
            sys.exit(1)
        with open(OUT_PATH) as f:
            result = json.load(f)
        result["samples"] = []
        print_calibration_report(result)
    else:
        run(
            pairs       = args.pairs,
            sample_every= args.sample_every,
            min_score   = args.min_score,
        )
