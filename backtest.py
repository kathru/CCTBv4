"""
Backtester V4 — Simulação Histórica
=====================================
Roda a pipeline V4 sobre dados históricos reais da OKX.

Componentes usados (baseados em OHLCV):
  - Regime Engine (ADX, BB width, ATR expansion)
  - Signal Engine (Volatility Expansion, Market Structure, Relative Strength)
  - Sizing Engine (Edge × Confidence / Vol × Correlation)
  - SL/TP dinâmico (ATR × regime)

Componentes simplificados (sem dados tick históricos):
  - Orderflow Proxy → estimado via volume delta de candles
  - Funding Rate    → neutro (0) — dado real-time
  - OI Expansion    → estimado via variação de volume

Saída: win rate, profit factor, P&L, drawdown, Sharpe, trade list
"""

import sys, os, time, math, statistics, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Parâmetros ──────────────────────────────────────────────────────────────
PAIRS          = ["BTC-USD", "ETH-USD", "SOL-USD"]
INITIAL_USD    = 1020.0       # R$ 5.000 / ~4.90
TRADE_PCT_BASE = 0.08         # 8% base por trade
from strategies.fee_model import FEE as _FEE
TAKER_FEE      = _FEE.taker    # 0.004 = 0.40% OKX Regular (era 0.0004 — bug 10×)
MAKER_FEE      = _FEE.maker    # 0.001 = 0.10% OKX Regular
DAYS           = 30
MIN_SCORE      = 0.55
COOLDOWN_BARS  = 4            # 4 ciclos (4h) entre BUYs no mesmo par
MAX_OPEN       = 3            # máx posições simultâneas

import requests
_session = requests.Session()

# ── Fetch histórico ──────────────────────────────────────────────────────────

def fetch_history(inst_id: str, bar: str = "1H", days: int = DAYS) -> list:
    """Busca candles históricos paginando a API da OKX."""
    all_c, after = [], None
    bars_per_day = {"1H": 24, "15m": 96, "6H": 4}[bar]
    target = days * bars_per_day
    for _ in range(30):
        p = {"instId": inst_id, "bar": bar, "limit": "300"}
        if after:
            p["after"] = after
        try:
            r = _session.get("https://www.okx.com/api/v5/market/history-candles",
                             params=p, timeout=12)
            data = r.json().get("data", [])
        except Exception:
            break
        if not data:
            break
        all_c.extend(data)
        after = data[-1][0]
        if len(all_c) >= target:
            break
        time.sleep(0.25)

    # Ordena ASC (mais antigo primeiro)
    all_c.sort(key=lambda x: int(x[0]))
    return [{
        "ts":     int(c[0]) // 1000,
        "open":   float(c[1]), "high": float(c[2]),
        "low":    float(c[3]), "close": float(c[4]),
        "volume": float(c[5]),
    } for c in all_c]


# ── Indicadores ──────────────────────────────────────────────────────────────

def ema(vals, span):
    k, r = 2/(span+1), [vals[0]]
    for v in vals[1:]:
        r.append(v*k + r[-1]*(1-k))
    return r

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period+1: return 0.0
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    return sum(trs[-period:]) / period

def calc_adx(highs, lows, closes, period=14):
    if len(closes) < period*2: return 20.0
    dm_p, dm_m, trs = [], [], []
    for i in range(1, len(closes)):
        h_d = highs[i]-highs[i-1]; l_d = lows[i-1]-lows[i]
        dm_p.append(max(h_d,0) if h_d > l_d else 0)
        dm_m.append(max(l_d,0) if l_d > h_d else 0)
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    def smooth(a, p):
        s = sum(a[:p]); r = [s]
        for v in a[p:]: r.append(r[-1] - r[-1]/p + v)
        return r
    atr_s = smooth(trs, period); dmp_s = smooth(dm_p, period); dmm_s = smooth(dm_m, period)
    dx_vals = []
    for i in range(len(atr_s)):
        if atr_s[i] == 0: continue
        dip = dmp_s[i]/atr_s[i]*100; dim = dmm_s[i]/atr_s[i]*100
        dx_vals.append(abs(dip-dim)/(dip+dim)*100 if (dip+dim)>0 else 0)
    if len(dx_vals) < period: return 20.0
    return sum(dx_vals[-period:]) / period

def bb_width_pct(closes, period=20, hist=80):
    if len(closes) < period+hist: return 0.5
    widths = []
    for i in range(hist, 0, -1):
        sub = closes[-(period+hist)+hist-i:-(hist)+hist-i+period] if i > 0 else closes[-period:]
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


# ── Signal Engine simplificado ───────────────────────────────────────────────

def compute_signal(candles: list, candles_other: dict, regime: str) -> dict:
    """Score probabilístico usando apenas OHLCV (sem dados tick)."""
    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]

    if len(closes) < 50:
        return {"score": 0.5, "ev": -0.01, "direction": "neutral"}

    # ── A. Volatility Expansion ──────────────────────────────────────────
    bb_pct   = bb_width_pct(closes)
    atr_cur  = calc_atr(highs, lows, closes, 14)
    atr_prev = calc_atr(highs[:-14], lows[:-14], closes[:-14], 14)
    atr_exp  = atr_cur / atr_prev if atr_prev > 0 else 1.0
    compress = max(0, 1 - bb_pct)
    exp_start = max(0, min(1, (atr_exp - 0.9) / 0.5)) if atr_exp > 0.9 else 0

    ema9  = ema(closes, 9);  ema21 = ema(closes, 21);  ema50 = ema(closes, 50)
    if ema9[-1] > ema21[-1] > ema50[-1]:
        dir_vol, dir_score = "long", 0.70
    elif ema9[-1] < ema21[-1] < ema50[-1]:
        dir_vol, dir_score = "short", 0.70
    elif ema9[-1] > ema21[-1]:
        dir_vol, dir_score = "long", 0.55
    else:
        dir_vol, dir_score = "short", 0.55

    vol_exp = (compress*0.35 + exp_start*0.30 + dir_score*0.35) * {
        "VOLATILITY_COMPRESSION": 1.30, "TREND_EXPANSION": 1.10,
        "MEAN_REVERTING_CHOP": 0.70, "PANIC_LIQUIDATION": 0.20
    }.get(regime, 1.0)

    # ── B. Market Structure ──────────────────────────────────────────────
    n = min(len(closes), 30)
    hh_score = ll_score = 0.0
    pivots_h = [max(highs[i-3:i+3]) for i in range(3, n-3, 3)]
    pivots_l = [min(lows[i-3:i+3])  for i in range(3, n-3, 3)]
    if len(pivots_h) >= 2:
        hh = sum(1 for i in range(1,len(pivots_h)) if pivots_h[i]>pivots_h[i-1])/(len(pivots_h)-1)
        hl = sum(1 for i in range(1,len(pivots_l))  if pivots_l[i]>pivots_l[i-1])/(len(pivots_l)-1)
        lh = sum(1 for i in range(1,len(pivots_h)) if pivots_h[i]<pivots_h[i-1])/(len(pivots_h)-1)
        ll = sum(1 for i in range(1,len(pivots_l))  if pivots_l[i]<pivots_l[i-1])/(len(pivots_l)-1)
        hh_score = (hh + hl) / 2; ll_score = (lh + ll) / 2

    avg_range = sum(highs[i]-lows[i] for i in range(-10,0)) / 10
    last_range = highs[-1] - lows[-1]
    disp = last_range / avg_range if avg_range > 0 else 1.0
    disp_bull = disp > 1.8 and closes[-1] > (closes[-1]+lows[-1])/2
    disp_bear = disp > 1.8 and closes[-1] < (closes[-1]+highs[-1])/2
    ob_imbal = sum(volumes[-5:])/sum(volumes[-10:-5]) - 1 if sum(volumes[-10:-5]) > 0 else 0

    mkt_long = min(hh_score*0.35 + (0.30 if disp_bull else 0) + max(0, ob_imbal)*0.15, 1.0)
    mkt_short = min(ll_score*0.35 + (0.30 if disp_bear else 0) + max(0, -ob_imbal)*0.15, 1.0)
    if mkt_long > mkt_short and mkt_long > 0.25:
        dir_mkt = "long";  mkt_score = 0.50 + min(mkt_long*0.40, 0.40)
    elif mkt_short > mkt_long and mkt_short > 0.25:
        dir_mkt = "short"; mkt_score = 0.50 + min(mkt_short*0.40, 0.40)
    else:
        dir_mkt = "neutral"; mkt_score = 0.50

    # ── C. Orderflow Proxy (via volume delta) ────────────────────────────
    bv = sum(c["volume"] for c in candles[-10:] if c["close"] >= c["open"])
    sv = sum(c["volume"] for c in candles[-10:] if c["close"] <  c["open"])
    tv = bv + sv
    taker_imbal = (bv - sv) / tv if tv > 0 else 0.0
    flow_long  = min(max(0, taker_imbal)*1.5 + (0.20 if candles[-1]["volume"] > sum(volumes[-5:])/5 else 0), 1.0)
    flow_short = min(max(0, -taker_imbal)*1.5, 1.0)
    if flow_long > flow_short and flow_long > 0.20:
        dir_flow = "long";  flow_score = 0.50 + min(flow_long*0.35, 0.35)
    elif flow_short > flow_long and flow_short > 0.20:
        dir_flow = "short"; flow_score = 0.50 + min(flow_short*0.35, 0.35)
    else:
        dir_flow = "neutral"; flow_score = 0.50

    # ── D. Relative Strength ─────────────────────────────────────────────
    rs_composite = 0.0
    if candles_other:
        btc_cl = [c["close"] for c in candles_other.get("BTC-USD", [])]
        this_cl = closes
        if len(btc_cl) >= 5 and len(this_cl) >= 5:
            perf_this = (this_cl[-1]-this_cl[-4])/this_cl[-4]
            perf_btc  = (btc_cl[-1]-btc_cl[-4])/btc_cl[-4]
            rs_composite = perf_this - perf_btc
    if rs_composite > 0.003:
        dir_rs = "long";  rs_score = 0.50 + min(rs_composite*20, 0.25)
    elif rs_composite < -0.003:
        dir_rs = "short"; rs_score = 0.50 + min(abs(rs_composite)*20, 0.25)
    else:
        dir_rs = "neutral"; rs_score = 0.50

    # ── Agregação ────────────────────────────────────────────────────────
    votes = {"long": 0.0, "short": 0.0, "neutral": 0.0}
    for d, s in [(dir_vol, vol_exp), (dir_mkt, mkt_score-0.5),
                 (dir_flow, flow_score-0.5), (dir_rs, rs_score-0.5)]:
        votes[d] += s

    dominant = max(votes, key=votes.get)
    if votes[dominant] <= 0.02: dominant = "neutral"

    def align(s, d): return s if d == dominant else (1-s)

    w = {"vol":0.20, "mkt":0.28, "flow":0.28, "rs":0.24}
    raw = (align(vol_exp, dir_vol)*w["vol"]*0.65 +
           align(mkt_score, dir_mkt)*w["mkt"]*0.65 +
           align(flow_score, dir_flow)*w["flow"]*0.70 +
           align(rs_score, dir_rs)*w["rs"]*0.60)
    total_w = sum(w.values())
    score = max(0.0, min(1.0, raw / (total_w * 0.65)))

    if regime in ("PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"):
        score = min(score, 0.40); dominant = "neutral"
    elif regime == "HIGH_CORRELATION_RISK":
        score = min(score, 0.58)

    ev = score * 2.0 - (1-score) * 1.0 - _FEE.signal_ev_cost(rr=2.0) * 3
    return {"score": round(score,4), "ev": round(ev,4), "direction": dominant,
            "vol_exp": round(vol_exp,3), "atr_exp": round(atr_exp,3)}


# ── Regime simplificado ───────────────────────────────────────────────────────

def detect_regime_simple(candles: list) -> str:
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    if len(closes) < 30: return "MEAN_REVERTING_CHOP"
    adx = calc_adx(highs, lows, closes)
    bb  = bb_width_pct(closes)
    atr_cur  = calc_atr(highs, lows, closes, 14)
    atr_prev = calc_atr(highs[:-14], lows[:-14], closes[:-14], 14)
    atr_exp  = atr_cur / atr_prev if atr_prev > 0 else 1.0
    vol_pct  = min(1.0, atr_cur / closes[-1] / 0.03)  # proxy
    if vol_pct > 0.90 and atr_exp > 2.0: return "PANIC_LIQUIDATION"
    if bb < 0.15 and adx < 18:           return "VOLATILITY_COMPRESSION"
    if adx > 30:                          return "TREND_EXPANSION"
    if adx > 25 and atr_exp < 0.90:      return "TREND_EXHAUSTION"
    if adx < 22:                          return "MEAN_REVERTING_CHOP"
    return "MEAN_REVERTING_CHOP"


# ── Backtester principal ──────────────────────────────────────────────────────

def run_backtest():
    print("\n" + "="*60)
    print("  BACKTEST V4 — SIMULAÇÃO HISTÓRICA")
    print("="*60)
    print(f"  Período:    {DAYS} dias | Ciclo: 1H")
    print(f"  Portfolio:  ${INITIAL_USD:.2f} USD")
    print(f"  Pares:      {', '.join(PAIRS)}")
    print("="*60)

    # ── Fetch dados ─────────────────────────────────────────────────────
    print("\n[1/3] Buscando dados históricos...")
    candles_map = {}
    for pair in PAIRS:
        inst = pair.replace("-USD", "-USDT")
        print(f"  {pair}...", end=" ", flush=True)
        c = fetch_history(inst, "1H", DAYS+5)
        candles_map[pair] = c
        print(f"{len(c)} velas ({(int(c[-1]['ts'])-int(c[0]['ts']))/86400:.1f}d)")

    n_bars = min(len(v) for v in candles_map.values())
    # Alinha todos ao mesmo tamanho
    for p in PAIRS:
        candles_map[p] = candles_map[p][-n_bars:]

    print(f"\n  Total: {n_bars} barras 1H por par")

    # ── Simulação ────────────────────────────────────────────────────────
    print("\n[2/3] Rodando simulação...")
    balance   = INITIAL_USD
    positions = {}   # {pair: {entry, sl, tp, qty, size_usd, bar_entry}}
    trades    = []
    equity    = [INITIAL_USD]
    cooldowns = {p: 0 for p in PAIRS}

    WARMUP = 100  # barras de aquecimento para indicadores

    for i in range(WARMUP, n_bars):
        # Snapshot de candles até a barra i (sem look-ahead)
        snap = {p: candles_map[p][:i+1] for p in PAIRS}

        for pair in PAIRS:
            c = snap[pair]
            price = c[-1]["close"]

            # ── Gerencia posição aberta ──────────────────────────────────
            if pair in positions:
                pos   = positions[pair]
                entry = pos["entry"]
                sl    = pos["sl"]
                tp    = pos["tp"]
                low   = c[-1]["low"]
                high  = c[-1]["high"]

                # TP hit (usa high da barra)
                if high >= tp:
                    exit_price = tp
                    pnl = (exit_price - entry) / entry * pos["size_usd"] - pos["size_usd"] * TAKER_FEE
                    balance += pos["size_usd"] + pnl
                    trades.append({"pair": pair, "entry": entry, "exit": exit_price,
                                   "pnl": pnl, "pct": (exit_price-entry)/entry*100,
                                   "reason": "TP", "bars": i - pos["bar_entry"]})
                    del positions[pair]
                    continue

                # SL hit (usa low da barra)
                if low <= sl:
                    exit_price = sl
                    pnl = (exit_price - entry) / entry * pos["size_usd"] - pos["size_usd"] * TAKER_FEE
                    balance += pos["size_usd"] + pnl
                    trades.append({"pair": pair, "entry": entry, "exit": exit_price,
                                   "pnl": pnl, "pct": (exit_price-entry)/entry*100,
                                   "reason": "SL", "bars": i - pos["bar_entry"]})
                    del positions[pair]
                    cooldowns[pair] = COOLDOWN_BARS
                    continue

            # ── Cooldown ─────────────────────────────────────────────────
            if cooldowns[pair] > 0:
                cooldowns[pair] -= 1
                continue

            # ── Verifica nova entrada ─────────────────────────────────────
            if pair in positions:
                continue
            if len(positions) >= MAX_OPEN:
                continue
            if balance < INITIAL_USD * 0.05:
                continue

            regime = detect_regime_simple(c)
            if regime in ("PANIC_LIQUIDATION",):
                continue

            others = {p: snap[p] for p in PAIRS if p != pair}
            sig = compute_signal(c, others, regime)

            if sig["direction"] != "long":
                continue
            if sig["score"] < MIN_SCORE or sig["ev"] <= 0:
                continue

            # Sizing
            atr_val   = calc_atr([x["high"] for x in c], [x["low"] for x in c],
                                  [x["close"] for x in c], 14)
            sl_pct    = max(0.015, min(0.07, atr_val * 2 / price))
            tp_pct    = sl_pct * 2.0

            # Kelly parcial
            p_win  = sig["score"]
            payoff = tp_pct / sl_pct
            kelly  = max(0, (p_win * payoff - (1-p_win)) / payoff) * 0.25
            size_pct = max(0.02, min(0.12, kelly if kelly > 0 else TRADE_PCT_BASE * sig["score"]))
            size_usd  = balance * size_pct
            if size_usd < 5: continue

            qty = size_usd / price
            fee = size_usd * TAKER_FEE
            if balance < size_usd + fee: continue

            balance -= size_usd + fee
            positions[pair] = {
                "entry":    price,
                "sl":       price * (1 - sl_pct),
                "tp":       price * (1 + tp_pct),
                "qty":      qty,
                "size_usd": size_usd,
                "bar_entry": i,
                "sl_pct":   sl_pct,
                "score":    sig["score"],
                "regime":   regime,
            }

        equity.append(balance + sum(
            pos["size_usd"] * (candles_map[p][min(i, n_bars-1)]["close"] / pos["entry"])
            for p, pos in positions.items()
        ))

    # Fecha posições abertas no último preço
    for pair, pos in list(positions.items()):
        price = candles_map[pair][-1]["close"]
        pnl = (price - pos["entry"]) / pos["entry"] * pos["size_usd"] - pos["size_usd"] * TAKER_FEE
        balance += pos["size_usd"] + pnl
        trades.append({"pair": pair, "entry": pos["entry"], "exit": price,
                       "pnl": pnl, "pct": (price-pos["entry"])/pos["entry"]*100,
                       "reason": "EOD", "bars": n_bars - pos["bar_entry"]})

    # ── Métricas ─────────────────────────────────────────────────────────
    print("\n[3/3] Calculando métricas...\n")
    final_usd  = balance
    total_pnl  = final_usd - INITIAL_USD
    pnl_pct    = total_pnl / INITIAL_USD * 100

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    n      = len(trades)
    win_rate = len(wins) / n if n > 0 else 0

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses)) or 0.001
    profit_factor = gross_profit / gross_loss

    avg_win  = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

    # Drawdown máximo
    peak, max_dd = equity[0], 0.0
    for v in equity:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd

    # Sharpe (retornos diários)
    daily = []
    step = 24  # 24 barras 1H = 1 dia
    for i in range(step, len(equity), step):
        r = (equity[i] - equity[i-step]) / equity[i-step]
        daily.append(r)
    sharpe = 0.0
    if len(daily) > 2:
        m = sum(daily)/len(daily); s = statistics.stdev(daily)
        sharpe = (m / s * math.sqrt(252)) if s > 0 else 0

    # Expectancy
    expectancy = (win_rate * avg_win + (1-win_rate) * avg_loss)

    # Por par
    by_pair = {}
    for p in PAIRS:
        pt = [t for t in trades if t["pair"] == p]
        pw = [t for t in pt if t["pnl"] > 0]
        by_pair[p] = {"n": len(pt), "wins": len(pw),
                      "wr": len(pw)/len(pt) if pt else 0,
                      "pnl": sum(t["pnl"] for t in pt)}

    # Por regime
    by_regime = {}
    for t in trades:
        r = positions.get(t["pair"], {}).get("regime", "?")
        by_regime.setdefault(r, {"n":0,"wins":0,"pnl":0.0})
        by_regime[r]["n"] += 1
        if t["pnl"] > 0: by_regime[r]["wins"] += 1
        by_regime[r]["pnl"] += t["pnl"]

    # Por motivo de saída
    by_reason = {}
    for t in trades:
        by_reason.setdefault(t["reason"], {"n":0,"pnl":0.0})
        by_reason[t["reason"]]["n"] += 1
        by_reason[t["reason"]]["pnl"] += t["pnl"]

    # ── Impressão ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  RESULTADO GERAL")
    print("=" * 60)
    print(f"  P&L Total:        ${total_pnl:+.2f}  ({pnl_pct:+.2f}%)")
    print(f"  Portfolio Final:  ${final_usd:.2f}")
    print(f"  Trades:           {n}  ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Win Rate:         {win_rate*100:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Avg Win:          ${avg_win:+.2f}")
    print(f"  Avg Loss:         ${avg_loss:+.2f}")
    print(f"  Expectancy/trade: ${expectancy:+.2f}")
    print(f"  Max Drawdown:     {max_dd*100:.2f}%")
    print(f"  Sharpe (anual):   {sharpe:.2f}")
    print()

    print("─" * 60)
    print("  POR PAR")
    print("─" * 60)
    for p, d in by_pair.items():
        print(f"  {p:<10} {d['n']:>3} trades | WR {d['wr']*100:.0f}% | P&L ${d['pnl']:+.2f}")

    print()
    print("─" * 60)
    print("  POR MOTIVO DE SAÍDA")
    print("─" * 60)
    for r, d in sorted(by_reason.items()):
        avg = d["pnl"]/d["n"] if d["n"] else 0
        print(f"  {r:<6} {d['n']:>3}x | P&L ${d['pnl']:+.2f} | avg ${avg:+.2f}/trade")

    print()
    print("─" * 60)
    print("  ÚLTIMOS 10 TRADES")
    print("─" * 60)
    for t in trades[-10:]:
        icon = "✓" if t["pnl"] > 0 else "✗"
        print(f"  {icon} {t['pair']:<9} {t['reason']:<4} | "
              f"entry ${t['entry']:,.2f} → exit ${t['exit']:,.2f} | "
              f"{t['pct']:+.2f}% | P&L ${t['pnl']:+.2f} | {t['bars']}h")

    print()
    print("=" * 60)

    # Análise e recomendações
    print("  DIAGNÓSTICO")
    print("=" * 60)
    issues = []
    if win_rate < 0.45:
        issues.append(f"⚠ Win Rate baixo ({win_rate*100:.1f}%) — score mínimo muito permissivo ou regime detection fraco")
    if profit_factor < 1.2:
        issues.append(f"⚠ Profit Factor baixo ({profit_factor:.2f}) — SL muito apertado ou TP muito distante")
    if max_dd < -0.15:
        issues.append(f"⚠ Drawdown alto ({max_dd*100:.1f}%) — sizing agressivo ou SL muito largo")
    if n < 10:
        issues.append(f"⚠ Poucos trades ({n}) — score threshold muito restritivo")
    if n > 100:
        issues.append(f"⚠ Trades excessivos ({n}) — possível overtrading com cooldown curto")

    tp_exits  = by_reason.get("TP", {}).get("n", 0)
    sl_exits  = by_reason.get("SL", {}).get("n", 0)
    tp_ratio  = tp_exits / (tp_exits + sl_exits) if (tp_exits+sl_exits) > 0 else 0
    if tp_ratio < 0.30:
        issues.append(f"⚠ Poucos TPs atingidos ({tp_ratio*100:.0f}%) — TP muito distante (2× SL pode ser excessivo)")

    sl_pnl = by_reason.get("SL", {}).get("pnl", 0)
    tp_pnl = by_reason.get("TP", {}).get("pnl", 0)

    if not issues:
        print("  ✓ Sem alertas críticos detectados")
    for issue in issues:
        print(f"  {issue}")

    print()
    print("  SUGESTÕES DE AJUSTE:")
    if win_rate > 0.55 and profit_factor > 1.5:
        print("  → Sistema saudável. Considerar aumentar sizing gradualmente.")
    if win_rate < 0.45:
        print("  → Aumentar MIN_SCORE para 0.60 para reduzir entradas fracas")
    if tp_ratio < 0.30:
        print("  → Reduzir TP de 2× para 1.5× SL para mais realizações")
    if max_dd < -0.20:
        print("  → Reduzir MAX_OPEN de 3 para 2 posições simultâneas")
    if n < 15:
        print("  → Reduzir MIN_SCORE para 0.52 ou COOLDOWN_BARS para 2")

    print("=" * 60)

    return {
        "trades": n, "win_rate": win_rate, "profit_factor": profit_factor,
        "pnl_usd": total_pnl, "pnl_pct": pnl_pct,
        "max_dd": max_dd, "sharpe": sharpe, "expectancy": expectancy
    }


if __name__ == "__main__":
    run_backtest()
