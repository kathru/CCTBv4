"""
Simulation for May 11, 2026 (Including BUY and SELL)
"""

import sys, os, time, math, statistics, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Parâmetros ──────────────────────────────────────────────────────────────
PAIRS          = ["BTC-USD", "ETH-USD", "SOL-USD"]
INITIAL_BRL    = 5000.0
USD_BRL        = 4.90
INITIAL_USD    = INITIAL_BRL / USD_BRL
TRADE_PCT_BASE = 0.08
TAKER_FEE      = 0.0004
MAKER_FEE      = 0.0001
BAR            = "15m"
START_TS       = 1778457600  # 2026-05-11 00:00:00
END_TS         = 1778543999  # 2026-05-11 23:59:59
MIN_SCORE      = 0.55
COOLDOWN_BARS  = 4
MAX_OPEN       = 3

import requests
_session = requests.Session()

def fetch_history(inst_id: str, bar: str = BAR, limit: int = 300) -> list:
    """Busca candles históricos."""
    p = {"instId": inst_id, "bar": bar, "limit": str(limit)}
    try:
        r = _session.get("https://www.okx.com/api/v5/market/history-candles",
                         params=p, timeout=12)
        data = r.json().get("data", [])
    except Exception as e:
        print(f"Error fetching {inst_id}: {e}")
        return []

    # Ordena ASC (mais antigo primeiro)
    all_c = [{
        "ts":     int(c[0]) // 1000,
        "open":   float(c[1]), "high": float(c[2]),
        "low":    float(c[3]), "close": float(c[4]),
        "volume": float(c[5]),
    } for c in data]
    all_c.sort(key=lambda x: x["ts"])
    return all_c

# Reusing indicators from backtest.py
def ema(vals, span):
    if not vals: return []
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

def compute_signal(candles: list, candles_other: dict, regime: str) -> dict:
    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]
    if len(closes) < 50: return {"score": 0.5, "ev": -0.01, "direction": "neutral"}
    bb_pct   = bb_width_pct(closes)
    atr_cur  = calc_atr(highs, lows, closes, 14)
    atr_prev = calc_atr(highs[:-14], lows[:-14], closes[:-14], 14)
    atr_exp  = atr_cur / atr_prev if atr_prev > 0 else 1.0
    compress = max(0, 1 - bb_pct)
    exp_start = max(0, min(1, (atr_exp - 0.9) / 0.5)) if atr_exp > 0.9 else 0
    ema9  = ema(closes, 9);  ema21 = ema(closes, 21);  ema50 = ema(closes, 50)
    if ema9[-1] > ema21[-1] > ema50[-1]: dir_vol, dir_score = "long", 0.70
    elif ema9[-1] < ema21[-1] < ema50[-1]: dir_vol, dir_score = "short", 0.70
    elif ema9[-1] > ema21[-1]: dir_vol, dir_score = "long", 0.55
    else: dir_vol, dir_score = "short", 0.55
    vol_exp = (compress*0.35 + exp_start*0.30 + dir_score*0.35) * {
        "VOLATILITY_COMPRESSION": 1.30, "TREND_EXPANSION": 1.10,
        "MEAN_REVERTING_CHOP": 0.70, "PANIC_LIQUIDATION": 0.20
    }.get(regime, 1.0)
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
    if mkt_long > mkt_short and mkt_long > 0.25: dir_mkt = "long";  mkt_score = 0.50 + min(mkt_long*0.40, 0.40)
    elif mkt_short > mkt_long and mkt_short > 0.25: dir_mkt = "short"; mkt_score = 0.50 + min(mkt_short*0.40, 0.40)
    else: dir_mkt = "neutral"; mkt_score = 0.50
    bv = sum(c["volume"] for c in candles[-10:] if c["close"] >= c["open"])
    sv = sum(c["volume"] for c in candles[-10:] if c["close"] <  c["open"])
    tv = bv + sv
    taker_imbal = (bv - sv) / tv if tv > 0 else 0.0
    flow_long  = min(max(0, taker_imbal)*1.5 + (0.20 if candles[-1]["volume"] > sum(volumes[-5:])/5 else 0), 1.0)
    flow_short = min(max(0, -taker_imbal)*1.5, 1.0)
    if flow_long > flow_short and flow_long > 0.20: dir_flow = "long";  flow_score = 0.50 + min(flow_long*0.35, 0.35)
    elif flow_short > flow_long and flow_short > 0.20: dir_flow = "short"; flow_score = 0.50 + min(flow_short*0.35, 0.35)
    else: dir_flow = "neutral"; flow_score = 0.50
    rs_composite = 0.0
    if candles_other:
        btc_cl = [c["close"] for c in candles_other.get("BTC-USD", [])]
        this_cl = closes
        if len(btc_cl) >= 5 and len(this_cl) >= 5:
            perf_this = (this_cl[-1]-this_cl[-4])/this_cl[-4]
            perf_btc  = (btc_cl[-1]-btc_cl[-4])/btc_cl[-4]
            rs_composite = perf_this - perf_btc
    if rs_composite > 0.003: dir_rs = "long";  rs_score = 0.50 + min(rs_composite*20, 0.25)
    elif rs_composite < -0.003: dir_rs = "short"; rs_score = 0.50 + min(abs(rs_composite)*20, 0.25)
    else: dir_rs = "neutral"; rs_score = 0.50
    votes = {"long": 0.0, "short": 0.0, "neutral": 0.0}
    for d, s in [(dir_vol, vol_exp), (dir_mkt, mkt_score-0.5), (dir_flow, flow_score-0.5), (dir_rs, rs_score-0.5)]:
        votes[d] += s
    dominant = max(votes, key=votes.get)
    if votes[dominant] <= 0.02: dominant = "neutral"
    def align(s, d): return s if d == dominant else (1-s)
    w = {"vol":0.20, "mkt":0.28, "flow":0.28, "rs":0.24}
    raw = (align(vol_exp, dir_vol)*w["vol"]*0.65 + align(mkt_score, dir_mkt)*w["mkt"]*0.65 +
           align(flow_score, dir_flow)*w["flow"]*0.70 + align(rs_score, dir_rs)*w["rs"]*0.60)
    total_w = sum(w.values())
    score = max(0.0, min(1.0, raw / (total_w * 0.65)))
    if regime in ("PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"): score = min(score, 0.40); dominant = "neutral"
    elif regime == "HIGH_CORRELATION_RISK": score = min(score, 0.58)
    ev = score * 2.0 - (1-score) * 1.0 - (TAKER_FEE * 3)
    return {"score": round(score,4), "ev": round(ev,4), "direction": dominant}

def detect_regime_simple(candles: list) -> str:
    closes = [c["close"] for c in candles]; highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
    if len(closes) < 30: return "MEAN_REVERTING_CHOP"
    adx = calc_adx(highs, lows, closes); bb  = bb_width_pct(closes)
    atr_cur  = calc_atr(highs, lows, closes, 14); atr_prev = calc_atr(highs[:-14], lows[:-14], closes[:-14], 14)
    atr_exp  = atr_cur / atr_prev if atr_prev > 0 else 1.0
    vol_pct  = min(1.0, atr_cur / closes[-1] / 0.03)
    if vol_pct > 0.90 and atr_exp > 2.0: return "PANIC_LIQUIDATION"
    if bb < 0.15 and adx < 18:           return "VOLATILITY_COMPRESSION"
    if adx > 30:                          return "TREND_EXPANSION"
    if adx > 25 and atr_exp < 0.90:      return "TREND_EXHAUSTION"
    return "MEAN_REVERTING_CHOP"

def run():
    print(f"Simulação para 11 de Maio de 2026 (BUY & SELL)")
    print(f"Portfolio Inicial: R$ {INITIAL_BRL:.2f} (US$ {INITIAL_USD:.2f} @ {USD_BRL})")

    candles_map = {}
    for pair in PAIRS:
        inst = pair.replace("-USD", "-USDT")
        candles_map[pair] = fetch_history(inst)

    # Find start and end indices for May 11
    n_bars = min(len(v) for v in candles_map.values())
    for p in PAIRS: candles_map[p] = candles_map[p][-n_bars:]

    start_idx = None
    end_idx = None
    for i, c in enumerate(candles_map[PAIRS[0]]):
        if c["ts"] >= START_TS and start_idx is None: start_idx = i
        if c["ts"] <= END_TS: end_idx = i

    if start_idx is None or end_idx is None:
        print("Erro: Dados para 11 de Maio não encontrados.")
        return

    print(f"Rodando de index {start_idx} a {end_idx} ({end_idx - start_idx + 1} bars)")

    balance = INITIAL_USD
    positions = {} # {pair: {type: 'long'|'short', entry, sl, tp, size_usd}}
    trades = []
    equity = [INITIAL_USD]
    cooldowns = {p: 0 for p in PAIRS}

    for i in range(start_idx, end_idx + 1):
        snap = {p: candles_map[p][:i+1] for p in PAIRS}
        for pair in PAIRS:
            c = snap[pair]
            price = c[-1]["close"]
            if pair in positions:
                pos = positions[pair]
                # Check exit
                hit_tp = False
                hit_sl = False
                exit_price = 0

                if pos["type"] == "long":
                    if c[-1]["high"] >= pos["tp"]:
                        hit_tp = True; exit_price = pos["tp"]
                    elif c[-1]["low"] <= pos["sl"]:
                        hit_sl = True; exit_price = pos["sl"]
                else: # short
                    if c[-1]["low"] <= pos["tp"]:
                        hit_tp = True; exit_price = pos["tp"]
                    elif c[-1]["high"] >= pos["sl"]:
                        hit_sl = True; exit_price = pos["sl"]

                if hit_tp or hit_sl:
                    reason = "TP" if hit_tp else "SL"
                    if pos["type"] == "long":
                        pnl = (exit_price - pos["entry"]) / pos["entry"] * pos["size_usd"] - pos["size_usd"] * TAKER_FEE
                    else:
                        pnl = (pos["entry"] - exit_price) / pos["entry"] * pos["size_usd"] - pos["size_usd"] * TAKER_FEE

                    balance += pos["size_usd"] + pnl
                    trades.append({"pair": pair, "type": pos["type"], "pnl": pnl, "pct": pnl/pos["size_usd"]*100, "reason": reason})
                    del positions[pair]
                    if hit_sl: cooldowns[pair] = COOLDOWN_BARS
                    continue

            if cooldowns[pair] > 0:
                cooldowns[pair] -= 1
                continue
            if pair in positions or len(positions) >= MAX_OPEN or balance < INITIAL_USD * 0.05:
                continue

            regime = detect_regime_simple(c)
            sig = compute_signal(c, {p: snap[p] for p in PAIRS if p != pair}, regime)

            if sig["score"] >= MIN_SCORE:
                atr_val = calc_atr([x["high"] for x in c], [x["low"] for x in c], [x["close"] for x in c], 14)
                sl_pct = max(0.015, min(0.07, atr_val * 2 / price))
                tp_pct = sl_pct * 2.0
                size_usd = balance * TRADE_PCT_BASE

                if sig["direction"] == "long":
                    if balance >= size_usd * (1 + TAKER_FEE):
                        balance -= size_usd * (1 + TAKER_FEE)
                        positions[pair] = {"type": "long", "entry": price, "sl": price*(1-sl_pct), "tp": price*(1+tp_pct), "size_usd": size_usd}
                        print(f"  BUY {pair} @ {price:.2f} (Score: {sig['score']})")
                elif sig["direction"] == "short":
                    if balance >= size_usd * (1 + TAKER_FEE):
                        balance -= size_usd * (1 + TAKER_FEE)
                        positions[pair] = {"type": "short", "entry": price, "sl": price*(1+sl_pct), "tp": price*(1-tp_pct), "size_usd": size_usd}
                        print(f"  SELL {pair} @ {price:.2f} (Score: {sig['score']})")

        current_val = balance
        for p, pos in positions.items():
            curr_price = candles_map[p][i]["close"]
            if pos["type"] == "long":
                current_val += pos["size_usd"] * (curr_price / pos["entry"])
            else:
                current_val += pos["size_usd"] * (2 - curr_price / pos["entry"])
        equity.append(current_val)

    # Close at end of day
    for pair, pos in list(positions.items()):
        price = candles_map[pair][end_idx]["close"]
        if pos["type"] == "long":
            pnl = (price - pos["entry"]) / pos["entry"] * pos["size_usd"] - pos["size_usd"] * TAKER_FEE
        else:
            pnl = (pos["entry"] - price) / pos["entry"] * pos["size_usd"] - pos["size_usd"] * TAKER_FEE
        balance += pos["size_usd"] + pnl
        trades.append({"pair": pair, "type": pos["type"], "pnl": pnl, "pct": pnl/pos["size_usd"]*100, "reason": "EOD"})

    final_brl = balance * USD_BRL
    print("\n--- RESULTADOS ---")
    print(f"Portfolio Final: R$ {final_brl:.2f}")
    print(f"P&L: R$ {final_brl - INITIAL_BRL:.2f} ({(final_brl/INITIAL_BRL - 1)*100:.2f}%)")

    wins = [t for t in trades if t["pnl"] > 0]
    wr = len(wins)/len(trades) if trades else 0
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0)) or 0.001
    pf = gp / gl

    print(f"Win Rate: {wr*100:.1f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Total Trades: {len(trades)}")

    for p in PAIRS:
        pt = [t for t in trades if t["pair"] == p]
        ppnl = sum(t["pnl"] for t in pt) * USD_BRL
        print(f"  {p}: {len(pt)} trades, P&L R$ {ppnl:+.2f}")

if __name__ == "__main__":
    run()
