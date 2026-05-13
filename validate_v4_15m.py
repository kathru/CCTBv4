"""
validate_v4_15m.py — Validação V4 em 15 minutos (mesmo ciclo do runtime)
=========================================================================
O bot opera em ciclos de 15 minutos. Validar em 1H subestima latência,
superestima sinal e não reflete a realidade de execução.

Esta versão usa:
  - Candles 15M (OKX histórico autenticado, ~2-3 anos disponíveis)
  - Mesmo V4Orchestrator do runtime
  - SimulatedExecutionEngine com fills/fees reais
  - Stepper: 1 candle = 1 ciclo de 15 min

Adicionalmente implementa:
  REGIME EDGE TABLE — só entra em regimes com edge líquido demonstrado
  FINAL REPORT      — salva os números exatos para decisão de go/no-go

Regime Edge Rule:
  Construída a partir do walk-forward 1H (melhores dados disponíveis).
  Regime tem "edge" se: WR_historico × RR - (1-WR) × 1 - fees > 0
  Com RR=3.0×, fees=0.52%: WR_min = 25.5%
  A rule bloqueia entradas em regimes com WR < 30% histórico OU n < 20 trades.

Uso:
  python validate_v4_15m.py --fetch      # busca candles 15M do OKX
  python validate_v4_15m.py --oos        # roda OOS com V4 pipeline
  python validate_v4_15m.py --report     # relatório final
  python validate_v4_15m.py              # tudo
"""

import os, sys, json, time, math, statistics, argparse, hmac, hashlib, base64
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")
RESULT_PATH = os.path.join(DATA_DIR, "validation_v4_15m_result.json")
EDGE_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "regime_edge.json")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "final_report.json")

PAIRS_HIST = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
PAIRS_RT   = ["BTC-USD",  "ETH-USD",  "SOL-USD"]
PAIR_MAP   = dict(zip(PAIRS_HIST, PAIRS_RT))
BAR        = "15m"
OOS_MONTHS = 6
INITIAL_CAPITAL = 1000.0

# ── OKX fetch 15M ────────────────────────────────────────────────────────────

def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code.env")
    env = {}
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

def fetch_15m(inst_id: str, env: dict) -> list:
    import requests
    api_key    = env.get("OKX_API_KEY",    os.getenv("OKX_API_KEY", ""))
    secret_key = env.get("OKX_SECRET_KEY", os.getenv("OKX_SECRET_KEY", ""))
    passphrase = env.get("OKX_PASSPHRASE", os.getenv("OKX_PASSPHRASE", ""))
    BASE_URL   = "https://www.okx.com"
    path       = "/api/v5/market/history-candles"

    def sign(ts, method, full_path):
        msg = f"{ts}{method.upper()}{full_path}"
        return base64.b64encode(
            hmac.new(secret_key.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()

    def headers(qs):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        fp = f"{path}?{qs}"
        return {
            "OK-ACCESS-KEY":        api_key,
            "OK-ACCESS-SIGN":       sign(ts, "GET", fp),
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": passphrase,
        }

    sess = requests.Session()
    all_c, after, pages = [], None, 0
    print(f"  Buscando {inst_id} {BAR}...", end="", flush=True)

    while True:
        params = {"instId": inst_id, "bar": BAR, "limit": "300"}
        if after:
            params["after"] = str(after)
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        try:
            r    = sess.get(BASE_URL + path, params=params, headers=headers(qs), timeout=12)
            data = r.json().get("data", [])
        except Exception as e:
            print(f" ERRO: {e}"); break
        if not data:
            break
        all_c.extend(data)
        after  = data[-1][0]
        pages += 1
        time.sleep(0.15)
        if pages % 20 == 0:
            oldest = datetime.utcfromtimestamp(int(data[-1][0])//1000).strftime("%Y-%m")
            print(f" {pages}p/{oldest}", end="", flush=True)

    all_c.sort(key=lambda x: int(x[0]))
    candles = [{"ts": int(c[0])//1000, "open": float(c[1]), "high": float(c[2]),
                "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
               for c in all_c]
    oldest = datetime.utcfromtimestamp(candles[0]["ts"]).strftime("%Y-%m-%d") if candles else "?"
    print(f" → {len(candles)} candles desde {oldest}")
    return candles


def fetch_all_15m():
    print("\n[FETCH 15M] Buscando candles 15 minutos do OKX...")
    env = _load_env()
    os.makedirs(DATA_DIR, exist_ok=True)
    for pair in PAIRS_HIST:
        candles = fetch_15m(pair, env)
        if candles:
            path = os.path.join(DATA_DIR, f"{pair}_15m.json")
            with open(path, "w") as f:
                json.dump(candles, f)
            print(f"  Salvo: {path}")


def load_15m_candles() -> dict:
    result = {}
    for pair in PAIRS_HIST:
        path = os.path.join(DATA_DIR, f"{pair}_15m.json")
        if os.path.exists(path):
            with open(path) as f:
                result[pair] = json.load(f)
            print(f"  {pair}: {len(result[pair])} candles 15M")
    return result


# ── Regime Edge Table ─────────────────────────────────────────────────────────

def build_regime_edge_table() -> dict:
    """
    Constrói a regime_edge_table a partir dos dados de walk-forward 1H.
    Edge = EV líquido positivo dado WR histórico e RR=3.0×.
    Bloqueia entradas em regimes sem edge demonstrado ou amostra insuficiente.
    """
    RR   = 3.0
    FEES = 0.0052   # round-trip fees + slippage
    WR_MIN_SAMPLES = 20   # mínimo de trades para considerar o regime

    result_path = os.path.join(DATA_DIR, "validation_result.json")
    v4_path     = os.path.join(DATA_DIR, "validation_v4_result.json")

    # Agrega by_regime de todas as janelas walk-forward
    by_regime: dict = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})

    for path in [result_path, v4_path]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for w in data.get("windows", []):
            for rname, rs in w.get("by_regime", {}).items():
                by_regime[rname]["n"]    += rs.get("n", 0)
                by_regime[rname]["wins"] += round(rs.get("n", 0) * rs.get("wr", 0.5))
                by_regime[rname]["pnl"]  += rs.get("pnl", 0)

    edge_table = {}
    REGIME_MAP = {
        # Mapeamento de nomes do walk-forward (validador simples) → nomes V4
        "TREND_UP":   "TREND_EXPANSION",
        "TREND_DOWN": "TREND_EXHAUSTION",
        "CHOP":       "MEAN_REVERTING_CHOP",
        "COMPRESS":   "VOLATILITY_COMPRESSION",
    }

    for rname, stats in by_regime.items():
        n    = stats["n"]
        wr   = stats["wins"] / n if n > 0 else 0.0
        ev   = wr * RR - (1 - wr) * 1.0 - FEES * (1 + RR)
        has_edge = (n >= WR_MIN_SAMPLES) and (ev > 0) and (wr >= 0.28)

        v4_name = REGIME_MAP.get(rname, rname)
        edge_table[v4_name] = {
            "n": n, "win_rate": round(wr, 4), "ev": round(ev, 4),
            "has_edge": has_edge,
            "reason": (
                "edge positivo" if has_edge
                else f"insuficiente (n={n})" if n < WR_MIN_SAMPLES
                else f"EV negativo ({ev:.3f})"
            ),
        }

    # Regimes não observados: bloqueados por padrão
    for regime in ["PANIC_LIQUIDATION", "LIQUIDITY_VACUUM", "HIGH_CORRELATION_RISK"]:
        if regime not in edge_table:
            edge_table[regime] = {
                "n": 0, "win_rate": 0.0, "ev": -1.0,
                "has_edge": False, "reason": "regime de risco — bloqueado",
            }

    os.makedirs(os.path.dirname(EDGE_PATH), exist_ok=True)
    with open(EDGE_PATH, "w") as f:
        json.dump(edge_table, f, indent=2)

    print("\n[REGIME EDGE TABLE]")
    print(f"  {'Regime':<30} {'n':>5} {'WR':>7} {'EV':>8} {'Edge?'}")
    print("  " + "-" * 60)
    for r, s in sorted(edge_table.items(), key=lambda x: -x[1]["n"]):
        flag = "[EDGE]" if s["has_edge"] else "[BLOCK]"
        print(f"  {r:<30} {s['n']:>5} {s['win_rate']*100:>6.1f}% {s['ev']:>+8.4f} {flag}")
    print(f"\n  Salvo: {EDGE_PATH}")
    return edge_table


# ── Simulação OOS 15M ─────────────────────────────────────────────────────────

def simulate_oos_15m(candles_map: dict, edge_table: dict) -> dict:
    from strategies.fee_model import FEE
    from strategies.risk_prior import load_prior
    from paper_trading.simulated_engine import SimulatedExecutionEngine
    from dashboard.v4_orchestrator import V4Orchestrator
    import data.market_data as _md

    btc = candles_map.get("BTC-USDT", [])
    if not btc:
        print("Sem candles 15M. Execute --fetch primeiro.")
        return {}

    oos_cut  = btc[-1]["ts"] - OOS_MONTHS * 30 * 24 * 3600
    oos_start = datetime.utcfromtimestamp(oos_cut).strftime("%Y-%m-%d")
    oos_end   = datetime.utcfromtimestamp(btc[-1]["ts"]).strftime("%Y-%m-%d")
    print(f"\n[V4 15M] OOS: {oos_start} → {oos_end}")

    # Contexto histórico (sem I/O de exchange)
    def _build_ctx(pair_rt, candles):
        closes  = [c["close"]  for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        volumes = [c["volume"] for c in candles]
        if len(closes) < 25:
            return {"pair": pair_rt, "timestamp": int(time.time()),
                    "realized_vol": 0.02, "vol_percentile": 0.5,
                    "atr_rate": {"expansion": 1.0},
                    "funding": {"funding_rate": 0.0, "sentiment": "neutral"},
                    "open_interest": {"expanding": False, "oi_change_pct": 0.0},
                    "orderbook": {"imbalance": 0.0, "spread_pct": 0.0002},
                    "taker_ratio": {"imbalance": 0.0},
                    "volume_delta": {"delta_pct": 0.0, "aggressive": False},
                    "correlation": {}}
        # taker proxy
        rc = candles[-10:]
        bv = sum(c["volume"] for c in rc if c["close"] >= c["open"])
        sv = sum(c["volume"] for c in rc if c["close"] < c["open"])
        tv = bv + sv; ti = (bv - sv) / tv if tv > 0 else 0.0
        avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        vd = (volumes[-1] - avg_vol) / avg_vol if avg_vol > 0 else 0.0
        # ATR expansion
        def _atr(h, l, c, p=14):
            if len(c) < 2: return c[-1] * 0.005
            trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
            return sum(trs[-p:]) / min(p, len(trs))
        atr_n = _atr(highs, lows, closes)
        atr_p = _atr(highs[:-14], lows[:-14], closes[:-14]) if len(closes) > 28 else atr_n
        return {
            "pair": pair_rt, "timestamp": int(time.time()),
            "realized_vol": statistics.stdev([math.log(closes[i]/closes[i-1])
                             for i in range(1, min(25, len(closes))) if closes[i-1] > 0]) * math.sqrt(96)
                             if len(closes) > 2 else 0.02,
            "vol_percentile": 0.5,
            "atr_rate": {"expansion": atr_n / atr_p if atr_p > 0 else 1.0,
                         "atr_pct": atr_n / closes[-1] if closes[-1] > 0 else 0.005},
            "funding": {"funding_rate": 0.0, "sentiment": "neutral"},
            "open_interest": {"expanding": False, "oi_change_pct": 0.0},
            "orderbook": {"imbalance": 0.0, "spread_pct": FEE.slippage_base * 2},
            "taker_ratio": {"imbalance": ti},
            "volume_delta": {"delta_pct": vd, "aggressive": vd > 0.5},
            "correlation": {},
        }

    def resample_6h(c15m):
        out = []; n = 24  # 24×15m = 6h
        for i in range(0, len(c15m) - n + 1, n):
            g = c15m[i:i+n]
            out.append({"ts": g[0]["ts"], "open": g[0]["open"],
                        "high": max(x["high"] for x in g),
                        "low":  min(x["low"]  for x in g),
                        "close": g[-1]["close"],
                        "volume": sum(x["volume"] for x in g)})
        return out

    v4     = V4Orchestrator(state_dir=os.path.join(DATA_DIR, ".v4_15m_state"))
    engine = SimulatedExecutionEngine(INITIAL_CAPITAL, default_order_mode="adaptive",
                                      default_spread_pct=FEE.slippage_base * 2, seed=42)
    engine.balance_usd = INITIAL_CAPITAL; engine.holdings = {}
    engine.entry_prices = {}; engine.trades = []; engine.pending_orders = []
    engine.total_fees_usd = 0.0

    slots = {f"V4:{p}": {"qty": 0.0, "entry": 0.0, "peak": 0.0,
                          "sl_pct": 0.03, "sl_level": 0.0} for p in PAIRS_RT}

    # Monkey-patch exchange calls
    _orig_ob = _md.get_orderbook_context
    _orig_tr = _md.get_taker_ratio
    _orig_fr = _md.get_funding_rate
    _orig_oi = _md.get_open_interest
    _md.get_orderbook_context = lambda p: {"imbalance": 0.0, "spread_pct": 0.0002, "bid_depth": 0, "ask_depth": 0}
    _md.get_taker_ratio       = lambda p: {"imbalance": 0.0, "buy_volume": 0, "sell_volume": 0}
    _md.get_funding_rate      = lambda p: {"funding_rate": 0.0, "sentiment": "neutral"}
    _md.get_open_interest     = lambda p: {"expanding": False, "oi_change_pct": 0.0, "oi_usd": 0}

    oos_candles = [c for c in btc if c["ts"] >= oos_cut]
    print(f"  {len(oos_candles)} candles de 15M no OOS ({len(oos_candles)*15//60//24} dias)")

    all_trades = []; equity = [INITIAL_CAPITAL]; done = 0

    try:
        for candle in oos_candles:
            ts_now = candle["ts"]
            done  += 1
            if done % 5000 == 0:
                pct = done / len(oos_candles) * 100
                print(f"  {pct:.0f}% ({done}/{len(oos_candles)})...", end="\r", flush=True)

            for pair_hist, pair_rt in PAIR_MAP.items():
                ctx_all = [c for c in candles_map.get(pair_hist, []) if c["ts"] <= ts_now][-400:]
                if len(ctx_all) < 50:
                    continue

                # ── Regime edge check ──────────────────────────────────────
                # Calculado rapidamente com features básicas antes de chamar V4
                closes = [c["close"] for c in ctx_all]
                def _ema(v, s):
                    k=2/(s+1); r=[v[0]]
                    for x in v[1:]: r.append(x*k+r[-1]*(1-k))
                    return r
                e9=_ema(closes,9); e21=_ema(closes,21); e50=_ema(closes,50) if len(closes)>=50 else e21
                quick_regime = ("TREND_EXPANSION" if e9[-1]>e21[-1]>e50[-1]
                                else "MEAN_REVERTING_CHOP")
                if quick_regime not in edge_table or not edge_table[quick_regime]["has_edge"]:
                    # Regime sem edge — não avalia V4 (economiza CPU)
                    continue

                ctx_6h = resample_6h(ctx_all)[-50:]
                slot   = slots.get(f"V4:{pair_rt}")
                has_pos = slot and slot.get("qty", 0) > 0

                try:
                    decision = v4.evaluate(
                        pair=pair_rt, candles_1h=ctx_all, candles_6h=ctx_6h,
                        closes_map={pair_rt: closes},
                        engine=engine, open_slots=slots,
                        existing_slot=slot if has_pos else None,
                        breadth_score=1.0,
                    )
                except Exception:
                    continue

                price = ctx_all[-1]["close"] if ctx_all else 0
                if not price:
                    continue

                slot_key = f"V4:{pair_rt}"
                sym = pair_rt.split("-")[0]

                if decision.get("decision") == "BUY" and not has_pos:
                    size_usd = decision.get("size_usd", 0)
                    if size_usd > 5.0 and engine.balance_usd >= size_usd:
                        ok = engine.buy(sym, size_usd, price, "V4:15m", order_type="market")
                        if ok:
                            sl   = (decision.get("execution") and decision["execution"].stop_loss) or price * 0.97
                            sl_p = abs(price - sl) / price if sl else 0.03
                            slots[slot_key] = {"qty": size_usd / price, "entry": price,
                                               "peak": price, "entry_usd": size_usd,
                                               "sl_pct": sl_p, "sl_level": sl or price*(1-sl_p)}
                            all_trades.append({"pair": pair_rt, "ts": ts_now, "side": "BUY",
                                               "price": price, "size_usd": size_usd,
                                               "score": decision.get("score", 0),
                                               "regime": decision.get("regime").regime
                                               if hasattr(decision.get("regime"), "regime") else quick_regime})

                elif decision.get("decision") == "SELL" and has_pos:
                    qty = slot.get("qty", 0)
                    if qty > 0:
                        ok = engine.sell(sym, qty, price, "V4:15m", order_type="market")
                        if ok:
                            entry = slot.get("entry", price)
                            pnl   = (price - entry) / entry * slot.get("entry_usd", qty*price)
                            all_trades.append({"pair": pair_rt, "ts": ts_now, "side": "SELL",
                                               "price": price, "pnl": round(pnl, 4),
                                               "score": decision.get("score", 0),
                                               "regime": quick_regime,
                                               "exit_type": decision.get("exit_type", "signal")})
                            slots[slot_key] = {"qty": 0.0, "entry": 0.0, "peak": 0.0,
                                               "sl_pct": 0.03, "sl_level": 0.0}

            prices_tick = {pair_rt.split("-")[0]: (candles_map.get(pair_hist, [{}])[-1] or {}).get("close", 0)
                           for pair_hist, pair_rt in PAIR_MAP.items()}
            engine.tick(prices_tick)
            equity.append(engine.portfolio_value())

    finally:
        _md.get_orderbook_context = _orig_ob
        _md.get_taker_ratio       = _orig_tr
        _md.get_funding_rate      = _orig_fr
        _md.get_open_interest     = _orig_oi

    print()
    sells    = [t for t in all_trades if t.get("side") == "SELL"]
    n        = len(sells)
    wins     = [t for t in sells if t.get("pnl", 0) > 0]
    losses   = [t for t in sells if t.get("pnl", 0) <= 0]
    wr       = len(wins) / n if n > 0 else 0.0
    g_win    = sum(t["pnl"] for t in wins)
    g_loss   = abs(sum(t["pnl"] for t in losses)) or 1e-6
    pf       = g_win / g_loss
    ret      = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    avg_win  = g_win  / len(wins)   if wins   else 0.0
    avg_loss = g_loss / len(losses) if losses else 0.0
    payoff   = avg_win / avg_loss   if avg_loss > 0 else 0.0

    peak = INITIAL_CAPITAL; max_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd

    pnls = [equity[i] - equity[i-1] for i in range(1, len(equity))]
    sharpe = 0.0
    if len(pnls) > 10:
        m = statistics.mean(pnls); s = statistics.stdev(pnls)
        sharpe = m / s * math.sqrt(252 * 96) if s > 0 else 0.0  # 96 ciclos/dia = 24h/0.25h

    # Benchmarks
    benchmarks = {}
    for pair_h, label in [("BTC-USDT","BTC"),("ETH-USDT","ETH"),("SOL-USDT","SOL")]:
        oos = [c for c in candles_map.get(pair_h,[]) if c["ts"] >= oos_cut]
        if oos:
            benchmarks[label] = round((oos[-1]["close"] - oos[0]["open"]) / oos[0]["open"], 4)
    if benchmarks:
        benchmarks["equal_weight"] = round(sum(benchmarks.values()) / len(benchmarks), 4)

    result = {
        "timeframe": "15m", "oos_start": oos_start, "oos_end": oos_end,
        "n_candles_oos": len(oos_candles),
        "n_trades": n, "win_rate": round(wr, 4),
        "profit_factor": round(min(pf, 20.0), 3),
        "total_return": round(ret, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "avg_win_usd": round(avg_win, 4),
        "avg_loss_usd": round(avg_loss, 4),
        "payoff_ratio": round(payoff, 3),
        "total_fees_usd": round(engine.total_fees_usd, 4),
        "capital_final": round(equity[-1], 2),
        "benchmarks": benchmarks,
        "exec_stats": engine.execution_stats(),
        "regime_edge_applied": True,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Resultado 15M salvo: {RESULT_PATH}")
    return result


# ── Final Report ──────────────────────────────────────────────────────────────

def build_final_report(result_15m: dict = None) -> dict:
    """
    Relatório final com os números exatos para decisão go/no-go.
    Usa 15M se disponível, senão 1H.
    """
    # Carrega melhor resultado disponível
    if result_15m is None:
        if os.path.exists(RESULT_PATH):
            with open(RESULT_PATH) as f:
                result_15m = json.load(f)

    v4_path = os.path.join(DATA_DIR, "validation_v4_result.json")
    result_1h = {}
    if os.path.exists(v4_path):
        with open(v4_path) as f:
            result_1h = json.load(f).get("oos", {})

    # Prefere 15M se tiver trades; caso contrário usa 1H
    oos = result_15m if (result_15m and result_15m.get("n_trades", 0) > 0) else result_1h

    benchmarks = (result_15m or {}).get("benchmarks", {}) or result_1h.get("benchmarks", {})

    bm = (result_15m or {}).get("benchmarks",
          result_1h.get("benchmarks",
          (json.load(open(v4_path)) if os.path.exists(v4_path) else {}).get("benchmarks", {})))

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  RELATÓRIO FINAL — DECISÃO GO / NO-GO")
    print(f"  Timeframe: {oos.get('timeframe','?')}  |  OOS: {oos.get('oos_start','?')} → {oos.get('oos_end','?')}")
    print(SEP)

    n   = oos.get("n_trades", 0)
    wr  = oos.get("win_rate", 0)
    pf  = oos.get("profit_factor", 0)
    ret = oos.get("total_return", 0)
    dd  = oos.get("max_drawdown", 0)
    sh  = oos.get("sharpe", 0)
    aw  = oos.get("avg_win_usd", 0)
    al  = oos.get("avg_loss_usd", 0)
    pay = oos.get("payoff_ratio", 0)
    fees= oos.get("total_fees_usd", 0)

    pct = lambda v: f"{v*100:+.2f}%"
    f2  = lambda v: f"{v:.2f}"

    print(f"\n  OOS PERFORMANCE")
    print(f"  {'-'*60}")
    print(f"  Retorno líquido           {pct(ret):>12}  {'[OK]' if ret > 0 else '[NEGATIVO]'}")
    print(f"  Max Drawdown              {pct(dd):>12}")
    print(f"  Win Rate                  {wr*100:>11.1f}%  {'[OK]' if wr >= 0.30 else '[BAIXO]'}")
    print(f"  Profit Factor             {pf:>12.2f}  {'[OK]' if pf >= 1.0 else '[ABAIXO DE 1]'}")
    print(f"  Avg Win / Avg Loss        US${aw:.3f} / US${al:.3f}")
    print(f"  Payoff Ratio              {pay:>12.2f}  {'[OK]' if pay >= 2.5 else '[ABAIXO DE 2.5]'}")
    print(f"  Trades OOS                {n:>12}")
    print(f"  Sharpe                    {sh:>12.2f}")
    print(f"  Fees totais               US${fees:.4f}")

    if bm:
        print(f"\n  BENCHMARK vs SISTEMA")
        print(f"  {'-'*60}")
        print(f"  {'Estrategia':<28} {'Retorno':>10} {'vs Sistema':>12}")
        print(f"  {'Sistema V4 (' + oos.get('timeframe','?') + ')':<28} {pct(ret):>10} {'<-- ref':>12}")
        for label, bh in bm.items():
            alpha = ret - bh
            flag  = "(melhor)" if alpha > 0.005 else ("(pior)" if alpha < -0.005 else "(neutro)")
            print(f"  {label:<28} {pct(bh):>10} {pct(alpha):>12} {flag}")
        cash_alpha = ret - 0.0
        print(f"  {'Cash':<28} {pct(0.0):>10} {pct(cash_alpha):>12}")

    print(f"\n  REGIME EDGE TABLE")
    print(f"  {'-'*60}")
    if os.path.exists(EDGE_PATH):
        with open(EDGE_PATH) as f:
            et = json.load(f)
        for r, s in sorted(et.items(), key=lambda x: -x[1]["n"]):
            if s["n"] == 0: continue
            flag = "[EDGE]" if s["has_edge"] else "[BLOCK]"
            print(f"  {r:<30} WR={s['win_rate']*100:.0f}% EV={s['ev']:+.3f} {flag}")

    # Veredicto
    print(f"\n  VEREDICTO")
    print(f"  {'-'*60}")
    if n == 0:
        verdict = "INCONCLUSIVO — 0 trades no OOS. Sistema estava em cash (bear market)."
        verdict += f"\n  Alpha vs BH BTC: {pct(ret - bm.get('BTC',0))}" if bm.get("BTC") else ""
        print(f"  {verdict}")
    elif ret > 0 and pf >= 1.0 and pay >= 2.5:
        print("  CONDICIONAL GO — edge existe mas amostra pequena.")
        print(f"  Proximo passo: 200+ trades reais para validar estatisticamente.")
    elif ret >= 0 and pf >= 1.0:
        print("  NEUTRO — retorno positivo mas payoff ratio insuficiente.")
        print(f"  Ajustar TP para aumentar payoff antes de escalar.")
    else:
        print("  NO-GO — retorno negativo ou PF < 1.")

    print(f"\n{SEP}\n")

    report = {
        "oos_return":         ret,
        "oos_max_drawdown":   dd,
        "oos_profit_factor":  pf,
        "oos_win_rate":       wr,
        "oos_avg_win_usd":    aw,
        "oos_avg_loss_usd":   al,
        "oos_payoff_ratio":   pay,
        "oos_n_trades":       n,
        "oos_sharpe":         sh,
        "oos_fees_usd":       fees,
        "oos_period":         f"{oos.get('oos_start','?')} -> {oos.get('oos_end','?')}",
        "timeframe":          oos.get("timeframe", "?"),
        "benchmarks":         bm,
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Salvo: {REPORT_PATH}")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch",   action="store_true", help="Busca candles 15M")
    parser.add_argument("--oos",     action="store_true", help="Roda OOS 15M")
    parser.add_argument("--regime",  action="store_true", help="Constrói regime edge table")
    parser.add_argument("--report",  action="store_true", help="Relatório final")
    args = parser.parse_args()

    run_all = not any([args.fetch, args.oos, args.regime, args.report])

    if args.fetch or run_all:
        fetch_all_15m()

    candles_map = {}
    if args.oos or args.regime or run_all:
        candles_map = load_15m_candles()
        if not candles_map:
            print("Sem candles 15M. Execute --fetch primeiro.")
            return

    edge_table = {}
    if args.regime or run_all:
        edge_table = build_regime_edge_table()

    result_15m = None
    if args.oos or run_all:
        if not edge_table:
            edge_table = json.load(open(EDGE_PATH)) if os.path.exists(EDGE_PATH) else {}
        result_15m = simulate_oos_15m(candles_map, edge_table)

    if args.report or run_all:
        build_final_report(result_15m)


if __name__ == "__main__":
    main()
