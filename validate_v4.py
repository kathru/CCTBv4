"""
validate_v4.py — Backtest usando o mesmo motor do runtime
=========================================================
Usa as peças EXATAS do runtime para garantir que o que se valida
é o que se opera.

  V4Orchestrator       — pipeline completa de decisão
  SimulatedExecutionEngine — fills realistas, fees, slippage
  FeeModel             — custos consistentes
  regime_engine        — 7 regimes (detect_regime)
  signal_engine        — score probabilístico + Platt calibration
  sizing_engine        — Kelly parcial × regime_mult
  thesis_invalidation  — saída por tese inválida (HARD/STRONG/SOFT)
  risk_prior           — VaR/ES bayesiano

O que é diferente em relação ao runtime:
  - orderbook/taker/OI/funding: substituídos por proxies de candles
    (dados tick não estão disponíveis historicamente — é conservador,
    não otimista: o sinal fica mais fraco sem orderflow real)
  - breadth: só usa alts_above_ema50 (sem CoinGecko/Binance histórico)
  - ciclo: 1H por candle (runtime é 15min)

Estrutura de walk-forward (igual ao validate.py original):
  6 meses treino (sem treino real — V4 não usa ML, só calibra prior)
  2 meses teste (execução V4 completa)
  1 mês step
  6 meses OOS final (completamente isolado)

Uso:
  python validate_v4.py                  # roda tudo
  python validate_v4.py --walkforward    # só walk-forward
  python validate_v4.py --report         # relatorio dos dados salvos
  python validate_v4.py --oos-only       # só OOS (rápido)
"""

import os, sys, json, time, math, statistics, argparse, random
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")
RESULT_PATH = os.path.join(DATA_DIR, "validation_v4_result.json")

PAIRS_HIST  = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
PAIRS_RT    = ["BTC-USD",  "ETH-USD",  "SOL-USD"]   # formato do runtime
PAIR_MAP    = dict(zip(PAIRS_HIST, PAIRS_RT))

INITIAL_CAPITAL = 1000.0
TRAIN_MONTHS    = 6
TEST_MONTHS     = 2
STEP_MONTHS     = 1
OOS_MONTHS      = 6

# Importa peças do runtime
from strategies.fee_model      import FEE
from strategies.risk_prior     import load_prior
from paper_trading.simulated_engine import SimulatedExecutionEngine
from dashboard.v4_orchestrator import V4Orchestrator


# ── Context histórico (sem I/O de exchange) ───────────────────────────────────

def _ema(values, span):
    if not values: return []
    k = 2 / (span + 1); r = [values[0]]
    for v in values[1:]: r.append(v * k + r[-1] * (1 - k))
    return r

def _atr(highs, lows, closes, period=14):
    if len(closes) < 2: return closes[-1] * 0.02 if closes else 0
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    return sum(trs[-period:]) / min(period, len(trs))

def build_historical_market_context(pair_rt: str, candles: list) -> dict:
    """
    Reconstrói market_context usando apenas candles OHLCV.
    Dados não disponíveis historicamente (orderbook, OI, funding) → neutral/zero.
    Isso é conservador: o edge observado será menor do que em runtime com dados reais.
    """
    if len(candles) < 25:
        return {
            "pair": pair_rt, "timestamp": int(time.time()),
            "realized_vol": 0.30, "vol_percentile": 0.50,
            "atr_rate": {"expansion": 1.0, "atr_pct": 0.02},
            "funding": {"funding_rate": 0.0, "sentiment": "neutral"},
            "open_interest": {"expanding": False, "oi_change_pct": 0.0},
            "orderbook": {"imbalance": 0.0, "spread_pct": 0.0002},
            "taker_ratio": {"imbalance": 0.0},
            "volume_delta": {"delta_pct": 0.0, "aggressive": False},
            "correlation": {},
        }

    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]

    # Volatilidade realizada
    rets = [math.log(closes[i]/closes[i-1]) for i in range(1, min(25, len(closes))) if closes[i-1] > 0]
    rvol = statistics.stdev(rets) * math.sqrt(24) if len(rets) > 2 else 0.02

    # BB width percentile
    if len(closes) >= 20:
        recent = closes[-20:]
        mid = sum(recent)/20; std = statistics.stdev(recent)
        bb_w = 2*std/mid if mid > 0 else 0
        widths = []
        for j in range(20, min(len(closes), 120)):
            s = closes[j-20:j]; m = sum(s)/20
            sd = statistics.stdev(s) if len(s) > 1 else 0
            widths.append(2*sd/m if m > 0 else 0)
        vol_pct = sum(1 for w in widths if w <= bb_w) / len(widths) if widths else 0.5
    else:
        vol_pct = 0.5

    # ATR expansion
    atr_now  = _atr(highs, lows, closes)
    atr_prev = _atr(highs[:-14], lows[:-14], closes[:-14]) if len(closes) > 28 else atr_now
    atr_exp  = atr_now / atr_prev if atr_prev > 0 else 1.0

    # Volume delta proxy (candle direction como proxy do taker)
    recent_c = candles[-10:]
    bv = sum(c["volume"] for c in recent_c if c["close"] >= c["open"])
    sv = sum(c["volume"] for c in recent_c if c["close"] <  c["open"])
    tv = bv + sv
    taker_imbal = (bv - sv) / tv if tv > 0 else 0.0
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
    vdelta  = (volumes[-1] - avg_vol) / avg_vol if avg_vol > 0 else 0.0

    return {
        "pair":       pair_rt,
        "timestamp":  int(time.time()),
        "realized_vol":   rvol,
        "vol_percentile": vol_pct,
        "atr_rate": {"expansion": atr_exp, "atr_pct": atr_now / closes[-1] if closes[-1] > 0 else 0.02},
        # Dados tick não disponíveis historicamente → neutral
        "funding":      {"funding_rate": 0.0, "sentiment": "neutral"},
        "open_interest":{"expanding": False, "oi_change_pct": 0.0},
        "orderbook":    {"imbalance": 0.0, "spread_pct": FEE.slippage_base * 2},
        "taker_ratio":  {"imbalance": taker_imbal},
        "volume_delta": {"delta_pct": vdelta, "aggressive": vdelta > 0.5},
        "correlation":  {},
    }


def resample_6h(candles_1h: list) -> list:
    """Agrega candles 1H em 6H para a Camada 2 do Regime Engine."""
    out = []
    for i in range(0, len(candles_1h) - 5, 6):
        grp = candles_1h[i:i+6]
        out.append({
            "ts":     grp[0]["ts"],
            "open":   grp[0]["open"],
            "high":   max(c["high"]  for c in grp),
            "low":    min(c["low"]   for c in grp),
            "close":  grp[-1]["close"],
            "volume": sum(c["volume"] for c in grp),
        })
    return out


# ── Monkey-patch: substitui get_market_context do runtime pelo histórico ───────

def _patch_orchestrator_context(v4: V4Orchestrator, candles_snapshot: dict):
    """
    Injeta contexto histórico nos métodos internos do V4Orchestrator.
    Usa monkey-patch no nível da instância para não afetar o runtime.
    """
    import data.market_data as _md

    def historical_get_market_context(pair, candles_1h, closes_map=None):
        return build_historical_market_context(pair, candles_1h)

    # Substitui apenas para esta instância via closure
    v4._historical_context_fn = historical_get_market_context


# ── Simulação de uma janela ───────────────────────────────────────────────────

def simulate_window(
    candles_map: dict,    # {pair_hist: [candles]}
    test_start_ts: int,
    test_end_ts:   int,
    window_id,
    step_hours: int = 1,
) -> dict:
    """
    Simula o runtime V4 sobre um período histórico.

    Passos:
      1. Inicializa V4Orchestrator + SimulatedExecutionEngine
      2. Para cada candle no período de teste:
         a. Constrói contexto histórico (sem I/O)
         b. Chama v4.evaluate() para cada par
         c. Executa decisão via engine
         d. Registra resultado
    """
    # Instâncias frescas por janela
    v4     = V4Orchestrator(state_dir=os.path.join(DATA_DIR, ".v4_state"))
    engine = SimulatedExecutionEngine(
        initial_balance_usd=INITIAL_CAPITAL,
        default_order_mode="adaptive",
        default_spread_pct=FEE.slippage_base * 2,
        seed=42,
    )
    engine.balance_usd   = INITIAL_CAPITAL
    engine.holdings      = {}
    engine.entry_prices  = {}
    engine.trades        = []
    engine.pending_orders = []
    engine.total_fees_usd = 0.0

    prior    = load_prior()
    slots    = {f"V4:{p}": {"qty": 0.0, "entry": 0.0, "peak": 0.0,
                             "sl_pct": 0.03, "sl_level": 0.0}
                for p in PAIRS_RT}

    # Pega índices do período de teste em BTC
    btc_hist    = candles_map.get("BTC-USDT", [])
    test_candles = [c for c in btc_hist if test_start_ts <= c["ts"] < test_end_ts]
    if not test_candles:
        return {"window": window_id, "n_trades": 0, "total_return": 0.0, "trades": []}

    all_trades   = []
    equity       = [INITIAL_CAPITAL]
    last_mc_ts   = 0

    for candle in test_candles[::step_hours]:
        ts_now = candle["ts"]

        for pair_hist, pair_rt in PAIR_MAP.items():
            candles_all = candles_map.get(pair_hist, [])
            # Contexto: apenas candles ANTERIORES ao candle atual (sem lookahead)
            ctx_candles = [c for c in candles_all if c["ts"] <= ts_now][-300:]
            if len(ctx_candles) < 50:
                continue

            ctx_6h     = resample_6h(ctx_candles)[-50:]
            closes_map = {p: [c["close"] for c in candles_map.get(p, []) if c["ts"] <= ts_now][-300:]
                          for p in PAIRS_HIST}

            # Override market_context via monkey-patch
            import data.market_data as _md
            _orig = _md.get_orderbook_context
            _md.get_orderbook_context = lambda pair: {"imbalance": 0.0, "spread_pct": 0.0002, "bid_depth": 0, "ask_depth": 0}
            _md.get_taker_ratio       = lambda pair: {"imbalance": build_historical_market_context(pair, ctx_candles)["taker_ratio"]["imbalance"], "buy_volume": 0, "sell_volume": 0}
            _md.get_funding_rate      = lambda pair: {"funding_rate": 0.0, "sentiment": "neutral"}
            _md.get_open_interest     = lambda pair: {"expanding": False, "oi_change_pct": 0.0, "oi_usd": 0}

            existing_slot = slots.get(f"V4:{pair_rt}")
            has_position  = existing_slot and existing_slot.get("qty", 0) > 0

            try:
                decision = v4.evaluate(
                    pair         = pair_rt,
                    candles_1h   = ctx_candles,
                    candles_6h   = ctx_6h,
                    closes_map   = {pair_rt: [c["close"] for c in ctx_candles]},
                    engine       = engine,
                    open_slots   = slots,
                    existing_slot= existing_slot if has_position else None,
                    breadth_score= 1.0,   # sem breadth histórico disponível
                )
            except Exception as e:
                continue
            finally:
                # Restaura funções originais
                _md.get_orderbook_context = _orig

            price = ctx_candles[-1]["close"] if ctx_candles else 0
            if not price:
                continue

            slot_key = f"V4:{pair_rt}"
            sym      = pair_rt.split("-")[0]

            if decision.get("decision") == "BUY" and not has_position:
                size_usd = decision.get("size_usd", 0)
                if size_usd > FEE._sym_params(pair_hist.replace("-","-")) if hasattr(FEE, '_sym_params') else 5.0:
                    if size_usd > 5.0 and engine.balance_usd >= size_usd:
                        ok = engine.buy(sym, size_usd, price, "V4:backtest",
                                        order_type="market",
                                        atr_pct=decision.get("signal",{}).get("factors",{}).get("volatility_expansion",{}).get("context",{}).get("atr_expansion",0.015) if isinstance(decision.get("signal",{}).get("factors",{}).get("volatility_expansion",{}), dict) else 0.015)
                        if ok:
                            sl = decision.get("execution") and decision["execution"].stop_loss or price * 0.97
                            sl_pct = abs(price - sl) / price if sl else 0.03
                            slots[slot_key] = {
                                "qty": size_usd / price, "entry": price,
                                "peak": price, "entry_usd": size_usd,
                                "sl_pct": sl_pct,
                                "sl_level": sl or price * (1 - sl_pct),
                            }
                            all_trades.append({
                                "pair": pair_rt, "ts": ts_now, "side": "BUY",
                                "price": price, "size_usd": size_usd,
                                "score": decision.get("score", 0),
                                "regime": decision.get("regime").regime if hasattr(decision.get("regime"), "regime") else "UNKNOWN",
                                "direction": decision.get("direction", "long"),
                                "reason": decision.get("reason", "")[:60],
                            })

            elif decision.get("decision") == "SELL" and has_position:
                qty = existing_slot.get("qty", 0)
                if qty > 0:
                    ok = engine.sell(sym, qty, price, "V4:backtest", order_type="market")
                    if ok:
                        entry    = existing_slot.get("entry", price)
                        pnl      = (price - entry) / entry * existing_slot.get("entry_usd", qty * price)
                        exit_type = decision.get("exit_type", "signal")
                        all_trades.append({
                            "pair": pair_rt, "ts": ts_now, "side": "SELL",
                            "price": price, "pnl": round(pnl, 4),
                            "score": decision.get("score", 0),
                            "regime": decision.get("regime").regime if hasattr(decision.get("regime"), "regime") else "UNKNOWN",
                            "exit_type": exit_type,
                            "reason": decision.get("reason", "")[:80],
                        })
                        slots[slot_key] = {"qty": 0.0, "entry": 0.0, "peak": 0.0,
                                           "sl_pct": 0.03, "sl_level": 0.0}

        # Processa tick de ordens pendentes
        prices_tick = {pair_rt.split("-")[0]: candles_map.get(pair_hist, [{}])[-1].get("close", 0)
                       for pair_hist, pair_rt in PAIR_MAP.items()}
        engine.tick(prices_tick)

        equity.append(engine.portfolio_value())

    # Métricas finais
    sells    = [t for t in all_trades if t.get("side") == "SELL"]
    n        = len(sells)
    wins     = [t for t in sells if t.get("pnl", 0) > 0]
    losses   = [t for t in sells if t.get("pnl", 0) <= 0]
    wr       = len(wins) / n if n > 0 else 0.0
    g_win    = sum(t["pnl"] for t in wins)
    g_loss   = abs(sum(t["pnl"] for t in losses)) or 1e-6
    pf       = g_win / g_loss
    ret      = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    peak  = INITIAL_CAPITAL; max_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = (v - peak) / peak
        if dd < max_dd: max_dd = dd

    pnl_series = [equity[i] - equity[i-1] for i in range(1, len(equity))]
    sharpe = 0.0
    if len(pnl_series) > 10:
        m = statistics.mean(pnl_series); s = statistics.stdev(pnl_series)
        sharpe = m / s * math.sqrt(252 * 24) if s > 0 else 0.0

    # Sortino
    down = [p for p in pnl_series if p < 0]
    sortino = 0.0
    if down and statistics.stdev(down) > 0:
        sortino = statistics.mean(pnl_series) / statistics.stdev(down) * math.sqrt(252 * 24)

    # Por regime e par
    by_regime: dict = defaultdict(lambda: {"n":0,"wins":0,"pnl":0.0})
    by_pair:   dict = defaultdict(lambda: {"n":0,"wins":0,"pnl":0.0})
    for t in sells:
        r = t.get("regime", "UNKNOWN"); p = t.get("pair", "?")
        pnl = t.get("pnl", 0)
        by_regime[r]["n"] += 1; by_pair[p]["n"] += 1
        if pnl > 0:
            by_regime[r]["wins"] += 1; by_pair[p]["wins"] += 1
        by_regime[r]["pnl"] += pnl;    by_pair[p]["pnl"] += pnl

    # Adiciona win_rate
    for d in list(by_regime.values()) + list(by_pair.values()):
        d["wr"] = round(d["wins"] / d["n"], 3) if d["n"] > 0 else 0.0

    start_str = datetime.utcfromtimestamp(test_start_ts).strftime("%Y-%m")
    end_str   = datetime.utcfromtimestamp(test_end_ts).strftime("%Y-%m")
    print(f"  W{str(window_id):<3} {start_str}->{end_str} | "
          f"n={n} WR={wr*100:.0f}% PF={pf:.2f} ret={ret*100:+.1f}% "
          f"DD={max_dd*100:.1f}% Sh={sharpe:.1f}")

    return {
        "window": window_id, "test_start": start_str, "test_end": end_str,
        "n_trades": n, "win_rate": round(wr, 4),
        "profit_factor": round(min(pf, 20.0), 3),
        "total_return": round(ret, 4), "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "capital_final": round(equity[-1], 2),
        "total_fees_usd": round(engine.total_fees_usd, 4),
        "by_regime": {k: dict(v) for k, v in by_regime.items()},
        "by_pair":   {k: dict(v) for k, v in by_pair.items()},
        "trades": [t for t in all_trades if t.get("side") == "SELL"],
        "exec_stats": engine.execution_stats(),
    }


# ── Walk-forward ──────────────────────────────────────────────────────────────

def months_to_seconds(n): return n * 30 * 24 * 3600

def run_walkforward(candles_map: dict) -> list:
    btc = candles_map.get("BTC-USDT", [])
    if not btc: return []

    t_start  = btc[0]["ts"]
    t_end    = btc[-1]["ts"]
    oos_cut  = t_end - months_to_seconds(OOS_MONTHS)

    windows = []
    t        = t_start + months_to_seconds(TRAIN_MONTHS)
    win_id   = 1

    while t + months_to_seconds(TEST_MONTHS) <= oos_cut:
        test_end = t + months_to_seconds(TEST_MONTHS)
        result   = simulate_window(candles_map, t, test_end, win_id)
        windows.append(result)
        t       += months_to_seconds(STEP_MONTHS)
        win_id  += 1

    return windows


def run_oos(candles_map: dict) -> dict:
    btc      = candles_map.get("BTC-USDT", [])
    oos_cut  = btc[-1]["ts"] - months_to_seconds(OOS_MONTHS)
    result   = simulate_window(candles_map, oos_cut, btc[-1]["ts"], "OOS")
    result["oos_start"] = datetime.utcfromtimestamp(oos_cut).strftime("%Y-%m-%d")
    result["oos_end"]   = datetime.utcfromtimestamp(btc[-1]["ts"]).strftime("%Y-%m-%d")
    return result


def calc_benchmarks(candles_map: dict) -> dict:
    btc = candles_map.get("BTC-USDT", [])
    if not btc: return {}
    oos_cut  = btc[-1]["ts"] - months_to_seconds(OOS_MONTHS)
    benchmarks = {}
    for pair, label in [("BTC-USDT","BTC"), ("ETH-USDT","ETH"), ("SOL-USDT","SOL")]:
        oos = [c for c in candles_map.get(pair,[]) if c["ts"] >= oos_cut]
        if oos:
            ret = (oos[-1]["close"] - oos[0]["open"]) / oos[0]["open"]
            benchmarks[label] = round(ret, 4)
    if benchmarks:
        ew = sum(benchmarks.values()) / len(benchmarks)
        benchmarks["equal_weight"] = round(ew, 4)
    return benchmarks


# ── Report ────────────────────────────────────────────────────────────────────

def print_v4_report(windows: list, oos: dict, benchmarks: dict):
    SEP  = "-" * 90
    SEP2 = "=" * 90
    pct  = lambda v: f"{v*100:+.2f}%"

    print(f"\n{SEP2}")
    print(f"  VALIDACAO V4 — MESMO MOTOR DO RUNTIME")
    print(f"  {len(windows)} janelas walk-forward | OOS: {oos.get('oos_start','?')} -> {oos.get('oos_end','?')}")
    print(f"  SimulatedExecutionEngine + V4Orchestrator + FeeModel + ThesisInvalidation")
    print(SEP2)

    # Walk-forward
    active = [w for w in windows if w.get("n_trades",0) >= 2]
    if active:
        rets  = [w["total_return"]   for w in active]
        wrs   = [w["win_rate"]       for w in active]
        pfs   = [min(w["profit_factor"],20.0) for w in active]
        dds   = [w["max_drawdown"]   for w in active]
        shs   = [w["sharpe"]         for w in active]

        avg  = lambda v: sum(v)/len(v) if v else 0
        med  = lambda v: sorted(v)[len(v)//2] if v else 0

        print(f"\n  1. WALK-FORWARD ({len(windows)} janelas, {len(active)} com >=2 trades)")
        print(SEP)
        print(f"  {'Metrica':<22} {'Media':>8} {'Mediana':>8} {'Pior':>8} {'Melhor':>8}")
        print(SEP)
        for name, vals in [
            ("Win Rate",      wrs),
            ("Profit Factor", pfs),
            ("Retorno",       rets),
            ("Max Drawdown",  dds),
            ("Sharpe",        shs),
        ]:
            m=avg(vals); md=med(vals); mn=min(vals); mx=max(vals)
            fmt = (lambda v: f"{v*100:.1f}%") if name in ("Win Rate","Retorno","Max Drawdown") else (lambda v: f"{v:.3f}")
            print(f"  {name:<22} {fmt(m):>8} {fmt(md):>8} {fmt(mn):>8} {fmt(mx):>8}")
        print(SEP)
        profitable = sum(1 for r in rets if r > 0)
        print(f"  Janelas lucrativas: {profitable}/{len(active)} ({profitable/len(active)*100:.0f}%)")
        print(f"  Janelas WR>50%:     {sum(1 for w in wrs if w>0.50)}/{len(active)}")

    # OOS
    print(f"\n  2. OUT-OF-SAMPLE FINAL ({oos.get('oos_start','?')} -> {oos.get('oos_end','?')})")
    print(SEP)
    if oos.get("n_trades",0) == 0:
        print(f"  Sem trades no OOS — threshold nao atingido no periodo")
    else:
        for k, v in [
            ("Trades",        oos["n_trades"]),
            ("Win Rate",      f"{oos['win_rate']*100:.1f}%"),
            ("Profit Factor", f"{oos['profit_factor']:.2f}"),
            ("Retorno",       pct(oos["total_return"])),
            ("Max Drawdown",  pct(oos["max_drawdown"])),
            ("Sharpe",        f"{oos['sharpe']:.2f}"),
            ("Sortino",       f"{oos.get('sortino',0):.2f}"),
            ("Fees pagas",    f"US$ {oos.get('total_fees_usd',0):.4f}"),
        ]:
            print(f"  {k:<20} {str(v):>12}")

    # Regime OOS
    if oos.get("by_regime"):
        print(f"\n  4. REGIME (OOS)")
        print(SEP)
        print(f"  {'Regime':<28} {'n':>5} {'WR':>7} {'PnL':>10} {'Exit types'}")
        print(SEP)
        for r, s in sorted(oos["by_regime"].items(), key=lambda x: -x[1]["n"]):
            print(f"  {r:<28} {s['n']:>5} {s['wr']*100:>6.0f}% {s['pnl']:>+10.3f}")

    # Pair OOS
    if oos.get("by_pair"):
        print(f"\n  5. PAR (OOS)")
        print(SEP)
        for p, s in sorted(oos["by_pair"].items(), key=lambda x: -x[1]["n"]):
            print(f"  {p:<12} n={s['n']:>4} WR={s['wr']*100:.0f}% PnL={s['pnl']:>+.3f}")

    # Exit types
    oos_trades = oos.get("trades", [])
    if oos_trades:
        exit_counts: dict = defaultdict(int)
        for t in oos_trades:
            exit_counts[t.get("exit_type","signal")] += 1
        print(f"\n  EXIT TYPES (OOS)")
        print(SEP)
        for et, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"  {et:<30} {cnt:>5} trades")

    # Exec stats
    es = oos.get("exec_stats", {})
    if es:
        print(f"\n  QUALIDADE DE EXECUCAO (OOS)")
        print(SEP)
        print(f"  Fill rate:          {es.get('fill_rate',0)*100:.1f}%")
        print(f"  Slippage medio:     {es.get('avg_slippage_bps',0):.2f} bps")
        print(f"  Slippage total:     US$ {es.get('total_slippage_usd',0):.4f}")
        print(f"  Maker/Taker fees:   US$ {es.get('total_maker_fee',0):.4f} / US$ {es.get('total_taker_fee',0):.4f}")

    # Benchmark
    if benchmarks:
        oos_ret = oos.get("total_return", 0)
        print(f"\n  7. BENCHMARK vs OOS")
        print(SEP)
        print(f"  {'Estrategia':<30} {'Retorno':>10} {'Alpha':>10}")
        print(SEP)
        print(f"  {'Sistema V4':<30} {pct(oos_ret):>10} {'<- referencia':>10}")
        for label, ret in benchmarks.items():
            alpha = oos_ret - ret
            flag  = "(melhor)" if alpha > 0.01 else ("(pior)" if alpha < -0.01 else "(neutro)")
            print(f"  {label:<30} {pct(ret):>10} {pct(alpha):>10} {flag}")
        print(f"  {'Cash':<30} {pct(0.0):>10} {pct(oos_ret):>10}")

    print(f"\n{SEP2}")
    print("  FIM DO RELATORIO V4")
    print(f"{SEP2}\n")


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_candles_map() -> dict:
    result = {}
    for pair in PAIRS_HIST:
        path = os.path.join(DATA_DIR, f"{pair}_1H.json")
        if os.path.exists(path):
            with open(path) as f:
                result[pair] = json.load(f)
            print(f"  {pair}: {len(result[pair])} candles")
    return result


def save_result(windows, oos, benchmarks):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump({
            "windows": windows, "oos": oos, "benchmarks": benchmarks,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine": "V4Orchestrator+SimulatedExecutionEngine",
        }, f, indent=2)
    print(f"\n  Resultado salvo: {RESULT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validacao V4 — mesmo motor do runtime")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--oos-only",    action="store_true")
    parser.add_argument("--report",      action="store_true")
    args = parser.parse_args()

    run_all = not any([args.walkforward, args.oos_only, args.report])

    if args.report:
        if not os.path.exists(RESULT_PATH):
            print("Resultado nao encontrado. Execute validate_v4.py primeiro.")
            return
        with open(RESULT_PATH) as f:
            data = json.load(f)
        print_v4_report(data["windows"], data["oos"], data.get("benchmarks", {}))
        return

    print("\n[V4] Carregando candles historicos...")
    candles_map = load_candles_map()
    if not candles_map:
        print("Sem candles. Execute: python validate.py --fetch")
        return

    windows, oos = [], {}

    if args.walkforward or run_all:
        print(f"\n[V4] Walk-forward ({TRAIN_MONTHS}m treino / {TEST_MONTHS}m teste / {STEP_MONTHS}m step)...")
        windows = run_walkforward(candles_map)

    if args.oos_only or run_all:
        print(f"\n[V4] Out-of-Sample (ultimos {OOS_MONTHS} meses)...")
        oos = run_oos(candles_map)

    benchmarks = calc_benchmarks(candles_map)
    save_result(windows, oos, benchmarks)
    print_v4_report(windows, oos, benchmarks)


if __name__ == "__main__":
    main()
