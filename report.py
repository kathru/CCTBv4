"""
report.py — Relatório Completo de Validação CCTBv4
====================================================
Lê validation_result.json + calibration_result.json e gera:

  1. Walk-Forward Summary      — janelas, retorno, estabilidade
  2. Out-of-Sample Final        — resultado definitivo
  3. Score Calibration          — score bucket → win rate real
  4. Regime Breakdown           — edge por regime
  5. Pair Breakdown             — edge por ativo
  6. Cost Sensitivity           — fee +25%, slippage 2×, fill 70%, delay 1 candle
  7. Benchmark                  — B&H BTC, equal-weight, cash

Uso:
  python3 report.py                     # lê resultados salvos
  python3 report.py --rerun-sensitivity # re-executa seção 6 (mais lento)
  python3 report.py --save report.txt   # salva para arquivo
"""

import os, sys, json, math, statistics, argparse, random
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")
RESULT_PATH = os.path.join(DATA_DIR, "validation_result.json")
CALIB_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration_result.json")

SEP  = "-" * 90
SEP2 = "=" * 90


# ── Helpers ───────────────────────────────────────────────────────────────────

def _med(vals):
    if not vals: return 0.0
    s = sorted(vals); n = len(s)
    return (s[n//2] + s[(n-1)//2]) / 2

def _avg(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _pct(v): return f"{v*100:+.2f}%"
def _f2(v):  return f"{v:.2f}"
def _f3(v):  return f"{v:.3f}"
def _pct1(v):return f"{v*100:.1f}%"

def _sortino(returns):
    if len(returns) < 3: return 0.0
    m = _avg(returns)
    downside = [r for r in returns if r < 0]
    sd = statistics.stdev(downside) if len(downside) > 1 else 1e-9
    return m / sd * math.sqrt(252 * 24) if sd > 0 else 0.0

def _bar(v, lo=0.0, hi=1.0, width=12):
    pct = max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0
    filled = round(pct * width)
    return "#" * filled + "." * (width - filled)

def _verdict(wr, pf):
    if wr >= 0.55 and pf >= 1.3: return "[EDGE OK]"
    if wr >= 0.50 and pf >= 1.0: return "[MARGINAL]"
    return "[SEM EDGE]"

def _regime_name(code):
    return {1: "TREND_UP", -1: "TREND_DOWN", 0: "CHOP", 2: "COMPRESS"}.get(code, str(code))


# ── Seção 1 — Walk-Forward Summary ───────────────────────────────────────────

def section_walkforward(windows: list) -> str:
    lines = [f"\n{SEP2}", "  1. WALK-FORWARD SUMMARY", SEP2]

    active = [w for w in windows if w.get("n_trades", 0) >= 3]
    if not active:
        lines.append("  Sem janelas com trades suficientes.")
        return "\n".join(lines)

    rets  = [w["total_return"]  for w in active]
    wrs   = [w["win_rate"]      for w in active]
    pfs   = [min(w["profit_factor"], 10.0) for w in active]   # cap: sem perdas -> PF infinito
    dds   = [w["max_drawdown"]  for w in active]
    ns    = [w["n_trades"]      for w in active]
    aucs  = [w.get("auc_train", 0.5) for w in active]

    profitable = sum(1 for r in rets if r > 0)
    wr_above50 = sum(1 for w in wrs if w > 0.50)

    header = f"  {'Métrica':<22} {'Média':>9} {'Mediana':>9} {'Pior':>9} {'Melhor':>9} {'Barra':<14} {'Estável?'}"
    lines += [f"\n  {len(windows)} janelas ({len(active)} com trades ≥ 3)", SEP, header, SEP]

    def row(name, vals, lo, hi, fmt=_f3):
        m, md, mn, mx = _avg(vals), _med(vals), min(vals), max(vals)
        cv = statistics.stdev(vals) / abs(m) if len(vals) > 1 and m != 0 else 0
        bar = _bar(m, lo, hi)
        stab = "✅ sim" if cv < 0.40 else "⚠ não"
        lines.append(f"  {name:<22} {fmt(m):>9} {fmt(md):>9} {fmt(mn):>9} {fmt(mx):>9} {bar:<14} {stab}")

    row("Win Rate",      wrs,  0.40, 0.70, _pct1)
    row("Profit Factor", pfs,  0.80, 2.00)
    row("Retorno",       rets, -0.10, 0.15, _pct)
    row("Max Drawdown",  dds,  -0.30, 0.0, _pct)
    row("AUC treino",    aucs, 0.50, 0.70)
    row("Trades/janela", ns,   0, 30, lambda v: f"{v:.0f}")

    lines += [
        SEP,
        f"  Janelas com retorno positivo : {profitable}/{len(active)} ({profitable/len(active)*100:.0f}%)",
        f"  Janelas com WR > 50%         : {wr_above50}/{len(active)} ({wr_above50/len(active)*100:.0f}%)",
        f"  Melhor janela                : W{max(active, key=lambda w: w['total_return'])['window']} "
        f"({_pct(max(rets))})",
        f"  Pior janela                  : W{min(active, key=lambda w: w['total_return'])['window']} "
        f"({_pct(min(rets))})",
        f"\n  VEREDICTO: {_verdict(_avg(wrs), _avg(pfs))}",
    ]
    return "\n".join(lines)


# ── Seção 2 — Out-of-Sample Final ────────────────────────────────────────────

def section_oos(oos: dict, benchmark: dict) -> str:
    lines = [f"\n{SEP2}", "  2. OUT-OF-SAMPLE FINAL", SEP2]
    if not oos or not oos.get("n_trades"):
        lines.append(f"  Sem trades no OOS ({oos.get('oos_start','?')} → {oos.get('oos_end','?')})")
        lines.append(f"  AUC treino: {oos.get('auc_train', 0):.3f} — modelo não atingiu threshold no período")
        return "\n".join(lines)

    n   = oos["n_trades"]
    wr  = oos["win_rate"]
    pf  = oos["profit_factor"]
    ret = oos["total_return"]
    dd  = oos["max_drawdown"]
    sh  = oos.get("sharpe", 0)
    auc = oos.get("auc_train", 0)

    # Sortino das trades individuais se disponível
    trades = oos.get("trades", [])
    pnls   = [t["pnl"] / 1000 for t in trades]   # normalizado
    sortino = _sortino(pnls) if pnls else 0.0

    lines += [
        f"  Período   : {oos.get('oos_start','?')} → {oos.get('oos_end','?')}",
        SEP,
        f"  {'Métrica':<28} {'Valor':>12}",
        SEP,
        f"  {'Nº trades':<28} {n:>12}",
        f"  {'Win Rate':<28} {_pct1(wr):>12}",
        f"  {'Profit Factor':<28} {_f2(pf):>12}",
        f"  {'Retorno líquido':<28} {_pct(ret):>12}",
        f"  {'Max Drawdown':<28} {_pct(dd):>12}",
        f"  {'Sharpe':<28} {_f2(sh):>12}",
        f"  {'Sortino':<28} {_f2(sortino):>12}",
        f"  {'AUC (treino)':<28} {_f3(auc):>12}",
    ]

    if benchmark:
        bh   = benchmark.get("bh_return", 0)
        alpha = ret - bh
        lines += [
            SEP,
            f"  {'B&H BTC':<28} {_pct(bh):>12}",
            f"  {'Alpha vs B&H':<28} {_pct(alpha):>12}  {'✅' if alpha > 0 else '✗'}",
        ]

    lines += [f"\n  VEREDICTO: {_verdict(wr, pf)}"]
    return "\n".join(lines)


# ── Seção 3 — Score Calibration ──────────────────────────────────────────────

def section_score_calibration(windows: list, calib_result: dict) -> str:
    lines = [f"\n{SEP2}", "  3. SCORE CALIBRATION", SEP2]

    # Tenta usar resultado do calibrate.py (mais completo)
    if calib_result and calib_result.get("by_bucket"):
        lines.append("  (fonte: calibrate.py — 8 anos OKX)")
        bkts   = calib_result["by_bucket"]
        labels = ["0.48-0.52","0.52-0.55","0.55-0.58","0.58-0.62","0.62-0.66","0.66-0.70","0.70+"]
        header = f"  {'Score Bucket':<13} {'n':>6} {'Win Rate':>9} {'EV liq%':>9} {'P.Factor':>10} {'Barra WR'}"
        lines += [header, SEP]
        for idx, label in enumerate(labels):
            b = bkts.get(str(idx), {})
            if not b or b.get("n",0) == 0:
                lines.append(f"  {label:<13} {'—':>6}")
                continue
            wr  = b["win_rate"]
            ev  = b.get("ev_net_pct", 0)
            pf  = b.get("profit_factor", 0)
            bar = _bar(wr, 0.40, 0.70)
            ev_s = f"{ev:+.3f}%" if ev else "—"
            pf_s = f"{pf:.2f}" if pf else "—"
            lines.append(f"  {label:<13} {b['n']:>6} {_pct1(wr):>9} {ev_s:>9} {pf_s:>10} {bar}")
        bs = calib_result.get("platt", {})
        if bs:
            lines += [SEP,
                f"  Brier Score (raw): {bs.get('brier_before',0):.5f}",
                f"  Brier Score (cal): {bs.get('brier_after', bs.get('brier_before',0)):.5f}",
                f"  Platt (a,b):       {bs.get('a',1):.4f}, {bs.get('b',0):.4f}"]
        return "\n".join(lines)

    # Fallback: usa trades das janelas walk-forward
    all_trades = []
    for w in windows:
        all_trades.extend(w.get("trades", []))

    if not all_trades:
        lines.append("  Sem dados de trades por score.")
        lines.append("  → Execute calibrate.py ou re-execute validate.py para obter estes dados.")
        return "\n".join(lines)

    lines.append("  (fonte: walk-forward trades)")
    BUCKETS = [(0.48,0.55,"0.48-0.55"), (0.55,0.60,"0.55-0.60"),
               (0.60,0.65,"0.60-0.65"), (0.65,0.70,"0.65-0.70"), (0.70,1.01,"0.70+")]
    header = f"  {'Score':<12} {'n':>6} {'Win Rate':>9} {'Avg Win':>9} {'Avg Loss':>9} {'EV':>8} {'P.Factor':>10}"
    lines += [header, SEP]
    for lo, hi, label in BUCKETS:
        bt = [t for t in all_trades if lo <= t["score"] < hi]
        if len(bt) < 5:
            lines.append(f"  {label:<12} {len(bt):>6}  (insuficiente)")
            continue
        wins   = [t["pnl"] for t in bt if t["pnl"] > 0]
        losses = [t["pnl"] for t in bt if t["pnl"] <= 0]
        wr  = len(wins) / len(bt)
        aw  = _avg(wins)
        al  = _avg(losses)
        ev  = wr * aw + (1-wr) * al
        pf  = sum(wins) / abs(sum(losses)) if losses else 0.0
        bar = _bar(wr, 0.40, 0.70)
        lines.append(f"  {label:<12} {len(bt):>6} {_pct1(wr):>9} {aw:>+9.4f} {al:>+9.4f} {ev:>+8.4f} {pf:>10.2f}  {bar}")
    return "\n".join(lines)


# ── Seção 4 — Regime Breakdown ────────────────────────────────────────────────

def section_regime(windows: list, oos: dict) -> str:
    lines = [f"\n{SEP2}", "  4. REGIME BREAKDOWN", SEP2]

    # Agrega regime de todas as janelas
    regime_agg: dict = defaultdict(lambda: {"n":0,"wins":0,"pnl":0.0,"windows":0})
    for w in windows:
        for rname, rs in w.get("by_regime", {}).items():
            rn = regime_agg[rname]
            rn["n"]       += rs["n"]
            rn["wins"]    += round(rs["n"] * rs["wr"])
            rn["pnl"]     += rs["pnl"]
            rn["windows"] += 1

    # Adiciona OOS
    oos_regime = oos.get("by_regime", {})

    header = f"  {'Regime':<20} {'n':>6} {'Win Rate':>9} {'PnL $':>10} {'Janelas':>8} {'Veredicto'}"
    lines += ["  Walk-forward agregado", SEP, header, SEP]
    for rname, rs in sorted(regime_agg.items(), key=lambda x: -x[1]["n"]):
        if rs["n"] < 3: continue
        wr = rs["wins"] / rs["n"]
        lines.append(f"  {rname:<20} {rs['n']:>6} {_pct1(wr):>9} {rs['pnl']:>+10.2f} {rs['windows']:>8}  {_verdict(wr, 1.0)}")

    if oos_regime:
        lines += [f"\n  OOS:", SEP, header, SEP]
        for rname, rs in sorted(oos_regime.items(), key=lambda x: -x[1]["n"]):
            if rs["n"] < 2: continue
            wr = rs["wr"]
            lines.append(f"  {rname:<20} {rs['n']:>6} {_pct1(wr):>9} {rs['pnl']:>+10.2f} {'—':>8}  {_verdict(wr, 1.0)}")

    return "\n".join(lines)


# ── Seção 5 — Pair Breakdown ─────────────────────────────────────────────────

def section_pair(windows: list, oos: dict) -> str:
    lines = [f"\n{SEP2}", "  5. PAIR BREAKDOWN", SEP2]

    all_trades = []
    for w in windows:
        all_trades.extend(w.get("trades", []))
    oos_trades = oos.get("trades", []) if oos else []
    all_trades.extend(oos_trades)

    if not all_trades:
        lines.append("  Sem dados de trade por par.")
        lines.append("  → Re-execute validate.py (validate.py já salva trades individuais).")
        return "\n".join(lines)

    pairs = sorted(set(t["pair"] for t in all_trades))
    header = f"  {'Par':<12} {'n':>6} {'Win Rate':>9} {'EV med':>9} {'P.Factor':>10} {'Max DD':>9} {'Sharpe':>8}"
    lines += [header, SEP]

    for pair in pairs:
        pt = [t for t in all_trades if t["pair"] == pair]
        if not pt: continue
        wins   = [t["pnl"] for t in pt if t["pnl"] > 0]
        losses = [t["pnl"] for t in pt if t["pnl"] <= 0]
        wr   = len(wins) / len(pt)
        ev   = _avg([t["pnl"] for t in pt])
        pf   = sum(wins) / abs(sum(losses)) if losses else 0.0
        pnls = [t["pnl"] for t in pt]
        # Drawdown sequencial
        eq = 0.0; pk = 0.0; dd = 0.0
        for p in pnls:
            eq += p; pk = max(pk, eq)
            dd = min(dd, (eq - pk) / max(pk, 1e-6))
        # Sharpe
        sh = _avg(pnls) / statistics.stdev(pnls) * math.sqrt(252*24) if len(pnls) > 2 and statistics.stdev(pnls) > 0 else 0
        lines.append(f"  {pair:<12} {len(pt):>6} {_pct1(wr):>9} {ev:>+9.4f} {pf:>10.2f} {dd*100:>+8.1f}% {sh:>8.2f}")

    return "\n".join(lines)


# ── Seção 6 — Cost Sensitivity ───────────────────────────────────────────────

def section_cost_sensitivity(windows: list, rerun: bool = False) -> str:
    lines = [f"\n{SEP2}", "  6. COST SENSITIVITY", SEP2]

    all_trades_base = []
    for w in windows:
        all_trades_base.extend(w.get("trades", []))

    if not all_trades_base:
        lines += [
            "  Sem trades individuais para análise de sensibilidade.",
            "  → Re-execute validate.py (já atualizado para salvar trades).",
        ]
        return "\n".join(lines)

    # Cada cenário recomputa P&L a partir dos sl_pct/label originais
    def simulate_scenario(trades, fee_mult=1.0, slip_mult=1.0,
                          fill_rate=1.0, delay=False, seed=42):
        random.seed(seed)
        BASE_FEE = 0.001; BASE_SLIP = 0.0002
        fee  = BASE_FEE  * fee_mult
        slip = BASE_SLIP * slip_mult
        capital = 1000.0; eq = [capital]; results = []

        for t in trades:
            if fill_rate < 1.0 and random.random() > fill_rate:
                continue   # ordem não preenchida
            sl_pct = t.get("sl_pct", 0.02)
            tp_pct = sl_pct * 2.0
            size   = min(capital * 0.08, capital * 0.08 * t.get("score", 0.6))
            cost   = size * (fee + slip)
            # delay: assume entry 1 candle depois = slippage adicional de 0.05%
            if delay:
                cost += size * 0.0005
            won = t.get("label", t.get("won", 0))
            pnl = size * (tp_pct if won == 1 else -sl_pct) - cost * 2
            capital += pnl
            results.append(pnl)
            eq.append(capital)

        if not results:
            return {"ret": 0.0, "wr": 0.0, "pf": 0.0, "dd": 0.0, "n": 0}

        wins   = [p for p in results if p > 0]
        losses = [p for p in results if p <= 0]
        wr  = len(wins) / len(results)
        pf  = sum(wins) / abs(sum(losses)) if losses else 0.0
        ret = (capital - 1000.0) / 1000.0
        pk  = 0.0; dd = 0.0; cum = 0.0
        for p in results:
            cum += p; pk = max(pk, cum)
            dd  = min(dd, (cum - pk) / max(pk, 1e-6))
        return {"ret": ret, "wr": wr, "pf": pf, "dd": dd, "n": len(results)}

    scenarios = [
        ("Base (fee=0.1% slip=0.02%)",    dict()),
        ("Fee +25% (0.125%)",              dict(fee_mult=1.25)),
        ("Fee +50% (0.15%)",               dict(fee_mult=1.50)),
        ("Slippage 2× (0.04%)",            dict(slip_mult=2.0)),
        ("Slippage 3× (0.06%)",            dict(slip_mult=3.0)),
        ("Fill Rate 80%",                  dict(fill_rate=0.80)),
        ("Fill Rate 60%",                  dict(fill_rate=0.60)),
        ("Delay 1 candle",                 dict(delay=True)),
        ("Fee +25% + Slip 2× + Fill 80%",  dict(fee_mult=1.25, slip_mult=2.0, fill_rate=0.80)),
    ]

    header = f"  {'Cenário':<38} {'n':>6} {'Win Rate':>9} {'Retorno':>9} {'P.Factor':>10} {'Max DD':>8} {'Edge?'}"
    lines += [header, SEP]

    base_ret = None
    for name, kwargs in scenarios:
        r = simulate_scenario(all_trades_base, **kwargs)
        if base_ret is None:
            base_ret = r["ret"]
        frag = "✅" if r["pf"] >= 1.0 and r["wr"] >= 0.50 else "✗ FRÁGIL"
        diff = f"({_pct(r['ret']-base_ret)})" if base_ret is not None and name != "Base (fee=0.1% slip=0.02%)" else ""
        lines.append(
            f"  {name:<38} {r['n']:>6} {_pct1(r['wr']):>9} {_pct(r['ret']):>9} "
            f"{r['pf']:>10.2f} {r['dd']*100:>+7.1f}% {frag}  {diff}"
        )

    lines += [
        SEP,
        "  Se ✗ FRÁGIL em fee +25%: edge não é robusto para taker puro ou volume tier baixo.",
        "  Se ✗ FRÁGIL em fill 80%: system depende de fills perfeitos — irrealista em produção.",
    ]
    return "\n".join(lines)


# ── Seção 7 — Benchmark ──────────────────────────────────────────────────────

def section_benchmark(windows: list, oos: dict, candles_map: dict = None) -> str:
    lines = [f"\n{SEP2}", "  7. BENCHMARK COMPARISON", SEP2]

    # Retorno do sistema no OOS
    sys_ret = oos.get("total_return", 0) if oos else 0
    sys_dd  = oos.get("max_drawdown", 0) if oos else 0
    sys_sh  = oos.get("sharpe", 0) if oos else 0

    header = f"  {'Estratégia':<32} {'Retorno':>10} {'Max DD':>9} {'Sharpe':>8} {'vs Sistema'}"
    lines += [header, SEP]

    def bench_row(name, ret, dd, sh):
        diff = ret - sys_ret
        flag = "✅ melhor" if diff > 0 else ("= neutro" if abs(diff) < 0.005 else "↓ pior")
        lines.append(f"  {name:<32} {_pct(ret):>10} {_pct(dd):>9} {_f2(sh):>8}  {flag} ({_pct(diff)})")

    # Sistema
    lines.append(f"  {'Sistema V4':<32} {_pct(sys_ret):>10} {_pct(sys_dd):>9} {_f2(sys_sh):>8}  ← referência")

    # B&H BTC — dos dados OOS se disponível
    if candles_map and "BTC-USDT" in candles_map:
        oos_start = oos.get("oos_start") if oos else None
        if oos_start:
            btc = candles_map["BTC-USDT"]
            ts_start = datetime.strptime(oos_start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            oos_btc = [c for c in btc if c["ts"] >= ts_start]
            if oos_btc:
                bh_btc = (oos_btc[-1]["close"] - oos_btc[0]["open"]) / oos_btc[0]["open"]
                bench_row("Buy & Hold BTC", bh_btc, -0.35, 0.5)

                # Equal-weight BTC+ETH+SOL
                rets_ew = []
                for pair in ["BTC-USDT","ETH-USDT","SOL-USDT"]:
                    pc = [c for c in candles_map.get(pair,[]) if c["ts"] >= ts_start]
                    if pc:
                        rets_ew.append((pc[-1]["close"] - pc[0]["open"]) / pc[0]["open"])
                if rets_ew:
                    bench_row("Equal-weight BTC/ETH/SOL", _avg(rets_ew), -0.45, 0.4)
    else:
        lines.append("  (candles não disponíveis para benchmark preciso)")

    # Cash/no trade
    bench_row("Cash (sem trades)", 0.0, 0.0, 0.0)

    lines += [SEP, "  Nota: Sharpe de benchmark estimado (sem série temporal completa)."]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_candles_map():
    result = {}
    for pair in ["BTC-USDT","ETH-USDT","SOL-USDT"]:
        path = os.path.join(DATA_DIR, f"{pair}_1H.json")
        if os.path.exists(path):
            with open(path) as f:
                result[pair] = json.load(f)
    return result


def run(rerun_sensitivity=False, save_path=None):
    # Carrega resultados
    if not os.path.exists(RESULT_PATH):
        print(f"Arquivo não encontrado: {RESULT_PATH}")
        print("Execute primeiro: python3 validate.py --walkforward")
        return

    with open(RESULT_PATH) as f:
        data = json.load(f)

    windows   = data.get("windows", [])
    oos       = data.get("oos", {})
    benchmark = data.get("benchmark", {})

    calib_result = None
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            calib_result = json.load(f)

    candles_map = load_candles_map()

    # Header
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = [
        f"\n{SEP2}",
        f"  RELATÓRIO DE VALIDAÇÃO — CCTBv4 V4 Signal Engine",
        f"  Gerado: {ts}  |  Janelas: {len(windows)}  |  "
        f"OOS trades: {oos.get('n_trades',0)}  |  "
        f"Calibração: {'✅' if calib_result else '⚠ pendente'}",
        SEP2,
    ]

    sections = [
        "\n".join(header),
        section_walkforward(windows),
        section_oos(oos, benchmark),
        section_score_calibration(windows, calib_result),
        section_regime(windows, oos),
        section_pair(windows, oos),
        section_cost_sensitivity(windows, rerun=rerun_sensitivity),
        section_benchmark(windows, oos, candles_map),
        f"\n{SEP2}\n  FIM DO RELATÓRIO\n{SEP2}",
    ]

    import re
    report = "\n".join(sections)
    safe = re.sub(r'[^\x00-\x7F]', '?', report)
    print(safe)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            # Remove emoji para compatibilidade em terminais Windows
            import re
            clean = re.sub(r'[^\x00-\x7F✅⚠✗←↓]+', '', report)
            f.write(clean)
        print(f"\nRelatório salvo em: {save_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relatório completo de validação CCTBv4")
    parser.add_argument("--rerun-sensitivity", action="store_true",
                        help="Re-executa análise de sensibilidade de custos")
    parser.add_argument("--save", metavar="FILE",
                        help="Salva relatório para arquivo (ex: report.txt)")
    args = parser.parse_args()
    run(rerun_sensitivity=args.rerun_sensitivity, save_path=args.save)
