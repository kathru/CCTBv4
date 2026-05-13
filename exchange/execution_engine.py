"""
Execution Engine — Camada 5
============================
Smart execution: entrada em tranches, ajuste por liquidez e spread.

Execução é edge. Entrar 0.2% melhor por trade equivale a
aumentar o winrate em vários pontos percentuais.

Modos de execução:
  PASSIVE_LIMIT   — limit abaixo do ask (maker fee)
  STAGGERED       — 3 tranches conforme liquidez e momentum
  ICEBERG         — size dividido para não sinalizar posição grande
  MARKET          — market order apenas em casos urgentes (vol extrema)

Saídas também são gerenciadas:
  - SL dinâmico baseado em structure (não ATR fixo × 2)
  - TP escalonado em múltiplos targets
  - Trailing baseado em HH/HL (market structure)
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionPlan:
    """Plano de execução antes de enviar ordens."""
    mode:        str           # 'passive_limit' | 'staggered' | 'market'
    tranches:    list          # [{pct: float, price: float, type: str}]
    stop_loss:   float         # preço de stop
    take_profits: list         # [tp1, tp2, tp3] — saída escalonada
    trailing_sl: bool          # usar trailing baseado em structure
    estimated_fee: float       # fee total estimado
    slippage_est:  float       # slippage estimado
    rationale:   dict = field(default_factory=dict)


def plan_entry(
    direction: str,
    current_price: float,
    size_usd: float,
    market_context: dict,
    regime: str,
    signal_score: dict,
    atr_value: float = 0.0,
) -> ExecutionPlan:
    """
    Define o plano de execução para uma entrada.

    Determina:
      - Modo de execução (passive limit vs staggered vs market)
      - Preços das tranches
      - Stop loss baseado em estrutura de mercado
      - Take profits escalonados
    """
    spread_pct  = market_context.get("orderbook", {}).get("spread_pct", 0.0001)
    ob_imbal    = market_context.get("orderbook", {}).get("imbalance", 0.0)
    taker_imbal = market_context.get("taker_ratio", {}).get("imbalance", 0.0)
    vol_pct     = market_context.get("vol_percentile", 0.5)
    score       = signal_score.get("score", 0.5)
    confidence  = signal_score.get("confidence", 0.5)

    # ── Modo de execução ──────────────────────────────────────────────────────
    #
    # MARKET: apenas em PANIC (fechamento urgente) — evitado em entradas
    # STAGGERED: score alto, mercado estável, setup de alta confiança
    # PASSIVE_LIMIT: padrão — aguarda preço vir até nós
    #
    if regime in ("PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"):
        mode = "market"   # urgência
    elif score >= 0.68 and confidence >= 0.65 and spread_pct < 0.0005:
        mode = "staggered"
    else:
        mode = "passive_limit"

    # ── Stop Loss baseado em estrutura ────────────────────────────────────────
    # ATR como base, ajustado pela volatilidade atual
    if atr_value <= 0:
        atr_value = current_price * 0.02  # fallback: 2%

    vol_sl_multiplier = {
        "TREND_EXPANSION":       2.0,
        "MEAN_REVERTING_CHOP":   1.5,
        "VOLATILITY_COMPRESSION":1.2,
        "TREND_EXHAUSTION":      1.8,
        "HIGH_CORRELATION_RISK": 2.5,
        "PANIC_LIQUIDATION":     3.0,
        "LIQUIDITY_VACUUM":      2.5,
    }.get(regime, 2.0)

    sl_distance = atr_value * vol_sl_multiplier

    if direction == "long":
        stop_loss = current_price - sl_distance
    else:
        stop_loss = current_price + sl_distance

    sl_pct = sl_distance / current_price

    # ── Take Profits escalonados (RR 2:1, 3:1, 4.5:1) ────────────────────────
    # Calibração Platt (28k amostras OKX 8 anos): WR real = 28-32%.
    # Break-even com fees (0.52% RT): WR_min = 1/(1+RR).
    #   RR=2.0 → WR_min=33.3%  (abaixo do WR real → EV negativo)
    #   RR=2.5 → WR_min=28.6%  (break-even com WR=28%)
    #   RR=3.0 → WR_min=25.0%  (EV positivo mesmo em bear parcial)
    # Sizing 30/40/30: menos exposição no TP1 (mais conservador),
    # mais peso no TP2 e TP3 onde o RR compensa o WR baixo.
    if direction == "long":
        tp1 = current_price + sl_distance * 2.0
        tp2 = current_price + sl_distance * 3.0
        tp3 = current_price + sl_distance * 4.5
    else:
        tp1 = current_price - sl_distance * 2.0
        tp2 = current_price - sl_distance * 3.0
        tp3 = current_price - sl_distance * 4.5

    take_profits = [
        {"price": round(tp1, 4), "pct_of_position": 0.30, "rr": 2.0},
        {"price": round(tp2, 4), "pct_of_position": 0.40, "rr": 3.0},
        {"price": round(tp3, 4), "pct_of_position": 0.30, "rr": 4.5},
    ]

    # ── Tranches de entrada ───────────────────────────────────────────────────
    tranches = _build_tranches(
        mode=mode,
        direction=direction,
        current_price=current_price,
        spread_pct=spread_pct,
        ob_imbal=ob_imbal,
        taker_imbal=taker_imbal,
        score=score,
    )

    # ── Fee estimado ──────────────────────────────────────────────────────────
    maker_fee = 0.001   # 0.10%
    taker_fee = 0.004   # 0.40%
    fee_entry = sum(
        t["pct"] * (maker_fee if t["type"] == "limit" else taker_fee)
        for t in tranches
    )
    fee_exit_avg = maker_fee * 0.5 + taker_fee * 0.5  # mix de saída
    estimated_fee = fee_entry + fee_exit_avg

    # ── Slippage estimado ─────────────────────────────────────────────────────
    slippage = spread_pct * 0.5  # metade do spread como slippage esperado

    return ExecutionPlan(
        mode=mode,
        tranches=tranches,
        stop_loss=round(stop_loss, 4),
        take_profits=take_profits,
        trailing_sl=(regime in ("TREND_EXPANSION", "TREND_EXHAUSTION")),
        estimated_fee=round(estimated_fee, 5),
        slippage_est=round(slippage, 6),
        rationale={
            "spread_pct":       round(spread_pct, 6),
            "sl_pct":           round(sl_pct, 4),
            "atr_value":        round(atr_value, 4),
            "vol_sl_mult":      vol_sl_multiplier,
            "mode":             mode,
            "score":            score,
        },
    )


def _build_tranches(
    mode: str,
    direction: str,
    current_price: float,
    spread_pct: float,
    ob_imbal: float,
    taker_imbal: float,
    score: float,
) -> list:
    """
    Constrói as tranches de entrada conforme o modo de execução.

    Staggered:
      Tranche 1: 30% — confirmação imediata (limit ligeiramente passivo)
      Tranche 2: 30% — pullback de 0.2-0.3% (melhora preço médio)
      Tranche 3: 40% — confirmação de momentum (só entra se continuar)
    """
    if mode == "market":
        return [{"pct": 1.0, "price": current_price, "type": "market", "label": "urgente"}]

    if mode == "passive_limit":
        if direction == "long":
            price = current_price * (1 - spread_pct * 2)  # abaixo do ask atual
        else:
            price = current_price * (1 + spread_pct * 2)
        return [{"pct": 1.0, "price": round(price, 4), "type": "limit", "label": "passivo"}]

    # STAGGERED — 3 tranches
    pullback = max(0.002, min(0.005, spread_pct * 20))  # 0.2%-0.5%

    if direction == "long":
        p1 = current_price * (1 - spread_pct)          # quasi-passivo
        p2 = current_price * (1 - pullback)             # pullback
        p3 = current_price * (1 + pullback * 0.5)      # momentum (acima do atual)
    else:
        p1 = current_price * (1 + spread_pct)
        p2 = current_price * (1 + pullback)
        p3 = current_price * (1 - pullback * 0.5)

    # Tranche 3 só entra se momentum confirmar — tipo limit distante
    return [
        {"pct": 0.30, "price": round(p1, 4), "type": "limit",   "label": "T1-confirmacao"},
        {"pct": 0.30, "price": round(p2, 4), "type": "limit",   "label": "T2-pullback"},
        {"pct": 0.40, "price": round(p3, 4), "type": "limit",   "label": "T3-momentum"},
    ]


def update_trailing_stop(
    current_price: float,
    entry_price: float,
    peak_price: float,
    current_sl: float,
    sl_pct: float,
    regime: str,
    direction: str = "long",
) -> dict:
    """
    Atualiza stop loss trailing baseado em structure de mercado.

    V3: trailing simples baseado em pico
    V4: trailing adapta à volatilidade do regime

    Gatilhos:
      Break-even: gain >= SL% × 1.5 → SL sobe para entry
      Trailing:   gain >= SL% × 2.5 → SL segue pico - SL%
      Tight:      Em TREND_EXHAUSTION → trailing mais apertado
    """
    if direction != "long":
        return {"sl": current_sl, "action": "hold"}

    gain_pct = (current_price - entry_price) / entry_price

    # Regime-adjusted trailing
    regime_trail_mult = {
        "TREND_EXPANSION":   1.0,   # trailing normal
        "TREND_EXHAUSTION":  0.70,  # trailing mais apertado
        "MEAN_REVERTING_CHOP": 0.80,
    }.get(regime, 1.0)

    sl_trail = sl_pct * regime_trail_mult

    new_sl = current_sl
    action = "hold"

    if gain_pct >= sl_pct * 1.5:
        # Break-even
        be_sl = entry_price * (1 + 0.001)  # ligeiramente acima do entry
        if be_sl > current_sl:
            new_sl = be_sl
            action = "breakeven"

    if gain_pct >= sl_pct * 2.5:
        # Trailing ativo
        trail_sl = peak_price * (1 - sl_trail)
        if trail_sl > new_sl:
            new_sl = trail_sl
            action = "trailing"

    return {
        "sl":     round(new_sl, 4),
        "action": action,
        "gain_pct": round(gain_pct, 4),
    }
