"""
Sizing Engine — Camada 4
========================
Position sizing institucional:

  Size ∝ Edge × Confidence / Volatility × Correlation Risk

Substitui o sizing fixo por regime (10% BULL, 7% CHOP, 5% BEAR)
por uma função contínua que reflete a qualidade real do setup.

Você aumenta posição quando:
  - Edge aumenta (score alto)
  - Correlação entre pares diminui (diversificação real)
  - Volatilidade é saudável (não em pânico)

Você reduz quando:
  - Caos sistêmico (correlação alta + vol extrema)
  - Confiança baixa nos dados
  - Kelly fraction sugere tamanho menor
"""

import math


# Limites de sizing
MIN_SIZE_PCT = 0.02   # 2% mínimo por trade
MAX_SIZE_PCT = 0.15   # 15% máximo por trade
BASE_SIZE_PCT = 0.10  # 10% base antes dos ajustes (backtest: sistema saudável, aumentar sizing)


def compute_position_size(
    signal_score: dict,
    regime: str,
    portfolio_context: dict,
    vol_percentile: float = 0.5,
    correlation_risk: float = 0.5,
    portfolio_value: float = 10000.0,
) -> dict:
    """
    Calcula o tamanho da posição usando a fórmula institucional.

    Parâmetros:
      signal_score      — output da Camada 3 (score, kelly_fraction, confidence, ev)
      regime            — regime atual da Camada 2
      portfolio_context — contexto de portfolio (VaR used, open positions, etc.)
      vol_percentile    — percentil de volatilidade atual (0-1)
      correlation_risk  — risco de correlação entre pares abertos (0-1)
      portfolio_value   — valor total do portfolio em USD

    Retorna:
      size_pct     — tamanho como % do portfolio
      size_usd     — tamanho em USD
      rationale    — dict explicando cada fator de ajuste
    """
    score      = signal_score.get("score", 0.5)
    confidence = signal_score.get("confidence", 0.5)
    kelly      = signal_score.get("kelly_fraction", 0.05)
    ev         = signal_score.get("expected_value", 0.0)
    direction  = signal_score.get("direction", "neutral")

    # ── Sem entrada se EV negativo ou direção neutra ──────────────────────────
    if ev <= 0 or direction == "neutral":
        return {
            "size_pct":  0.0,
            "size_usd":  0.0,
            "blocked":   True,
            "reason":    f"EV negativo ({ev:.4f}) ou direção neutra",
            "rationale": {},
        }

    # ── Componente Edge × Confidence ─────────────────────────────────────────
    edge_factor = score * confidence

    # ── Componente Volatility (denominador) ──────────────────────────────────
    # Vol percentil alto = penalidade; vol saudável (0.3-0.6) = neutro
    if vol_percentile < 0.20:
        vol_factor = 0.70  # muito comprimida — reduz um pouco (incerteza de direção)
    elif vol_percentile < 0.60:
        vol_factor = 1.0   # zona saudável
    elif vol_percentile < 0.80:
        vol_factor = 0.75  # elevada
    elif vol_percentile < 0.90:
        vol_factor = 0.50  # muito elevada
    else:
        vol_factor = 0.25  # extrema — modo defensivo

    # ── Componente Correlation Risk (denominador) ─────────────────────────────
    # Correlação alta = pares não diversificam → penaliza tamanho
    if correlation_risk < 0.50:
        corr_factor = 1.20  # baixa correlação = diversificação real → bônus
    elif correlation_risk < 0.70:
        corr_factor = 1.0
    elif correlation_risk < 0.85:
        corr_factor = 0.75
    elif correlation_risk < 0.92:
        corr_factor = 0.50
    else:
        corr_factor = 0.25  # correlação sistêmica extrema

    # ── Fórmula central: Size ∝ Edge × Confidence / Vol × CorrelationRisk ────
    if vol_factor * (1 / corr_factor if corr_factor > 0 else 4) > 0:
        raw_size = BASE_SIZE_PCT * edge_factor / (vol_factor * (1 / corr_factor))
    else:
        raw_size = 0.0

    # Simplificando: numerador = edge_factor, denominador = vol_penalty × corr_penalty
    vol_corr_denominator = (1 / vol_factor) * (1 / corr_factor) if vol_factor > 0 and corr_factor > 0 else 4
    raw_size = BASE_SIZE_PCT * edge_factor / max(vol_corr_denominator, 0.25)

    # ── Ajuste por Kelly ──────────────────────────────────────────────────────
    # Kelly sugere tamanho máximo — usamos como cap superior
    kelly_cap = min(kelly, MAX_SIZE_PCT)
    size = min(raw_size, kelly_cap)

    # ── Ajuste por regime ─────────────────────────────────────────────────────
    regime_mult = {
        "TREND_EXPANSION":       1.10,
        "TREND_EXHAUSTION":      0.70,
        "VOLATILITY_COMPRESSION":0.75,
        "PANIC_LIQUIDATION":     0.0,   # suspende entradas
        "MEAN_REVERTING_CHOP":   0.85,
        "HIGH_CORRELATION_RISK": 0.50,
        "LIQUIDITY_VACUUM":      0.0,   # suspende entradas
    }.get(regime, 1.0)

    size *= regime_mult

    # ── Ajuste por VaR disponível no portfolio ────────────────────────────────
    var_headroom = portfolio_context.get("var_headroom", 1.0)
    size *= min(var_headroom, 1.0)

    # ── Ajuste por número de posições abertas ─────────────────────────────────
    open_positions = portfolio_context.get("open_positions", 0)
    if open_positions >= 3:
        size *= 0.70
    elif open_positions >= 2:
        size *= 0.85

    # ── Clamp final ───────────────────────────────────────────────────────────
    size = max(MIN_SIZE_PCT, min(MAX_SIZE_PCT, size))

    # Se regime bloqueia, retorna zero
    if regime_mult == 0.0:
        return {
            "size_pct":  0.0,
            "size_usd":  0.0,
            "blocked":   True,
            "reason":    f"Regime {regime} suspende entradas",
            "rationale": {"regime": regime, "regime_mult": 0.0},
        }

    size_usd = size * portfolio_value

    return {
        "size_pct":  round(size, 4),
        "size_usd":  round(size_usd, 2),
        "blocked":   False,
        "reason":    "OK",
        "rationale": {
            "edge_factor":    round(edge_factor, 4),
            "vol_factor":     round(vol_factor, 3),
            "corr_factor":    round(corr_factor, 3),
            "raw_size":       round(raw_size, 4),
            "kelly_cap":      round(kelly_cap, 4),
            "regime_mult":    regime_mult,
            "var_headroom":   round(var_headroom, 3),
            "open_positions": open_positions,
        },
    }
