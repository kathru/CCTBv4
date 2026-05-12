"""
Regime Engine — Camada 2
========================
Detecta o regime de mercado atual com muito mais granularidade que o V3.

V3: 3 estados (bull / chop / bear) via EMA200 + ADX
V4: 7 regimes com vetor de probabilidade, usando variáveis de microestrutura

Regimes possíveis:
  TREND_EXPANSION      — tendência forte e saudável → trend-following agressivo
  TREND_EXHAUSTION     — tendência madura, sinais de fraqueza → reduz size
  VOLATILITY_COMPRESSION — squeeze pré-movimento → aguarda direção
  PANIC_LIQUIDATION    — crash com liquidações → fecha posições, para
  MEAN_REVERTING_CHOP  — mercado lateral → estratégias de reversão
  HIGH_CORRELATION_RISK — mercado sistêmico → limita slots
  LIQUIDITY_VACUUM     — baixa liquidez → suspende execução

Output: RegimeResult com regime dominante + vetor de probabilidades + confiança
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegimeResult:
    regime: str                        # regime dominante
    probabilities: dict                # {regime: float 0-1}
    confidence: float                  # 0-1: qualidade dos dados para detecção
    signals: dict = field(default_factory=dict)  # sinais intermediários para debug
    action: str = "normal"            # 'normal' | 'reduce' | 'close' | 'suspend'


REGIMES = [
    "TREND_EXPANSION",
    "TREND_EXHAUSTION",
    "VOLATILITY_COMPRESSION",
    "PANIC_LIQUIDATION",
    "MEAN_REVERTING_CHOP",
    "HIGH_CORRELATION_RISK",
    "LIQUIDITY_VACUUM",
]

# Ações por regime
REGIME_ACTIONS = {
    "TREND_EXPANSION":       "normal",
    "TREND_EXHAUSTION":      "reduce",
    "VOLATILITY_COMPRESSION":"reduce",
    "PANIC_LIQUIDATION":     "close",
    "MEAN_REVERTING_CHOP":   "normal",
    "HIGH_CORRELATION_RISK": "reduce",
    "LIQUIDITY_VACUUM":      "suspend",
}


def _ema(values: list, span: int) -> list:
    """EMA simples."""
    if not values:
        return []
    k = 2 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """ADX clássico."""
    if len(closes) < period + 1:
        return 20.0
    dm_plus, dm_minus, trs = [], [], []
    for i in range(1, len(closes)):
        h_diff = highs[i] - highs[i - 1]
        l_diff = lows[i - 1] - lows[i]
        dm_plus.append(max(h_diff, 0) if h_diff > l_diff else 0)
        dm_minus.append(max(l_diff, 0) if l_diff > h_diff else 0)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)

    def _smooth(lst, p):
        s = sum(lst[:p])
        result = [s]
        for v in lst[p:]:
            result.append(result[-1] - result[-1] / p + v)
        return result

    sm_tr  = _smooth(trs, period)
    sm_dmp = _smooth(dm_plus, period)
    sm_dmn = _smooth(dm_minus, period)

    adx_vals = []
    for i in range(len(sm_tr)):
        if sm_tr[i] == 0:
            continue
        dip = sm_dmp[i] / sm_tr[i] * 100
        din = sm_dmn[i] / sm_tr[i] * 100
        dx  = abs(dip - din) / (dip + din) * 100 if (dip + din) > 0 else 0
        adx_vals.append(dx)

    if len(adx_vals) < period:
        return 20.0
    return sum(adx_vals[-period:]) / period


def _bb_width_percentile(closes: list, period: int = 20, window: int = 100) -> float:
    """
    Percentil da largura atual das Bollinger Bands vs histórico.
    0 = mais comprimido já visto, 1 = mais expandido já visto.
    """
    if len(closes) < period + window:
        return 0.5

    widths = []
    for i in range(window):
        subset = closes[-(period + window) + i: -(window) + i + period]
        if len(subset) < period:
            continue
        mean = sum(subset) / period
        std  = statistics.stdev(subset)
        widths.append(std * 4 / mean if mean > 0 else 0)  # BB width normalizado

    if not widths:
        return 0.5

    current_subset = closes[-period:]
    mean = sum(current_subset) / period
    std  = statistics.stdev(current_subset)
    current_width = std * 4 / mean if mean > 0 else 0

    below = sum(1 for w in widths if w <= current_width)
    return below / len(widths)


def detect_regime(
    candles_1h: list,
    candles_6h: list,
    market_context: dict,
    closes_map: Optional[dict] = None,
) -> RegimeResult:
    """
    Detecta o regime de mercado atual.

    Parâmetros:
      candles_1h      — candles 1H do par principal (BTC)
      candles_6h      — candles 6H para contexto macro
      market_context  — output da Camada 1 (microestrutura)
      closes_map      — {pair: [closes]} para correlação

    Retorna RegimeResult com regime dominante e vetor de probabilidades.
    """
    # Extrai séries de preço
    if len(candles_1h) < 30:
        return RegimeResult(
            regime="MEAN_REVERTING_CHOP",
            probabilities={r: 1/7 for r in REGIMES},
            confidence=0.1,
            action="reduce",
        )

    closes  = [float(c["close"])  for c in candles_1h]
    highs   = [float(c["high"])   for c in candles_1h]
    lows    = [float(c["low"])    for c in candles_1h]
    volumes = [float(c["volume"]) for c in candles_1h]

    # ── Indicadores base ─────────────────────────────────────────────────────
    adx_val    = _adx(highs, lows, closes)
    bb_pct     = _bb_width_percentile(closes)
    vol_pct    = market_context.get("vol_percentile", 0.5)
    atr_exp    = market_context.get("atr_rate", {}).get("expansion", 1.0)
    funding    = market_context.get("funding", {})
    oi         = market_context.get("open_interest", {})
    ob         = market_context.get("orderbook", {})
    taker      = market_context.get("taker_ratio", {})
    rvol       = market_context.get("realized_vol", 0.0)
    corr       = market_context.get("correlation", {})

    funding_rate = funding.get("funding_rate", 0.0)
    oi_expanding = oi.get("expanding", False)
    spread_pct   = ob.get("spread_pct", 0.0)
    taker_imbal  = taker.get("imbalance", 0.0)
    avg_corr     = corr.get("avg_correlation", 0.5)

    # Volume médio recente vs histórico
    avg_vol_recent = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0
    avg_vol_hist   = sum(volumes[-50:]) / 50 if len(volumes) >= 50 else avg_vol_recent
    vol_ratio = avg_vol_recent / avg_vol_hist if avg_vol_hist > 0 else 1.0

    # EMA200 6H para contexto macro
    above_ema200 = True
    ema200_slope = 0.0
    if len(candles_6h) >= 50:
        closes_6h = [float(c["close"]) for c in candles_6h]
        ema200 = _ema(closes_6h, 200)
        if len(ema200) >= 2:
            above_ema200 = closes_6h[-1] > ema200[-1]
            ema200_slope = (ema200[-1] - ema200[-10]) / ema200[-10] if ema200[-10] > 0 else 0

    # ── Scores por regime (0-1) ──────────────────────────────────────────────
    scores = {}

    # PANIC_LIQUIDATION: vol extrema + spread alto + taker vendedor + OI caindo
    panic_score = 0.0
    if vol_pct > 0.90:       panic_score += 0.35
    if atr_exp > 2.0:        panic_score += 0.25
    if spread_pct > 0.001:   panic_score += 0.15
    if taker_imbal < -0.30:  panic_score += 0.15
    if not oi_expanding and vol_pct > 0.85: panic_score += 0.10
    scores["PANIC_LIQUIDATION"] = min(panic_score, 1.0)

    # LIQUIDITY_VACUUM: spread alto + volume baixo + sem direcionalidade
    vacuum_score = 0.0
    if spread_pct > 0.0005:  vacuum_score += 0.30
    if vol_ratio < 0.40:     vacuum_score += 0.35
    if abs(taker_imbal) < 0.05: vacuum_score += 0.20
    if adx_val < 15:         vacuum_score += 0.15
    scores["LIQUIDITY_VACUUM"] = min(vacuum_score, 1.0)

    # HIGH_CORRELATION_RISK: correlação alta entre pares + vol elevada
    corr_score = 0.0
    if avg_corr > 0.85:      corr_score += 0.50
    if avg_corr > 0.90:      corr_score += 0.20  # bônus por correlação extrema
    if vol_pct > 0.70:       corr_score += 0.20
    if not above_ema200:     corr_score += 0.10
    scores["HIGH_CORRELATION_RISK"] = min(corr_score, 1.0)

    # VOLATILITY_COMPRESSION: BB width no percentil baixo + ADX baixo + vol caindo
    compress_score = 0.0
    if bb_pct < 0.15:        compress_score += 0.40
    if adx_val < 18:         compress_score += 0.25
    if atr_exp < 0.80:       compress_score += 0.20
    if vol_ratio < 0.70:     compress_score += 0.15
    scores["VOLATILITY_COMPRESSION"] = min(compress_score, 1.0)

    # TREND_EXPANSION: ADX alto + above EMA200 + OI crescendo + taker comprador
    expand_score = 0.0
    if adx_val > 30:         expand_score += 0.30
    if adx_val > 40:         expand_score += 0.10  # bônus tendência forte
    if above_ema200:         expand_score += 0.20
    if ema200_slope > 0.001: expand_score += 0.15
    if oi_expanding:         expand_score += 0.15
    if taker_imbal > 0.15:   expand_score += 0.10
    scores["TREND_EXPANSION"] = min(expand_score, 1.0)

    # TREND_EXHAUSTION: ADX alto mas caindo + funding extremo + vol desacelerando
    exhaust_score = 0.0
    if adx_val > 25 and atr_exp < 0.90: exhaust_score += 0.35  # força mas desacelerando
    if abs(funding_rate) > 0.0008:      exhaust_score += 0.25  # funding extremo
    if bb_pct > 0.85 and atr_exp < 1.0: exhaust_score += 0.25  # vol alta mas comprimindo
    if above_ema200 and adx_val < 25:   exhaust_score += 0.15
    scores["TREND_EXHAUSTION"] = min(exhaust_score, 1.0)

    # MEAN_REVERTING_CHOP: ADX baixo + BB médio + sem direcionalidade de OI
    chop_score = 0.0
    if adx_val < 22:              chop_score += 0.35
    if 0.30 < bb_pct < 0.70:     chop_score += 0.25
    if abs(taker_imbal) < 0.15:  chop_score += 0.20
    if not oi_expanding:          chop_score += 0.10
    if 0.30 < vol_pct < 0.70:    chop_score += 0.10
    scores["MEAN_REVERTING_CHOP"] = min(chop_score, 1.0)

    # ── Normaliza para vetor de probabilidade ────────────────────────────────
    total = sum(scores.values()) or 1.0
    probs = {r: round(scores[r] / total, 4) for r in REGIMES}

    # ── Regime dominante ─────────────────────────────────────────────────────
    # PANIC e VACUUM têm prioridade absoluta por segurança
    if probs["PANIC_LIQUIDATION"] > 0.35:
        dominant = "PANIC_LIQUIDATION"
    elif probs["LIQUIDITY_VACUUM"] > 0.40:
        dominant = "LIQUIDITY_VACUUM"
    else:
        dominant = max(probs, key=probs.get)

    # ── Confiança: qualidade dos dados disponíveis ───────────────────────────
    confidence = 0.5
    if len(candles_1h) >= 100: confidence += 0.15
    if len(candles_6h) >= 50:  confidence += 0.10
    if market_context.get("funding"):  confidence += 0.10
    if market_context.get("open_interest"): confidence += 0.10
    if avg_corr > 0:           confidence += 0.05
    confidence = min(confidence, 1.0)

    signals = {
        "adx":           round(adx_val, 1),
        "bb_percentile": round(bb_pct, 3),
        "vol_percentile": round(vol_pct, 3),
        "atr_expansion": round(atr_exp, 3),
        "taker_imbal":   round(taker_imbal, 3),
        "funding_rate":  round(funding_rate, 6),
        "oi_expanding":  oi_expanding,
        "above_ema200":  above_ema200,
        "spread_pct":    round(spread_pct, 6),
        "vol_ratio":     round(vol_ratio, 3),
        "avg_corr":      round(avg_corr, 3),
    }

    return RegimeResult(
        regime=dominant,
        probabilities=probs,
        confidence=round(confidence, 3),
        signals=signals,
        action=REGIME_ACTIONS[dominant],
    )
