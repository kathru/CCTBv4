"""
Signal Engine — Camada 3
========================
Substitui os indicadores clássicos isolados por modelos probabilísticos
condicionais ao contexto de mercado.

Remove (V3):
  MACD crossover, RSI threshold fixo, BB Reversion clássica

Adiciona (V4):
  A. Volatility Expansion Model  — compressão → expansão
  B. Market Structure Engine     — HH/HL, liquidity sweeps, displacement
  C. Orderflow Proxy             — taker imbalance, volume delta, OI
  D. Relative Strength Engine    — inter-asset alpha

Output de cada modelo:
  {
    "probability":  float 0-1,   # P(edge se realizar)
    "confidence":   float 0-1,   # qualidade dos dados
    "edge":         float,       # probability - 0.50 (excess over random)
    "direction":    "long"|"short"|"neutral",
    "context":      dict,        # fatores usados
  }

Score final combinado:
  {
    "score":          float 0-1,   # score ponderado calibrado
    "expected_value": float,       # edge × payoff esperado - fees
    "kelly_fraction": float,       # sizing sugerido (Kelly parcial)
    "direction":      str,
    "factors":        dict,        # contribuição de cada modelo
  }
"""

import math
import json
import os
import statistics
from typing import Optional

# ── Platt Scaling (calibração) ────────────────────────────────────────────────
# Carregado de data/calibration_coef.json quando disponível.
# Produzido por calibrate.py após rodar sobre 8 anos de dados OKX.
# Enquanto não existir, score raw é usado (sem calibração).
_CALIB_COEF: Optional[dict] = None

def _load_calibration() -> Optional[dict]:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration_coef.json"
    )
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def _apply_calibration(raw_score: float) -> tuple:
    """
    Aplica Platt Scaling se disponível.
    Retorna (calibrated_score, is_calibrated).
    Se coeficientes não existem, retorna (raw_score, False).
    """
    global _CALIB_COEF
    if _CALIB_COEF is None:
        _CALIB_COEF = _load_calibration() or {}
    a = _CALIB_COEF.get("platt_a")
    b = _CALIB_COEF.get("platt_b")
    if a is None or b is None:
        return raw_score, False
    cal = 1.0 / (1.0 + math.exp(-(a * raw_score + b)))
    return round(cal, 4), True


# Pesos dos fatores no score final (dinâmicos por regime — ajustados no compute)
BASE_WEIGHTS = {
    "market_structure":     0.20,
    "trend_strength":       0.18,
    "volatility_expansion": 0.16,
    "relative_strength":    0.14,
    "liquidity_conditions": 0.12,
    "funding_imbalance":    0.11,
    "breadth":              0.09,
}

# Ajuste de pesos por regime
REGIME_WEIGHT_OVERRIDES = {
    "TREND_EXPANSION": {
        "trend_strength":       0.28,
        "market_structure":     0.22,
        "volatility_expansion": 0.14,
        "relative_strength":    0.14,
        "liquidity_conditions": 0.10,
        "funding_imbalance":    0.07,
        "breadth":              0.05,
    },
    "VOLATILITY_COMPRESSION": {
        "volatility_expansion": 0.35,
        "market_structure":     0.25,
        "liquidity_conditions": 0.15,
        "trend_strength":       0.10,
        "relative_strength":    0.08,
        "funding_imbalance":    0.05,
        "breadth":              0.02,
    },
    "MEAN_REVERTING_CHOP": {
        "market_structure":     0.25,
        "liquidity_conditions": 0.20,
        "orderflow":            0.20,
        "volatility_expansion": 0.15,
        "funding_imbalance":    0.12,
        "trend_strength":       0.05,
        "breadth":              0.03,
    },
}


def _ema(values: list, span: int) -> list:
    if not values:
        return []
    k = 2 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_g  = sum(gains) / period
    avg_l  = sum(losses) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100 - (100 / (1 + rs))


# ── A. Volatility Expansion Model ─────────────────────────────────────────────

def volatility_expansion_signal(
    candles: list,
    market_context: dict,
    regime: str,
) -> dict:
    """
    Detecta setups de compressão → expansão.
    Muito mais poderoso que MACD porque captura o MOMENTO do movimento,
    não o movimento depois que já aconteceu.

    Alta probabilidade de long quando:
      - BB width no percentil baixo (compressão)
      - ATR acelerando (expansão começando)
      - Taker ratio favorável (compra agressiva)
      - OI expandindo (novo dinheiro entrando)
    """
    closes  = [float(c["close"]) for c in candles]
    highs   = [float(c["high"])  for c in candles]
    lows    = [float(c["low"])   for c in candles]

    if len(closes) < 25:
        return {"probability": 0.5, "confidence": 0.1, "edge": 0.0,
                "direction": "neutral", "context": {}}

    # BB width percentile
    bb_pct  = market_context.get("vol_percentile", 0.5)
    atr_exp = market_context.get("atr_rate", {}).get("expansion", 1.0)
    taker   = market_context.get("taker_ratio", {}).get("imbalance", 0.0)
    oi_exp  = market_context.get("open_interest", {}).get("expanding", False)
    rvol    = market_context.get("realized_vol", 0.0)

    # EMA direcional — determina direção da expansão esperada
    ema_fast = _ema(closes, 9)
    ema_slow = _ema(closes, 21)
    ema_long = _ema(closes, 50)

    direction = "neutral"
    dir_score = 0.0

    if len(ema_fast) > 1 and len(ema_slow) > 1:
        if ema_fast[-1] > ema_slow[-1] > ema_long[-1]:
            direction = "long"
            dir_score = 0.70
        elif ema_fast[-1] < ema_slow[-1] < ema_long[-1]:
            direction = "short"
            dir_score = 0.70
        elif ema_fast[-1] > ema_slow[-1]:
            direction = "long"
            dir_score = 0.55
        else:
            direction = "short"
            dir_score = 0.55

    # Score de compressão (quanto mais comprimido, maior o potencial)
    compression_score = max(0, 1 - bb_pct)  # baixo percentil = boa compressão

    # Score de expansão começando (ATR acelerando)
    expansion_starting = max(0, min(1, (atr_exp - 0.9) / 0.5)) if atr_exp > 0.9 else 0.0

    # Confirmação de fluxo
    flow_score = 0.0
    if direction == "long":
        flow_score = max(0, taker)            # taker comprador
        if oi_exp: flow_score = min(1, flow_score + 0.3)
    elif direction == "short":
        flow_score = max(0, -taker)           # taker vendedor
        if not oi_exp: flow_score = min(1, flow_score + 0.2)

    # Regime favorável para expansão?
    regime_mult = {
        "VOLATILITY_COMPRESSION": 1.30,
        "TREND_EXPANSION":        1.10,
        "MEAN_REVERTING_CHOP":    0.70,
        "PANIC_LIQUIDATION":      0.20,
    }.get(regime, 1.0)

    raw_prob = (
        compression_score * 0.35 +
        expansion_starting * 0.30 +
        flow_score * 0.20 +
        dir_score * 0.15
    ) * regime_mult

    probability = max(0.0, min(1.0, raw_prob))
    confidence  = 0.7 if len(closes) >= 100 else 0.5

    return {
        "probability": round(probability, 4),
        "confidence":  round(confidence, 3),
        "edge":        round(probability - 0.50, 4),
        "direction":   direction,
        "context": {
            "bb_percentile":    round(bb_pct, 3),
            "atr_expansion":    round(atr_exp, 3),
            "compression":      round(compression_score, 3),
            "expansion_start":  round(expansion_starting, 3),
            "taker_imbalance":  round(taker, 3),
            "oi_expanding":     oi_exp,
        },
    }


# ── B. Market Structure Engine ────────────────────────────────────────────────

def market_structure_signal(
    candles: list,
    market_context: dict,
    regime: str,
) -> dict:
    """
    Detecta estrutura de mercado institucional:
      - Higher highs / higher lows (tendência intacta)
      - Liquidity sweeps (fakeout antes do movimento real)
      - Failed breakouts (armadilha → inversão)
      - Displacement candles (candles de movimento institucional)

    Não usa MACD, RSI ou BB isolados. Usa a estrutura do preço.
    """
    if len(candles) < 20:
        return {"probability": 0.5, "confidence": 0.1, "edge": 0.0,
                "direction": "neutral", "context": {}}

    closes  = [float(c["close"]) for c in candles[-30:]]
    highs   = [float(c["high"])  for c in candles[-30:]]
    lows    = [float(c["low"])   for c in candles[-30:]]
    opens   = [float(c["open"])  for c in candles[-30:]]
    volumes = [float(c["volume"])for c in candles[-30:]]

    n = len(closes)

    # ── 1. Higher Highs / Higher Lows ──────────────────────────────────────
    hh_score = 0.0  # tendência de alta intacta
    ll_score = 0.0  # tendência de baixa intacta

    # Pivots locais (últimas 3 janelas de 5 velas)
    pivots_high = []
    pivots_low  = []
    window = 5
    for i in range(window, n - window, window):
        local_high = max(highs[i - window: i + window])
        local_low  = min(lows[i - window: i + window])
        pivots_high.append(local_high)
        pivots_low.append(local_low)

    if len(pivots_high) >= 2:
        hh_count = sum(1 for i in range(1, len(pivots_high)) if pivots_high[i] > pivots_high[i-1])
        hl_count = sum(1 for i in range(1, len(pivots_low))  if pivots_low[i]  > pivots_low[i-1])
        hh_score = (hh_count + hl_count) / (2 * (len(pivots_high) - 1))

        lh_count = sum(1 for i in range(1, len(pivots_high)) if pivots_high[i] < pivots_high[i-1])
        ll_count = sum(1 for i in range(1, len(pivots_low))  if pivots_low[i]  < pivots_low[i-1])
        ll_score = (lh_count + ll_count) / (2 * (len(pivots_high) - 1))

    # ── 2. Displacement Candles (movimento institucional) ──────────────────
    avg_range = sum(highs[i] - lows[i] for i in range(n)) / n if n > 0 else 0
    last_range = highs[-1] - lows[-1]
    displacement = last_range / avg_range if avg_range > 0 else 1.0

    # Vela de deslocamento: range > 2× média e fecha próximo ao extremo
    displacement_bullish = (
        displacement > 2.0 and
        closes[-1] > opens[-1] and
        (closes[-1] - lows[-1]) / (highs[-1] - lows[-1] + 1e-9) > 0.70
    )
    displacement_bearish = (
        displacement > 2.0 and
        closes[-1] < opens[-1] and
        (highs[-1] - closes[-1]) / (highs[-1] - lows[-1] + 1e-9) > 0.70
    )

    # ── 3. Liquidity Sweep Detection ───────────────────────────────────────
    # Sweep: vela que rompe mínimo/máximo recente mas fecha dentro da range anterior
    recent_low  = min(lows[-10:-1])
    recent_high = max(highs[-10:-1])

    bullish_sweep = (
        lows[-1] < recent_low and       # rompeu abaixo
        closes[-1] > recent_low and     # mas fechou acima (rejeição)
        volumes[-1] > sum(volumes[-5:]) / 5  # volume acima da média
    )
    bearish_sweep = (
        highs[-1] > recent_high and     # rompeu acima
        closes[-1] < recent_high and   # mas fechou abaixo
        volumes[-1] > sum(volumes[-5:]) / 5
    )

    # ── Score final ─────────────────────────────────────────────────────────
    long_score  = 0.0
    short_score = 0.0

    long_score  += hh_score * 0.30
    short_score += ll_score * 0.30

    if displacement_bullish: long_score  += 0.30
    if displacement_bearish: short_score += 0.30

    if bullish_sweep: long_score  += 0.25
    if bearish_sweep: short_score += 0.25

    # Contexto de regime
    ob_imbalance = market_context.get("orderbook", {}).get("imbalance", 0.0)
    if ob_imbalance > 0.20:  long_score  += 0.15
    if ob_imbalance < -0.20: short_score += 0.15

    direction = "neutral"
    probability = 0.50

    if long_score > short_score and long_score > 0.30:
        direction   = "long"
        probability = 0.50 + min(long_score * 0.40, 0.40)
    elif short_score > long_score and short_score > 0.30:
        direction   = "short"
        probability = 0.50 + min(short_score * 0.40, 0.40)

    confidence = 0.65 if len(candles) >= 50 else 0.45

    return {
        "probability": round(probability, 4),
        "confidence":  round(confidence, 3),
        "edge":        round(probability - 0.50, 4),
        "direction":   direction,
        "context": {
            "hh_score":            round(hh_score, 3),
            "ll_score":            round(ll_score, 3),
            "displacement":        round(displacement, 3),
            "displacement_bull":   displacement_bullish,
            "displacement_bear":   displacement_bearish,
            "bullish_sweep":       bullish_sweep,
            "bearish_sweep":       bearish_sweep,
        },
    }


# ── C. Orderflow Proxy ────────────────────────────────────────────────────────

def orderflow_signal(
    candles: list,
    market_context: dict,
    regime: str,
) -> dict:
    """
    Proxy de orderflow institucional usando dados públicos:
      - Taker buy/sell imbalance
      - Volume delta (candles up vs down)
      - OI expansion (novo dinheiro)
      - Funding divergence (posicionamento alavancado)

    Captura quem está sendo agressivo no mercado —
    compradores ou vendedores institucionais.
    """
    taker    = market_context.get("taker_ratio", {})
    vol_d    = market_context.get("volume_delta", {})
    funding  = market_context.get("funding", {})
    oi       = market_context.get("open_interest", {})

    taker_imbal  = taker.get("imbalance", 0.0)
    vol_delta    = vol_d.get("delta_pct", 0.0)
    aggressive   = vol_d.get("aggressive", False)
    funding_rate = funding.get("funding_rate", 0.0)
    oi_expanding = oi.get("expanding", False)
    oi_change    = oi.get("oi_change_pct", 0.0)

    # ── Long signals ──────────────────────────────────────────────────────
    long_score = 0.0
    if taker_imbal > 0.10:   long_score += min(taker_imbal * 1.5, 0.35)
    if vol_delta > 0.15:     long_score += min(vol_delta * 1.2, 0.30)
    if oi_expanding:         long_score += 0.20
    if oi_change > 1.0:      long_score += 0.10
    # Funding negativo com preço subindo = shorts sendo forçados (squeeze)
    if funding_rate < -0.0001 and taker_imbal > 0.10:
        long_score += 0.15

    # ── Short signals ─────────────────────────────────────────────────────
    short_score = 0.0
    if taker_imbal < -0.10:  short_score += min(abs(taker_imbal) * 1.5, 0.35)
    if vol_delta < -0.15:    short_score += min(abs(vol_delta) * 1.2, 0.30)
    if not oi_expanding and oi_change < -1.0: short_score += 0.20
    # Funding muito positivo = longs excessivos (reversão possível)
    if funding_rate > 0.0005:
        short_score += 0.15

    direction = "neutral"
    probability = 0.50

    # Regime de pânico: orderflow vendedor domina qualquer sinal de compra
    if regime == "PANIC_LIQUIDATION":
        if taker_imbal < -0.20:
            return {
                "probability": 0.80, "confidence": 0.85, "edge": 0.30,
                "direction": "short",
                "context": {"panic_confirmed": True, "taker": taker_imbal},
            }

    if long_score > short_score and long_score > 0.25:
        direction   = "long"
        probability = 0.50 + min(long_score * 0.35, 0.35)
    elif short_score > long_score and short_score > 0.25:
        direction   = "short"
        probability = 0.50 + min(short_score * 0.35, 0.35)

    confidence = 0.70  # orderflow é dado direto — alta confiança

    return {
        "probability": round(probability, 4),
        "confidence":  round(confidence, 3),
        "edge":        round(probability - 0.50, 4),
        "direction":   direction,
        "context": {
            "taker_imbalance":  round(taker_imbal, 4),
            "vol_delta_pct":    round(vol_delta, 4),
            "aggressive":       aggressive,
            "oi_expanding":     oi_expanding,
            "oi_change_pct":    round(oi_change, 4),
            "funding_rate":     round(funding_rate, 6),
        },
    }


# ── D. Relative Strength Engine ───────────────────────────────────────────────

def relative_strength_signal(
    pair: str,
    closes_map: dict,
    regime: str,
) -> dict:
    """
    Detecta alpha inter-asset:
      - ETH outperforming BTC → rotação para risco
      - SOL outperforming ETH → apetite especulativo crescente
      - Inverso → fuga para qualidade

    Performance relativa calculada sobre janelas múltiplas:
      1H, 4H, 24H
    """
    if len(closes_map) < 2:
        return {"probability": 0.5, "confidence": 0.2, "edge": 0.0,
                "direction": "neutral", "context": {}}

    def _perf(closes: list, window: int) -> float:
        if len(closes) < window + 1:
            return 0.0
        return (closes[-1] - closes[-window]) / closes[-window]

    # Performance do par atual vs BTC
    target_closes = closes_map.get(pair, [])
    btc_closes    = closes_map.get("BTC-USD", [])

    if not target_closes or not btc_closes:
        return {"probability": 0.5, "confidence": 0.2, "edge": 0.0,
                "direction": "neutral", "context": {}}

    perf_target_1h  = _perf(target_closes, 1)
    perf_btc_1h     = _perf(btc_closes, 1)
    perf_target_4h  = _perf(target_closes, 4)
    perf_btc_4h     = _perf(btc_closes, 4)
    perf_target_24h = _perf(target_closes, 24)
    perf_btc_24h    = _perf(btc_closes, 24)

    # RS = performance relativa ao BTC
    rs_1h  = perf_target_1h  - perf_btc_1h
    rs_4h  = perf_target_4h  - perf_btc_4h
    rs_24h = perf_target_24h - perf_btc_24h

    # Pesos maiores para timeframes mais curtos
    rs_composite = rs_1h * 0.50 + rs_4h * 0.30 + rs_24h * 0.20

    # Score: outperformance → long, underperformance → short/skip
    direction = "neutral"
    probability = 0.50

    # Se é BTC vs BTC → sempre neutro
    if pair == "BTC-USD":
        # BTC vs altcoins: BTC liderando = risco-off, short alts
        eth_closes = closes_map.get("ETH-USD", [])
        sol_closes = closes_map.get("SOL-USD", [])
        if eth_closes and sol_closes:
            rs_eth = _perf(btc_closes, 4) - _perf(eth_closes, 4)
            if rs_eth > 0.005:   # BTC outperforming significativamente
                direction   = "long"
                probability = 0.55 + min(rs_eth * 10, 0.15)
            elif rs_eth < -0.005:
                direction   = "short"
                probability = 0.55 + min(abs(rs_eth) * 10, 0.15)
    else:
        # Alt vs BTC
        if rs_composite > 0.003:
            direction   = "long"
            probability = 0.50 + min(rs_composite * 20, 0.30)
        elif rs_composite < -0.003:
            direction   = "short"
            probability = 0.50 + min(abs(rs_composite) * 20, 0.30)

    # Em regime TREND_EXPANSION, RS tem mais peso
    if regime == "TREND_EXPANSION" and direction == "long":
        probability = min(1.0, probability * 1.10)

    confidence = 0.60 if len(target_closes) >= 24 else 0.35

    return {
        "probability": round(probability, 4),
        "confidence":  round(confidence, 3),
        "edge":        round(probability - 0.50, 4),
        "direction":   direction,
        "context": {
            "rs_1h":        round(rs_1h, 5),
            "rs_4h":        round(rs_4h, 5),
            "rs_24h":       round(rs_24h, 5),
            "rs_composite": round(rs_composite, 5),
        },
    }


# ── Score Final Combinado ─────────────────────────────────────────────────────

def compute_signal_score(
    pair: str,
    candles_1h: list,
    market_context: dict,
    regime: str,
    closes_map: Optional[dict] = None,
    fee_rate: float = 0.001,
    expected_rr: float = 2.0,
) -> dict:
    """
    Combina os 4 modelos em um score final probabilístico.

    Parâmetros:
      fee_rate     — custo total de execução (entrada + saída)
      expected_rr  — risk/reward esperado do setup

    Retorna:
      score         — 0-1, probabilidade calibrada
      expected_value — EV real após fees
      kelly_fraction — sizing sugerido (Kelly parcial 25%)
      direction      — 'long' | 'short' | 'neutral'
      factors        — contribuição de cada modelo
    """
    closes_map = closes_map or {}

    # ── Roda os 4 modelos ────────────────────────────────────────────────────
    vol_exp = volatility_expansion_signal(candles_1h, market_context, regime)
    mkt_str = market_structure_signal(candles_1h, market_context, regime)
    order_f = orderflow_signal(candles_1h, market_context, regime)
    rel_str = relative_strength_signal(pair, closes_map, regime)

    # ── Pesos dinâmicos por regime ────────────────────────────────────────────
    weights = REGIME_WEIGHT_OVERRIDES.get(regime, BASE_WEIGHTS).copy()

    # Normaliza pesos disponíveis
    available = ["volatility_expansion", "market_structure", "orderflow",
                 "relative_strength", "trend_strength", "liquidity_conditions",
                 "funding_imbalance", "breadth"]
    total_w = sum(weights.get(k, 0) for k in available)
    if total_w == 0:
        total_w = 1.0

    # ── Determina direção dominante ───────────────────────────────────────────
    direction_votes = {"long": 0.0, "short": 0.0, "neutral": 0.0}
    for model in [vol_exp, mkt_str, order_f, rel_str]:
        d = model["direction"]
        p = model["probability"]
        direction_votes[d] += (p - 0.50) * model["confidence"]

    dominant_direction = max(direction_votes, key=direction_votes.get)
    if direction_votes[dominant_direction] <= 0.02:
        dominant_direction = "neutral"

    # ── Score ponderado pela direção dominante ────────────────────────────────
    model_scores = {
        "volatility_expansion": vol_exp["probability"] if vol_exp["direction"] == dominant_direction else (1 - vol_exp["probability"]),
        "market_structure":     mkt_str["probability"] if mkt_str["direction"] == dominant_direction else (1 - mkt_str["probability"]),
        "orderflow":            order_f["probability"] if order_f["direction"] == dominant_direction else (1 - order_f["probability"]),
        "relative_strength":    rel_str["probability"] if rel_str["direction"] == dominant_direction else (1 - rel_str["probability"]),
    }

    # Peso de cada modelo pelo score e confiança
    weighted_sum = (
        model_scores["volatility_expansion"] * weights.get("volatility_expansion", 0.16) * vol_exp["confidence"] +
        model_scores["market_structure"]     * weights.get("market_structure", 0.20)     * mkt_str["confidence"] +
        model_scores["orderflow"]            * weights.get("orderflow", 0.15)            * order_f["confidence"] +
        model_scores["relative_strength"]    * weights.get("relative_strength", 0.14)   * rel_str["confidence"]
    )
    weight_total = (
        weights.get("volatility_expansion", 0.16) * vol_exp["confidence"] +
        weights.get("market_structure", 0.20)     * mkt_str["confidence"] +
        weights.get("orderflow", 0.15)            * order_f["confidence"] +
        weights.get("relative_strength", 0.14)   * rel_str["confidence"]
    )

    score = weighted_sum / weight_total if weight_total > 0 else 0.50

    # ── Regime blockers ───────────────────────────────────────────────────────
    if regime in ("PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"):
        score = min(score, 0.40)
        dominant_direction = "neutral"
    elif regime == "HIGH_CORRELATION_RISK":
        score = min(score, 0.55)

    # ── EV e Kelly ────────────────────────────────────────────────────────────
    p = score
    q = 1 - p
    b = expected_rr  # payoff por unidade de risco

    # EV = p × b - q × 1 - fee
    ev = p * b - q * 1.0 - fee_rate * (1 + b)

    # Kelly fraction = (p × b - q) / b
    kelly_full = (p * b - q) / b if b > 0 else 0.0
    kelly_fraction = max(0.0, kelly_full * 0.25)  # Kelly parcial 25% para segurança

    # Confidence global
    confidence = (
        vol_exp["confidence"] * 0.25 +
        mkt_str["confidence"] * 0.30 +
        order_f["confidence"] * 0.25 +
        rel_str["confidence"] * 0.20
    )

    # Aplica Platt Scaling se calibration_coef.json disponível
    score_raw = score
    score_cal, is_calibrated = _apply_calibration(score)
    score = score_cal   # usa score calibrado se disponível, raw caso contrário

    # Recalcula EV com score calibrado
    p = score
    q = 1 - p
    ev = p * b - q * 1.0 - fee_rate * (1 + b)
    kelly_full = (p * b - q) / b if b > 0 else 0.0
    kelly_fraction = max(0.0, kelly_full * 0.25)

    return {
        "score":           round(score, 4),
        "score_raw":       round(score_raw, 4),
        "calibrated":      is_calibrated,
        "expected_value":  round(ev, 4),
        "kelly_fraction":  round(kelly_fraction, 4),
        "confidence":      round(confidence, 3),
        "direction":       dominant_direction,
        "factors": {
            "volatility_expansion": {
                "score":     round(model_scores["volatility_expansion"], 4),
                "confidence": vol_exp["confidence"],
                "direction":  vol_exp["direction"],
                "context":    vol_exp["context"],
            },
            "market_structure": {
                "score":     round(model_scores["market_structure"], 4),
                "confidence": mkt_str["confidence"],
                "direction":  mkt_str["direction"],
                "context":    mkt_str["context"],
            },
            "orderflow": {
                "score":     round(model_scores["orderflow"], 4),
                "confidence": order_f["confidence"],
                "direction":  order_f["direction"],
                "context":    order_f["context"],
            },
            "relative_strength": {
                "score":     round(model_scores["relative_strength"], 4),
                "confidence": rel_str["confidence"],
                "direction":  rel_str["direction"],
                "context":    rel_str["context"],
            },
        },
        "regime":  regime,
        "weights": {k: round(v, 3) for k, v in weights.items()},
    }
