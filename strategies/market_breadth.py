"""
Market Breadth — Indicadores de Amplitude de Mercado
=====================================================
Agrega 4 dimensões de saúde do mercado crypto:

1. % Alts acima da EMA50   — breadth interno (usa candles já coletados)
2. BTC Dominance           — peso do BTC no mercado total (CoinGecko free)
3. Funding Rate            — sentimento alavancado (Binance Futures free)
4. Open Interest Expansion — expansão de posições (Binance Futures free)

Retorna um MarketBreadthSnapshot com score composto (0–1) e sinais individuais.

Interpretação do score composto:
  ≥ 0.70 → mercado saudável       → permite entradas normais
  0.40–0.69 → mercado misto       → reduz size em 30%
  < 0.40 → breadth fraco          → bloqueia novas compras em alts

APIs utilizadas (todas gratuitas, sem API key):
  CoinGecko: https://api.coingecko.com/api/v3/global
  Binance:   https://fapi.binance.com/fapi/v1/fundingRate
             https://fapi.binance.com/futures/data/openInterestHist
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger("market_breadth")

# ── Cache TTL ──────────────────────────────────────────────────────────────
_cache: Dict = {}
_cache_ts: Dict[str, float] = {}
CACHE_TTL = {
    "breadth":    900,   # % alts acima EMA50 → 15 min (calculado internamente)
    "dominance":  1800,  # BTC dominance → 30 min
    "funding":    600,   # Funding rate → 10 min
    "oi":         900,   # Open Interest → 15 min
}

# Símbolos no Binance Futures para cada par (apenas os ativos atualmente monitorados)
BINANCE_SYMBOL = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
}


@dataclass
class MarketBreadthSnapshot:
    """Snapshot do estado de amplitude do mercado."""

    # ── Breadth interno ──────────────────────────────────────────────────
    alts_above_ema50_pct: float = 0.5   # 0–1: proporção de alts acima da EMA50
    alts_above_ema50_n:   int   = 0     # número absoluto de alts bullish

    # ── BTC Dominance ────────────────────────────────────────────────────
    btc_dominance: float = 0.50         # 0–1: dominância do BTC no market cap total
    # Interpretação: > 0.55 = risco-off (BTC domina, alts sofrendo)
    #                < 0.45 = risco-on (capital fluindo para alts)
    #                0.45–0.55 = neutro

    # ── Funding Rate ─────────────────────────────────────────────────────
    funding_rate_btc: float = 0.0       # taxa de funding BTC (% por 8h)
    funding_rate_avg: float = 0.0       # média das alts monitoradas
    # Interpretação: > +0.05% = longs sobrecarregados (risk de liquidação)
    #                < -0.05% = shorts sobrecarregados (short squeeze possível)
    #                -0.05% a +0.05% = equilibrado

    # ── Open Interest ────────────────────────────────────────────────────
    oi_expansion_btc: float = 0.0       # variação % do OI em 1h (BTC)
    oi_expansion_avg: float = 0.0       # variação % do OI em 1h (média alts)
    # Interpretação: > +2% = expansão de posições → tendência sustentável
    #                < -2% = fechamento de posições → fraqueza

    # ── Score composto ───────────────────────────────────────────────────
    score: float = 0.5                  # 0–1: saúde geral do mercado
    label: str = "NEUTRAL"              # "STRONG" | "MODERATE" | "WEAK" | "DANGER"

    # ── Metadados ────────────────────────────────────────────────────────
    ts: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)

    def size_multiplier(self) -> float:
        """Multiplicador de tamanho baseado na amplitude do mercado."""
        if self.score >= 0.70:
            return 1.0     # mercado saudável → size normal
        elif self.score >= 0.40:
            return 0.70    # mercado misto → reduz 30%
        else:
            return 0.0     # breadth fraco → bloqueia alts (retorna 0 = não entra)

    def should_block_alts(self) -> bool:
        """True se as condições de breadth recomendam evitar alts."""
        return self.score < 0.40 or self.btc_dominance > 0.60

    def to_dict(self) -> dict:
        return {
            # pairs_above_ema50_pct: enviado como % real (0–100) para o frontend
            "alts_above_ema50_pct": round(self.alts_above_ema50_pct * 100, 1),
            "alts_above_ema50_n":   self.alts_above_ema50_n,
            # btc_dominance: enviado como % real (0–100), ex: 60.5 para 60.5%
            "btc_dominance":        round(self.btc_dominance * 100, 1),
            "funding_rate_btc":     round(self.funding_rate_btc, 6),
            "funding_rate_avg":     round(self.funding_rate_avg, 6),
            "oi_expansion_btc":     round(self.oi_expansion_btc, 4),
            "oi_expansion_avg":     round(self.oi_expansion_avg, 4),
            "score":                round(self.score, 3),
            "label":                self.label,
            "size_multiplier":      self.size_multiplier(),
            "ts":                   self.ts,
        }


# ── 1. % Alts acima da EMA50 (calculado internamente com candles) ──────────

def calc_alts_above_ema50(candles_by_pair: Dict[str, list]) -> Tuple[float, int]:
    """
    Calcula a proporção de pares cujo preço atual está acima da EMA50.

    Args:
      candles_by_pair: {"BTC-USD": [...candles...], "ETH-USD": [...], ...}

    Returns:
      (proporção 0-1, contagem absoluta)
    """
    try:
        import pandas as pd
    except ImportError:
        return 0.5, 0

    total = 0
    above = 0

    for pair, candles in candles_by_pair.items():
        if not candles or len(candles) < 55:
            continue
        try:
            closes = [float(c["close"] if isinstance(c, dict) else c[4]) for c in candles]
            ser    = pd.Series(closes)
            ema50  = float(ser.ewm(span=50, adjust=False).mean().iloc[-1])
            price  = closes[-1]
            total += 1
            if price > ema50:
                above += 1
        except Exception:
            continue

    if total == 0:
        return 0.5, 0
    return above / total, above


# ── 2. BTC Dominance (CoinGecko) ──────────────────────────────────────────

def fetch_btc_dominance() -> float:
    """
    Busca a dominância do BTC via CoinGecko API (gratuita, sem API key).
    Cache de 30 min.
    """
    cache_key = "dominance"
    if cache_key in _cache and time.time() - _cache_ts.get(cache_key, 0) < CACHE_TTL[cache_key]:
        return _cache[cache_key]

    try:
        import requests
        url  = "https://api.coingecko.com/api/v3/global"
        resp = requests.get(url, timeout=8, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json().get("data", {})
        pct  = data.get("market_cap_percentage", {}).get("btc", 50.0)
        dom  = float(pct) / 100.0  # converte % para decimal
        _cache[cache_key] = dom
        _cache_ts[cache_key] = time.time()
        return dom
    except Exception as e:
        logger.debug(f"[Breadth] BTC dominance erro: {e}")
        return _cache.get(cache_key, 0.50)


# ── 3. Funding Rate (Binance Futures) ─────────────────────────────────────

def fetch_funding_rates(pairs: List[str]) -> Tuple[float, float]:
    """
    Busca as taxas de funding via Binance Futures API (pública, sem API key).
    Retorna (funding_rate_btc, funding_rate_avg_alts).
    Cache de 10 min.
    """
    cache_key = "funding"
    if cache_key in _cache and time.time() - _cache_ts.get(cache_key, 0) < CACHE_TTL[cache_key]:
        return _cache[cache_key]

    try:
        import requests
        rates = {}
        for pair in pairs:
            symbol = BINANCE_SYMBOL.get(pair)
            if not symbol:
                continue
            url  = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            resp = requests.get(url, timeout=6)
            if resp.ok:
                data = resp.json()
                if data:
                    rates[pair] = float(data[0].get("fundingRate", 0))

        btc_rate = rates.get("BTC-USD", 0.0)
        alt_rates = [v for k, v in rates.items() if k != "BTC-USD"]
        avg_rate  = sum(alt_rates) / len(alt_rates) if alt_rates else 0.0

        result = (btc_rate, avg_rate)
        _cache[cache_key] = result
        _cache_ts[cache_key] = time.time()
        return result
    except Exception as e:
        logger.debug(f"[Breadth] Funding rate erro: {e}")
        return _cache.get(cache_key, (0.0, 0.0))


# ── 4. Open Interest Expansion (Binance Futures) ──────────────────────────

def fetch_oi_expansion(pairs: List[str]) -> Tuple[float, float]:
    """
    Calcula a variação % do Open Interest em 1h via Binance Futures.
    Retorna (oi_change_btc_pct, oi_change_avg_alts_pct).
    Cache de 15 min.
    """
    cache_key = "oi"
    if cache_key in _cache and time.time() - _cache_ts.get(cache_key, 0) < CACHE_TTL[cache_key]:
        return _cache[cache_key]

    try:
        import requests
        changes = {}
        for pair in pairs:
            symbol = BINANCE_SYMBOL.get(pair)
            if not symbol:
                continue
            url  = (f"https://fapi.binance.com/futures/data/openInterestHist"
                    f"?symbol={symbol}&period=1h&limit=2")
            resp = requests.get(url, timeout=6)
            if not resp.ok:
                continue
            data = resp.json()
            if len(data) >= 2:
                oi_now  = float(data[-1].get("sumOpenInterest", 0))
                oi_prev = float(data[-2].get("sumOpenInterest", 0))
                if oi_prev > 0:
                    changes[pair] = (oi_now - oi_prev) / oi_prev * 100

        btc_change  = changes.get("BTC-USD", 0.0)
        alt_changes = [v for k, v in changes.items() if k != "BTC-USD"]
        avg_change  = sum(alt_changes) / len(alt_changes) if alt_changes else 0.0

        result = (btc_change, avg_change)
        _cache[cache_key] = result
        _cache_ts[cache_key] = time.time()
        return result
    except Exception as e:
        logger.debug(f"[Breadth] Open Interest erro: {e}")
        return _cache.get(cache_key, (0.0, 0.0))


# ── Score composto ─────────────────────────────────────────────────────────

def _compute_score(snap: MarketBreadthSnapshot) -> Tuple[float, str]:
    """
    Pontuação composta (0–1) ponderada pelos 4 indicadores.

    Pesos:
      35% Breadth alts EMA50 — quantas alts estão bullish
      25% BTC Dominance       — risco-on vs risco-off
      25% Funding Rate        — excesso de alavancagem
      15% OI Expansion        — confirmação de tendência
    """
    score = 0.0

    # ── 1. Alts acima EMA50 (35%) ─────────────────────────────────────
    # 100% acima = 1.0 | 50% = 0.5 | 0% = 0.0
    score += snap.alts_above_ema50_pct * 0.35

    # ── 2. BTC Dominance (25%) ────────────────────────────────────────
    # dom < 45% = risco-on (alts fluindo) = 1.0
    # dom 45-55% = neutro = 0.5
    # dom > 60% = risco-off = 0.0
    dom = snap.btc_dominance
    if dom <= 0.45:
        dom_score = 1.0
    elif dom <= 0.55:
        dom_score = 1.0 - (dom - 0.45) / 0.10  # linear 45-55%
    elif dom <= 0.65:
        dom_score = 0.5 - (dom - 0.55) / 0.10 * 0.5
    else:
        dom_score = 0.0
    score += dom_score * 0.25

    # ── 3. Funding Rate (25%) ─────────────────────────────────────────
    # Funding muito positivo = longs sobrecarregados = risco de dump
    # Funding negativo = short squeeze possível = oportunidade
    avg_funding = (snap.funding_rate_btc + snap.funding_rate_avg) / 2
    if avg_funding < -0.001:      # muito negativo → oportunidade
        fund_score = 1.0
    elif avg_funding < 0.0005:    # levemente positivo → ok
        fund_score = 0.75
    elif avg_funding < 0.001:     # moderado → cautela
        fund_score = 0.50
    elif avg_funding < 0.002:     # alto → risco
        fund_score = 0.25
    else:                          # excessivo → danger
        fund_score = 0.0
    score += fund_score * 0.25

    # ── 4. OI Expansion (15%) ─────────────────────────────────────────
    # OI crescendo = mais posições abertas = tendência sustentável
    # OI caindo = fechamento de posições = fraqueza
    avg_oi = (snap.oi_expansion_btc + snap.oi_expansion_avg) / 2
    if avg_oi > 3.0:
        oi_score = 1.0    # expansão forte
    elif avg_oi > 1.0:
        oi_score = 0.75
    elif avg_oi > -1.0:
        oi_score = 0.50   # estável
    elif avg_oi > -3.0:
        oi_score = 0.25
    else:
        oi_score = 0.0    # contração forte
    score += oi_score * 0.15

    # ── Label ──────────────────────────────────────────────────────────
    if score >= 0.70:
        label = "STRONG"
    elif score >= 0.55:
        label = "MODERATE"
    elif score >= 0.40:
        label = "WEAK"
    else:
        label = "DANGER"

    return round(score, 3), label


# ── Função principal ───────────────────────────────────────────────────────

def get_market_breadth(candles_by_pair: Dict[str, list],
                       pairs: Optional[List[str]] = None) -> MarketBreadthSnapshot:
    """
    Calcula o snapshot completo de amplitude de mercado.

    Args:
      candles_by_pair: candles 1H por par {"BTC-USD": [...], ...}
      pairs: lista de pares para funding/OI (padrão: todos)

    Returns:
      MarketBreadthSnapshot com todos os indicadores e score composto
    """
    if pairs is None:
        pairs = list(candles_by_pair.keys())

    snap   = MarketBreadthSnapshot()
    errors = []

    # ── Breadth interno (sempre disponível) ───────────────────────────
    try:
        snap.alts_above_ema50_pct, snap.alts_above_ema50_n = \
            calc_alts_above_ema50(candles_by_pair)
    except Exception as e:
        errors.append(f"breadth: {e}")

    # ── BTC Dominance ────────────────────────────────────────────────
    try:
        snap.btc_dominance = fetch_btc_dominance()
    except Exception as e:
        errors.append(f"dominance: {e}")

    # ── Funding Rate ─────────────────────────────────────────────────
    try:
        snap.funding_rate_btc, snap.funding_rate_avg = \
            fetch_funding_rates(pairs)
    except Exception as e:
        errors.append(f"funding: {e}")

    # ── Open Interest ────────────────────────────────────────────────
    try:
        snap.oi_expansion_btc, snap.oi_expansion_avg = \
            fetch_oi_expansion(pairs)
    except Exception as e:
        errors.append(f"oi: {e}")

    # ── Score composto ────────────────────────────────────────────────
    snap.score, snap.label = _compute_score(snap)
    snap.errors = errors
    snap.ts     = time.time()

    if errors:
        logger.debug(f"[Breadth] Erros parciais (não-críticos): {errors}")

    return snap
