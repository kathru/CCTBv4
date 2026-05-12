"""
Data Engine — Camada 1
======================
Transforma dados brutos da OKX em contexto estatístico estruturado:
  - OHLCV multi-timeframe
  - Microestrutura (orderbook imbalance, spread, taker buy/sell ratio)
  - Derivativos (funding rate, open interest, liquidation clusters)
  - Volatilidade realizada

Todos os métodos retornam dicts normalizados prontos para consumo
pelo Regime Engine (Camada 2) e Signal Engine (Camada 3).
"""

import time
import statistics
import math
import requests
from typing import Optional


# Símbolos OKX para derivativos (perp swaps)
_PERP_MAP = {
    "BTC-USD": "BTC-USDT-SWAP",
    "ETH-USD": "ETH-USDT-SWAP",
    "SOL-USD": "SOL-USDT-SWAP",
}

_SPOT_MAP = {
    "BTC-USD": "BTC-USDT",
    "ETH-USD": "ETH-USDT",
    "SOL-USD": "SOL-USDT",
}

BASE_URL = "https://www.okx.com"
_SESSION = requests.Session()
_SESSION.headers.update({"Content-Type": "application/json", "Accept": "application/json"})


def _get(path: str, params: dict = None, timeout: int = 8) -> Optional[dict]:
    try:
        r = _SESSION.get(f"{BASE_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        if d.get("code") == "0":
            return d
        return None
    except Exception:
        return None


# ── Realized Volatility ───────────────────────────────────────────────────────

def calc_realized_vol(closes: list, window: int = 24) -> float:
    """
    Volatilidade realizada anualizada usando log-returns.
    Janela padrão: 24 candles 1H = 1 dia de dados.
    """
    if len(closes) < window + 1:
        return 0.0
    recent = closes[-(window + 1):]
    log_returns = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]
    if len(log_returns) < 2:
        return 0.0
    std = statistics.stdev(log_returns)
    # Anualiza: 1H candles → 8760 períodos/ano
    return std * math.sqrt(8760)


def calc_vol_percentile(closes: list, window_short: int = 24, window_long: int = 168) -> float:
    """
    Percentil da volatilidade atual vs histórico de 7 dias (168 candles 1H).
    Retorna 0.0-1.0: onde 1.0 = volatilidade máxima histórica recente.
    """
    if len(closes) < window_long + 1:
        return 0.5
    vol_history = []
    for i in range(window_long - window_short, window_long):
        subset = closes[i: i + window_short + 1]
        if len(subset) > window_short:
            vol_history.append(calc_realized_vol(subset, window_short))
    if not vol_history:
        return 0.5
    current_vol = calc_realized_vol(closes[-window_short - 1:], window_short)
    below = sum(1 for v in vol_history if v <= current_vol)
    return below / len(vol_history)


def calc_atr_rate(highs: list, lows: list, closes: list, period: int = 14) -> dict:
    """
    ATR atual vs ATR período anterior.
    Retorna:
      atr_current  — ATR das últimas `period` velas
      atr_prev     — ATR do período anterior
      expansion    — atr_current / atr_prev (>1 = expansão, <1 = compressão)
    """
    if len(closes) < period * 2 + 1:
        return {"atr_current": 0.0, "atr_prev": 0.0, "expansion": 1.0}

    def _atr(h, l, c, start, end):
        trs = []
        for i in range(start + 1, end + 1):
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0

    n = len(closes)
    atr_cur  = _atr(highs, lows, closes, n - period - 1, n - 1)
    atr_prev = _atr(highs, lows, closes, n - period * 2 - 1, n - period - 1)
    expansion = (atr_cur / atr_prev) if atr_prev > 0 else 1.0

    return {
        "atr_current":  round(atr_cur, 6),
        "atr_prev":     round(atr_prev, 6),
        "expansion":    round(expansion, 4),
    }


# ── Orderbook Microstructure ──────────────────────────────────────────────────

def get_orderbook_context(pair: str) -> dict:
    """
    Busca orderbook e calcula:
      imbalance   — pressão compradora vs vendedora (-1 a +1)
      spread_pct  — spread como % do mid price
      bid_depth   — liquidez nos primeiros 1% abaixo do mid
      ask_depth   — liquidez nos primeiros 1% acima do mid
    """
    inst_id = _SPOT_MAP.get(pair, pair)
    data = _get("/api/v5/market/books", {"instId": inst_id, "sz": "50"})

    empty = {"imbalance": 0.0, "spread_pct": 0.0, "bid_depth": 0.0, "ask_depth": 0.0, "mid_price": 0.0}
    if not data or not data.get("data"):
        return empty

    d = data["data"][0]
    bids = [[float(b[0]), float(b[1])] for b in d.get("bids", [])]
    asks = [[float(a[0]), float(a[1])] for a in d.get("asks", [])]

    if not bids or not asks:
        return empty

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2
    spread_pct = (best_ask - best_bid) / mid if mid > 0 else 0.0

    threshold = mid * 0.01  # 1% do mid
    bid_depth = sum(p * q for p, q in bids if p >= mid - threshold)
    ask_depth = sum(p * q for p, q in asks if p <= mid + threshold)
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

    return {
        "imbalance":  round(imbalance, 4),    # positivo = pressão compradora
        "spread_pct": round(spread_pct, 6),
        "bid_depth":  round(bid_depth, 2),
        "ask_depth":  round(ask_depth, 2),
        "mid_price":  round(mid, 4),
    }


# ── Taker Buy/Sell Ratio ──────────────────────────────────────────────────────

def get_taker_ratio(pair: str) -> dict:
    """
    Ratio de volume taker comprador vs vendedor (via trades recentes).
    Retorna:
      buy_ratio   — % do volume taker que foi compra (0-1)
      sell_ratio  — % do volume taker que foi venda (0-1)
      imbalance   — buy_ratio - sell_ratio (-1 a +1)
    """
    inst_id = _SPOT_MAP.get(pair, pair)
    data = _get("/api/v5/market/trades", {"instId": inst_id, "limit": "100"})

    empty = {"buy_ratio": 0.5, "sell_ratio": 0.5, "imbalance": 0.0}
    if not data or not data.get("data"):
        return empty

    buy_vol = sell_vol = 0.0
    for t in data["data"]:
        sz = float(t.get("sz", 0))
        side = t.get("side", "")
        if side == "buy":
            buy_vol += sz
        else:
            sell_vol += sz

    total = buy_vol + sell_vol
    if total == 0:
        return empty

    buy_r = buy_vol / total
    sell_r = sell_vol / total
    return {
        "buy_ratio":  round(buy_r, 4),
        "sell_ratio": round(sell_r, 4),
        "imbalance":  round(buy_r - sell_r, 4),
    }


# ── Funding Rate ──────────────────────────────────────────────────────────────

def get_funding_rate(pair: str) -> dict:
    """
    Funding rate atual do perp swap.
    Retorna:
      funding_rate     — taxa atual (ex: 0.0001 = 0.01%)
      annualized_pct   — taxa anualizada em %
      sentiment        — 'bullish_excess' | 'bearish_excess' | 'neutral'
    """
    inst_id = _PERP_MAP.get(pair)
    if not inst_id:
        return {"funding_rate": 0.0, "annualized_pct": 0.0, "sentiment": "neutral"}

    data = _get("/api/v5/public/funding-rate", {"instId": inst_id})
    empty = {"funding_rate": 0.0, "annualized_pct": 0.0, "sentiment": "neutral"}

    if not data or not data.get("data"):
        return empty

    rate = float(data["data"][0].get("fundingRate", 0))
    # 3x por dia × 365 dias
    annualized = rate * 3 * 365 * 100

    if rate > 0.0005:
        sentiment = "bullish_excess"   # longs pagando caro — reversão possível
    elif rate < -0.0002:
        sentiment = "bearish_excess"   # shorts pagando — pressão short
    else:
        sentiment = "neutral"

    return {
        "funding_rate":   round(rate, 6),
        "annualized_pct": round(annualized, 2),
        "sentiment":      sentiment,
    }


# ── Open Interest ─────────────────────────────────────────────────────────────

def get_open_interest(pair: str) -> dict:
    """
    Open interest atual do perp swap + variação vs média recente.
    Retorna:
      oi_usd         — OI em USD
      oi_change_pct  — variação % vs hora anterior
      expanding      — True se OI crescendo (novo dinheiro entrando)
    """
    inst_id = _PERP_MAP.get(pair)
    if not inst_id:
        return {"oi_usd": 0.0, "oi_change_pct": 0.0, "expanding": False}

    # Histórico de OI (última hora)
    data = _get("/api/v5/rubik/stat/contracts/open-interest-history", {
        "instId": inst_id, "period": "1H", "limit": "5"
    })

    empty = {"oi_usd": 0.0, "oi_change_pct": 0.0, "expanding": False}
    if not data or not data.get("data"):
        # Fallback: OI atual apenas
        cur = _get("/api/v5/public/open-interest", {"instId": inst_id})
        if cur and cur.get("data"):
            oi = float(cur["data"][0].get("oiCcy", 0))
            return {"oi_usd": oi, "oi_change_pct": 0.0, "expanding": False}
        return empty

    rows = data["data"]
    if len(rows) < 2:
        return empty

    # rows são DESC por tempo
    oi_now  = float(rows[0][1]) if len(rows[0]) > 1 else 0.0
    oi_prev = float(rows[1][1]) if len(rows[1]) > 1 else 0.0
    change_pct = ((oi_now - oi_prev) / oi_prev * 100) if oi_prev > 0 else 0.0

    return {
        "oi_usd":        round(oi_now, 2),
        "oi_change_pct": round(change_pct, 4),
        "expanding":     change_pct > 0.5,   # >0.5% expansão = novo dinheiro
    }


# ── Correlation Matrix ────────────────────────────────────────────────────────

def calc_correlation_matrix(closes_map: dict, window: int = 20) -> dict:
    """
    Calcula matriz de correlação rolling entre pares.
    closes_map: {"BTC-USD": [closes...], "ETH-USD": [...], "SOL-USD": [...]}
    Retorna dict com correlações par-a-par e effective_positions.
    """
    pairs = list(closes_map.keys())
    result = {}

    for i, p1 in enumerate(pairs):
        for j, p2 in enumerate(pairs):
            if j <= i:
                continue
            c1 = closes_map[p1][-window:]
            c2 = closes_map[p2][-window:]
            if len(c1) < window or len(c2) < window:
                result[f"{p1}_{p2}"] = 0.5
                continue
            # Log returns
            r1 = [math.log(c1[k] / c1[k - 1]) for k in range(1, len(c1))]
            r2 = [math.log(c2[k] / c2[k - 1]) for k in range(1, len(c2))]
            # Pearson correlation
            n = len(r1)
            mean1, mean2 = sum(r1) / n, sum(r2) / n
            cov = sum((r1[k] - mean1) * (r2[k] - mean2) for k in range(n)) / n
            std1 = math.sqrt(sum((x - mean1) ** 2 for x in r1) / n)
            std2 = math.sqrt(sum((x - mean2) ** 2 for x in r2) / n)
            corr = cov / (std1 * std2) if std1 * std2 > 0 else 0.0
            result[f"{p1}_{p2}"] = round(corr, 4)

    # Effective positions: Σ(1 / (1 + avg_corr_of_pair))
    # Métrica de diversificação real
    if len(pairs) > 1:
        avg_corr = sum(result.values()) / len(result) if result else 0.5
        effective = len(pairs) / (1 + (len(pairs) - 1) * avg_corr)
    else:
        effective = 1.0

    result["effective_positions"] = round(effective, 2)
    result["avg_correlation"]     = round(sum(v for k, v in result.items()
                                              if "effective" not in k and "avg" not in k) /
                                          max(len(result) - 2, 1), 4)
    return result


# ── Volume Delta ─────────────────────────────────────────────────────────────

def calc_volume_delta(candles: list, window: int = 10) -> dict:
    """
    Proxy de volume delta: compara volume de candles de alta vs baixa.
    Retorna:
      delta        — pressão líquida (positivo = compradora)
      delta_pct    — delta como % do volume total
      aggressive   — True se delta > 30% do volume total
    """
    if len(candles) < window:
        return {"delta": 0.0, "delta_pct": 0.0, "aggressive": False}

    recent = candles[-window:]
    buy_vol = sell_vol = 0.0
    for c in recent:
        vol = float(c.get("volume", 0))
        op  = float(c.get("open", 0))
        cl  = float(c.get("close", 0))
        if cl >= op:
            buy_vol += vol
        else:
            sell_vol += vol

    total = buy_vol + sell_vol
    delta = buy_vol - sell_vol
    delta_pct = delta / total if total > 0 else 0.0

    return {
        "delta":      round(delta, 4),
        "delta_pct":  round(delta_pct, 4),
        "aggressive": abs(delta_pct) > 0.30,
    }


# ── Market Context Bundle ─────────────────────────────────────────────────────

def get_market_context(pair: str, candles_1h: list, closes_map: dict = None) -> dict:
    """
    Agrega todos os dados de microestrutura em um único contexto.
    É o output principal da Camada 1 — consumido por Camada 2 e 3.
    """
    closes = [float(c["close"]) for c in candles_1h]
    highs  = [float(c["high"])  for c in candles_1h]
    lows   = [float(c["low"])   for c in candles_1h]

    # Dados de microestrutura (chamadas paralelas possíveis)
    ob      = get_orderbook_context(pair)
    taker   = get_taker_ratio(pair)
    funding = get_funding_rate(pair)
    oi      = get_open_interest(pair)

    # Métricas de volatilidade (computacionais — sem I/O)
    rvol        = calc_realized_vol(closes, window=24)
    vol_pct     = calc_vol_percentile(closes)
    atr_rate    = calc_atr_rate(highs, lows, closes)
    vol_delta   = calc_volume_delta(candles_1h, window=10)

    # Correlação (opcional — requer closes de múltiplos pares)
    correlation = {}
    if closes_map and len(closes_map) > 1:
        correlation = calc_correlation_matrix(closes_map)

    return {
        "pair":            pair,
        "timestamp":       int(time.time()),

        # Microestrutura
        "orderbook":       ob,
        "taker_ratio":     taker,

        # Derivativos
        "funding":         funding,
        "open_interest":   oi,

        # Volatilidade
        "realized_vol":    round(rvol, 4),
        "vol_percentile":  round(vol_pct, 4),
        "atr_rate":        atr_rate,

        # Fluxo
        "volume_delta":    vol_delta,

        # Correlação
        "correlation":     correlation,
    }
