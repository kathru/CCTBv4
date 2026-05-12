"""
Market Regime Detection & Technical Filters
============================================
Funções utilitárias para detecção de regime de mercado e filtros de confirmação.

Módulos implementados:
  1. ADX  — detecta se mercado está em tendência ou lateralização
  2. ATR  — calcula volatilidade real do ativo para SL dinâmico
  3. OBV  — confirma breakouts com volume institucional
  4. MTF  — verifica alinhamento com timeframe maior (EMA200 6H)
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. ADX — Average Directional Index (força de tendência)
# ─────────────────────────────────────────────────────────────────────────────

def calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calcula o ADX (0–100).
    ADX > 25 → tendência forte   → Donchian + EMA Pullback habilitados
    ADX < 20 → mercado lateral   → apenas Stoch Bounce habilitado
    """
    if len(df) < period * 2 + 5:
        return 20.0  # neutro como fallback

    df = df.copy()
    high = df["high"]
    low  = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Directional Movements
    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    # Smoothed (Wilder EMA)
    atr_s   = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean()  / atr_s.replace(0, 1e-9)
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, 1e-9)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return float(adx.iloc[-1])


def detect_regime(df: pd.DataFrame,
                  adx_trend: float = 25.0,
                  adx_range: float = 20.0) -> str:
    """
    Retorna o regime de mercado com base no ADX:
      'trending'  → ADX > 25   (usar Donchian + EMA Pullback)
      'ranging'   → ADX < 20   (usar apenas Stoch Bounce)
      'neutral'   → 20 ≤ ADX ≤ 25  (posição reduzida, qualquer estratégia)
    """
    adx = calc_adx(df)
    if adx > adx_trend:
        return "trending"
    if adx < adx_range:
        return "ranging"
    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ATR — Average True Range (SL baseado em volatilidade real)
# ─────────────────────────────────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calcula o ATR (Average True Range) atual.
    Usado para SL dinâmico: SL = entry - (multiplier × ATR)
    """
    if len(df) < period + 5:
        return 0.0

    df = df.copy()
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-1])


def atr_stop_loss(entry_price: float, df: pd.DataFrame,
                  multiplier: float = 2.0, period: int = 14,
                  min_pct: float = 0.03, max_pct: float = 0.12) -> float:
    """
    Calcula o preço de Stop Loss baseado em ATR.
    SL = entry_price - (multiplier × ATR)

    Parâmetros:
      multiplier : quanto de ATR abaixo da entrada (padrão 2.0×)
      min_pct    : SL mínimo de 3% (não deixa SL muito próximo)
      max_pct    : SL máximo de 12% (não deixa SL excessivamente largo)

    Retorna: preço do SL (float)
    """
    atr = calc_atr(df, period)
    if atr <= 0 or entry_price <= 0:
        return entry_price * (1 - min_pct)

    sl_price = entry_price - (multiplier * atr)
    sl_pct   = (entry_price - sl_price) / entry_price

    # Clamp entre min e max
    sl_pct = max(min_pct, min(max_pct, sl_pct))
    return entry_price * (1 - sl_pct)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OBV — On-Balance Volume (confirma breakouts com volume)
# ─────────────────────────────────────────────────────────────────────────────

def calc_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV):
    +volume se close > close anterior, -volume se close < close anterior.
    """
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["volume"]).cumsum()


def obv_rising(df: pd.DataFrame, lookback: int = 5) -> bool:
    """
    Retorna True se o OBV está em tendência de alta nas últimas `lookback` barras.
    Usado para confirmar breakouts do Donchian com volume institucional real.
    """
    if len(df) < lookback + 5:
        return True  # sem dados suficientes → não bloqueia

    obv = calc_obv(df)
    # Compara OBV atual com a média das últimas N barras
    obv_now  = float(obv.iloc[-1])
    obv_prev = float(obv.iloc[-(lookback + 1)])
    return obv_now > obv_prev


def mfi_bullish(df: pd.DataFrame, period: int = 14, threshold: float = 40.0) -> bool:
    """
    Money Flow Index > threshold → fluxo de dinheiro positivo (compradores ativos).
    MFI < 20 = sobrevenda extrema, > 80 = sobrecompra extrema.
    Usado como filtro adicional para confirmar entradas.
    """
    if len(df) < period + 5:
        return True  # sem dados → não bloqueia

    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow    = typical_price * df["volume"]

    pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    pos_mf = pos_flow.rolling(period).sum()
    neg_mf = neg_flow.rolling(period).sum()

    mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, 1e-9)))
    return float(mfi.iloc[-1]) > threshold


# ─────────────────────────────────────────────────────────────────────────────
# 4. MTF — Multi-Timeframe Check (EMA200 no timeframe maior)
# ─────────────────────────────────────────────────────────────────────────────

def price_above_ema200(df_higher_tf: pd.DataFrame) -> bool:
    """
    Retorna True se o preço atual está ACIMA da EMA200 no timeframe maior (6H/diário).
    Garante que só compramos a favor da tendência principal.
    'The trend is your friend' — operar apenas na direção do fluxo macro.
    """
    if len(df_higher_tf) < 200:
        return True  # sem dados suficientes → não bloqueia

    df = df_higher_tf.copy()
    ema200 = df["close"].ewm(span=200, adjust=False).mean()
    return float(df["close"].iloc[-1]) > float(ema200.iloc[-1])


def mtf_trend_bullish(df_higher_tf: pd.DataFrame, ema_fast: int = 50, ema_slow: int = 200) -> bool:
    """
    Confirmação de tendência no timeframe maior:
    EMA50 > EMA200 (Golden Cross macro) AND preço acima da EMA50.
    Mais relaxado que EMA200 puro para não bloquear excessivamente.
    """
    if len(df_higher_tf) < ema_slow:
        # Fallback: se não tiver dados suficientes, usa EMA50 apenas
        if len(df_higher_tf) < ema_fast:
            return True
        df = df_higher_tf.copy()
        ema50 = df["close"].ewm(span=ema_fast, adjust=False).mean()
        return float(df["close"].iloc[-1]) > float(ema50.iloc[-1])

    df = df_higher_tf.copy()
    ema50_s  = df["close"].ewm(span=ema_fast, adjust=False).mean()
    ema200_s = df["close"].ewm(span=ema_slow, adjust=False).mean()
    price    = float(df["close"].iloc[-1])

    # Preço acima da EMA50 E EMA50 acima da EMA200
    return price > float(ema50_s.iloc[-1]) and float(ema50_s.iloc[-1]) > float(ema200_s.iloc[-1])
