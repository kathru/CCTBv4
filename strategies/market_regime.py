"""
Technical Utilities
===================
Funções utilitárias de baixo nível usadas pelo app.py.
Cálculos de ADX e ATR via pandas para compatibilidade com o loop V3 legado.
"""

import pandas as pd


def calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX (0–100). Retorna 20.0 como fallback se dados insuficientes."""
    if len(df) < period * 2 + 5:
        return 20.0

    df = df.copy()
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr_s    = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, 1e-9)
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, 1e-9)

    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return float(adx.iloc[-1])


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR atual. Retorna 0.0 se dados insuficientes."""
    if len(df) < period + 5:
        return 0.0

    df = df.copy()
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-1])
