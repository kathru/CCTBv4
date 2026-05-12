import pandas as pd
from .base import BaseStrategy


class TrendFilter(BaseStrategy):
    """
    Filtro de tendência baseado na MA20 diária (mais rápida que MA50).
    BUY  → preço acima da MA + MA em alta (uptrend) — autoriza entradas.
    SELL → preço abaixo da MA ou MA em baixa (downtrend) — bloqueia entradas.
    HOLD → sem dados suficientes.
    """

    def __init__(self, period: int = 20):
        super().__init__("Trend")
        self.period = period

    def analyze(self, df: pd.DataFrame) -> str:
        if len(df) < self.period + 2:
            return "HOLD"

        df = df.copy()
        df["ma"] = df["close"].rolling(self.period).mean()
        df = df.dropna()

        price    = df["close"].iloc[-1]
        ma_curr  = df["ma"].iloc[-1]
        ma_prev  = df["ma"].iloc[-2]

        above_ma    = price > ma_curr
        ma_rising   = ma_curr > ma_prev

        if above_ma and ma_rising:
            return "BUY"    # uptrend confirmado
        return "SELL"       # downtrend ou neutro — bloqueia entradas
