import pandas as pd
from .base import BaseStrategy


class EMAPullback(BaseStrategy):
    """
    EMA Pullback Trend — 1H candles.
    Compra pullback ao EMA21 dentro de tendência confirmada.

    BUY  → EMA9 > EMA21 > EMA50 (uptrend alinhado)
           + low tocou EMA21 (pullback real, com tolerância)
           + close >= EMA21 (rejeição/recompra acima da média)
           + vela verde (close > open)
    SELL → EMA9 cruzou abaixo EMA21 por 2 velas consecutivas (anti-whipsaw)

    Fase 2 — simplificações aplicadas:
      Slope EMA50: removido — regime (bull/chop) já filtra tendência macro.
                  Filtrar slope dentro da estratégia duplica a lógica do regime.
      Volume pullback/breakout: removidos — dois filtros de volume numa
                  estratégia que já tem RVOL no Donchian.
                  Volume é mais relevante para breakouts (Donchian) do que
                  para pullbacks onde a vela de retomada nem sempre tem
                  volume extraordinário.
    Resultado: 4 condições limpas e independentes.
    """

    def __init__(self, fast: int = 9, mid: int = 21, slow: int = 50,
                 touch_tolerance_pct: float = 0.3):
        super().__init__("EMA Pullback")
        self.fast = fast
        self.mid  = mid
        self.slow = slow
        self.tol  = touch_tolerance_pct / 100.0

    def analyze(self, df: pd.DataFrame) -> str:
        min_bars = self.slow + 10
        if len(df) < min_bars:
            return "HOLD"

        df = df.copy()
        df["ema_f"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_m"] = df["close"].ewm(span=self.mid,  adjust=False).mean()
        df["ema_s"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df = df.dropna().reset_index(drop=True)
        if len(df) < 3:
            return "HOLD"

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # ── 1. Uptrend: EMAs alinhadas ────────────────────────────────────────
        in_uptrend = curr["ema_f"] > curr["ema_m"] > curr["ema_s"]

        # ── 2. Pullback real: low tocou a EMA21 ───────────────────────────────
        touched_ema21 = curr["low"] <= curr["ema_m"] * (1 + self.tol)

        # ── 3. Rejeição: close voltou acima da EMA21 ──────────────────────────
        reclaimed = curr["close"] >= curr["ema_m"]

        # ── 4. Vela verde: momentum de alta ───────────────────────────────────
        bull_candle = curr["close"] > curr["open"]

        if in_uptrend and touched_ema21 and reclaimed and bull_candle:
            return "BUY"

        # ── SELL: EMA9 abaixo EMA21 por 2 velas consecutivas (anti-whipsaw) ──
        if len(df) >= 3:
            prev2 = df.iloc[-3]
            two_candle_cross = (
                prev2["ema_f"] >= prev2["ema_m"]
                and prev["ema_f"]  < prev["ema_m"]
                and curr["ema_f"]  < curr["ema_m"]
            )
            if two_candle_cross:
                return "SELL"

        return "HOLD"
