import pandas as pd
from .base import BaseStrategy


class DonchianBreakout(BaseStrategy):
    """
    Donchian Channel Breakout (estilo Turtle Traders) — 1H principal, confirmação 6H.

    BUY  → Candle FECHADO acima da banda superior (close confirmado, não wick)
           + RVOL >= 1.3× média (volume acima da média — participação real)
           + ADX > 20 (tendência confirmada — sem lateralização)
           + RSI(14) > 45 (momentum positivo)
    SELL → Close < menor low das últimas N barras (saída técnica)

    Fase 2 — simplificações aplicadas:
      RVOL: threshold simples >= 1.3 (era: aceleração crescente 2-3 candles).
            Mais estável e menos path-dependent. 30% acima da média
            já indica breakout com participação real.
      OBV:  removido — RVOL já confirma participação de volume.
            Dois indicadores de volume = redundância sem informação extra.

    CLOSE-CONFIRMATION (mantida):
      Descarta candle em formação (iloc[:-1]) — apenas fechamentos reais.
    """

    def __init__(self, period: int = 20, rsi_period: int = 14,
                 rsi_min: float = 45.0, vol_mult: float = 1.0,
                 adx_period: int = 14, adx_min: float = 20.0,
                 rvol_period: int = 20, rvol_min: float = 1.3):
        super().__init__("Donchian Breakout")
        self.period      = period
        self.rsi_period  = rsi_period
        self.rsi_min     = rsi_min
        self.vol_mult    = vol_mult
        self.adx_period  = adx_period
        self.adx_min     = adx_min
        self.rvol_period = rvol_period
        self.rvol_min    = rvol_min

    # ── Indicadores internos ──────────────────────────────────────────────────

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs    = gain / loss.replace(0, float("inf"))
        return 100 - (100 / (1 + rs))

    def _adx(self, df: pd.DataFrame) -> float:
        """ADX — mede a força da tendência. > 20 = tendência presente."""
        p = self.adx_period
        high, low, close = df["high"], df["low"], df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        dm_plus  = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)
        dm_plus  = dm_plus.where(dm_plus > dm_minus, 0.0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0.0)

        atr_s    = tr.ewm(alpha=1 / p, adjust=False).mean()
        di_plus  = 100 * dm_plus.ewm(alpha=1 / p, adjust=False).mean() / atr_s.replace(0, 1e-9)
        di_minus = 100 * dm_minus.ewm(alpha=1 / p, adjust=False).mean() / atr_s.replace(0, 1e-9)
        dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-9)
        adx      = dx.ewm(alpha=1 / p, adjust=False).mean()
        return float(adx.iloc[-1])

    def _rvol(self, df: pd.DataFrame) -> float:
        """
        Relative Volume = volume_atual / média_N_períodos_anteriores.
        RVOL >= 1.3 → breakout com participação real (30% acima da média).
        """
        if len(df) < self.rvol_period + 2:
            return 1.0
        vol_now = float(df["volume"].iloc[-1])
        vol_avg = float(df["volume"].iloc[-(self.rvol_period + 1):-1].mean())
        if vol_avg <= 0:
            return 1.0
        return vol_now / vol_avg

    # ── Análise principal ─────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> str:
        min_bars = self.period + self.rsi_period + self.adx_period * 2 + self.rvol_period + 5
        if len(df) < min_bars + 1:
            return "HOLD"

        df = df.copy()

        # Close-confirmation: descarta candle em formação
        df = df.iloc[:-1].reset_index(drop=True)

        df["dc_upper"] = df["high"].rolling(self.period).max().shift(1)
        df["dc_lower"] = df["low"].rolling(self.period).min().shift(1)
        df["rsi"]      = self._rsi(df["close"])
        df = df.dropna().reset_index(drop=True)
        if len(df) < self.rvol_period + 2:
            return "HOLD"

        curr = df.iloc[-1]

        # ── BUY: close acima da banda (não wick) ─────────────────────────────
        price_breakout = curr["close"] > curr["dc_upper"]
        rsi_ok         = curr["rsi"] >= self.rsi_min

        if not (price_breakout and rsi_ok):
            if curr["close"] < curr["dc_lower"]:
                return "SELL"
            return "HOLD"

        # ── RVOL: threshold simples ≥ 1.3 (Fase 2) ───────────────────────────
        rvol_value = self._rvol(df)
        if rvol_value < self.rvol_min:
            return "HOLD"

        # ── ADX: tendência presente ───────────────────────────────────────────
        adx_value = self._adx(df)
        if adx_value < self.adx_min:
            return "HOLD"

        return "BUY"
