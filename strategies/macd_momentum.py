import pandas as pd
from .base import BaseStrategy


class MACDMomentum(BaseStrategy):
    """
    MACD Momentum Surge — 1H candles.

    BUY  → Histograma cruza de negativo para positivo (impulso inicial)
           + close > EMA20 (preço acima filtro de curto prazo)
           + momentum positivo (close > close 3 barras atrás)
           + RSI < 75 (não sobrecomprado)
           + ADX rising (força da tendência em crescimento — não em exaustão)
    Nota: macd_above_zero removido — captura inicio de recuperação mesmo antes
    de a linha MACD cruzar zero (histograma + ADX já confirmam direção)
    SELL → Histograma cruza de positivo para negativo

    ADX Rising:
      ADX atual > ADX N barras atrás → tendência ganhando força (confirma momentum)
      ADX caindo → tendência em enfraquecimento → MACD pode ser sinal falso
      Combinado com MACD > 0, elimina entradas em rebounds fracos sem follow-through.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 ema_filter: int = 12,
                 rsi_period: int = 14,
                 rsi_max: float = 75.0,
                 adx_period: int = 14,
                 adx_rising_bars: int = 3):
        super().__init__("MACD Momentum")
        self.fast            = fast
        self.slow            = slow
        self.signal          = signal
        self.ema_filter      = ema_filter
        self.rsi_period      = rsi_period
        self.rsi_max         = rsi_max
        self.adx_period      = adx_period
        self.adx_rising_bars = adx_rising_bars  # ADX atual > ADX N barras atrás

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs    = gain / loss.replace(0, float("inf"))
        return 100 - (100 / (1 + rs))

    def _adx_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula a série completa do ADX para verificar se está subindo (rising).
        ADX rising = tendência ganhando força = momentum mais confiável.
        """
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
        return dx.ewm(alpha=1 / p, adjust=False).mean()

    def analyze(self, df: pd.DataFrame) -> str:
        min_bars = self.slow + self.signal + max(self.ema_filter, self.rsi_period) + self.adx_period * 2 + self.adx_rising_bars + 5
        if len(df) < min_bars:
            return "HOLD"

        df = df.copy()
        ema_fast      = df["close"].ewm(span=self.fast,   adjust=False).mean()
        ema_slow      = df["close"].ewm(span=self.slow,   adjust=False).mean()
        macd_line     = ema_fast - ema_slow
        sig_line      = macd_line.ewm(span=self.signal, adjust=False).mean()
        df["hist"]    = macd_line - sig_line
        df["macd"]    = macd_line
        df["ema_flt"] = df["close"].ewm(span=self.ema_filter, adjust=False).mean()
        df["rsi"]     = self._rsi(df["close"])
        df["adx"]     = self._adx_series(df)
        df = df.dropna().reset_index(drop=True)
        if len(df) < self.adx_rising_bars + 4:
            return "HOLD"

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # ── BUY ──────────────────────────────────────────────────────────────

        # 1. Histograma cruzou de negativo para positivo
        hist_cross_up = prev["hist"] <= 0 and curr["hist"] > 0

        # 2. Preço acima da EMA de curto prazo
        above_filter = curr["close"] > curr["ema_flt"]

        # 3. Momentum: preço maior que 3 barras atrás
        momentum_pos = curr["close"] > df["close"].iloc[-4]

        # 4. RSI não sobrecomprado
        rsi_ok = curr["rsi"] < self.rsi_max

        # 5. ADX rising: força da tendência crescendo nos últimos N períodos
        adx_now    = float(curr["adx"])
        adx_prev   = float(df["adx"].iloc[-(self.adx_rising_bars + 1)])
        adx_rising = adx_now > adx_prev

        if hist_cross_up and above_filter and momentum_pos and rsi_ok and adx_rising:
            return "BUY"

        # ── SELL: histograma cruzou de positivo para negativo ────────────────
        hist_cross_dn = prev["hist"] >= 0 and curr["hist"] < 0
        if hist_cross_dn:
            return "SELL"

        return "HOLD"
