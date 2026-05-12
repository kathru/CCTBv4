import pandas as pd
from .base import BaseStrategy


class BBReversion(BaseStrategy):
    """
    Bollinger Band Mean Reversion — 1H candles.

    Estratégia de reversão à média projetada para mercados CHOP (laterais).
    Quando o preço se afasta significativamente da média, tende a retornar.

    BUY  → Close ABAIXO da banda inferior BB(20,2)  — preço oversold
           + RSI(14) < 35                            — momentum fraco confirmado
           + Candle fechado (close-confirmation)     — não apenas wick

    SELL → Close >= BB middle (EMA20)               — reversão completada
           (saída natural na média — TP técnico, não %)

    SL   → ATR × 2 (via _calc_exit no app.py) — same system as other strategies

    Regime gate (aplicado no app.py):
      CHOP → ativo (este é o regime alvo da estratégia)
      BULL → bloqueado (trend-following domina em tendência)
      BEAR → bloqueado (gate G1 padrão)

    Execução: Limit Order (maker 0.10%)
      Limit price = banda inferior BB (já estamos no nível exato)
      Alta probabilidade de fill: preço já está no nível alvo
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, rsi_oversold: float = 35.0):
        super().__init__("BB Reversion")
        self.bb_period   = bb_period
        self.bb_std      = bb_std
        self.rsi_period  = rsi_period
        self.rsi_oversold = rsi_oversold

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs    = gain / loss.replace(0, float("inf"))
        return 100 - (100 / (1 + rs))

    def analyze(self, df: pd.DataFrame) -> str:
        min_bars = self.bb_period + self.rsi_period + 5
        if len(df) < min_bars + 1:
            return "HOLD"

        df = df.copy()

        # Close-confirmation: descarta candle em formação
        df = df.iloc[:-1].reset_index(drop=True)

        closes = df["close"].astype(float)

        # Bollinger Bands
        df["bb_mid"]   = closes.rolling(self.bb_period).mean()
        df["bb_std"]   = closes.rolling(self.bb_period).std()
        df["bb_lower"] = df["bb_mid"] - self.bb_std * df["bb_std"]

        # RSI
        df["rsi"] = self._rsi(closes)
        df = df.dropna().reset_index(drop=True)
        if len(df) < 3:
            return "HOLD"

        curr = df.iloc[-1]

        # ── SELL: preço atingiu a média → reversão completa ──────────────────
        if curr["close"] >= curr["bb_mid"]:
            return "SELL"

        # ── BUY: preço abaixo da banda inferior + RSI oversold ──────────────
        below_lower = curr["close"] < curr["bb_lower"]
        rsi_ok      = curr["rsi"]   < self.rsi_oversold

        if below_lower and rsi_ok:
            return "BUY"

        return "HOLD"
