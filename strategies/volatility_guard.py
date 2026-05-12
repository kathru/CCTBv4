import pandas as pd
from .base import BaseStrategy


class VolatilityGuard(BaseStrategy):
    """
    Monitor de volatilidade e spread intradiário — candles 1H.

    Usa desvio padrão histórico em vez de threshold fixo:
      range_pct = (high - low) / close  →  proxy de spread intrabar

    Dois níveis de ação:
      BLOCK  → range_pct > mean + 1σ  (volatilidade elevada)
               Bloqueia novos BUYs — spread amplo aumenta custo efetivo
               e sinaliza instabilidade que reduz probabilidade de fill.

      SELL   → range_pct > mean + 2σ  (volatilidade extrema)
               Fecha posições abertas — risco de liquidação e slippage
               intenso justificam saída preventiva.

      HOLD   → volatilidade dentro do normal histórico.

    Vantagem sobre threshold fixo (era 25%):
      Auto-calibrado por ativo: BTC tolera menos range que SOL.
      Adapta ao regime atual: períodos calmos têm threshold menor,
      períodos agitados permitem mais margem sem disparar falso alarme.
    """

    def __init__(self, lookback: int = 24, block_std: float = 1.0,
                 sell_std: float = 2.0):
        """
        lookback  : janela histórica em candles 1H (24 = últimas 24h)
        block_std : múltiplo de σ para bloquear entradas (padrão 1σ)
        sell_std  : múltiplo de σ para fechar posições (padrão 2σ)
        """
        super().__init__("Vol Guard")
        self.lookback   = lookback
        self.block_std  = block_std
        self.sell_std   = sell_std

    def analyze(self, df: pd.DataFrame) -> str:
        min_bars = self.lookback + 5
        if len(df) < min_bars:
            return "HOLD"

        df = df.copy()
        df["high"]  = pd.to_numeric(df["high"],  errors="coerce")
        df["low"]   = pd.to_numeric(df["low"],   errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        # Spread intrabar: quanto o preço oscilou dentro da vela
        # Normalizado pelo close — comparável entre ativos e preços
        df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, float("nan"))
        df = df.dropna(subset=["range_pct"]).reset_index(drop=True)

        if len(df) < self.lookback + 2:
            return "HOLD"

        # Estatísticas históricas: exclui a vela atual (usa lookback anterior)
        hist = df["range_pct"].iloc[-(self.lookback + 1):-1]
        mean = float(hist.mean())
        std  = float(hist.std())

        if std <= 0:
            return "HOLD"

        # Candle mais recente fechado
        current = float(df["range_pct"].iloc[-2])

        threshold_sell  = mean + self.sell_std  * std   # 2σ → extremo
        threshold_block = mean + self.block_std * std   # 1σ → elevado

        if current >= threshold_sell:
            return "SELL"   # volatilidade extrema → fecha posições

        if current >= threshold_block:
            return "BLOCK"  # volatilidade elevada → bloqueia entradas

        return "HOLD"
