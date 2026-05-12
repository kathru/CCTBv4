from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> str:
        """Returns 'BUY', 'SELL', or 'HOLD'"""
        pass

    def candles_to_df(self, candles: list) -> pd.DataFrame:
        df = pd.DataFrame(candles, columns=["start", "low", "high", "open", "close", "volume"])
        df = df.astype({"start": int, "low": float, "high": float,
                        "open": float, "close": float, "volume": float})
        df = df.sort_values("start").reset_index(drop=True)
        return df
