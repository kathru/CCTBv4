"""
OKX Market Data Client
======================
Cliente para a API pública da OKX (market data — sem autenticação necessária).
Mantém a mesma interface do CoinbaseClient para compatibilidade com o bot.

Endpoints utilizados (todos públicos):
  GET /api/v5/market/ticker?instId=BTC-USDT
  GET /api/v5/market/candles?instId=BTC-USDT&bar=1H&limit=200

Mapeamento de pares (interno → OKX):
  BTC-USD → BTC-USDT
  ETH-USD → ETH-USDT
  SOL-USD → SOL-USDT

Formato de candle OKX: [ts_ms, open, high, low, close, vol, volCcy, ...]
Normalizado para dict: {"start": ts_s, "low": float, "high": float,
                         "open": float, "close": float, "volume": float}
"""

import time
import hmac
import hashlib
import base64
import requests
from datetime import datetime, timezone


class OKXClient:
    BASE_URL = "https://www.okx.com"

    # Mapeamento: par interno (BTC-USD) → símbolo OKX (BTC-USDT)
    PAIR_MAP = {
        "BTC-USD":  "BTC-USDT",
        "ETH-USD":  "ETH-USDT",
        "SOL-USD":  "SOL-USDT",
        "AVAX-USD": "AVAX-USDT",
        "LINK-USD": "LINK-USDT",
        "DOGE-USD": "DOGE-USDT",
    }

    # Mapeamento: granularidade Coinbase → bar OKX
    GRAN_MAP = {
        "ONE_MINUTE":     "1m",
        "FIVE_MINUTE":    "5m",
        "FIFTEEN_MINUTE": "15m",
        "THIRTY_MINUTE":  "30m",
        "ONE_HOUR":       "1H",
        "TWO_HOUR":       "2H",
        "SIX_HOUR":       "6H",
        "ONE_DAY":        "1D",
    }

    def __init__(self, api_key: str = "", secret_key: str = "", passphrase: str = ""):
        """
        api_key, secret_key, passphrase: credenciais OKX.
        Para endpoints públicos (market data), não são necessárias.
        """
        self.api_key    = api_key or ""
        self.secret_key = secret_key or ""
        self.passphrase = passphrase or ""
        self._session   = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
        })

    # ── Autenticação (para endpoints privados, não usada no paper trading) ─
    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        msg = f"{timestamp}{method.upper()}{path}{body}"
        sig = hmac.new(self.secret_key.encode(), msg.encode(), hashlib.sha256).digest()
        return base64.b64encode(sig).decode()

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return {
            "OK-ACCESS-KEY":        self.api_key,
            "OK-ACCESS-SIGN":       self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }

    # ── Helpers ────────────────────────────────────────────────────────────
    def _inst_id(self, product_id: str) -> str:
        """Converte par interno (BTC-USD) para símbolo OKX (BTC-USDT)."""
        return self.PAIR_MAP.get(product_id, product_id.replace("-USD", "-USDT"))

    def _get_public(self, path: str, params: dict = None) -> dict:
        """Request GET público — sem autenticação."""
        resp = self._session.get(
            f"{self.BASE_URL}{path}",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise ValueError(f"OKX API erro {data.get('code')}: {data.get('msg')}")
        return data

    # ── Interface pública (compatível com CoinbaseClient) ──────────────────

    def get_ticker(self, product_id: str) -> dict:
        """
        Busca ticker (preço atual, variação 24h, volume 24h).
        Retorna dict compatível com o que o bot espera:
          price, price_percentage_change_24h, volume_24h
        """
        inst_id = self._inst_id(product_id)
        data    = self._get_public("/api/v5/market/ticker", {"instId": inst_id})
        d       = data["data"][0]

        price    = float(d.get("last", 0))
        open24h  = float(d.get("open24h", price) or price)
        pct_chg  = ((price - open24h) / open24h * 100) if open24h else 0.0
        vol24h   = float(d.get("vol24h", 0))   # volume em moeda base (BTC, ETH...)

        return {
            "price":                       price,
            "price_percentage_change_24h": round(pct_chg, 4),
            "volume_24h":                  vol24h,
            # campos extras para compatibilidade
            "product_id": product_id,
            "bid":  float(d.get("bidPx", 0)),
            "ask":  float(d.get("askPx", 0)),
        }

    def get_candles(self, product_id: str, granularity: str = "ONE_HOUR",
                    limit: int = 200) -> list:
        """
        Busca candles históricos.

        Retorna lista de dicts normalizados:
          [{"start": ts_s, "low": float, "high": float,
            "open": float, "close": float, "volume": float}, ...]

        OKX retorna candles mais recentes primeiro; inverte para ordem cronológica.
        """
        inst_id = self._inst_id(product_id)
        bar     = self.GRAN_MAP.get(granularity, "1H")
        lim     = min(limit, 300)   # OKX max: 300 candles por request

        data = self._get_public("/api/v5/market/candles", {
            "instId": inst_id,
            "bar":    bar,
            "limit":  str(lim),
        })

        candles = []
        for row in reversed(data["data"]):   # inverte: OKX retorna DESC
            # OKX format: [ts_ms, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
            try:
                candles.append({
                    "start":  int(row[0]) // 1000,   # ms → seconds
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                })
            except (IndexError, ValueError):
                continue

        return candles

    def get_order_book(self, product_id: str, limit: int = 50) -> dict:
        """Order book (bids/asks). Compatibilidade com CoinbaseClient."""
        inst_id = self._inst_id(product_id)
        data    = self._get_public("/api/v5/market/books", {
            "instId": inst_id, "sz": str(min(limit, 400))
        })
        d = data["data"][0]
        return {
            "bids": [[float(b[0]), float(b[1])] for b in d.get("bids", [])],
            "asks": [[float(a[0]), float(a[1])] for a in d.get("asks", [])],
        }
