import jwt
import time
import hashlib
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


class CoinbaseClient:
    BASE_URL = "https://api.coinbase.com"

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key

    def _build_jwt(self, method: str, path: str) -> str:
        now = int(time.time())
        pem = self.secret_key.replace("\\n", "\n").encode()
        private_key = serialization.load_pem_private_key(
            pem, password=None, backend=default_backend()
        )
        payload = {
            "sub": self.api_key,
            "iss": "coinbase-cloud",
            "nbf": now,
            "exp": now + 120,
            "iat": now,
            "uri": f"{method} api.coinbase.com{path}",
        }
        return jwt.encode(payload, private_key, algorithm="ES256",
                          headers={"kid": self.api_key})

    def _get(self, path: str, params: dict = None) -> dict:
        token = self._build_jwt("GET", path)
        resp = requests.get(
            f"{self.BASE_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        token = self._build_jwt("POST", path)
        resp = requests.post(
            f"{self.BASE_URL}{path}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_accounts(self) -> list:
        data = self._get("/api/v3/brokerage/accounts")
        return data.get("accounts", [])

    def get_candles(self, product_id: str, granularity: str = "ONE_HOUR", limit: int = 200) -> list:
        end = int(time.time())
        granularity_seconds = {
            "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800, "ONE_HOUR": 3600, "TWO_HOUR": 7200,
            "SIX_HOUR": 21600, "ONE_DAY": 86400,
        }
        start = end - limit * granularity_seconds.get(granularity, 3600)
        data = self._get(f"/api/v3/brokerage/products/{product_id}/candles", {
            "start": str(start), "end": str(end), "granularity": granularity
        })
        return data.get("candles", [])

    def get_best_bid_ask(self, product_ids: list) -> dict:
        data = self._get("/api/v3/brokerage/best_bid_ask", {"product_ids": product_ids})
        return {p["product_id"]: p for p in data.get("pricebooks", [])}

    def get_ticker(self, product_id: str) -> dict:
        data = self._get(f"/api/v3/brokerage/products/{product_id}")
        return data

    def get_order_book(self, product_id: str, limit: int = 50) -> dict:
        data = self._get("/api/v3/brokerage/product_book", {
            "product_id": product_id, "limit": limit
        })
        return data.get("pricebook", {})
