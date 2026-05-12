"""
OKX Trading Client — Produção
==============================
Client completo para trading real na OKX Spot.

Módulos implementados:
  RateLimiter          — token bucket por endpoint, respeita limites da OKX
  RetryPolicy          — exponential backoff com jitter para 5xx e rate-limit
  InstrumentCache      — precisão (tickSz, lotSz, minSz) por par, com TTL
  TimeDriftGuard       — detecta e corrige desvio de clock local vs OKX server
  OKXTradingClient     — client principal com todos os endpoints de trading
    place_order()      — cria ordem limit/market com clOrdId idempotente
    cancel_order()     — cancela por ordId ou clOrdId
    cancel_all()       — cancela todas as ordens abertas (kill switch parcial)
    get_order()        — status de uma ordem
    get_pending()      — lista ordens pendentes
    get_fills()        — fills recentes para reconciliação
    get_balance()      — saldo spot atual
    reconcile()        — compara estado local vs exchange, retorna divergências
    kill_switch()      — cancela tudo + fecha posições + para o bot

Idempotência:
  Cada ordem recebe clOrdId = uuid4 gerado ANTES do request.
  Em retry, o mesmo clOrdId é reenviado — OKX deduplica dentro de 24h.
  Arquivo local orders_log.json registra clOrdId → resposta para auditoria.

Rate limits OKX (spot, conta regular):
  /trade/order         — 60 req/2s  (usamos token bucket conservador: 20/2s)
  /trade/cancel-order  — 60 req/2s
  /trade/orders-*      — 20 req/2s
  /account/balance     — 10 req/2s
  /public/*            — 20 req/2s
"""

import time
import hmac
import hashlib
import base64
import json
import math
import random
import uuid
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger("okx_trading")

ORDERS_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "orders_log.json"
)


# ── Rate Limiter (token bucket) ───────────────────────────────────────────────

class RateLimiter:
    """
    Token bucket por endpoint.
    OKX agrupa rate limits por família de endpoint.
    Usamos limites conservadores (50% do máximo) para margem de segurança.
    """

    # (tokens_por_janela, janela_segundos) — conservador: 50% do limite oficial
    LIMITS = {
        "order":     (20, 2),    # /trade/order, /trade/cancel-order
        "query":     (10, 2),    # /trade/orders-*, /trade/fills
        "account":   ( 5, 2),    # /account/balance
        "public":    (10, 2),    # /public/*
    }

    def __init__(self):
        self._tokens  = {k: v[0] for k, v in self.LIMITS.items()}
        self._last_refill = {k: time.monotonic() for k in self.LIMITS}
        self._lock = threading.Lock()

    def _family(self, path: str) -> str:
        if "/trade/order" in path or "/trade/cancel" in path or "/trade/batch" in path:
            return "order"
        if "/trade/" in path:
            return "query"
        if "/account/" in path:
            return "account"
        return "public"

    def acquire(self, path: str):
        """Bloqueia até ter token disponível para o endpoint."""
        family = self._family(path)
        cap, window = self.LIMITS[family]
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill[family]
                # Refill proporcional ao tempo passado
                refill = elapsed / window * cap
                self._tokens[family] = min(cap, self._tokens[family] + refill)
                self._last_refill[family] = now
                if self._tokens[family] >= 1.0:
                    self._tokens[family] -= 1.0
                    return
            time.sleep(0.05)


# ── Retry com exponential backoff ─────────────────────────────────────────────

class RetryPolicy:
    """
    Retry automático para erros transitórios.

    Retenta em:
      - HTTP 429 (rate limit)
      - HTTP 5xx (erro do servidor OKX)
      - OKX error codes: 50011 (too many requests), 50001 (service unavailable)
      - Timeout / ConnectionError

    Não retenta em:
      - HTTP 4xx (erro do cliente — ordem inválida, saldo insuficiente, etc.)
      - OKX codes de erro de negócio (51000+)
    """

    RETRYABLE_HTTP  = {429, 500, 502, 503, 504}
    RETRYABLE_CODES = {"50011", "50001", "50013"}   # OKX internal errors

    def __init__(self, max_attempts: int = 4, base_delay: float = 0.5, max_delay: float = 30.0):
        self.max_attempts = max_attempts
        self.base_delay   = base_delay
        self.max_delay    = max_delay

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        if isinstance(exc, requests.HTTPError):
            return exc.response.status_code in self.RETRYABLE_HTTP
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, OKXAPIError):
            return exc.code in self.RETRYABLE_CODES
        return False

    def delay(self, attempt: int) -> float:
        """Exponential backoff com full jitter."""
        exp   = min(self.max_delay, self.base_delay * (2 ** attempt))
        jitter = random.uniform(0, exp)
        return jitter


# ── Erros específicos OKX ─────────────────────────────────────────────────────

class OKXAPIError(Exception):
    def __init__(self, code: str, msg: str, data=None):
        self.code = code
        self.msg  = msg
        self.data = data
        super().__init__(f"OKX [{code}]: {msg}")

class OKXInsufficientFunds(OKXAPIError): pass
class OKXOrderNotFound(OKXAPIError):     pass
class OKXMinSizeError(OKXAPIError):      pass
class OKXPrecisionError(OKXAPIError):    pass
class OKXKillSwitchError(Exception):     pass


# ── Instrument Cache (precisão por par) ───────────────────────────────────────

class InstrumentCache:
    """
    Cache de parâmetros de instrumentos OKX.
    Evita fetch repetido; TTL de 1 hora (precisão muda raramente).

    Campos por instrumento:
      tickSz   — mínimo incremento de preço  (ex: "0.1" para BTC → $0.1)
      lotSz    — mínimo incremento de qty    (ex: "0.00001" para BTC)
      minSz    — quantidade mínima de ordem  (ex: "0.00001" BTC)
      maxSz    — quantidade máxima por ordem
    """

    TTL = 3600  # segundos

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._fetched_at: dict[str, float] = {}

    def get(self, inst_id: str) -> Optional[dict]:
        ts = self._fetched_at.get(inst_id, 0)
        if time.time() - ts < self.TTL:
            return self._cache.get(inst_id)
        return None

    def set(self, inst_id: str, params: dict):
        self._cache[inst_id]       = params
        self._fetched_at[inst_id]  = time.time()

    def round_qty(self, inst_id: str, qty: float) -> float:
        """Arredonda qty para lotSz do instrumento."""
        p = self._cache.get(inst_id, {})
        lot = float(p.get("lotSz", "0.00001") or "0.00001")
        if lot <= 0:
            return qty
        # floor para o múltiplo inferior de lotSz
        factor = 1 / lot
        return math.floor(qty * factor) / factor

    def round_price(self, inst_id: str, price: float) -> float:
        p = self._cache.get(inst_id, {})
        tick = float(p.get("tickSz", "0.01") or "0.01")
        if tick <= 0:
            return price
        factor = 1 / tick
        return round(price * factor) / factor

    def min_size(self, inst_id: str) -> float:
        p = self._cache.get(inst_id, {})
        return float(p.get("minSz", "0.00001") or "0.00001")

    def check_min_size(self, inst_id: str, qty: float):
        mn = self.min_size(inst_id)
        if qty < mn:
            raise OKXMinSizeError("51000", f"{inst_id} qty {qty} < minSz {mn}")


# ── Time Drift Guard ──────────────────────────────────────────────────────────

class TimeDriftGuard:
    """
    OKX rejeita requests com timestamp fora de ±30s do server time.
    Mede o drift periodicamente e ajusta o timestamp gerado.
    """

    MAX_DRIFT_S = 25   # alerta se drift > 25s (limite OKX é 30s)
    MEASURE_INTERVAL = 300   # re-mede a cada 5 min

    def __init__(self):
        self._drift_s   = 0.0    # offset: local_time + drift = server_time
        self._last_check = 0.0

    def measure(self, server_time_ms: int):
        """Atualiza drift dado o server time em ms retornado pela OKX."""
        server_s = server_time_ms / 1000.0
        local_s  = time.time()
        self._drift_s   = server_s - local_s
        self._last_check = local_s
        if abs(self._drift_s) > self.MAX_DRIFT_S:
            logger.warning(
                f"[TimeDrift] Drift elevado: {self._drift_s:+.1f}s — "
                f"requests podem ser rejeitados pela OKX"
            )

    def now_iso(self) -> str:
        """Timestamp ISO ajustado pelo drift para uso nos headers."""
        adjusted = time.time() + self._drift_s
        dt = datetime.fromtimestamp(adjusted, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

    def needs_remeasure(self) -> bool:
        return time.time() - self._last_check > self.MEASURE_INTERVAL


# ── OKX Trading Client ────────────────────────────────────────────────────────

class OKXTradingClient:
    """
    Client de trading real para OKX Spot.

    Requer credenciais: api_key, secret_key, passphrase.
    Para paper trading, use SimulatedExecutionEngine em vez deste client.
    """

    BASE_URL = "https://www.okx.com"

    PAIR_MAP = {
        "BTC-USD":  "BTC-USDT",
        "ETH-USD":  "ETH-USDT",
        "SOL-USD":  "SOL-USDT",
        "AVAX-USD": "AVAX-USDT",
        "BNB-USD":  "BNB-USDT",
        "XRP-USD":  "XRP-USDT",
    }

    def __init__(
        self,
        api_key:    str,
        secret_key: str,
        passphrase: str,
        kill_switch_active: bool = False,
    ):
        if not api_key or not secret_key or not passphrase:
            raise ValueError("OKXTradingClient requer api_key, secret_key e passphrase")

        self.api_key    = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self._kill_switch_active = kill_switch_active

        self._session   = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
        })

        self._rate_limiter  = RateLimiter()
        self._retry_policy  = RetryPolicy(max_attempts=4)
        self._instruments   = InstrumentCache()
        self._drift_guard   = TimeDriftGuard()

        # Ordens pendentes locais: clOrdId → {ordId, symbol, side, status, ...}
        self._local_orders: dict[str, dict] = {}
        self._orders_log: list[dict] = self._load_orders_log()

        # Mede drift inicial
        self._sync_server_time()

    # ── Autenticação ──────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        msg = f"{timestamp}{method.upper()}{path}{body}"
        sig = hmac.new(self.secret_key.encode(), msg.encode(), hashlib.sha256).digest()
        return base64.b64encode(sig).decode()

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict:
        ts = self._drift_guard.now_iso()
        return {
            "OK-ACCESS-KEY":        self.api_key,
            "OK-ACCESS-SIGN":       self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }

    # ── HTTP com rate limit + retry ───────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: dict = None,
        body: dict = None,
        auth: bool = True,
        timeout: float = 10.0,
    ) -> dict:
        """
        Executa request com rate limiting, retry e tratamento de erro.
        Lança OKXAPIError para erros de negócio (não retriáveis).
        """
        if self._kill_switch_active:
            raise OKXKillSwitchError(
                "Kill switch ATIVO — todas as operações bloqueadas"
            )

        url       = f"{self.BASE_URL}{path}"
        body_str  = json.dumps(body) if body else ""

        # Re-mede drift se necessário
        if self._drift_guard.needs_remeasure():
            try:
                self._sync_server_time()
            except Exception:
                pass   # não bloqueia a operação

        for attempt in range(self._retry_policy.max_attempts):
            try:
                self._rate_limiter.acquire(path)

                headers = {}
                if auth:
                    headers = self._auth_headers(method, path, body_str)

                if method.upper() == "GET":
                    resp = self._session.get(url, params=params, headers=headers,
                                             timeout=timeout)
                else:
                    resp = self._session.post(url, data=body_str, headers=headers,
                                              timeout=timeout)

                resp.raise_for_status()
                data = resp.json()

                # OKX erro de negócio (code != "0")
                code = str(data.get("code", "0"))
                if code != "0":
                    self._raise_okx_error(code, data.get("msg", ""), data.get("data"))

                return data

            except Exception as exc:
                if self._retry_policy.should_retry(exc, attempt):
                    delay = self._retry_policy.delay(attempt)
                    logger.warning(
                        f"[OKX] {method} {path} — tentativa {attempt+1} falhou "
                        f"({type(exc).__name__}: {exc}), retry em {delay:.2f}s"
                    )
                    time.sleep(delay)
                    continue
                raise

        raise RuntimeError(f"[OKX] {method} {path} falhou após {self._retry_policy.max_attempts} tentativas")

    def _raise_okx_error(self, code: str, msg: str, data=None):
        """Mapeia código OKX para exceção específica."""
        # https://www.okx.com/docs-v5/en/#error-code
        if code in ("58110", "58111", "58112"):   # insufficient balance
            raise OKXInsufficientFunds(code, msg, data)
        if code in ("51603", "51604", "51000"):   # order not found
            raise OKXOrderNotFound(code, msg, data)
        if code in ("51020", "51021"):            # min size
            raise OKXMinSizeError(code, msg, data)
        if code in ("51022",):                   # precision
            raise OKXPrecisionError(code, msg, data)
        raise OKXAPIError(code, msg, data)

    # ── Server time ───────────────────────────────────────────────────────────

    def _sync_server_time(self):
        data = self._request("GET", "/api/v5/public/time", auth=False)
        server_ms = int(data["data"][0]["ts"])
        self._drift_guard.measure(server_ms)
        logger.debug(f"[OKX] Server time sync — drift: {self._drift_guard._drift_s:+.3f}s")

    # ── Instrument precision ──────────────────────────────────────────────────

    def _ensure_instrument(self, inst_id: str):
        """Busca e cacheia parâmetros de precisão do instrumento."""
        if self._instruments.get(inst_id) is not None:
            return
        data = self._request("GET", "/api/v5/public/instruments", {
            "instType": "SPOT",
            "instId":   inst_id,
        }, auth=False)
        if not data.get("data"):
            raise OKXAPIError("0", f"Instrumento {inst_id} não encontrado")
        d = data["data"][0]
        self._instruments.set(inst_id, {
            "tickSz": d.get("tickSz"),
            "lotSz":  d.get("lotSz"),
            "minSz":  d.get("minSz"),
            "maxSz":  d.get("maxSz"),
        })
        logger.info(f"[OKX] Instrumento {inst_id}: "
                    f"tickSz={d.get('tickSz')} lotSz={d.get('lotSz')} minSz={d.get('minSz')}")

    def _inst_id(self, pair: str) -> str:
        return self.PAIR_MAP.get(pair, pair.replace("-USD", "-USDT"))

    # ── Ordens ────────────────────────────────────────────────────────────────

    def place_order(
        self,
        pair:        str,
        side:        str,       # "buy" | "sell"
        order_type:  str,       # "limit" | "market"
        qty:         float,
        price:       float = 0.0,
        cl_ord_id:   str  = None,   # idempotência: reutilize em retry
    ) -> dict:
        """
        Coloca ordem spot na OKX.

        Idempotência:
          Se cl_ord_id for fornecido e a ordem já existir (OKX deduplica por 24h),
          a OKX retorna o ordId existente em vez de criar duplicata.

        Retorna:
          {"ordId": str, "clOrdId": str, "sCode": "0", ...}
        """
        inst_id = self._inst_id(pair)
        self._ensure_instrument(inst_id)

        # Arredondamento por precisão do instrumento
        qty   = self._instruments.round_qty(inst_id, qty)
        self._instruments.check_min_size(inst_id, qty)

        if order_type == "limit" and price > 0:
            price = self._instruments.round_price(inst_id, price)

        # clOrdId idempotente — UUID sem hífens, máx 32 chars (OKX limit)
        if not cl_ord_id:
            cl_ord_id = uuid.uuid4().hex

        body = {
            "instId":  inst_id,
            "tdMode":  "cash",            # spot (sem alavancagem)
            "side":    side.lower(),
            "ordType": order_type.lower(),
            "sz":      str(qty),
            "clOrdId": cl_ord_id,
        }
        if order_type == "limit" and price > 0:
            body["px"] = str(price)

        logger.info(f"[OKX] place_order {side.upper()} {qty} {inst_id} "
                    f"@ {'market' if order_type == 'market' else price} "
                    f"clOrdId={cl_ord_id}")

        data   = self._request("POST", "/api/v5/trade/order", body=body)
        result = data["data"][0]

        # Registra localmente para reconciliação
        record = {
            "clOrdId":  cl_ord_id,
            "ordId":    result.get("ordId", ""),
            "pair":     pair,
            "inst_id":  inst_id,
            "side":     side,
            "type":     order_type,
            "qty":      qty,
            "price":    price,
            "status":   "live",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._local_orders[cl_ord_id] = record
        self._log_order(record)

        logger.info(f"[OKX] Ordem criada: ordId={result.get('ordId')} "
                    f"clOrdId={cl_ord_id}")
        return result

    def cancel_order(
        self,
        pair:      str,
        ord_id:    str = None,
        cl_ord_id: str = None,
    ) -> dict:
        """Cancela ordem por ordId ou clOrdId."""
        if not ord_id and not cl_ord_id:
            raise ValueError("cancel_order requer ordId ou clOrdId")

        inst_id = self._inst_id(pair)
        body = {"instId": inst_id}
        if ord_id:
            body["ordId"]   = ord_id
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id

        logger.info(f"[OKX] cancel_order {inst_id} ordId={ord_id} clOrdId={cl_ord_id}")

        try:
            data   = self._request("POST", "/api/v5/trade/cancel-order", body=body)
            result = data["data"][0]
            # Atualiza estado local
            if cl_ord_id and cl_ord_id in self._local_orders:
                self._local_orders[cl_ord_id]["status"] = "cancelled"
            return result
        except OKXOrderNotFound:
            logger.warning(f"[OKX] cancel_order: ordem não encontrada "
                           f"ordId={ord_id} — já executada ou cancelada?")
            return {"sCode": "51603", "sMsg": "order already filled or cancelled"}

    def cancel_all(self, pair: str = None) -> list:
        """
        Cancela todas as ordens pendentes.
        Se pair for fornecido, cancela só aquele par.
        """
        pending = self.get_pending(pair=pair)
        results = []
        for order in pending:
            try:
                r = self.cancel_order(
                    pair   = order["instId"].replace("-USDT", "-USD"),
                    ord_id = order["ordId"],
                )
                results.append({"ordId": order["ordId"], "ok": True, "result": r})
                logger.info(f"[OKX] cancel_all: cancelou ordId={order['ordId']}")
            except Exception as exc:
                results.append({"ordId": order["ordId"], "ok": False, "error": str(exc)})
                logger.error(f"[OKX] cancel_all: falha ao cancelar {order['ordId']}: {exc}")
        return results

    def get_order(self, pair: str, ord_id: str = None, cl_ord_id: str = None) -> dict:
        """Busca status de uma ordem específica."""
        inst_id = self._inst_id(pair)
        params  = {"instId": inst_id}
        if ord_id:
            params["ordId"]   = ord_id
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id

        data = self._request("GET", "/api/v5/trade/order", params=params)
        return data["data"][0] if data.get("data") else {}

    def get_pending(self, pair: str = None) -> list:
        """Lista todas as ordens pendentes (live + partially filled)."""
        params = {"instType": "SPOT"}
        if pair:
            params["instId"] = self._inst_id(pair)

        data = self._request("GET", "/api/v5/trade/orders-pending", params=params)
        return data.get("data", [])

    def get_fills(self, pair: str = None, limit: int = 100) -> list:
        """
        Busca fills recentes (últimas 3 horas; até 3 dias com after/before).
        Usado para reconciliação de P&L e auditar execuções.
        """
        params = {"instType": "SPOT", "limit": str(min(limit, 100))}
        if pair:
            params["instId"] = self._inst_id(pair)

        data = self._request("GET", "/api/v5/trade/fills", params=params)
        fills = data.get("data", [])

        # Normaliza
        result = []
        for f in fills:
            result.append({
                "fill_id":  f.get("fillId"),
                "ord_id":   f.get("ordId"),
                "pair":     f.get("instId", "").replace("-USDT", "-USD"),
                "side":     f.get("side"),
                "price":    float(f.get("fillPx", 0)),
                "qty":      float(f.get("fillSz", 0)),
                "fee":      float(f.get("fee", 0)),
                "fee_ccy":  f.get("feeCcy"),
                "ts":       int(f.get("ts", 0)) // 1000,
                "is_maker": f.get("execType") == "M",
            })
        return result

    # ── Saldo ─────────────────────────────────────────────────────────────────

    def get_balance(self) -> dict:
        """
        Saldo spot atual por moeda.
        Retorna dict: {ccy: {"available": float, "frozen": float, "total": float}}
        """
        data    = self._request("GET", "/api/v5/account/balance")
        details = data.get("data", [{}])[0].get("details", [])
        result  = {}
        for d in details:
            ccy = d.get("ccy", "")
            if not ccy:
                continue
            result[ccy] = {
                "available": float(d.get("availBal",  0) or 0),
                "frozen":    float(d.get("frozenBal", 0) or 0),
                "total":     float(d.get("cashBal",   0) or 0),
            }
        return result

    # ── Reconciliação ─────────────────────────────────────────────────────────

    def reconcile(self, local_holdings: dict, local_balance_usd: float) -> dict:
        """
        Compara estado local (PaperTradingEngine) com saldo real na OKX.

        local_holdings   — {symbol: qty}  ex: {"BTC": 0.001, "ETH": 0.5}
        local_balance_usd — saldo USD disponível localmente

        Retorna:
          {
            "ok": bool,
            "divergences": [{"asset": str, "local": float, "exchange": float, "diff": float}],
            "exchange_balance": dict,
            "timestamp": str,
          }
        """
        try:
            exchange_bal = self.get_balance()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "divergences": []}

        divergences = []
        threshold   = 0.001   # ignora diferenças < 0.1% (arredondamento)

        # Checa USDT (caixa)
        ex_usdt   = exchange_bal.get("USDT", {}).get("available", 0.0)
        diff_usdt = abs(ex_usdt - local_balance_usd)
        if diff_usdt / max(local_balance_usd, 1) > threshold:
            divergences.append({
                "asset":    "USDT",
                "local":    round(local_balance_usd, 4),
                "exchange": round(ex_usdt, 4),
                "diff":     round(ex_usdt - local_balance_usd, 4),
                "severity": "HIGH" if diff_usdt > 10 else "LOW",
            })

        # Checa cada cripto
        for symbol, local_qty in local_holdings.items():
            ccy      = symbol.replace("-USD", "").replace("-USDT", "")
            ex_qty   = exchange_bal.get(ccy, {}).get("available", 0.0)
            diff_qty = abs(ex_qty - local_qty)
            if local_qty > 0 and diff_qty / max(local_qty, 1e-10) > threshold:
                divergences.append({
                    "asset":    ccy,
                    "local":    round(local_qty, 8),
                    "exchange": round(ex_qty, 8),
                    "diff":     round(ex_qty - local_qty, 8),
                    "severity": "HIGH" if diff_qty > 0.001 else "LOW",
                })

        ok = len([d for d in divergences if d["severity"] == "HIGH"]) == 0

        if not ok:
            logger.warning(
                f"[OKX] Reconciliação: {len(divergences)} divergência(s) encontrada(s): "
                + ", ".join(f"{d['asset']} local={d['local']} exchange={d['exchange']}"
                            for d in divergences)
            )
        else:
            logger.info("[OKX] Reconciliação OK — estado local consistente com exchange")

        return {
            "ok":               ok,
            "divergences":      divergences,
            "exchange_balance": exchange_bal,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }

    # ── Kill Switch ───────────────────────────────────────────────────────────

    def kill_switch(
        self,
        reason: str = "manual",
        cancel_orders: bool = True,
        close_positions: bool = False,  # True = market sell de tudo
    ) -> dict:
        """
        Para todas as operações imediatamente.

          cancel_orders    — cancela todas as ordens abertas
          close_positions  — vende tudo a mercado (CUIDADO: slippage alto)

        Após chamar este método, TODAS as operações futuras são bloqueadas
        até que kill_switch_active seja resetado manualmente.

        Use close_positions=True apenas em emergência (DD extremo, bug crítico).
        """
        logger.critical(f"[KILL SWITCH] ATIVADO — motivo: {reason}")

        result = {
            "reason":           reason,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "cancelled_orders": [],
            "closed_positions": [],
            "errors":           [],
        }

        if cancel_orders:
            try:
                cancelled = self.cancel_all()
                result["cancelled_orders"] = cancelled
                n_ok  = sum(1 for c in cancelled if c.get("ok"))
                n_err = len(cancelled) - n_ok
                logger.critical(f"[KILL SWITCH] Cancelou {n_ok} ordens, {n_err} erros")
            except Exception as exc:
                result["errors"].append(f"cancel_all falhou: {exc}")
                logger.critical(f"[KILL SWITCH] cancel_all FALHOU: {exc}")

        if close_positions:
            try:
                balance = self.get_balance()
                for ccy, bal in balance.items():
                    if ccy == "USDT":
                        continue
                    qty = bal.get("available", 0)
                    if qty < 1e-8:
                        continue
                    pair = f"{ccy}-USD"
                    if pair not in self.PAIR_MAP:
                        continue
                    logger.critical(f"[KILL SWITCH] Vendendo {qty} {ccy} a mercado")
                    try:
                        r = self.place_order(pair, "sell", "market", qty)
                        result["closed_positions"].append({"ccy": ccy, "qty": qty, "result": r})
                    except Exception as exc:
                        result["errors"].append(f"close {ccy}: {exc}")
            except Exception as exc:
                result["errors"].append(f"get_balance falhou: {exc}")

        # Bloqueia todas as operações futuras
        self._kill_switch_active = True

        # Persiste log do kill switch
        log_path = os.path.join(os.path.dirname(ORDERS_LOG), "kill_switch_log.json")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                json.dump(result, f)
                f.write("\n")
        except Exception:
            pass

        logger.critical(f"[KILL SWITCH] Concluído. Erros: {result['errors']}")
        return result

    def reset_kill_switch(self):
        """Re-habilita operações após kill switch. Requer confirmação explícita."""
        self._kill_switch_active = False
        logger.warning("[KILL SWITCH] Reset — operações re-habilitadas")

    # ── Orders log ────────────────────────────────────────────────────────────

    def _load_orders_log(self) -> list:
        try:
            if os.path.exists(ORDERS_LOG):
                with open(ORDERS_LOG) as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _log_order(self, record: dict):
        self._orders_log.append(record)
        # Mantém últimas 1000 ordens
        self._orders_log = self._orders_log[-1000:]
        try:
            os.makedirs(os.path.dirname(ORDERS_LOG), exist_ok=True)
            with open(ORDERS_LOG, "w") as f:
                json.dump(self._orders_log, f, indent=2)
        except Exception as exc:
            logger.warning(f"[OKX] Falha ao salvar orders_log: {exc}")

    def get_orders_log(self, limit: int = 100) -> list:
        """Retorna histórico de ordens enviadas (auditoria)."""
        return self._orders_log[-limit:]

    # ── Status summary ────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Resumo do estado do client para o dashboard."""
        return {
            "kill_switch":    self._kill_switch_active,
            "drift_s":        round(self._drift_guard._drift_s, 3),
            "pending_local":  len(self._local_orders),
            "orders_logged":  len(self._orders_log),
            "rate_tokens":    {k: round(v, 1)
                               for k, v in self._rate_limiter._tokens.items()},
        }
