import os
import sys
import time
import json
import asyncio
import requests
import pandas as pd
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv

from exchange.okx     import OKXClient
from paper_trading.engine import PaperTradingEngine, TAKER_FEE
# ── Estratégias ativas: trend-following + momentum ─────────────────
from strategies.donchian_breakout      import DonchianBreakout
from strategies.ema_pullback           import EMAPullback
from strategies.macd_momentum          import MACDMomentum
from strategies.volatility_guard       import VolatilityGuard
from strategies.trend_filter           import TrendFilter
from strategies.news_guard             import is_news_blackout, next_event
from strategies.news_sync              import sync_if_needed as _news_sync_if_needed
from strategies.market_breadth         import get_market_breadth
from strategies.market_regime          import calc_adx, calc_atr
from strategies.bb_reversion           import BBReversion
from logger import setup_logger, log_cycle, log_trade, log_portfolio
from notifier import notify_trade

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "code.env"))

app = FastAPI()
HTML_FILE    = os.path.join(os.path.dirname(__file__), "templates", "index.html")
STATIC_DIR   = os.path.join(os.path.dirname(__file__), "static")

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
HISTORY_FILE      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio_history.json")
NEWS_EVENTS_FILE  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "news_events.json")


# ── Cache de cotação USD/BRL ──────────────────────────────────────
USD_BRL_TTL = 1800           # atualiza a cada 30 minutos
_usd_brl_cache: dict = {"rate": 5.70, "ts": 0.0}

def _fetch_usd_brl() -> float:
    now = time.time()
    if now - _usd_brl_cache["ts"] < USD_BRL_TTL:
        return _usd_brl_cache["rate"]
    # Tenta múltiplas APIs como fallback
    apis = [
        ("https://api.frankfurter.dev/v1/latest?from=USD&to=BRL", lambda d: float(d["rates"]["BRL"])),
        ("https://open.er-api.com/v6/latest/USD", lambda d: float(d["rates"]["BRL"])),
    ]
    for url, parser in apis:
        try:
            r = requests.get(url, timeout=5)
            rate = parser(r.json())
            if 3.0 < rate < 10.0:   # sanity check
                _usd_brl_cache["rate"] = rate
                _usd_brl_cache["ts"]   = now
                return rate
        except Exception:
            continue
    return _usd_brl_cache["rate"]


# ── Fear & Greed Index (alternative.me) ──────────────────────────
_fg_cache: dict = {"value": 50, "label": "Neutral", "ts": 0.0}

def _fetch_fear_greed() -> dict:
    now = time.time()
    if now - _fg_cache["ts"] < FG_TTL:
        return _fg_cache
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        d = r.json()["data"][0]
        _fg_cache["value"] = int(d["value"])
        _fg_cache["label"] = d["value_classification"]
        _fg_cache["ts"]    = now
    except Exception:
        pass   # mantém cache anterior em caso de falha
    return _fg_cache


# ── Cache de candles por par ──────────────────────────────────────
# Refresh a cada 3 minutos para acompanhar cycle de 180s e capturar breakouts
CANDLE_TTL = 240             # 4 min — suficiente para 30min/1H candles
_candle_cache: dict = {}     # {pair: {"data": [...], "ts": float}}

def _get_candles(pair: str, granularity: str, limit: int = 100) -> list:
    key = f"{pair}:{granularity}"
    now = time.time()
    cached = _candle_cache.get(key)
    if cached and (now - cached["ts"]) < CANDLE_TTL:
        return cached["data"]
    data = client.get_candles(pair, granularity=granularity, limit=limit)
    _candle_cache[key] = {"data": data, "ts": now}
    return data


# ── Preço anterior por par (para log de variação) ────────────────
_last_prices: dict = {}      # {pair: float}


# _dynamic_tp removida — obsoleta após Fase 3 (TP = SL × 2)

# _dynamic_tp_by_regime removida na Fase 3 — TP agora = SL × 2 (RR 2:1 fixo).

def _detect_market_regime(candles_1h: list, candles_6h: list,
                           breadth=None) -> tuple:
    """
    Detecta o regime global de mercado baseado no BTC (Fase 1 — simplificado).

    Retorna: (regime: str, bear_signals: list[str])
      regime       → 'bull' | 'chop' | 'bear'
      bear_signals → lista informativa (não mais usada para reclassificação)

    Regras simples e testáveis:
      'bear' → BTC < EMA200 no 6H  (sinal estrutural único)
      'bull' → BTC > EMA200 + EMA50 > EMA200 + ADX > 20
      'chop' → qualquer outra condição (ADX < 20 ou EMA50 < EMA200)

    Fase 1: removidos 4 sinais adicionais e reclassificação chop→bear.
    Motivo: sinais correlacionados criavam false negatives sem adicionar
    informação independente. Menos parâmetros = menos overfitting.
    """
    bear_signals: list = []

    if not candles_6h or len(candles_6h) < 50:
        return "chop", bear_signals
    try:
        import pandas as _pd

        _COLS = ["start", "low", "high", "open", "close", "volume"]
        _NUM  = ["low", "high", "open", "close", "volume"]

        def _to_df(candles: list) -> "_pd.DataFrame":
            """Cria DataFrame de candles garantindo colunas numéricas."""
            df = _pd.DataFrame(candles, columns=_COLS)
            for col in _NUM:
                df[col] = _pd.to_numeric(df[col], errors="coerce")
            return df

        df6 = _to_df(candles_6h)
        closes6 = df6["close"]
        ema50_s  = closes6.ewm(span=50,  adjust=False).mean()
        ema200_s = closes6.ewm(span=200, adjust=False).mean()
        price    = float(closes6.iloc[-1])
        e50      = float(ema50_s.iloc[-1])
        e200     = float(ema200_s.iloc[-1])

        # ── Sinal único bear: preço vs EMA200 6H ─────────────────────────────
        if price < e200:
            bear_signals.append("Preço < EMA200 6H")
            return "bear", bear_signals

        # ── ADX para distinguir bull vs chop ──────────────────────────────────
        adx_val = 20.0
        if candles_1h and len(candles_1h) >= 30:
            df1h    = _to_df(candles_1h)
            adx_val = calc_adx(df1h)

        # ── Bull: EMA alinhadas + tendência forte ──────────────────────────────
        if price > e200 and e50 > e200 and adx_val >= 20:
            return "bull", bear_signals

        # ── Chop: qualquer outra condição ─────────────────────────────────────
        return "chop", bear_signals

        return "chop", bear_signals

    except Exception as _ex:
        import traceback as _tb
        logger.warning(f"[Regime] Erro na detecção: {_ex}\n{_tb.format_exc()}")
        return "chop", []


def _get_fee_rates() -> tuple:
    """Determina maker/taker fee pelo volume dos últimos 30 dias."""
    cutoff  = time.time() - 30 * 86400
    vol_30d = sum(t.get("usd", 0) for t in engine.trades if (t.get("ts") or 0) >= cutoff)
    maker, taker = 0.0010, 0.0040
    for min_vol, m, t_ in reversed(COINBASE_FEE_TIERS):
        if vol_30d >= min_vol:
            maker, taker = m, t_
            break
    return maker, taker

def _current_taker_fee() -> float:
    _, taker = _get_fee_rates(); return taker

def _current_maker_fee() -> float:
    maker, _ = _get_fee_rates(); return maker


def _calc_confidence_score(signals: dict, regime: str, adx: float) -> float:
    """Score 0-1 ponderado por regime. Ajusta tamanho da posição."""
    weights = STRATEGY_WEIGHTS.get(regime, STRATEGY_WEIGHTS["neutral"])
    max_w   = sum(weights.values()) or 1.0
    buy_score = sum(weights.get(s, 1.0) for s, sig in signals.items() if sig == "BUY")
    normalized = buy_score / max_w
    if regime == "trending" and adx > 20:
        normalized = min(1.0, normalized * (1 + min(0.25, (adx - 20) / 80)))
    return normalized


# _dynamic_sl removida na Fase 3 — SL agora = ATR × 2 clampado em PAIR_SL_RANGE.


def _load_history() -> list:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_history(history: list):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except Exception:
        pass


SP_OFFSET = -3 * 3600   # UTC-3 fixo — SP aboliu horário de verão em 2019


def _current_cycle() -> int:
    """Número da hora atual em SP (UTC-3). Vai de 0 (meia-noite SP) a 23."""
    return (int(time.time()) + SP_OFFSET) % 86400 // CYCLE_INTERVAL


def _seconds_to_next_sp_hour() -> float:
    """
    Segundos até o início da próxima hora cheia em SP (UTC-3).
    Garante que os ciclos de 1H fiquem alinhados com o relógio:
      ex.: se são 14:47 SP → aguarda 13min até 15:00 SP.
    Mínimo de 30s para evitar re-execução imediata no final de uma hora.
    """
    now_sp       = time.time() + SP_OFFSET
    secs_in_hour = now_sp % 3600
    wait         = 3600 - secs_in_hour
    return wait if wait >= 30 else wait + 3600


PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD"]  # 3 pares — foco em ativos de maior liquidez

# ── Portfolio em Real é FIXO em R$ 5.000 ────────────────────────
TOTAL_BRL_INITIAL = 5000.0  # Portfolio inicial em BRL — FIXO, nunca muda
# Portfolio em USD varia com cotação: USD_atual = TOTAL_BRL_INITIAL / usd_brl_atual

# ── Ciclo e candles ─────────────────────────────────────────────
CYCLE_INTERVAL    = 3600     # ciclo de 3600s (1 hora)
CANDLE_30M        = "THIRTY_MINUTE"
CANDLE_1H         = "ONE_HOUR"       # EMA Pullback, MACD
CANDLE_6H         = "SIX_HOUR"
CANDLE_1D         = "ONE_DAY"        # Trend, VolGuard

# ── Execução por estratégia (independente, sem consenso) ──────────
TRADE_PCT          = 0.10   # 10% do portfolio por trade — reduzido para diminuir taxas e risco


def _calculate_dynamic_position_size(pair: str, candles: list, base_pct: float = None) -> float:
    """Calcula tamanho de posição dinamicamente baseado em volatilidade.

    Alta volatilidade → posição menor (2%)
    Baixa volatilidade → posição maior (até 10%)
    """
    if base_pct is None:
        base_pct = TRADE_PCT

    if len(candles) < 20:
        return base_pct  # Fallback ao base

    df = pd.DataFrame(candles, columns=["start", "low", "high", "open", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    # Calcular ATR (Average True Range)
    df["tr"] = df["high"] - df["low"]
    atr_current = df["tr"].iloc[-1]
    atr_avg = df["tr"].tail(14).mean()

    # Ratio de volatilidade (inverso: mais vol = posição menor)
    if atr_current > 0:
        vol_ratio = atr_avg / atr_current
    else:
        vol_ratio = 1.0

    # Aplicar ratio com limites (2% mín, TRADE_PCT máx)
    size = base_pct * vol_ratio
    size = max(0.02, min(TRADE_PCT, size))

    return size


# ── Gestão de risco (Fase 3 — sistema unificado ATR-based) ───────
# SL = ATR × 2, clampado entre min e max por ativo
# TP = SL × 2  (RR fixo 2:1)
# Break-even: gain >= SL% × BE_TRIGGER_MULT → SL sobe para entrada
# Trailing:   gain >= SL% × TRAIL_TRIGGER_MULT → segue pico a SL% de distância
#
# BE_TRIGGER_MULT = 1.5  → BE só ativa com 1.5× o SL de distância (era 1.0×)
#   Exemplo BTC (SL=3%): precisa de +4.5% de ganho antes de proteger o zero a zero
#   Evita que correções normais de mercado acionem o stop no break-even
BE_TRIGGER_MULT    = 1.5   # gatilho do break-even (múltiplo do SL%)
TRAIL_TRIGGER_MULT = 2.5   # gatilho do trailing   (múltiplo do SL%) — era 2.0×
PAIR_SL_RANGE = {
    "BTC-USD":    (0.02, 0.04),   # SL entre 2% e 4%
    "ETH-USD":    (0.03, 0.05),   # SL entre 3% e 5%
    "SOL-USD":    (0.05, 0.07),   # SL entre 5% e 7%
}

# ── OKX — Fee System Regular (Spot Trading, 2026) ────────────────
# Nível Regular: Maker 0.10% / Taker 0.40%
# vol_30d em USD → (min_vol, maker_fee, taker_fee)
OKX_FEE_TIERS = [
    (           0,  0.0010, 0.0040),  # Regular: Maker 0.10% / Taker 0.40%
]
# Alias para compatibilidade com código que usa COINBASE_FEE_TIERS
COINBASE_FEE_TIERS = OKX_FEE_TIERS

# ── Score ponderado por regime ────────────────────────────────────
STRATEGY_WEIGHTS = {
    # Chaves alinhadas com o retorno de _detect_market_regime(): "bull"/"chop"/"bear"
    "bull":  {"Donchian Breakout": 1.5, "EMA Pullback": 1.3, "MACD Momentum": 1.2},
    "chop":  {"Donchian Breakout": 0.8, "EMA Pullback": 1.0, "MACD Momentum": 0.9},
    "bear":  {"Donchian Breakout": 0.5, "EMA Pullback": 0.7, "MACD Momentum": 0.8},
    # Aliases legados (fallback)
    "trending": {"Donchian Breakout": 1.5, "EMA Pullback": 1.3, "MACD Momentum": 1.2},
    "ranging":  {"Donchian Breakout": 0.5, "EMA Pullback": 0.9, "MACD Momentum": 0.8},
    "neutral":  {"Donchian Breakout": 1.0, "EMA Pullback": 1.0, "MACD Momentum": 1.0},
}
SCORE_MIN_THRESHOLD = 0.33   # abaixo de 33% → BUY bloqueado (1 estratégia = ~33%)
SCORE_SIZE_BOOST    = 1.4    # score 85%+ → tamanho +40%

# ── Classificação de pares ───────────────────────────────────────
ALT_PAIRS = {"SOL-USD"}
BTC_PAIRS  = {"BTC-USD", "ETH-USD"}
# PAIR_TRAILING e PAIR_BREAKEVEN removidos na Fase 3 —
# substituídos pela regra unificada ATR-based em _calc_exit()
SL_COOLDOWN_CYCLES    = 3     # SL normal: 3h = 3 ciclos de 1h

# ── Circuit breaker + controles de risco ─────────────────────────
MAX_DAILY_TRADES      = 10    # máximo de trades por dia (BUY+SELL)
MAX_OPEN_SLOTS        = 4     # máximo de slots abertos simultaneamente
BUY_COOLDOWN_SECONDS  = 7200   # 2h entre BUYs no mesmo par/estratégia (Fase 4)
_daily_trade_count: dict = {}  # {"YYYY-MM-DD": count}
last_buy_time:      dict = {}  # {f"{strat}:{pair}": timestamp}

# ── Pyramid (scale-in em posição lucrativa) ──────────────────────
# Pyramid removido na Fase 4 — adiciona complexidade sem edge claro em 3 pares

# ── Fear & Greed ─────────────────────────────────────────────────
FG_GREED_MIN   = 70   # Acima de 70: bloqueia novas entradas (euforia = risco de topo)
FG_TTL         = 3600 # cache de 1 hora (índice atualiza 1×/dia)

# ── Abordagem híbrida Limit/Market ───────────────────────────────
# EMA Pullback → Limit order ao nível EMA21 (maker 0.10%) — alta prob. de fill
# Donchian + MACD → Market order (taker 0.40%) — breakouts exigem execução imediata
LIMIT_STRATEGIES      = {"EMA Pullback", "BB Reversion"}  # limit order (maker 0.10%)
LIMIT_ORDER_TIMEOUT_H = 2                  # cancela se não preencher em 2 ciclos (2h)
pending_orders: dict  = {}                 # {f"{strat}:{pair}": {limit_price, ...}}

client = OKXClient(
    api_key    = os.getenv("OKX_API_KEY",    os.getenv("API_KEY", "")),
    secret_key = os.getenv("OKX_SECRET_KEY", os.getenv("SECRET_KEY", "")),
    passphrase = os.getenv("OKX_PASSPHRASE", ""),
)
engine = PaperTradingEngine(initial_balance_usd=10000.0)

# ── 5 estratégias AGRESSIVAS 65/35 independentes ──────────────────────
# 3 estratégias de tendência — foco em qualidade, menos fees
all_strategies = [
    DonchianBreakout(period=20, rsi_min=45.0, adx_min=20.0,
                     rvol_period=20, rvol_min=1.3),   # breakout — market order
    MACDMomentum(fast=12, slow=26, signal=9, ema_filter=12, rsi_max=75.0),  # momentum — market
    EMAPullback(fast=9, mid=21, slow=50,
                touch_tolerance_pct=0.3),              # pullback — limit order
    BBReversion(bb_period=20, bb_std=2.0,
                rsi_period=14, rsi_oversold=35.0),    # mean reversion CHOP — limit order
]

# Mapa de candles por estratégia
STRAT_CANDLES = {
    "Donchian Breakout": CANDLE_1H,   # 1H principal + confirmação 6H (proxy 4H)
    "EMA Pullback":      CANDLE_1H,
    "MACD Momentum":     CANDLE_1H,
}

# Guard de risco global — só dispara em crashes REAIS (>25% vol = mercado caindo)
vol_guard    = VolatilityGuard(lookback=24, block_std=1.0, sell_std=2.0)  # 1H candles: BLOCK>1σ, SELL>2σ
trend_filter = TrendFilter(period=50)   # EMA50 1H — alinhado com EMA Pullback e MACD Momentum

# ── Cooldown anti-whipsaw após SL (por slot) ─────────────────────
sl_cooldowns: dict = {}   # {f"{strat}:{pair}": cycles_remaining}
# sl_history removido na Fase 4 — cooldown escalante substituído por cooldown fixo

# Sinaliza ao trading_loop para rodar o próximo ciclo imediatamente (sem esperar CYCLE_INTERVAL)
_immediate_cycle = asyncio.Event()

# Quando True, o próximo ciclo força feed entries para TODOS os sinais (ignora change-detection)
_force_feed_populate: bool = False

# ── Slots independentes: 4 estratégias × 3 pares + 3 manuais ────
def _empty_slot():
    return {"qty": 0.0, "entry": 0.0, "peak": 0.0,
            "realized": 0.0, "unrealized": 0.0, "pyramids": 0, "be_sl": 0.0,
            "entry_usd": 0.0, "sl_pct": 0.0}  # sl_pct: ATR-based SL% fixado na entrada


def _calc_exit(slot: dict, price: float, pair: str) -> tuple:
    """
    Sistema de saída unificado (Fase 3) — baseado em ATR.

    Regra única derivada do SL% fixado na entrada:
      SL hard:    entry × (1 - sl_pct%)
      Break-even: quando gain >= sl_pct% × BE_TRIGGER_MULT (1.5×), SL sobe para entry
      Trailing:   quando gain >= sl_pct% × TRAIL_TRIGGER_MULT (2.5×), segue pico a sl_pct%
      TP:         entry × (1 + sl_pct% × 2)  → RR fixo 2:1

    Retorna: (tp_hit, sl_hit, sl_level, tp_level, sl_pct)
    """
    entry   = slot["entry"]
    peak    = slot["peak"]
    be_sl   = slot.get("be_sl", 0.0)
    sl_pct  = slot.get("sl_pct") or 0.0

    # Fallback: se sl_pct não foi salvo, usar máximo do range do par
    if sl_pct <= 0:
        sl_min_pct, sl_max_pct = PAIR_SL_RANGE.get(pair, (0.03, 0.07))
        sl_pct = sl_max_pct * 100

    gain_pct = (price - entry) / entry * 100 if entry > 0 else 0.0

    # Nível de SL progressivo
    sl_level = entry * (1 - sl_pct / 100)          # SL base

    if gain_pct >= sl_pct * BE_TRIGGER_MULT:        # Break-even (1.5× SL%)
        sl_level = max(sl_level, entry)
    if gain_pct >= sl_pct * TRAIL_TRIGGER_MULT:     # Trailing (2.5× SL%)
        sl_level = max(sl_level, peak * (1 - sl_pct / 100))

    # Ratchet: nunca desce o SL
    sl_level = max(sl_level, be_sl)

    tp_level = entry * (1 + sl_pct * 2 / 100)      # TP = 2× SL (RR 2:1)

    tp_hit = price >= tp_level
    sl_hit = price <= sl_level

    return tp_hit, sl_hit, sl_level, tp_level, sl_pct

SLOTS_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "strategy_slots.json")

def _load_slots() -> dict:
    slots = {}
    for s in all_strategies:
        for p in PAIRS:
            slots[f"{s.name}:{p}"] = _empty_slot()
    for p in PAIRS:
        slots[f"manual:{p}"] = _empty_slot()
    try:
        if os.path.exists(SLOTS_FILE):
            saved = json.load(open(SLOTS_FILE))
            for k, v in saved.items():
                if k in slots:
                    slots[k].update(v)
    except Exception:
        pass
    return slots

def _save_slots(slots: dict):
    try:
        os.makedirs(os.path.dirname(SLOTS_FILE), exist_ok=True)
        with open(SLOTS_FILE, "w") as f:
            json.dump(slots, f, indent=2)
    except Exception:
        pass

strategy_slots = _load_slots()

# compat aliases para endpoints manuais
def _save_manual(s): _save_slots(s)

# ── P&L por estratégia (atribuição proporcional) ─────────────────
STRAT_PNL_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "strategy_pnl.json")

def _load_strategy_pnl() -> dict:
    pnl = {s.name: {"realized": 0.0, "trades": 0, "buys": 0, "sells": 0}
           for s in all_strategies}
    try:
        if os.path.exists(STRAT_PNL_FILE):
            saved = json.load(open(STRAT_PNL_FILE))
            for k, v in saved.items():
                if k in pnl:
                    pnl[k].update(v)
    except Exception:
        pass
    return pnl

def _save_strategy_pnl(pnl: dict):
    try:
        os.makedirs(os.path.dirname(STRAT_PNL_FILE), exist_ok=True)
        with open(STRAT_PNL_FILE, "w") as f:
            json.dump(pnl, f, indent=2)
    except Exception:
        pass

strategy_pnl = _load_strategy_pnl()

def _attr_pnl(strat_name: str, pnl_usd: float):
    """Registra P&L realizado e contabiliza o sell na estratégia."""
    if strat_name in strategy_pnl:
        strategy_pnl[strat_name]["realized"] += pnl_usd
        strategy_pnl[strat_name]["sells"]    = strategy_pnl[strat_name].get("sells", 0) + 1
        strategy_pnl[strat_name]["trades"]   = (strategy_pnl[strat_name].get("buys", 0)
                                                + strategy_pnl[strat_name]["sells"])
    _save_strategy_pnl(strategy_pnl)
    state["strategy_pnl"] = strategy_pnl

def _count_buy(strat_name: str):
    """Contabiliza um BUY na estratégia."""
    if strat_name in strategy_pnl:
        strategy_pnl[strat_name]["buys"]   = strategy_pnl[strat_name].get("buys", 0) + 1
        strategy_pnl[strat_name]["trades"] = (strategy_pnl[strat_name]["buys"]
                                              + strategy_pnl[strat_name].get("sells", 0))
    _save_strategy_pnl(strategy_pnl)
    state["strategy_pnl"] = strategy_pnl

last_signals: dict = {}   # {f"{pair}:{strat}": signal}

logger = setup_logger("dashboard")
connected_clients: List[WebSocket] = []


def _update_portfolio_state():
    """Calcula P&L apenas de TRADES — variação cambial USD/BRL não conta.

    Lógica:
    - P&L_USD = portfolio_atual_USD - initial_balance_USD  (puro resultado de trades)
    - P&L_BRL = P&L_USD × cotação_atual                   (converte só o lucro/perda)
    - Total_BRL = portfolio_total_USD × cotação_atual      (valor atual em BRL)

    Assim, se não houve trades, P&L = R$0,00 independente do câmbio.
    A oscilação cambial afeta o "Portfolio Total" (valor patrimonial) mas NÃO o P&L.
    """
    total_usd = engine.portfolio_value()
    usd_brl_current = state.get("usd_brl", 5.70)

    # P&L somente de trades (USD): diferença entre valor atual e capital inicial
    pnl_usd = total_usd - engine.initial_balance

    # P&L em BRL: converte apenas o resultado de trades — NÃO subtrai R$5.000 do total
    # Isso garante que variação cambial não afeta o P&L exibido
    pnl_brl = pnl_usd * usd_brl_current
    pnl_pct  = (pnl_brl / TOTAL_BRL_INITIAL) * 100 if TOTAL_BRL_INITIAL > 0 else 0

    # Valor total atual em BRL (patrimônio — pode variar com câmbio, isso é normal)
    total_brl = total_usd * usd_brl_current

    state["portfolio"] = {
        "usd":               round(engine.balance_usd, 2),
        "total_usd":         round(total_usd, 2),
        "total_brl":         round(total_brl, 2),
        "pnl_usd":           round(pnl_usd, 2),
        "pnl_brl":           round(pnl_brl, 2),
        "pnl_pct":           round(pnl_pct, 2),
        "initial_balance_usd": round(engine.initial_balance, 2),
        "initial_balance_brl": round(TOTAL_BRL_INITIAL, 2),
        "holdings":          {k: round(v, 8) for k, v in engine.holdings.items()},
        "total_fees_usd":    round(engine.total_fees_usd, 4),
    }
    return total_usd, pnl_usd


def _calculate_kpis() -> dict:
    """
    Calcula métricas avançadas de performance por estratégia e globais.

    Métricas por estratégia:
      win_rate        → % trades vencedores
      profit_factor   → gross_profit / gross_loss
      edge_decay      → win_rate últimos 10 trades vs win_rate total (detecta deterioração)
      drawdown_contrib → % do drawdown máximo atribuível à estratégia
      mfe_capture     → P&L realizado / MFE estimado (quanto do movimento foi capturado)
    """
    all_trades_list = list(engine.trades)
    sell_trades     = [t for t in all_trades_list if t.get("side") == "SELL"]

    # ── Helper: preço médio de entrada na época do SELL ─────────────────────
    def _avg_entry_at_sell(history, sell_idx):
        sell   = history[sell_idx]
        symbol = sell.get("symbol") or sell.get("pair", "").replace("-USD", "")
        rqty, rcost = 0.0, 0.0
        for t in history[:sell_idx]:
            tsym = t.get("symbol") or t.get("pair", "").replace("-USD", "")
            if tsym != symbol:
                continue
            if t.get("side") == "BUY":
                q = t.get("qty", 0)
                rqty  += q
                rcost += q * t.get("price", 0)
            elif t.get("side") == "SELL":
                q = min(t.get("qty", 0), rqty)
                if rqty > 1e-10:
                    rcost *= (rqty - q) / rqty
                rqty = max(0, rqty - q)
        return rcost / rqty if rqty > 1e-10 else 0.0

    # ── Calcula P&L real por SELL com estratégia e metadados ────────────────
    trade_records = []   # {pnl, strategy, sell_price, entry_price, qty, pnl_pct}
    for idx, t in enumerate(all_trades_list):
        if t.get("side") != "SELL":
            continue
        sell_usd = t.get("usd", 0)
        qty      = t.get("qty", 0)
        sell_px  = t.get("price", 0)
        entry    = _avg_entry_at_sell(all_trades_list, idx)
        strategy = t.get("strategy", t.get("note", "")).split(":")[0] or "unknown"
        if entry > 0 and qty > 0:
            buy_fee  = qty * entry * TAKER_FEE
            cost_usd = qty * entry + buy_fee
            pnl      = sell_usd - cost_usd
            pnl_pct  = (sell_px - entry) / entry * 100 if entry > 0 else 0.0
        else:
            pnl = pnl_pct = 0.0
        trade_records.append({
            "pnl": pnl, "pnl_pct": pnl_pct,
            "strategy": strategy,
            "entry": entry, "sell_px": sell_px, "qty": qty,
        })

    # ── Global overview ──────────────────────────────────────────────────────
    all_pnls  = [r["pnl"] for r in trade_records]
    win_pnls  = [p for p in all_pnls if p > 0]
    loss_pnls = [p for p in all_pnls if p <= 0]
    n         = len(all_pnls)
    wins      = len(win_pnls)
    losses    = len(loss_pnls)
    avg_win   = sum(win_pnls)  / wins   if wins   else 0.0
    avg_loss  = sum(loss_pnls) / losses if losses else 0.0
    sum_wins  = sum(win_pnls)
    sum_loss  = abs(sum(loss_pnls)) if loss_pnls else 0.0

    # ── Drawdown máximo global (sequência de equity) ─────────────────────────
    equity   = [0.0]
    for r in trade_records:
        equity.append(equity[-1] + r["pnl"])
    peak  = 0.0
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    # ── Métricas por estratégia ──────────────────────────────────────────────
    strat_names = ["Donchian Breakout", "EMA Pullback", "MACD Momentum"]
    by_strat = {}

    for sname in strat_names:
        recs = [r for r in trade_records if sname in r["strategy"]]
        if not recs:
            by_strat[sname] = {
                "win_rate": None, "profit_factor": None,
                "edge_decay": None, "drawdown_contrib": None,
                "mfe_capture": None, "n": 0,
            }
            continue

        s_pnls  = [r["pnl"] for r in recs]
        s_wins  = [p for p in s_pnls if p > 0]
        s_loss  = [p for p in s_pnls if p <= 0]
        s_n     = len(s_pnls)
        s_wr    = len(s_wins) / s_n if s_n else 0.0
        s_gp    = sum(s_wins)
        s_gl    = abs(sum(s_loss)) if s_loss else 0.0
        s_pf    = round(s_gp / s_gl, 3) if s_gl > 0 else None

        # Edge decay: win_rate últimos N vs total
        # Detecta se a estratégia está perdendo efetividade recentemente
        recent_n = min(10, s_n)
        recent   = [r["pnl"] for r in recs[-recent_n:]]
        recent_wr = len([p for p in recent if p > 0]) / len(recent) if recent else s_wr
        edge_decay = round(recent_wr - s_wr, 4)   # negativo = edge caindo

        # Drawdown contribution: % do drawdown máximo que veio desta estratégia
        s_equity = [0.0]
        for r in recs:
            s_equity.append(s_equity[-1] + r["pnl"])
        s_peak = 0.0
        s_dd   = 0.0
        for e in s_equity:
            if e > s_peak: s_peak = e
            s_dd = max(s_dd, s_peak - e)
        dd_contrib = round(s_dd / max_dd * 100, 1) if max_dd > 0 else 0.0

        # MFE Capture: quanto do movimento máximo favorável foi capturado
        # Estimativa: MFE = TP alvo × qty × entry_price (usa TP configurado no momento)
        # Como não temos MFE histórico salvo, estimamos via pnl_pct vs avg_win_pct
        # MFE proxy = avg(pnl_pct_dos_winners) / TP_atual (% capturado do alvo)
        win_pcts = [r["pnl_pct"] for r in recs if r["pnl"] > 0]
        tp_ref   = 10.0  # referência para MFE capture (Fase 3: TP médio estimado)
        mfe_capture = round(sum(win_pcts) / len(win_pcts) / tp_ref * 100, 1) if win_pcts else None

        by_strat[sname] = {
            "win_rate":        round(s_wr, 4),
            "profit_factor":   s_pf,
            "edge_decay":      edge_decay,
            "drawdown_contrib": dd_contrib,
            "mfe_capture":     mfe_capture,
            "n":               s_n,
            "wins":            len(s_wins),
            "losses":          len(s_loss),
            "realized_usd":    round(sum(s_pnls), 2),
        }

    return {
        # Global
        "total_trades":   len(all_trades_list),
        "sell_trades":    n,
        "win_rate":       round(wins / n, 4) if n else 0.0,
        "win_count":      wins,
        "loss_count":     losses,
        "avg_win":        round(avg_win,  2),
        "avg_loss":       round(avg_loss, 2),
        "profit_factor":  round(sum_wins / sum_loss, 3) if sum_loss else 0.0,
        "expected_value": round((avg_win * wins + avg_loss * losses) / n, 2) if n else 0.0,
        "max_drawdown":   round(max_dd, 2),
        # Por estratégia
        "by_strategy":    by_strat,
    }


def _load_trades_from_engine() -> list:
    """Converte trades salvos no engine para o formato do dashboard."""
    result = []
    for t in reversed(engine.trades[-50:]):
        pair = t.get("symbol", "") + "-USD"
        result.append({
            "time":     t.get("time", "")[:19].replace("T", " ")[11:],
            "side":     t.get("side", ""),
            "pair":     pair,
            "price":    t.get("price", 0),
            "usd":      t.get("usd", 0),
            "fee":      t.get("fee", 0),
            "strategy": t.get("strategy", ""),
        })
    return result


state = {
    "prices":    {},
    "signals":   {},
    "slots":     strategy_slots,   # 12 slots independentes + 3 manuais
    # FIX: campos BRL completos no state inicial — evita saldo US$ 10.000 na conexão WebSocket
    "portfolio": {
        "usd":                 round(engine.balance_usd, 2),
        "total_usd":           round(engine.portfolio_value(), 2),
        "total_brl":           round(engine.portfolio_value() * 5.70, 2),
        "pnl_usd":             0.0,
        "pnl_brl":             0.0,
        "pnl_pct":             0.0,
        "initial_balance_usd": round(engine.initial_balance, 2),
        "initial_balance_brl": round(TOTAL_BRL_INITIAL, 2),
        "holdings":            {k: round(v, 8) for k, v in engine.holdings.items()},
        "total_fees_usd":      round(engine.total_fees_usd, 4),
    },
    "trades":    _load_trades_from_engine(),
    "feed":      [],
    "history":   _load_history(),
    "cycle":     _current_cycle(),
    "status":        "running",
    "last_update":   "",
    "cycle_start_ts": 0,
    "cycle_interval": CYCLE_INTERVAL,
    "usd_brl":          5.70,
    "trade_amount_brl": 0.0,
    "strategy_pnl":     strategy_pnl,
    "fear_greed":       {"value": 50, "label": "Neutral"},
    "kpis":             _calculate_kpis(),  # Métricas de performance
    # ── Campos de controle — inicializados para evitar undefined no frontend ──
    "market_mode":      "chop",             # bull / chop / bear
    "bear_signals":     [],                 # lista de sinais bear ativos
    "scores":           {p: 0.0 for p in PAIRS},
    "trades_today":     0,
    "max_daily_trades": MAX_DAILY_TRADES,
    "open_slots_count": 0,
    "max_open_slots":   MAX_OPEN_SLOTS,
    "trade_pct":        TRADE_PCT,
    "tp_objective":     {"info": "SL×2 (RR 2:1) por par", "regime": "chop"},
    "sl_objective":     {"info": "ATR×2 por par — ver PAIR_SL_RANGE"},
    "fee_taker":        round(_current_taker_fee() * 100, 4),
    "fee_maker":        round(_current_maker_fee() * 100, 4),
    "fee_vol_30d":      0.0,
    "news_blackout":    False,
    "news_reason":      "",
    "next_news_event":  None,
    "market_breadth":   {
        "alts_above_ema50_pct": None, "btc_dominance": None,
        "funding_rate_btc": None, "funding_rate_avg": None,
        "oi_expansion_btc": None, "oi_expansion_avg": None,
        "score": None, "label": "N/A", "size_multiplier": 1.0,
    },
}


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(HTML_FILE)




@app.get("/candles/{pair}")
async def get_candles(pair: str, granularity: str = "FIVE_MINUTE", limit: int = 150):
    try:
        candles = client.get_candles(pair, granularity=granularity, limit=limit)
        result = []
        for c in sorted(candles, key=lambda x: int(x["start"])):
            result.append({
                "time":   int(c["start"]),
                "open":   float(c["open"]),
                "high":   float(c["high"]),
                "low":    float(c["low"]),
                "close":  float(c["close"]),
                "volume": float(c["volume"]),
            })
        return result
    except Exception as e:
        logger.error(f"get_candles error: {e}")
        return []


@app.post("/trade/buy")
async def manual_buy(pair: str, brl: float = 62.5):
    symbol = pair.split("-")[0]
    ticker = client.get_ticker(pair)
    price  = float(ticker.get("price", 0))
    if not price:
        return {"ok": False, "error": "Preço indisponível"}
    usd = brl / state["usd_brl"]
    qty = usd / price
    if not engine.buy(symbol, usd, price, "manual"):
        return {"ok": False, "error": "Saldo insuficiente"}
    engine.update_price(symbol, price)
    slot_key = f"manual:{pair}"
    strategy_slots[slot_key] = {"qty": qty, "entry": price, "peak": price,
                                 "realized": 0.0, "unrealized": 0.0}
    _save_manual(strategy_slots)
    _record_trade("BUY", pair, qty, price, usd, "manual")
    _update_portfolio_state()
    await broadcast(state)
    return {"ok": True, "qty": qty, "price": price, "usd": usd}


@app.post("/trade/sell")
async def manual_sell(pair: str, qty: float = 0, brl: float = 0):
    symbol = pair.split("-")[0]
    held   = engine.holdings.get(symbol, 0)
    if held <= 0:
        return {"ok": False, "error": f"Sem {symbol} para vender"}
    ticker = client.get_ticker(pair)
    price  = float(ticker.get("price", 0))
    if not price:
        return {"ok": False, "error": "Preço indisponível"}
    if brl > 0:
        # Converte valor em BRL para qty de cripto
        usd_value = brl / state["usd_brl"]
        sell_qty = min(usd_value / price, held)
    else:
        sell_qty = min(qty, held) if qty > 0 else held   # 0 = vender tudo
    usd = sell_qty * price * (1 - _current_taker_fee())
    if not engine.sell(symbol, sell_qty, price, "manual"):
        return {"ok": False, "error": "Falha na venda"}
    # Venda total: zera slot manual
    if sell_qty >= held:
        slot_k = f"manual:{pair}"
        if slot_k in strategy_slots:
            strategy_slots[slot_k].update({"qty": 0.0, "entry": 0.0, "peak": 0.0})
    _save_slots(strategy_slots)
    _record_trade("SELL", pair, sell_qty, price, usd, "manual")
    _update_portfolio_state()
    await broadcast(state)
    return {"ok": True, "qty": sell_qty, "price": price, "usd": usd}


@app.post("/admin/reset-portfolio")
async def reset_portfolio(token: str = "", brl: float = 0.0):
    """Reset completo com portfolio inicial em BRL (padrão: TOTAL_BRL_INITIAL)."""
    expected = os.getenv("RESET_TOKEN", "reset2026")
    if token != expected:
        return {"ok": False, "error": "Token inválido"}

    # Permite sobrescrever o valor inicial via parâmetro
    global TOTAL_BRL_INITIAL
    if brl > 0:
        TOTAL_BRL_INITIAL = float(brl)

    # ── Busca preços e câmbio ────────────────────────────────────
    usd_brl = _fetch_usd_brl()
    prices  = {}
    for pair in PAIRS:
        try:
            t = client.get_ticker(pair)
            prices[pair] = float(t.get("price", 0))
        except Exception as e:
            logger.error(f"reset: preço de {pair} indisponível: {e}")
    if not all(prices.get(p) for p in PAIRS):
        return {"ok": False, "error": f"Preços indisponíveis: {prices}"}

    # ── Portfolio em BRL é FIXO (R$ 4.000) ────────────────────────────
    ALLOC_BRL = TOTAL_BRL_INITIAL / 4      # R$1.000 por cripto (1000 + 1000 + 1000 + 1000)
    alloc_usd = ALLOC_BRL / usd_brl        # valor em USD na cotação atual
    total_usd = TOTAL_BRL_INITIAL / usd_brl  # portfolio total em USD na cotação atual

    # ── Reinicia engine DIRETAMENTE — sem compra, sem taxas ──────
    # Portfólio: 100% em caixa — sem posições pré-carregadas
    # Estratégias compram naturalmente via sinais → histórico 100% real
    # P&L = portfolio_value() - initial_balance = total_usd - total_usd = 0,00 ✅
    engine.initial_balance = total_usd
    engine.balance_usd     = total_usd   # tudo em caixa
    engine.holdings        = {}
    engine.entry_prices    = {}
    engine.trades          = []
    engine.total_fees_usd  = 0.0
    engine.prices          = {p.split("-")[0]: prices[p] for p in PAIRS}
    engine._save_state()

    # ── Todos os slots zerados — sem posições artificiais ────────
    for s in all_strategies:
        for pair in PAIRS:
            strategy_slots[f"{s.name}:{pair}"] = _empty_slot()
    for pair in PAIRS:
        strategy_slots[f"manual:{pair}"] = _empty_slot()
    _save_slots(strategy_slots)
    state["slots"] = strategy_slots

    # ── Reinicia P&L por estratégia ──────────────────────────────
    for name in strategy_pnl:
        strategy_pnl[name] = {"realized": 0.0, "trades": 0, "buys": 0, "sells": 0}
    _save_strategy_pnl(strategy_pnl)
    state["strategy_pnl"] = strategy_pnl

    # ── Reinicia histórico e feed ─────────────────────────────────
    state["history"] = []
    state["trades"]  = []
    state["feed"]    = []
    _save_history(state["history"])

    # ── Reinicia cooldowns, sinais e contadores ───────────────────
    sl_cooldowns.clear()
    last_signals.clear()
    _daily_trade_count.clear()
    last_buy_time.clear()
    pending_orders.clear()       # cancela limit orders pendentes no reset

    _update_portfolio_state()
    await broadcast(state)

    # Insere entrada de reset no feed imediatamente (visível antes do ciclo completar)
    state["feed"].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "cycle": state["cycle"],
        "pair": "—", "strategy": "Sistema",
        "signal": "HOLD", "price": 0,
        "executed": False,
        "note": f"Reset R$ {TOTAL_BRL_INITIAL:,.0f} · aguardando sinais...",
    })

    # Dispara ciclo imediato + flag para forçar feed population
    global _force_feed_populate
    _force_feed_populate = True
    _immediate_cycle.set()
    logger.info("[Reset] Ciclo imediato agendado — dashboard será populado em instantes")

    summary = {
        "ok":        True,
        "total_brl": round(TOTAL_BRL_INITIAL, 2),
        "usd_brl":   round(usd_brl, 4),
        "total_usd": round(total_usd, 2),
        "cash_usd":  round(total_usd, 2),
        "cash_brl":  round(TOTAL_BRL_INITIAL, 2),
        "note":      f"Portfolio em R$ {TOTAL_BRL_INITIAL:,.0f} (fixo) — variação USD/BRL não afeta P&L",
        "slots":     "todos zerados — estratégias operam por sinal",
    }
    logger.info(f"✅ RESET COMPLETO — {summary}")
    return summary


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    await websocket.send_json(state)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def broadcast(data: dict):
    """Broadcast estado com timeout para evitar travamentos de clientes lentos"""
    dead = []
    for ws in connected_clients:
        try:
            await asyncio.wait_for(ws.send_json(data), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timeout - removendo cliente")
            dead.append(ws)
        except Exception as e:
            logger.debug(f"WebSocket send error: {e}")
            dead.append(ws)
    for ws in dead:
        try:
            connected_clients.remove(ws)
        except ValueError:
            pass


def get_rsi_value(candles, period=14):
    try:
        import pandas as pd
        df = pd.DataFrame(candles, columns=["start","low","high","open","close","volume"])
        df = df.astype({"close": float}).sort_values("start")
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, float("inf"))
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 1)
    except Exception:
        return 50.0


def _record_trade(side, pair, qty, price, usd, strategy):
    symbol = pair.split("-")[0]
    fee = usd * TAKER_FEE if side == "BUY" else usd / (1 - TAKER_FEE) * TAKER_FEE
    log_trade(logger, side, pair, qty, price, usd, strategy)
    notify_trade(side, pair, qty, price, usd)
    state["trades"].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "side": side, "pair": pair,
        "price": price, "usd": usd,
        "fee":  round(fee, 6),
        "strategy": strategy,
    })
    state["trades"] = state["trades"][:50]


async def trading_loop():
    logger.info("Loop independente — 4 estratégias × 25%%, ciclo %ds", CYCLE_INTERVAL)
    loop = asyncio.get_event_loop()
    while True:
        state["cycle"] = _current_cycle()
        now_str = datetime.now().strftime("%H:%M:%S")

        # Auto-sync do calendário de notícias (a cada 12h se API configurada)
        _news_sync_if_needed()
        state["last_update"]    = now_str
        state["cycle_start_ts"] = int(time.time())

        try:
            usd_brl = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_usd_brl),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("USD/BRL fetch timeout - usando cache")
            usd_brl = _usd_brl_cache["rate"]
        state["usd_brl"] = round(usd_brl, 4)
        state["trade_pct"] = TRADE_PCT  # máximo — o real por par é dinâmico (2-10%)

        # Calcula portfolio_total
        portfolio_total = engine.portfolio_value()
        state["trade_amount_brl"] = round(portfolio_total * TRADE_PCT * usd_brl, 2)

        # Fear & Greed (cache 1h — non-blocking via executor)
        try:
            fg = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_fear_greed),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Fear&Greed fetch timeout - usando cache")
            fg = _fg_cache
        state["fear_greed"] = {"value": fg["value"], "label": fg["label"]}
        fg_value       = fg["value"]
        # Fase 3: SL/TP via ATR por par — sem dinâmica F&G global
        state["sl_objective"] = {"info": "ATR×2 por par — ver PAIR_SL_RANGE"}

        # Fase 4: sizing = TRADE_PCT × regime_mult (sem dynamic_pct)

        # ── Pre-fetch paralelo de candles — garante cache populado no 1º ciclo ─
        # Breadth e Regime usam _candle_cache ANTES do loop de pares.
        # Sem este bloco, no primeiro ciclo após restart/reset o cache está vazio
        # e breadth mostra N/A enquanto regime cai para "chop" sem dados reais.
        _pf_jobs = (
            [loop.run_in_executor(None, _get_candles, p, CANDLE_1H, 250) for p in PAIRS] +
            [loop.run_in_executor(None, _get_candles, p, CANDLE_6H, 100) for p in PAIRS] +
            [loop.run_in_executor(None, _get_candles, "BTC-USD", CANDLE_1D, 250)]
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*_pf_jobs, return_exceptions=True),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            logger.warning("[Loop] Pre-fetch candles timeout — usando cache existente")

        # ── Market Breadth — calculado ANTES do regime (alimenta bear signals) ─
        try:
            _breadth_candles = {
                p: _candle_cache.get(f"{p}:{CANDLE_1H}", {}).get("data", [])
                for p in PAIRS
            }
            _breadth = await asyncio.wait_for(
                loop.run_in_executor(None, get_market_breadth, _breadth_candles),
                timeout=15.0
            )
        except Exception as _be:
            logger.warning(f"[MarketBreadth] Erro: {_be}")
            _breadth = None
        state["market_breadth"] = _breadth.to_dict() if _breadth else {
            "alts_above_ema50_pct": None, "btc_dominance": None,
            "funding_rate_btc": None, "funding_rate_avg": None,
            "oi_expansion_btc": None, "oi_expansion_avg": None,
            "score": None, "label": "N/A", "size_multiplier": 1.0,
        }

        # ── Market Regime Engine — bull/chop/bear com 4 sinais adicionais ─────
        try:
            _btc_1h = _candle_cache.get(f"BTC-USD:{CANDLE_1H}", {}).get("data", [])
            _btc_6h = _candle_cache.get(f"BTC-USD:{CANDLE_6H}", {}).get("data", [])
            market_mode, _bear_signals = _detect_market_regime(_btc_1h, _btc_6h, _breadth)
        except Exception:
            market_mode, _bear_signals = "chop", []
        state["market_mode"]    = market_mode
        state["bear_signals"]   = _bear_signals
        if _bear_signals:
            logger.info(f"[Regime] {market_mode.upper()} | bear signals: {_bear_signals}")

        # TP dinâmico pelo regime (substitui _dynamic_tp simples)
        # Fase 3: TP = SL×2 fixado na entrada por par — sem dinâmica de regime
        state["tp_objective"] = {"info": "SL×2 (RR 2:1) por par", "regime": market_mode}

        # Fase 1: force-close em bear removido.
        # Posições abertas são protegidas pelo ATR stop-loss normal.
        # Fechar na força em regime bear causava perdas desnecessárias.

        for pair in PAIRS:
            symbol = pair.split("-")[0]
            try:
                # Fetch ticker com timeout via executor (evita bloquear event loop)
                ticker = await asyncio.wait_for(
                    loop.run_in_executor(None, client.get_ticker, pair),
                    timeout=8.0
                )
                price  = float(ticker.get("price", 0))
                if not price:
                    continue

                _last_prices[pair] = price
                engine.update_price(symbol, price)
                state["prices"][pair] = {
                    "price":          price,
                    "price_pct_chg":  float(ticker.get("price_percentage_change_24h", 0)),
                    "volume_24h":     float(ticker.get("volume_24h", 0)),
                }

                # Fetch candles com timeout (evita bloquear se API está lenta)
                try:
                    candles_1h = await asyncio.wait_for(
                        loop.run_in_executor(None, _get_candles, pair, CANDLE_1H, 250),
                        timeout=8.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{pair}] Candles 1H timeout - usando cache")
                    candles_1h = _candle_cache.get(f"{pair}:{CANDLE_1H}", {}).get("data", [])

                try:
                    candles_6h = await asyncio.wait_for(
                        loop.run_in_executor(None, _get_candles, pair, CANDLE_6H, 100),
                        timeout=8.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{pair}] Candles 6H timeout - usando cache")
                    candles_6h = _candle_cache.get(f"{pair}:{CANDLE_6H}", {}).get("data", [])

                try:
                    candles_1d = await asyncio.wait_for(
                        loop.run_in_executor(None, _get_candles, pair, CANDLE_1D, 250),
                        timeout=8.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{pair}] Candles 1D timeout - usando cache")
                    candles_1d = _candle_cache.get(f"{pair}:{CANDLE_1D}", {}).get("data", [])

                # Macro: tendência EMA9·21·50 em 1H (idêntico ao EMA Pullback) + vol diária
                df_1h_trend = trend_filter.candles_to_df(candles_1h)
                try:
                    import pandas as _pd
                    _c = df_1h_trend["close"]
                    _e9  = _c.ewm(span=9,  adjust=False).mean()
                    _e21 = _c.ewm(span=21, adjust=False).mean()
                    _e50 = _c.ewm(span=50, adjust=False).mean()
                    if len(_c) >= 50:
                        if float(_e9.iloc[-1]) > float(_e21.iloc[-1]) > float(_e50.iloc[-1]):
                            trend = "BUY"    # EMAs alinhadas em alta
                        elif float(_e9.iloc[-1]) < float(_e21.iloc[-1]):
                            trend = "SELL"   # EMA9 abaixo de EMA21 — perda de tendência
                        else:
                            trend = "HOLD"
                    else:
                        trend = "HOLD"
                except Exception:
                    trend = trend_filter.analyze(df_1h_trend)  # fallback
                df_1h_vg   = vol_guard.candles_to_df(candles_1h)
                vol_signal = vol_guard.analyze(df_1h_vg)
                pair_signals = {"Trend": trend, "Vol Guard": vol_signal}
                # Defaults caso vol_guard dispare e o bloco else seja pulado
                pair_score           = 0.0
                _adx_val             = 20.0
                donchian_6h_confirmed = True

                # ── Verificação de Pending Limit Orders (EMA Pullback) ──────────
                # Verifica se alguma limit order pendente foi preenchida neste ciclo.
                # Fill condition: candle low ≤ limit_price (preço chegou ao nivel desejado)
                for _strat in all_strategies:
                    _pk = f"{_strat.name}:{pair}"
                    _po = pending_orders.get(_pk)
                    if not _po:
                        continue
                    _candle_low = float(candles_1h[-2][3] if candles_1h and len(candles_1h) >= 2
                                        else (candles_1h[-1].get("low", price) if candles_1h and isinstance(candles_1h[-1], dict)
                                              else price))
                    _now_ts = time.time()
                    if _candle_low <= _po["limit_price"]:
                        # ✅ FILLED — executa ao preço limite com MAKER fee
                        _fill_px  = _po["limit_price"]
                        _fill_usd = _po["trade_usd"]
                        _fill_qty = _fill_usd / _fill_px
                        if engine.buy(symbol, _fill_usd, _fill_px, f"{_strat.name}:limit"):
                            _slot = strategy_slots.setdefault(_pk, _empty_slot())
                            _slot["qty"]       = _fill_qty
                            _slot["entry"]     = _fill_px
                            _slot["peak"]      = _fill_px
                            _slot["entry_usd"] = _fill_usd
                            _slot["sl_pct"]    = _po.get("sl_pct", 0.0)
                            _slot["be_sl"]     = 0.0
                            _record_trade("BUY", pair, _fill_qty, _fill_px, _fill_usd, f"{_strat.name}:limit")
                            _count_buy(_strat.name)
                            last_buy_time[_pk] = _now_ts
                            _daily_trade_count[datetime.now().strftime("%Y-%m-%d")] = \
                                _daily_trade_count.get(datetime.now().strftime("%Y-%m-%d"), 0) + 1
                            maker_fee_pct = _current_maker_fee() * 100
                            logger.info(f"[{pair}][{_strat.name}] ✅ LIMIT FILLED @ ${_fill_px:,.2f} "
                                        f"| fee={maker_fee_pct:.2f}% (maker)")
                            state["feed"].insert(0, {
                                "time": now_str, "cycle": state["cycle"],
                                "pair": pair, "strategy": _strat.name,
                                "signal": "BUY", "price": _fill_px,
                                "executed": True,
                                "note": f"LIMIT filled · maker {maker_fee_pct:.2f}%",
                            })
                            state["feed"] = state["feed"][:100]
                        del pending_orders[_pk]
                    elif _now_ts > _po["expires_at"]:
                        # ❌ EXPIRED — cancela sem executar
                        logger.info(f"[{pair}][{_strat.name}] ⏰ LIMIT expirado sem fill "
                                    f"(limit=${_po['limit_price']:,.2f} vs low=${_candle_low:,.2f})")
                        state["feed"].insert(0, {
                            "time": now_str, "cycle": state["cycle"],
                            "pair": pair, "strategy": _strat.name,
                            "signal": "HOLD", "price": _po["limit_price"],
                            "executed": False,
                            "note": f"LIMIT expirado (não preencheu em {LIMIT_ORDER_TIMEOUT_H}h)",
                        })
                        state["feed"] = state["feed"][:100]
                        del pending_orders[_pk]

                # ── Volatilidade: BLOCK (1σ) bloqueia entradas; SELL (2σ) fecha posições ──
                _vol_buy_blocked = vol_signal in ("SELL", "BLOCK")
                if vol_signal == "SELL":
                    # vol_guard: reduz 5%/ciclo em cada slot aberto deste par
                    max_usd = portfolio_total * TRADE_PCT
                    for strat in all_strategies:
                        vkey = f"{strat.name}:{pair}"
                        vslot = strategy_slots.get(vkey, _empty_slot())
                        if vslot["qty"] > 0:
                            vqty = min(vslot["qty"], max_usd / price if price > 0 else vslot["qty"])
                            vnet = vqty * price * (1 - _current_taker_fee())
                            if engine.sell(symbol, vqty, price, f"vol_guard:{strat.name}"):
                                pnl_vg = vnet - vslot["entry"] * vqty
                                vslot["realized"] += pnl_vg
                                _attr_pnl(strat.name, pnl_vg)
                                _record_trade("SELL", pair, vqty, price, vnet, f"vol_guard:{strat.name}")
                                # ← slot só atualiza se venda foi executada
                                rem = vslot["qty"] - vqty
                                if rem < 1e-8:
                                    vslot.update({"qty": 0.0, "entry": 0.0, "peak": 0.0, "pyramids": 0, "be_sl": 0.0})
                                else:
                                    vslot["qty"] = rem
                    _save_slots(strategy_slots)
                    logger.info(f"[{pair}] VOLATILIDADE EXTREMA — reduzindo posições {TRADE_PCT*100:.0f}%/ciclo")
                    # ← Não executa estratégias neste ciclo para evitar re-abertura imediata

                else:
                  # ── PASSO 1: coleta TODOS os sinais antes de agir ─────────
                  # Donchian: 1H principal, confirmação 6H (proxy 4H)
                  # Calcular confirmação 6H para Donchian
                  def _donchian_6h_bullish(candles_6h_data: list) -> bool:
                      """Confirmação 6H (proxy 4H): preço acima da EMA20 no 6H."""
                      if not candles_6h_data or len(candles_6h_data) < 20:
                          return True  # sem dados → não bloqueia
                      try:
                          import pandas as _pd6
                          df6 = _pd6.DataFrame(candles_6h_data,
                                               columns=["start","low","high","open","close","volume"])
                          closes6 = df6["close"].astype(float)
                          ema20_6h = float(closes6.ewm(span=20, adjust=False).mean().iloc[-1])
                          price_6h = float(closes6.iloc[-1])
                          return price_6h > ema20_6h
                      except Exception:
                          return True

                  donchian_6h_confirmed = _donchian_6h_bullish(candles_6h)

                  candle_map  = {
                      "Donchian Breakout": candles_1h,   # análise 1H
                      "EMA Pullback":      candles_1h,
                      "MACD Momentum":     candles_1h,
                  }
                  signals_this_cycle = {}

                  for strat in all_strategies:
                      raw    = candle_map.get(strat.name, candles_1h)
                      df     = strat.candles_to_df(raw)
                      signal = strat.analyze(df)
                      pair_signals[strat.name]        = signal
                      signals_this_cycle[strat.name]  = signal

                  # Confidence score e ADX para este par
                  try:
                      import pandas as _pdx
                      _df_adx = _pdx.DataFrame(candles_1h, columns=["start","low","high","open","close","volume"])
                      for _c in ["low","high","open","close","volume"]:
                          _df_adx[_c] = _pdx.to_numeric(_df_adx[_c], errors="coerce")
                      _adx_val = calc_adx(_df_adx)
                  except Exception:
                      _adx_val = 20.0
                  pair_score = _calc_confidence_score(signals_this_cycle, market_mode, _adx_val)

                  # Feed: registra mudanças de sinal (ou força todos no 1º ciclo pós-reset)
                  global _force_feed_populate
                  for strat in all_strategies:
                      signal  = signals_this_cycle[strat.name]
                      sig_key = f"{pair}:{strat.name}"
                      changed = signal != last_signals.get(sig_key)
                      if changed or _force_feed_populate:
                          last_signals[sig_key] = signal
                          trade_usd = portfolio_total * TRADE_PCT
                          state["feed"].insert(0, {
                              "time":     now_str,
                              "cycle":    state["cycle"],
                              "pair":     pair,
                              "strategy": strat.name,
                              "signal":   signal,
                              "price":    price,
                              "executed": False,
                              "note":     f"R${trade_usd*usd_brl:.0f} ({TRADE_PCT*100:.0f}% PL)",
                          })
                          state["feed"] = state["feed"][:100]
                  # Após processar o último par, desliga o flag
                  if pair == PAIRS[-1] and _force_feed_populate:
                      _force_feed_populate = False
                      logger.info("[Feed] Force-populate concluído — feed populado com sinais iniciais")

                  # ── PASSO 2: execução independente por estratégia ────────
                  extreme_greed = fg_value >= FG_GREED_MIN
                  # extreme_fear removido — não era usado em nenhuma lógica

                  # Circuit breaker diário
                  today_str    = now_str[:8] if len(now_str) == 8 else str(now_str)[:10]
                  today_key    = datetime.now().strftime("%Y-%m-%d")
                  daily_trades = _daily_trade_count.get(today_key, 0)
                  circuit_open = daily_trades >= MAX_DAILY_TRADES

                  # Slots abertos
                  open_slots_count = sum(1 for s in strategy_slots.values() if s.get("qty", 0) > 0)

                  # Tamanho dinâmico por par — usa candles 1h (mais estável)
                  # dynamic_pct removido — sizing usa TRADE_PCT × regime_mult

                  # Expor ao frontend
                  state["trades_today"]    = daily_trades
                  state["max_daily_trades"] = MAX_DAILY_TRADES
                  state["open_slots_count"] = open_slots_count
                  state["max_open_slots"]   = MAX_OPEN_SLOTS

                  for strat in all_strategies:
                      key    = f"{strat.name}:{pair}"
                      slot   = strategy_slots.setdefault(key, _empty_slot())
                      signal = signals_this_cycle.get(strat.name, "HOLD")

                      if slot["qty"] > 0:
                          # ── Limpa posições de "pó" (valor < $0.50) ──────────────
                          slot_value_usd = slot["qty"] * price
                          if slot_value_usd < 0.50:
                              logger.info(f"[{pair}][{strat.name}] Zerando posição de pó: {slot['qty']:.10f} val=${slot_value_usd:.4f}")
                              engine.holdings[symbol] = max(0, engine.holdings.get(symbol, 0) - slot["qty"])
                              slot.update({"qty": 0.0, "entry": 0.0, "peak": 0.0, "pyramids": 0, "be_sl": 0.0})
                              continue

                          slot["peak"] = max(slot["peak"], price)
                          gain_pct = (price - slot["entry"]) / slot["entry"] * 100

                          # ── Sistema de saída unificado (Fase 3) ────────────────
                          tp_hit, sl_hit, effective_sl, tp_level, _sl_pct = \
                              _calc_exit(slot, price, pair)
                          slot["be_sl"] = effective_sl  # persiste nível máximo de SL

                          def _sell_slot(qty, label, is_sl=False):
                              net = qty * price * (1 - _current_taker_fee())
                              sold = engine.sell(symbol, qty, price, f"{strat.name}:{label}")
                              if sold:
                                  pnl = net - slot["entry"] * qty
                                  slot["realized"] += pnl
                                  _attr_pnl(strat.name, pnl)
                                  _record_trade("SELL", pair, qty, price, net, f"{strat.name}:{label}")
                                  _daily_trade_count[today_key] = _daily_trade_count.get(today_key, 0) + 1
                                  logger.info(f"[{pair}][{strat.name}] SELL {label} gain={gain_pct:+.2f}% P&L ${pnl:+.2f}")
                                  if is_sl:
                                      # Fase 4: cooldown fixo pós-SL (3 ciclos = 3h)
                                      sl_cooldowns[key] = SL_COOLDOWN_CYCLES
                                  # Sempre insere novo entry no topo — mesma lógica do BUY executado
                                  state["feed"].insert(0, {
                                      "time": now_str, "cycle": state["cycle"],
                                      "pair": pair, "strategy": strat.name,
                                      "signal": "SELL", "price": price,
                                      "executed": True,
                                      "note": label,
                                  })
                                  state["feed"] = state["feed"][:100]
                                  # ← Só atualiza o slot SE a venda foi executada no engine
                                  rem = slot["qty"] - qty
                                  if rem < 1e-8:
                                      slot.update({"qty": 0.0, "entry": 0.0, "peak": 0.0, "pyramids": 0, "be_sl": 0.0})
                                  else:
                                      slot["qty"] = rem
                                      slot["peak"] = price  # reseta peak após venda parcial
                              else:
                                  logger.warning(f"[{pair}][{strat.name}] SELL {label} FALHOU — engine rejeitou (held insuficiente?)")

                          if tp_hit:
                              _sell_slot(slot["qty"], f"TP+{_sl_pct*2:.1f}%")
                          elif sl_hit:
                              lbl = "BE-stop" if gain_pct >= 0 else f"SL-{_sl_pct:.1f}%"
                              _sell_slot(slot["qty"], lbl, is_sl=True)
                          elif signal == "SELL":
                              max_qty = portfolio_total * TRADE_PCT / price
                              sell_q  = min(slot["qty"], max_qty)
                              if sell_q > 1e-8:
                                  _sell_slot(sell_q, "SELL")
                          # Pyramid removido na Fase 4 — sem re-entry em posição aberta

                          slot["unrealized"] = (price - slot["entry"]) * slot["qty"] if slot["qty"] > 0 else 0.0

                      elif signal == "BUY" and not extreme_greed:
                          # ── Gates de qualidade — avaliados em sequência ──────────
                          _buy_blocked = None

                          # G-Vol: VolatilityGuard BLOCK (1σ) — spread/volatilidade elevada
                          if _vol_buy_blocked:
                              _buy_blocked = f"VolGuard: volatilidade {'extrema (SELL)' if vol_signal == 'SELL' else 'elevada (BLOCK)'} — entradas bloqueadas"

                          # G-0.5: Market Breadth — apenas reduz size (nunca bloqueia hard)
                          # _breadth_mult já aplica 0.5/0.7/1.0 na execução abaixo.
                          # Hard-block removido na Fase 1: breadth DANGER não é sinal
                          # suficiente para zero-trade; o SL protege posições abertas.

                          # G-1: News Volatility Guard — bloqueia antes/depois de eventos macro
                          _news_blocked, _news_reason = is_news_blackout(
                              custom_events_path=NEWS_EVENTS_FILE
                          )
                          if _news_blocked:
                              _buy_blocked = _news_reason

                          # G0: Circuit breaker diário
                          if circuit_open:
                              _buy_blocked = f"circuit breaker ({daily_trades}/{MAX_DAILY_TRADES} trades hoje)"

                          # G0b: Max slots abertos
                          elif open_slots_count >= MAX_OPEN_SLOTS:
                              _buy_blocked = f"max slots ({open_slots_count}/{MAX_OPEN_SLOTS})"

                          # G0c: Cooldown de 1h entre BUYs
                          elif time.time() - last_buy_time.get(key, 0) < BUY_COOLDOWN_SECONDS:
                              _buy_blocked = f"cooldown 3h ativo"

                          # G1: Bear market — bloqueia BUYs EXCETO se o par está
                          # individualmente bullish (acima da própria EMA200 1H).
                          # Permite capturar SOL/ETH em recuperação mesmo com BTC bear.
                          elif market_mode == "bear":
                              try:
                                  import pandas as _pd_g1
                                  _c1h_g1 = _pd_g1.DataFrame(
                                      candles_1h, columns=["start","low","high","open","close","volume"]
                                  )
                                  for _col in ["close"]:
                                      _c1h_g1[_col] = _pd_g1.to_numeric(_c1h_g1[_col], errors="coerce")
                                  _ema200_1h = float(
                                      _c1h_g1["close"].ewm(span=200, adjust=False).mean().iloc[-1]
                                  )
                                  _pair_above_ema200 = price > _ema200_1h
                              except Exception:
                                  _pair_above_ema200 = False
                              if not _pair_above_ema200:
                                  _buy_blocked = f"bear market (par < EMA200 1H)"

                          # G1b: Donchian — confirmação obrigatória no 6H (proxy 4H)
                          elif strat.name == "Donchian Breakout" and not donchian_6h_confirmed:
                              _buy_blocked = f"Donchian 6H bearish (proxy 4H)"

                          # G1c: BB Reversion — exclusivo do regime CHOP
                          # Em BULL, trend-following domina; mean reversion luta contra a tendência
                          elif strat.name == "BB Reversion" and market_mode == "bull":
                              _buy_blocked = f"BB Reversion bloqueado em BULL (use Donchian/MACD)"

                          # G2: Score mínimo 60%
                          elif pair_score < SCORE_MIN_THRESHOLD:
                              _buy_blocked = f"score {pair_score:.0%} < {SCORE_MIN_THRESHOLD:.0%}"

                          # G3: Min Risk/Reward 2.5:1
                          else:
                              _sl_max_pct = PAIR_SL_RANGE.get(pair, (0.05, 0.08))[1]
                              # RR fixo 2:1 pelo sistema unificado — gate G3 satisfeito
                              _rr = 2.0
                              if False:  # gate G3 desativado na Fase 3 (RR é fixo 2:1)
                                  _buy_blocked = f"RR={_rr:.2f} < 2.5:1"

                          # G4: Correlação de alts
                          if _buy_blocked is None and pair in ALT_PAIRS:
                              _open_alts = sum(1 for k2, s2 in strategy_slots.items()
                                               if s2.get("qty", 0) > 0
                                               and k2.split(":")[-1] in ALT_PAIRS)
                              _alt_sym = {p.split("-")[0] for p in ALT_PAIRS}
                              _alt_val = sum(
                                  s2["qty"] * (_last_prices.get(k2.split(":")[-1], {}).get("price") or 0)
                                  for k2, s2 in strategy_slots.items()
                                  if s2.get("qty", 0) > 0 and k2.split(":")[-1] in _alt_sym
                              )
                              _alt_exp = _alt_val / portfolio_total if portfolio_total > 0 else 0
                              if _open_alts >= 2:
                                  _buy_blocked = f"max 2 alts ({_open_alts} abertas)"
                              elif _alt_exp >= 0.25:
                                  _buy_blocked = f"alt exposure {_alt_exp:.0%} >= 25%"

                          # G4b removido na Fase 1: com apenas 3 pares (BTC/ETH/SOL),
                          # o cap de 35% BTC+ETH bloqueava demais. G4 (max alts) é suficiente.

                          # G5: SL cooldown
                          cooldown = sl_cooldowns.get(key, 0)
                          if _buy_blocked is None and cooldown > 0:
                              sl_cooldowns[key] = cooldown - 1
                              _buy_blocked = f"SL cooldown ({cooldown} ciclos)"

                          if _buy_blocked:
                              logger.debug(f"[{pair}][{strat.name}] BUY bloqueado — {_buy_blocked}")
                          else:
                              # ── Sizing: TRADE_PCT × regime_mult ──────────────────
                              _regime_mult = 1.0 if market_mode == "bull" else \
                                             0.7 if market_mode == "chop" else 0.5
                              trade_usd = portfolio_total * TRADE_PCT * _regime_mult

                              # SL% para usar na ordem
                              _sl_min, _sl_max = PAIR_SL_RANGE.get(pair, (0.03, 0.07))
                              _sl_at_entry = _atr_sl_pct if _atr_sl_pct else _sl_max * 100
                              _sl_pct_entry = max(_sl_min * 100, min(_sl_max * 100, _sl_at_entry))

                              if strat.name in LIMIT_STRATEGIES and key not in pending_orders:
                                  # ── LIMIT ORDER (EMA Pullback → maker 0.10%) ───────
                                  # Limit price: EMA21 para EMA Pullback, Banda Inferior BB para BB Reversion
                                  try:
                                      _df_lim = pd.DataFrame(candles_1h,
                                          columns=["start","low","high","open","close","volume"])
                                      _df_lim["close"] = pd.to_numeric(_df_lim["close"], errors="coerce")
                                      _closes_lim = _df_lim["close"]
                                      if strat.name == "BB Reversion":
                                          # Limit = Banda Inferior BB(20,2) do último candle fechado
                                          _bb_mid = _closes_lim.rolling(20).mean()
                                          _bb_std = _closes_lim.rolling(20).std()
                                          _limit_px = float((_bb_mid - 2.0 * _bb_std).iloc[-2])
                                      else:
                                          # EMA Pullback: limit = EMA21
                                          _limit_px = float(_closes_lim.ewm(span=21, adjust=False).mean().iloc[-2])
                                  except Exception:
                                      _limit_px = price * 0.9995   # fallback: 0.05% abaixo

                                  fee_margin = _current_maker_fee()
                                  if engine.balance_usd >= trade_usd * (1 + fee_margin) and trade_usd > 1.0:
                                      pending_orders[key] = {
                                          "limit_price": round(_limit_px, 4),
                                          "trade_usd":   round(trade_usd, 4),
                                          "expires_at":  time.time() + LIMIT_ORDER_TIMEOUT_H * CYCLE_INTERVAL,
                                          "sl_pct":      _sl_pct_entry,
                                      }
                                      last_buy_time[key] = time.time()  # previne re-sinalização imediata
                                      pct_used = (trade_usd / engine.initial_balance) * 100 if engine.initial_balance > 0 else 0
                                      logger.info(f"[{pair}][{strat.name}] 📋 LIMIT @ ${_limit_px:,.4f} "
                                                  f"(market=${price:,.2f}) | R${trade_usd*usd_brl:.0f} | maker fee")
                                      state["feed"].insert(0, {
                                          "time": now_str, "cycle": state["cycle"],
                                          "pair": pair, "strategy": strat.name,
                                          "signal": "BUY", "price": _limit_px,
                                          "executed": False,
                                          "note": f"LIMIT R${trade_usd*usd_brl:.0f} | maker 0.10%",
                                      })
                                      state["feed"] = state["feed"][:100]

                              elif strat.name not in LIMIT_STRATEGIES:
                                  # ── MARKET ORDER (Donchian, MACD → taker 0.40%) ────
                                  fee_margin = _current_taker_fee()
                                  if engine.balance_usd < trade_usd * (1 + fee_margin):
                                      trade_usd = max(0, engine.balance_usd * (1 - fee_margin))

                                  if trade_usd > 1.0:
                                      qty = trade_usd / price
                                      if engine.buy(symbol, trade_usd, price, strat.name):
                                          slot["qty"]       = qty
                                          slot["entry"]     = price
                                          slot["peak"]      = price
                                          slot["entry_usd"] = trade_usd
                                          slot["sl_pct"]    = _sl_pct_entry
                                          slot["be_sl"]     = 0.0
                                          _record_trade("BUY", pair, qty, price, trade_usd, strat.name)
                                          _count_buy(strat.name)
                                          last_buy_time[key] = time.time()
                                          _daily_trade_count[today_key] = daily_trades + 1
                                          pct_used = (trade_usd / engine.initial_balance) * 100 if engine.initial_balance > 0 else 0
                                          logger.info(f"[{pair}][{strat.name}] ✅ MARKET BUY {pct_used:.1f}% "
                                                      f"R${trade_usd*usd_brl:.0f} @ ${price:,.2f} | taker fee")
                                          state["feed"].insert(0, {
                                              "time": now_str, "cycle": state["cycle"],
                                              "pair": pair, "strategy": strat.name,
                                              "signal": "BUY", "price": price,
                                              "executed": True,
                                              "note": f"MARKET R${trade_usd*usd_brl:.0f} | taker 0.40%",
                                          })
                                          state["feed"] = state["feed"][:100]
                                  else:
                                      logger.info(f"[{pair}][{strat.name}] BUY negado — saldo insuficiente")

                # ── Slot manual: usa mesma regra unificada (Fase 3) ─────────────
                ms = strategy_slots.get(f"manual:{pair}")
                if ms and ms.get("qty", 0) > 0:
                    ms["peak"] = max(ms["peak"], price)
                    g = (price - ms["entry"]) / ms["entry"] * 100
                    ms_tp, ms_sl, ms_eff_sl, ms_tp_lvl, ms_sl_pct = _calc_exit(ms, price, pair)
                    ms["be_sl"] = ms_eff_sl
                    rsn = (f"TP+{ms_sl_pct*2:.1f}%" if ms_tp else
                           f"BE-stop"               if ms_sl and g >= 0 else
                           f"SL-{ms_sl_pct:.1f}%"  if ms_sl else None)
                    if rsn:
                        net = ms["qty"] * price * (1 - _current_taker_fee())
                        if engine.sell(symbol, ms["qty"], price, f"manual:{rsn}"):
                            ms["realized"] += net - ms["entry"] * ms["qty"]
                            _record_trade("SELL", pair, ms["qty"], price, net, f"manual:{rsn}")
                            logger.info(f"[{pair}][manual] {rsn} @ ${price:,.2f}")
                            ms.update({"qty": 0.0, "entry": 0.0, "peak": 0.0, "be_sl": 0.0})
                        else:
                            logger.warning(f"[{pair}][manual] {rsn} FALHOU — engine rejeitou")
                    else:
                        ms["unrealized"] = (price - ms["entry"]) * ms["qty"]

                # ── Verificação de consistência slots ↔ engine ───────────────
                # Garante que slots não acumulem qty quando engine não tem posição
                held_in_engine = engine.holdings.get(symbol, 0)
                slots_total_qty = sum(
                    strategy_slots.get(f"{s.name}:{pair}", {}).get("qty", 0)
                    for s in all_strategies
                ) + strategy_slots.get(f"manual:{pair}", {}).get("qty", 0)

                if held_in_engine < slots_total_qty - 1e-6:
                    # Engine tem menos que os slots declaram — corrige proporcionalmente
                    logger.warning(f"[{pair}] INCONSISTÊNCIA: engine={held_in_engine:.6f} slots={slots_total_qty:.6f} — corrigindo")
                    if slots_total_qty > 1e-10:
                        ratio = held_in_engine / slots_total_qty
                        for s in all_strategies:
                            sk = f"{s.name}:{pair}"
                            if strategy_slots.get(sk, {}).get("qty", 0) > 0:
                                strategy_slots[sk]["qty"] *= ratio
                                if strategy_slots[sk]["qty"] < 1e-8:
                                    strategy_slots[sk].update({"qty": 0.0, "entry": 0.0, "peak": 0.0, "pyramids": 0, "be_sl": 0.0})
                        ms_key = f"manual:{pair}"
                        if strategy_slots.get(ms_key, {}).get("qty", 0) > 0:
                            strategy_slots[ms_key]["qty"] *= ratio
                            if strategy_slots[ms_key]["qty"] < 1e-8:
                                strategy_slots[ms_key].update({"qty": 0.0, "entry": 0.0, "peak": 0.0, "be_sl": 0.0})

                # ── Salva slots e atualiza signals no state ───────────────────
                _save_slots(strategy_slots)
                state["slots"] = strategy_slots

                rsi_val     = get_rsi_value(candles_1h)
                entry_price = engine.entry_prices.get(symbol)
                change_pct  = ((price - entry_price) / entry_price * 100) if entry_price else None

                # ATR Stop Loss dinâmico
                try:
                    _df_atr = pd.DataFrame(candles_1h, columns=["start","low","high","open","close","volume"]).astype(
                        {"low": float, "high": float, "open": float, "close": float, "volume": float}
                    )
                    _atr_val = calc_atr(_df_atr)
                    _atr_sl_level = round(price - 2.0 * _atr_val, 2) if _atr_val > 0 else None
                    _atr_sl_pct   = round((_atr_val * 2.0 / price) * 100, 2) if price > 0 and _atr_val > 0 else None
                except Exception:
                    _atr_sl_level = None
                    _atr_sl_pct   = None

                # Score e MTF para o frontend
                state["scores"][pair] = round(pair_score, 3)

                state["signals"][pair] = {
                    "strategies":  pair_signals,
                    "trend":       trend,
                    "vol_guard":   vol_signal,
                    "rsi":         rsi_val,
                    "entry_price": round(entry_price, 2) if entry_price else None,
                    "change_pct":  round(change_pct,  2) if change_pct is not None else None,
                    "sl_level":    round(entry_price * (1 - (_atr_sl_pct or 5.0) / 100), 2) if entry_price else None,
                    "tp_level":    round(entry_price * (1 + (_atr_sl_pct or 5.0) * 2 / 100), 2) if entry_price else None,
                    "atr_sl_level": _atr_sl_level,
                    "atr_sl_pct":   _atr_sl_pct,
                    "mtf_ok":       donchian_6h_confirmed,
                    "score":        round(pair_score, 3),
                    "adx":          round(_adx_val, 1),
                }
                log_cycle(logger, state["cycle"], pair, price, pair_signals, trend)

            except Exception as e:
                logger.error(f"[{pair}] Erro: {e}")

        total, pnl = _update_portfolio_state()
        log_portfolio(logger, engine.balance_usd, total, pnl,
                      (pnl / engine.initial_balance) * 100, engine.holdings)
        state["history"].append({"time": now_str, "ts": int(time.time()), "total": round(total, 2)})
        state["history"] = state["history"][-90000:]
        _save_history(state["history"])

        # Atualizar KPIs e tamanho médio de trade a cada ciclo
        state["kpis"] = _calculate_kpis()
        # Fase 4: trade_pct fixo (TRADE_PCT × regime_mult)
        _regime_display = 1.0 if market_mode == "bull" else 0.7 if market_mode == "chop" else 0.5
        state["trade_pct"] = round(TRADE_PCT * _regime_display, 4)

        # News Guard status para o dashboard
        _nb, _nr = is_news_blackout(custom_events_path=NEWS_EVENTS_FILE)
        _nxt = next_event(custom_events_path=NEWS_EVENTS_FILE)
        state["news_blackout"]  = _nb
        state["news_reason"]    = _nr if _nb else ""
        state["next_news_event"] = {
            "name": _nxt["name"], "mins_to": _nxt["mins_to"]
        } if _nxt else None
        state["pending_limit_orders"] = {
            k: {"limit_price": v["limit_price"], "trade_usd": v["trade_usd"],
                "expires_in_min": max(0, int((v["expires_at"] - time.time()) / 60))}
            for k, v in pending_orders.items()
        }

        # Expõe timestamp da próxima hora cheia SP para o countdown do frontend
        _next_sp = _seconds_to_next_sp_hour()
        state["next_cycle_ts"]    = int(time.time() + _next_sp)
        state["cycle_interval"]   = CYCLE_INTERVAL   # mantido para compatibilidade JS

        await broadcast(state)

        # Sleep alinhado ao relógio SP (UTC-3):
        # aguarda até a próxima hora cheia em vez de dormir exatamente 3600s.
        # Interrompível via _immediate_cycle (reset dispara execução imediata).
        logger.info(f"[Loop] Próximo ciclo em {_next_sp:.0f}s "
                    f"({_next_sp/60:.1f} min) — alinhado ao relógio SP")
        try:
            await asyncio.wait_for(_immediate_cycle.wait(), timeout=_next_sp)
            _immediate_cycle.clear()
            logger.info("[Loop] Ciclo imediato solicitado — executando agora")
        except asyncio.TimeoutError:
            pass  # hora cheia SP atingida → próximo ciclo


@app.on_event("startup")
async def startup():
    # ── Fix Bug 2: Sincronizar initial_balance com cotação BRL atual ──────
    usd_brl_now = _fetch_usd_brl()
    correct_initial_usd = TOTAL_BRL_INITIAL / usd_brl_now
    if abs(engine.initial_balance - correct_initial_usd) > 1.0:
        logger.info(f"[STARTUP] Sincronizando initial_balance: {engine.initial_balance:.4f} → {correct_initial_usd:.4f} (R${TOTAL_BRL_INITIAL} @ {usd_brl_now:.4f})")
        engine.initial_balance = correct_initial_usd
        # Só ajusta balance_usd se não houver posições abertas (reset limpo)
        if not engine.holdings:
            engine.balance_usd = correct_initial_usd
        engine._save_state()
    state["usd_brl"] = usd_brl_now

    # Inicializa preços antes de iniciar trading loop
    logger.info(f"[STARTUP] Iniciando inicialização de preços para {PAIRS}")
    for pair in PAIRS:
        try:
            ticker = client.get_ticker(pair)
            price = float(ticker.get("price", 0))
            if price:
                state["prices"][pair] = {
                    "price": price,
                    "price_pct_chg": float(ticker.get("price_percentage_change_24h", 0)),
                    "volume_24h": float(ticker.get("volume_24h", 0)),
                }
                engine.update_price(pair.split("-")[0], price)
                logger.info(f"[STARTUP] {pair}: ${price:.2f}")
            else:
                logger.warning(f"[STARTUP] Preço inválido para {pair}: {ticker.get('price')}")
        except Exception as e:
            logger.error(f"[STARTUP] Erro ao buscar {pair}: {type(e).__name__}: {e}")

    # ── Pre-fetch candles no startup para que o 1º ciclo tenha dados ────────
    logger.info("[STARTUP] Pre-fetching candles (1H + 6H) para todos os pares...")
    loop_startup = asyncio.get_event_loop()
    _startup_jobs = (
        [loop_startup.run_in_executor(None, _get_candles, p, CANDLE_1H, 250) for p in PAIRS] +
        [loop_startup.run_in_executor(None, _get_candles, p, CANDLE_6H, 100) for p in PAIRS]
    )
    try:
        await asyncio.wait_for(asyncio.gather(*_startup_jobs, return_exceptions=True), timeout=30.0)
        logger.info("[STARTUP] Candles pre-fetched com sucesso")
    except asyncio.TimeoutError:
        logger.warning("[STARTUP] Pre-fetch candles timeout — ciclo inicial usará cache parcial")

    # ── Atualizar portfolio state ANTES do primeiro ciclo ─────────────────
    _update_portfolio_state()
    logger.info(f"[STARTUP] Portfolio inicializado — USD: ${engine.balance_usd:.2f} | Total: ${engine.portfolio_value():.2f} | initial_balance: ${engine.initial_balance:.2f}")
    logger.info(f"[STARTUP] state['prices'] após inicialização: {list(state['prices'].keys())}")
    asyncio.create_task(trading_loop())
