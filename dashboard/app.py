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
from paper_trading.engine           import PaperTradingEngine, TAKER_FEE
from paper_trading.simulated_engine import SimulatedExecutionEngine
from exchange.okx_trading           import OKXTradingClient
from strategies.fee_model           import FEE
from strategies.news_guard             import is_news_blackout, next_event
from strategies.market_breadth         import get_market_breadth, MarketBreadthSnapshot
from strategies.market_regime          import calc_adx, calc_atr
from logger import setup_logger, log_cycle, log_trade, log_portfolio
from notifier import notify_trade
from dashboard.v4_orchestrator import V4Orchestrator

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



def _detect_market_regime(candles_1h: list, candles_6h: list,
                           breadth=None) -> tuple:
    """Wrapper legado — delega ao V4 Regime Engine via state."""
    regime = state.get("v4", {}).get("BTC-USD", {}).get("regime", "MEAN_REVERTING_CHOP")
    regime_map = {
        "TREND_EXPANSION": "bull", "TREND_EXHAUSTION": "bull",
        "VOLATILITY_COMPRESSION": "chop", "MEAN_REVERTING_CHOP": "chop",
        "HIGH_CORRELATION_RISK": "chop", "PANIC_LIQUIDATION": "bear",
        "LIQUIDITY_VACUUM": "bear",
    }
    return regime_map.get(regime, "chop"), []


def _get_fee_rates() -> tuple:
    """Retorna (maker, taker) do FeeModel canônico."""
    return FEE.maker, FEE.taker

def _current_taker_fee() -> float:
    return FEE.taker

def _current_maker_fee() -> float:
    return FEE.maker


def _calc_confidence_score(signals: dict, regime: str, adx: float) -> float:
    """Score V3 legado — retorna score V4 se disponível, fallback por pesos."""
    v4_score = state.get("v4", {}).get("BTC-USD", {}).get("score")
    if v4_score is not None:
        return v4_score
    weights = STRATEGY_WEIGHTS.get(regime, STRATEGY_WEIGHTS["neutral"])
    max_w   = sum(weights.values()) or 1.0
    buy_score = sum(weights.get(s, 1.0) for s, sig in signals.items() if sig == "BUY")
    return buy_score / max_w


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
    """Número do ciclo de 15min atual em SP (UTC-3). 0–95 por dia."""
    return (int(time.time()) + SP_OFFSET) % 86400 // CYCLE_INTERVAL


def _seconds_to_next_sp_hour() -> float:
    """
    Segundos até o próximo múltiplo de 15min em SP (UTC-3).
    Alinha ciclos em :00, :15, :30, :45 de cada hora.
    Mínimo de 10s para evitar re-execução imediata.
    """
    now_sp        = time.time() + SP_OFFSET
    secs_in_cycle = now_sp % CYCLE_INTERVAL
    wait          = CYCLE_INTERVAL - secs_in_cycle
    return wait if wait >= 10 else wait + CYCLE_INTERVAL


PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD"]  # 3 pares — foco em ativos de maior liquidez

# ── Portfolio em Real é FIXO em R$ 5.000 ────────────────────────
TOTAL_BRL_INITIAL = 5000.0  # Portfolio inicial em BRL — FIXO, nunca muda
# Portfolio em USD varia com cotação: USD_atual = TOTAL_BRL_INITIAL / usd_brl_atual

# ── Ciclo e candles ─────────────────────────────────────────────
CYCLE_INTERVAL    = 900      # ciclo de 900s (15 minutos), alinhado ao UTC-3
CANDLE_30M        = "THIRTY_MINUTE"
CANDLE_1H         = "ONE_HOUR"       # EMA Pullback, MACD
CANDLE_6H         = "SIX_HOUR"
CANDLE_1D         = "ONE_DAY"        # Trend, VolGuard

# ── Execução por estratégia (independente, sem consenso) ──────────
TRADE_PCT          = 0.10   # 10% do portfolio por trade — reduzido para diminuir taxas e risco



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

SCORE_MIN_THRESHOLD = 0.55   # score V4 mínimo para BUY

# ── Classificação de pares ───────────────────────────────────────
ALT_PAIRS = {"SOL-USD"}
BTC_PAIRS  = {"BTC-USD", "ETH-USD"}
SL_COOLDOWN_CYCLES    = 3

# ── Circuit breaker + controles de risco ─────────────────────────
MAX_DAILY_TRADES      = 20
MAX_OPEN_SLOTS        = 4
BUY_COOLDOWN_SECONDS  = 3600   # 1h — alinhado com ciclos 15min (backtest: -4h → -2h)
_daily_trade_count: dict = {}  # {"YYYY-MM-DD": count}
last_buy_time:      dict = {}  # {f"{strat}:{pair}": timestamp}

# ── Pyramid (scale-in em posição lucrativa) ──────────────────────
# Pyramid removido na Fase 4 — adiciona complexidade sem edge claro em 3 pares

# ── Fear & Greed ─────────────────────────────────────────────────
FG_GREED_MIN   = 70   # Acima de 70: bloqueia novas entradas (euforia = risco de topo)
FG_TTL         = 3600 # cache de 1 hora (índice atualiza 1×/dia)


_okx_key  = os.getenv("OKX_API_KEY",    os.getenv("API_KEY", ""))
_okx_sec  = os.getenv("OKX_SECRET_KEY", os.getenv("SECRET_KEY", ""))
_okx_pass = os.getenv("OKX_PASSPHRASE", "")

client = OKXClient(api_key=_okx_key, secret_key=_okx_sec, passphrase=_okx_pass)

# OKXTradingClient: usado apenas para ler precisão real de instrumentos (tickSz/lotSz/minSz).
# Nenhuma ordem real é enviada — o bot continua em paper trading.
_trading_client = None
if _okx_key and _okx_sec and _okx_pass:
    try:
        _trading_client = OKXTradingClient(_okx_key, _okx_sec, _okx_pass)
        logger.info("[STARTUP] OKXTradingClient iniciado — precisão real de instrumentos ativa")
    except Exception as _e:
        logger.warning(f"[STARTUP] OKXTradingClient não iniciado: {_e}")

# Converte capital inicial de BRL para USD na taxa atual de mercado.
_startup_usd_brl = _fetch_usd_brl()
_initial_usd     = round(TOTAL_BRL_INITIAL / _startup_usd_brl, 2)
engine = SimulatedExecutionEngine(
    initial_balance_usd=_initial_usd,
    default_order_mode="adaptive",
    default_spread_pct=0.0002,
    trading_client=_trading_client,   # injeta precisão real de instrumentos OKX
)

# Cache de spreads reais por par (bid/ask do último ticker)
_last_spreads: dict[str, float] = {p: 0.0002 for p in ["BTC-USD", "ETH-USD", "SOL-USD"]}

# ── V4 Orchestrator — pipeline probabilística completa ────────────
v4 = V4Orchestrator(state_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

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
    for p in PAIRS:
        slots[f"V4:{p}"]     = _empty_slot()
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

last_signals: dict = {}

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
        "total_brl":           round(engine.portfolio_value() * _startup_usd_brl, 2),
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
    "usd_brl":          _startup_usd_brl,
    "trade_amount_brl": 0.0,
    "strategy_pnl":     {},
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
    "exec_stats":       {},
    "pending_orders":   [],
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

    # ── Converte capital inicial de BRL → USD na cotação atual ────────
    # Motor opera sempre em USD; dashboard exibe BRL apenas na UI.
    total_usd = TOTAL_BRL_INITIAL / usd_brl

    # ── Reinicia engine 100% em caixa — sem posições pré-carregadas ──
    engine.initial_balance = total_usd
    engine.balance_usd     = total_usd
    engine.holdings        = {}
    engine.entry_prices    = {}
    engine.trades          = []
    engine.total_fees_usd  = 0.0
    engine.prices          = {p.split("-")[0]: prices[p] for p in PAIRS}
    engine._save_state()

    # ── Todos os slots zerados — sem posições artificiais ────────
    for pair in PAIRS:
        strategy_slots[f"V4:{pair}"]     = _empty_slot()
        strategy_slots[f"manual:{pair}"] = _empty_slot()
    _save_slots(strategy_slots)
    state["slots"] = strategy_slots

    state["strategy_pnl"] = {}

    # ── Reinicia histórico e feed ─────────────────────────────────
    state["history"] = []
    state["trades"]  = []
    state["feed"]    = []
    _save_history(state["history"])

    # ── Reinicia cooldowns, sinais e contadores ───────────────────
    last_signals.clear()
    _daily_trade_count.clear()
    last_buy_time.clear()

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
    logger.info("V4 Motor Probabilístico — ciclo %ds", CYCLE_INTERVAL)
    loop = asyncio.get_event_loop()
    while True:
        state["cycle"] = _current_cycle()
        now_str = datetime.now().strftime("%H:%M:%S")

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

        # ── Market Breadth — circuit breaker para compras em momentos críticos ─
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

        # ── V4 Risk Engine — atualiza estado global antes do loop de pares ──
        try:
            _v4_risk = await asyncio.wait_for(
                loop.run_in_executor(None, v4.update_risk_state, engine),
                timeout=15.0
            )
            state["v4_risk"] = {
                "action":      _v4_risk.get("action", "normal"),
                "sizing_mult": _v4_risk.get("sizing_mult", 1.0),
                "alerts":      _v4_risk.get("alerts", []),
                "var_pct":     _v4_risk.get("var_result", {}).get("var_pct", 0),
                "ruin_prob":   _v4_risk.get("mc_result", {}).get("ruin_probability", 0),
                "sharpe":      _v4_risk.get("mc_result", {}).get("sharpe_rolling", 0),
                "winrate_7d":  _v4_risk.get("mc_result", {}).get("winrate_7d", 0.5),
                "dd_alert":    _v4_risk.get("dd_result", {}).get("alert", False),
                "meta":        _v4_risk.get("meta_summary", {}),
            }
        except Exception as _re:
            logger.warning(f"[V4 Risk] Erro: {_re}")
            state.setdefault("v4_risk", {"action": "normal", "sizing_mult": 1.0, "alerts": []})

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

                # Spread real bid/ask — alimenta SimulatedExecutionEngine
                _bid = float(ticker.get("bid", 0) or 0)
                _ask = float(ticker.get("ask", 0) or 0)
                if _bid > 0 and _ask > 0 and _ask > _bid:
                    _last_spreads[pair] = (_ask - _bid) / price
                else:
                    _last_spreads[pair] = 0.0002   # fallback 2 bps

                state["prices"][pair] = {
                    "price":          price,
                    "price_pct_chg":  float(ticker.get("price_percentage_change_24h", 0)),
                    "volume_24h":     float(ticker.get("volume_24h", 0)),
                    "spread_pct":     round(_last_spreads[pair] * 100, 4),
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

                # ── V4 Pipeline: avaliação probabilística por par ─────────────
                _closes_map = {
                    p: [float(c["close"]) for c in _candle_cache.get(f"{p}:{CANDLE_1H}", {}).get("data", [])]
                    for p in PAIRS
                }
                _v4_slot = strategy_slots.get(f"V4:{pair}", _empty_slot())
                _v4_open_slots = {
                    p: strategy_slots.get(f"V4:{p}", _empty_slot())
                    for p in PAIRS
                }
                _breadth_score = _breadth.score if _breadth else 1.0
                try:
                    import functools as _ft
                    _v4_decision = await asyncio.wait_for(
                        loop.run_in_executor(None, _ft.partial(
                            v4.evaluate,
                            pair, candles_1h, candles_6h, _closes_map,
                            engine, _v4_open_slots,
                            _v4_slot if _v4_slot.get("qty", 0) > 0 else None,
                            _breadth_score,
                        )),
                        timeout=12.0
                    )
                except Exception as _v4_err:
                    logger.warning(f"[V4][{pair}] Erro na pipeline: {_v4_err}")
                    _v4_decision = {"decision": "HOLD", "score": 0.5, "reason": str(_v4_err)}

                # Armazena decisão V4 no state para o dashboard
                if "v4" not in state:
                    state["v4"] = {}
                state["v4"][pair] = {
                    "decision":  _v4_decision.get("decision", "HOLD"),
                    "score":     _v4_decision.get("score", 0.5),
                    "regime":    _v4_decision.get("regime").regime if hasattr(_v4_decision.get("regime"), "regime") else "UNKNOWN",
                    "direction": _v4_decision.get("direction", "neutral"),
                    "ev":        _v4_decision.get("expected_value", 0.0),
                    "size_pct":  _v4_decision.get("size_pct", 0.0),
                    "reason":    _v4_decision.get("reason", ""),
                    "factors":   _v4_decision.get("signal", {}).get("factors", {}),
                    "execution_mode": _v4_decision.get("execution", None) and _v4_decision["execution"].mode,
                }
                logger.info(f"[V4][{pair}] {_v4_decision.get('decision')} | score={_v4_decision.get('score', 0):.3f} | {_v4_decision.get('reason', '')}")

                # ── Feed: registra sinal V4 de cada ciclo ────────────────────
                _v4_dec   = _v4_decision.get("decision", "HOLD")
                _v4_score = _v4_decision.get("score", 0.5)
                _v4_regime = state["v4"][pair].get("regime", "—")
                _v4_ev    = _v4_decision.get("expected_value", 0.0) or 0.0
                _v4_reason = _v4_decision.get("reason", "")
                _sig_key  = f"V4:{pair}:signal"
                _prev_dec = last_signals.get(_sig_key)
                if _v4_dec != _prev_dec:   # só insere quando decisão muda
                    last_signals[_sig_key] = _v4_dec
                    state["feed"].insert(0, {
                        "time":     now_str,
                        "cycle":    state["cycle"],
                        "pair":     pair,
                        "strategy": "V4:signal",
                        "signal":   _v4_dec,
                        "price":    price,
                        "executed": False,
                        "note":     f"score={_v4_score:.0%} ev={_v4_ev:+.4f} | {_v4_regime} | {_v4_reason[:40]}",
                    })
                    state["feed"] = state["feed"][:100]

                # ── Parâmetros de execução realista (compartilhados BUY/SELL) ──
                _v4_atr_pct = (state.get("v4", {}).get(pair, {})
                               .get("signal", {}).get("factors", {})
                               .get("atr_pct", 0.015))
                _v4_spread  = _last_spreads.get(pair, 0.0002)

                # ── V4 BUY execution ──────────────────────────────────────────
                _today = datetime.now().strftime("%Y-%m-%d")
                _v4_key = f"V4:{pair}"
                _breadth_score = (_breadth.score if _breadth else 1.0)
                _breadth_block = _breadth_score < 0.40   # DANGER — bloqueia compras
                if _breadth_block:
                    logger.info(f"[MarketBreadth] BLOQUEIO de compra — score={_breadth_score:.2f} ({_breadth.label if _breadth else 'N/A'})")
                if (
                    _v4_decision.get("decision") == "BUY"
                    and not _breadth_block
                    and _v4_slot.get("qty", 0) == 0
                    and _daily_trade_count.get(_today, 0) < MAX_DAILY_TRADES
                    and sum(1 for s in strategy_slots.values() if s.get("qty", 0) > 0) < MAX_OPEN_SLOTS
                    and time.time() - last_buy_time.get(_v4_key, 0) > BUY_COOLDOWN_SECONDS
                ):
                    _v4_size_usd = _v4_decision.get("size_usd", 0)
                    _v4_sl = _v4_decision.get("execution") and _v4_decision["execution"].stop_loss or 0
                    if _v4_size_usd > 10 and price > 0:
                        _v4_qty = _v4_size_usd / price
                        _v4_atr_pct = (state.get("v4", {}).get(pair, {})
                                       .get("signal", {}).get("factors", {})
                                       .get("atr_pct", 0.015))
                        if engine.buy(
                            symbol, _v4_size_usd, price, "V4:signal",
                            atr_pct=_v4_atr_pct, spread_pct=_v4_spread,
                            score=_v4_score, regime=_v4_regime,
                        ):
                            _sl_pct = abs(price - _v4_sl) / price if _v4_sl and price else 0.03
                            strategy_slots[_v4_key] = {
                                "qty":       _v4_qty,
                                "entry":     price,
                                "peak":      price,
                                "entry_usd": _v4_size_usd,
                                "sl_pct":    _sl_pct,
                                "sl_level":  _v4_sl or price * (1 - _sl_pct),
                                "be_sl":     0.0,
                                "realized":  0.0,
                                "unrealized":0.0,
                            }
                            last_buy_time[_v4_key] = time.time()
                            _daily_trade_count[_today] = _daily_trade_count.get(_today, 0) + 1
                            _record_trade("BUY", pair, _v4_qty, price, _v4_size_usd, "V4:signal")
                            logger.info(f"[V4][{pair}] ✅ BUY ${_v4_size_usd:.2f} @ ${price:,.2f} | sl={_v4_sl:.2f}")

                # ── V4 SELL execution ─────────────────────────────────────────
                elif _v4_decision.get("decision") == "SELL" and _v4_slot.get("qty", 0) > 0:
                    _v4_qty = _v4_slot["qty"]
                    _v4_sell_usd = _v4_qty * price
                    _v4_entry = _v4_slot.get("entry", price)
                    _v4_pnl   = (_v4_sell_usd - _v4_slot.get("entry_usd", _v4_sell_usd)) * (1 - TAKER_FEE)
                    _exit_type  = _v4_decision.get("exit_type", "signal")
                    _exit_tier  = _v4_decision.get("thesis_tier", "")
                    _exit_label = f"V4:{_exit_type}" + (f":{_exit_tier}" if _exit_tier else "")
                    if engine.sell(
                        symbol, _v4_qty, price, _exit_label,
                        atr_pct=_v4_atr_pct, spread_pct=_v4_spread,
                        score=_v4_score, regime=_v4_regime,
                    ):
                        strategy_slots[_v4_key] = _empty_slot()
                        _record_trade("SELL", pair, _v4_qty, price, _v4_sell_usd, _exit_label)
                        logger.info(f"[V4][{pair}] ✅ SELL ${_v4_sell_usd:.2f} @ ${price:,.2f} "
                                    f"| pnl={_v4_pnl:.2f} | {_exit_label} | {_v4_decision.get('reason','')}")

                pair_signals = {}
                pair_score   = state.get("v4", {}).get(pair, {}).get("score", 0.0)

                today_key        = datetime.now().strftime("%Y-%m-%d")
                daily_trades     = _daily_trade_count.get(today_key, 0)
                open_slots_count = sum(1 for s in strategy_slots.values() if s.get("qty", 0) > 0)
                state["trades_today"]     = daily_trades
                state["max_daily_trades"] = MAX_DAILY_TRADES
                state["open_slots_count"] = open_slots_count
                state["max_open_slots"]   = MAX_OPEN_SLOTS

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
                held_in_engine  = engine.holdings.get(symbol, 0)
                v4_slot_qty     = strategy_slots.get(f"V4:{pair}", {}).get("qty", 0)
                manual_slot_qty = strategy_slots.get(f"manual:{pair}", {}).get("qty", 0)
                slots_total_qty = v4_slot_qty + manual_slot_qty

                if held_in_engine < slots_total_qty - 1e-6 and slots_total_qty > 1e-10:
                    logger.warning(f"[{pair}] INCONSISTÊNCIA: engine={held_in_engine:.6f} slots={slots_total_qty:.6f} — corrigindo")
                    ratio = held_in_engine / slots_total_qty
                    for sk in [f"V4:{pair}", f"manual:{pair}"]:
                        if strategy_slots.get(sk, {}).get("qty", 0) > 0:
                            strategy_slots[sk]["qty"] *= ratio
                            if strategy_slots[sk]["qty"] < 1e-8:
                                strategy_slots[sk].update({"qty": 0.0, "entry": 0.0, "peak": 0.0, "be_sl": 0.0})

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
                    "strategies":   pair_signals,
                    "rsi":          rsi_val,
                    "entry_price":  round(entry_price, 2) if entry_price else None,
                    "change_pct":   round(change_pct,  2) if change_pct is not None else None,
                    "atr_sl_level": _atr_sl_level,
                    "atr_sl_pct":   _atr_sl_pct,
                    "score":        round(pair_score, 3),
                    "regime":       state.get("v4", {}).get(pair, {}).get("regime", "UNKNOWN"),
                }
                log_cycle(logger, state["cycle"], pair, price, pair_signals, "")

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
        state["pending_limit_orders"] = {}

        # Expõe timestamp da próxima hora cheia SP para o countdown do frontend
        _next_sp = _seconds_to_next_sp_hour()
        state["next_cycle_ts"]    = int(time.time() + _next_sp)
        state["cycle_interval"]   = CYCLE_INTERVAL   # mantido para compatibilidade JS

        # Processa ordens limit pendentes contra preços atuais
        if hasattr(engine, "tick"):
            _tick_prices = {p: state["prices"][p]["price"] for p in PAIRS if state["prices"].get(p, {}).get("price")}
            _tick_events = engine.tick(_tick_prices)
            if _tick_events:
                logger.info(f"[SIM] tick: {len(_tick_events)} evento(s) — {_tick_events}")
            state["exec_stats"]   = engine.execution_stats() if hasattr(engine, "execution_stats") else {}
            state["pending_orders"] = engine.pending_summary() if hasattr(engine, "pending_summary") else []

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
