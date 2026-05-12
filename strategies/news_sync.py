"""
News Calendar Sync
==================
Sincroniza o calendário de eventos econômicos com APIs externas gratuitas
e salva em data/news_events.json para o NewsVolatilityGuard usar.

APIs suportadas (configure via .env):
  FINNHUB_TOKEN  → https://finnhub.io  (free: 60 req/min)
  EODHD_TOKEN    → https://eodhd.com   (free: 20 req/dia)

Se nenhuma API estiver configurada, usa apenas o calendário embutido
no news_guard.py (eventos FOMC/CPI/NFP/PPI 2026 hardcoded).

Como usar:
  python strategies/news_sync.py          # executa uma vez
  — ou —
  importar e chamar sync_news_events()    # integrado ao bot
"""

import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("news_sync")

# ── Configuração ───────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
NEWS_EVENTS_FILE = os.path.join(DATA_DIR, "news_events.json")

# Keywords que sozinhos (sem precisar de impact=high) já são críticos
CRITICAL_KEYWORDS = [
    "FOMC", "Federal Reserve", "Fed Rate", "Interest Rate Decision",
    "Nonfarm Payroll", "Non-Farm Payroll", "NFP",
    "Bitcoin", "Crypto", "SEC", "ETF",
]

# Keywords que só bloqueiam se impact=high (eventos macro importantes mas não críticos sozinhos)
HIGH_IMPACT_KEYWORDS = [
    "CPI", "Consumer Price Index",
    "PPI", "Producer Price Index",
    "PCE Price Index",
    "GDP Growth Rate",
    "Retail Sales MoM",
    "Core Inflation Rate",
    "Inflation Rate MoM", "Inflation Rate YoY",
]

# Países a monitorar
COUNTRIES_WATCH = {"US", "USD", "United States", "EU", "EUR", "Europe"}

# Janelas por tipo de evento
EVENT_WINDOWS = {
    "FOMC":     (45, 90),
    "FED":      (45, 90),
    "CPI":      (30, 60),
    "NFP":      (30, 60),
    "PAYROLL":  (30, 60),
    "PPI":      (20, 40),
    "GDP":      (20, 40),
    "PCE":      (20, 40),
    "BITCOIN":  (30, 60),
    "CRYPTO":   (30, 60),
    "ETF":      (30, 60),
    "DEFAULT":  (25, 50),
}


def _get_window(event_name: str):
    name_upper = event_name.upper()
    for key, window in EVENT_WINDOWS.items():
        if key in name_upper:
            return window
    return EVENT_WINDOWS["DEFAULT"]


def _is_high_impact(event_name: str, impact: str = "") -> bool:
    """
    Verifica se o evento deve bloquear operações.
    Critério duplo:
      1. Evento CRÍTICO por nome (FOMC, NFP, Bitcoin ETF…) → sempre bloqueia
      2. Evento HIGH_IMPACT por nome + impact=high da API → bloqueia
    Evita bloquear eventos medium/low que aparecem com nomes macro genéricos.
    """
    name_upper = event_name.upper()
    # Críticos por nome — sempre bloqueiam independente do impact
    if any(kw.upper() in name_upper for kw in CRITICAL_KEYWORDS):
        return True
    # Outros macro: só bloqueia se a API também classifica como high
    if impact.lower() == "high":
        if any(kw.upper() in name_upper for kw in HIGH_IMPACT_KEYWORDS):
            return True
    return False


# ── Finnhub ────────────────────────────────────────────────────────────────
def fetch_finnhub(token: str, days_ahead: int = 30) -> List[dict]:
    """
    Busca eventos do calendário econômico via Finnhub.
    Free tier: 60 req/min — mais que suficiente.

    Registre em https://finnhub.io para obter token gratuito.
    Adicione ao .env: FINNHUB_TOKEN=seu_token
    """
    if not requests:
        logger.error("requests não instalado. pip install requests")
        return []

    now   = datetime.now(tz=timezone.utc)
    start = now.strftime("%Y-%m-%d")
    end   = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url   = f"https://finnhub.io/api/v1/calendar/economic?from={start}&to={end}&token={token}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events_raw = data.get("economicCalendar", [])
    except Exception as e:
        logger.error(f"[Finnhub] Erro ao buscar calendário: {e}")
        return []

    events = []
    for ev in events_raw:
        name    = ev.get("event", "")
        country = ev.get("country", "")
        impact  = ev.get("impact", "")
        time_s  = ev.get("time", "")  # "2026-06-18 14:00:00"

        # Filtrar apenas eventos de alto impacto de países relevantes
        if country not in COUNTRIES_WATCH and country not in {"", "US"}:
            continue
        if not _is_high_impact(name, impact):
            continue

        try:
            dt = datetime.strptime(time_s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        pre_min, post_min = _get_window(name)
        events.append({
            "date":     dt.strftime("%Y-%m-%d"),
            "time_utc": dt.strftime("%H:%M"),
            "name":     name,
            "pre_min":  pre_min,
            "post_min": post_min,
            "impact":   impact,
            "source":   "finnhub",
        })

    logger.info(f"[Finnhub] {len(events)} eventos de alto impacto nos próximos {days_ahead} dias")
    return events


# ── EODHD ──────────────────────────────────────────────────────────────────
def fetch_eodhd(token: str, days_ahead: int = 30) -> List[dict]:
    """
    Busca eventos via EODHD Economic Events API.
    Free tier: 20 req/dia.

    Registre em https://eodhd.com para obter token gratuito.
    Adicione ao .env: EODHD_TOKEN=seu_token
    """
    if not requests:
        return []

    now   = datetime.now(tz=timezone.utc)
    start = now.strftime("%Y-%m-%d")
    end   = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    url   = (f"https://eodhd.com/api/economic-events"
             f"?api_token={token}&from={start}&to={end}&fmt=json")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        events_raw = resp.json()
    except Exception as e:
        logger.error(f"[EODHD] Erro ao buscar calendário: {e}")
        return []

    events = []
    for ev in events_raw:
        name   = ev.get("type", ev.get("event", ""))
        time_s = ev.get("date", "")  # "2026-06-18 13:30:00"

        if not _is_high_impact(name):
            continue

        try:
            dt = datetime.strptime(time_s[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        pre_min, post_min = _get_window(name)
        events.append({
            "date":     dt.strftime("%Y-%m-%d"),
            "time_utc": dt.strftime("%H:%M"),
            "name":     name,
            "pre_min":  pre_min,
            "post_min": post_min,
            "source":   "eodhd",
        })

    logger.info(f"[EODHD] {len(events)} eventos de alto impacto nos próximos {days_ahead} dias")
    return events


# ── Sync principal ──────────────────────────────────────────────────────────
def sync_news_events(days_ahead: int = 14) -> int:
    """
    Sincroniza o calendário de eventos com APIs externas.
    Salva em data/news_events.json.

    Tenta Finnhub primeiro, depois EODHD como fallback.
    Retorna o número de eventos salvos.
    """
    finnhub_token = os.getenv("FINNHUB_TOKEN", "")
    eodhd_token   = os.getenv("EODHD_TOKEN", "")

    events = []

    # Tenta Finnhub
    if finnhub_token:
        events = fetch_finnhub(finnhub_token, days_ahead)
        if events:
            logger.info(f"[NewsSync] {len(events)} eventos via Finnhub")

    # Fallback para EODHD se Finnhub não disponível
    if not events and eodhd_token:
        events = fetch_eodhd(eodhd_token, days_ahead)
        if events:
            logger.info(f"[NewsSync] {len(events)} eventos via EODHD")

    if not events:
        if not finnhub_token and not eodhd_token:
            logger.warning("[NewsSync] Nenhuma API configurada. "
                           "Adicione FINNHUB_TOKEN ou EODHD_TOKEN ao .env. "
                           "Usando calendário embutido (2026 hardcoded).")
        else:
            logger.warning("[NewsSync] APIs retornaram 0 eventos. Usando calendário embutido.")
        return 0

    # Remover duplicatas por (date + time_utc + name)
    seen = set()
    unique = []
    for ev in events:
        key = (ev["date"], ev["time_utc"], ev["name"])
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    # Salvar JSON
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(NEWS_EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    logger.info(f"[NewsSync] {len(unique)} eventos salvos em news_events.json")
    return len(unique)


# ── Agendamento automático ──────────────────────────────────────────────────
_last_sync_ts: float = 0.0
SYNC_INTERVAL_HOURS: int = 12  # sincroniza a cada 12h


def sync_if_needed() -> None:
    """Chama sync_news_events() se já passou SYNC_INTERVAL_HOURS desde a última sync."""
    global _last_sync_ts
    if time.time() - _last_sync_ts > SYNC_INTERVAL_HOURS * 3600:
        try:
            count = sync_news_events()
            _last_sync_ts = time.time()
            if count > 0:
                logger.info(f"[NewsSync] Auto-sync concluído: {count} eventos.")
        except Exception as e:
            logger.error(f"[NewsSync] Erro no auto-sync: {e}")


# ── Execução direta ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Carrega .env manual se executado diretamente
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code.env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    n = sync_news_events(days_ahead=30)
    if n > 0:
        print(f"\n✅ {n} eventos salvos em data/news_events.json")
        with open(NEWS_EVENTS_FILE) as f:
            events = json.load(f)
        for ev in events[:10]:
            print(f"  {ev['date']} {ev['time_utc']} UTC — {ev['name']} "
                  f"(pre={ev['pre_min']}min post={ev['post_min']}min)")
        if len(events) > 10:
            print(f"  ... e mais {len(events)-10} eventos")
    else:
        print("\n⚠️  Nenhum evento sincronizado via API.")
        print("Configure FINNHUB_TOKEN ou EODHD_TOKEN no .env e execute novamente.")
        print("Registre grátis em:")
        print("  Finnhub: https://finnhub.io  (recomendado — 60 req/min)")
        print("  EODHD:   https://eodhd.com   (alternativa — 20 req/dia)")
