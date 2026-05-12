"""
News Volatility Guard
=====================
Bloqueia entradas em janelas de alta volatilidade ao redor de eventos econômicos
e de crypto programados.

Janela padrão: 30 min ANTES e 60 min DEPOIS do evento (total: 90 min).

Eventos monitorados:
  - FOMC (Federal Reserve — decisão de juros)
  - CPI  (Índice de Preços ao Consumidor — EUA)
  - NFP  (Non-Farm Payroll — empregos EUA)
  - PPI  (Índice de Preços ao Produtor)
  - Eventos custom via data/news_events.json

Formato do JSON custom:
  [
    {"date": "2026-06-01", "time_utc": "18:00", "name": "FOMC", "pre_min": 30, "post_min": 60},
    {"date": "2026-06-15", "time_utc": "08:30", "name": "Bitcoin ETF", "pre_min": 60, "post_min": 120}
  ]
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional

# ── Calendário econômico 2026 (horários em UTC) ────────────────────────────
# Fonte: Federal Reserve, BLS (Bureau of Labor Statistics)
# Formato: (ano, mês, dia, hora_utc, min_utc, nome_evento)
#
# ET (Eastern Time):
#   Inverno (Nov–Mar): UTC-5  → 08:30 ET = 13:30 UTC
#   Verão  (Mar–Nov): UTC-4  → 08:30 ET = 12:30 UTC
#   FOMC:  14:00 ET  → Inverno: 19:00 UTC | Verão: 18:00 UTC

ECONOMIC_CALENDAR_2026: List[Tuple] = [
    # ── FOMC (Federal Open Market Committee) — 8 reuniões em 2026 ────────
    (2026,  1, 29, 19,  0, "FOMC"),   # Jan — inverno (UTC-5)
    (2026,  3, 19, 18,  0, "FOMC"),   # Mar — verão (UTC-4)
    (2026,  5,  7, 18,  0, "FOMC"),   # Mai
    (2026,  6, 18, 18,  0, "FOMC"),   # Jun
    (2026,  7, 30, 18,  0, "FOMC"),   # Jul
    (2026,  9, 17, 18,  0, "FOMC"),   # Set
    (2026, 11,  5, 19,  0, "FOMC"),   # Nov — inverno (UTC-5)
    (2026, 12, 16, 19,  0, "FOMC"),   # Dez

    # ── CPI (Consumer Price Index) — mensal ───────────────────────────────
    (2026,  1, 15, 13, 30, "CPI"),
    (2026,  2, 12, 13, 30, "CPI"),
    (2026,  3, 12, 12, 30, "CPI"),    # verão
    (2026,  4, 10, 12, 30, "CPI"),
    (2026,  5, 13, 12, 30, "CPI"),
    (2026,  6, 11, 12, 30, "CPI"),
    (2026,  7, 10, 12, 30, "CPI"),
    (2026,  8, 13, 12, 30, "CPI"),
    (2026,  9, 11, 12, 30, "CPI"),
    (2026, 10, 15, 12, 30, "CPI"),
    (2026, 11, 13, 13, 30, "CPI"),    # inverno
    (2026, 12, 11, 13, 30, "CPI"),

    # ── NFP (Non-Farm Payroll) — primeira sexta do mês ───────────────────
    (2026,  1,  9, 13, 30, "NFP"),
    (2026,  2,  6, 13, 30, "NFP"),
    (2026,  3,  6, 13, 30, "NFP"),
    (2026,  4,  3, 12, 30, "NFP"),    # verão
    (2026,  5,  1, 12, 30, "NFP"),
    (2026,  6,  5, 12, 30, "NFP"),
    (2026,  7, 10, 12, 30, "NFP"),    # 4 jul → shift
    (2026,  8,  7, 12, 30, "NFP"),
    (2026,  9,  4, 12, 30, "NFP"),
    (2026, 10,  2, 12, 30, "NFP"),
    (2026, 11,  6, 13, 30, "NFP"),    # inverno
    (2026, 12,  4, 13, 30, "NFP"),

    # ── PPI (Producer Price Index) — mensal ──────────────────────────────
    (2026,  1, 16, 13, 30, "PPI"),
    (2026,  2, 13, 13, 30, "PPI"),
    (2026,  3, 13, 12, 30, "PPI"),
    (2026,  4, 11, 12, 30, "PPI"),
    (2026,  5, 14, 12, 30, "PPI"),
    (2026,  6, 12, 12, 30, "PPI"),
    (2026,  7, 14, 12, 30, "PPI"),
    (2026,  8, 14, 12, 30, "PPI"),
    (2026,  9, 11, 12, 30, "PPI"),
    (2026, 10, 14, 12, 30, "PPI"),
    (2026, 11, 13, 13, 30, "PPI"),
    (2026, 12, 11, 13, 30, "PPI"),
]

# ── Defaults de janela de bloqueio ────────────────────────────────────────
DEFAULT_PRE_MINUTES  = 30   # bloqueia 30 min ANTES do evento
DEFAULT_POST_MINUTES = 60   # bloqueia 60 min DEPOIS do evento

# Eventos mais impactantes podem ter janelas maiores
EVENT_WINDOWS = {
    "FOMC": (45, 90),   # FOMC: 45 min antes, 90 min depois (muito volátil)
    "CPI":  (30, 60),
    "NFP":  (30, 60),
    "PPI":  (20, 40),
}


def _load_custom_events(json_path: str) -> List[dict]:
    """Carrega eventos custom do arquivo JSON editável."""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _event_to_utc(year: int, month: int, day: int,
                   hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def is_news_blackout(now_ts: Optional[float] = None,
                     custom_events_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verifica se o momento atual está dentro de uma janela de bloqueio por eventos.

    Retorna:
      (True, "motivo") → BLOQUEADO — não abrir posições
      (False, "")      → LIVRE — pode operar normalmente

    Args:
      now_ts: timestamp Unix (padrão: agora)
      custom_events_path: caminho para o JSON de eventos custom
    """
    if now_ts is None:
        now_ts = time.time()

    now = datetime.fromtimestamp(now_ts, tz=timezone.utc)

    # ── Verificar calendário embutido ─────────────────────────────────────
    for (yr, mo, dy, hr, mi, name) in ECONOMIC_CALENDAR_2026:
        pre_min, post_min = EVENT_WINDOWS.get(name, (DEFAULT_PRE_MINUTES, DEFAULT_POST_MINUTES))
        event_dt = _event_to_utc(yr, mo, dy, hr, mi)
        window_start = event_dt - timedelta(minutes=pre_min)
        window_end   = event_dt + timedelta(minutes=post_min)

        if window_start <= now <= window_end:
            if now < event_dt:
                mins_to = int((event_dt - now).total_seconds() / 60)
                return True, f"NewsGuard: {name} em {mins_to}min"
            else:
                mins_after = int((now - event_dt).total_seconds() / 60)
                return True, f"NewsGuard: {name} há {mins_after}min"

    # ── Verificar eventos custom (JSON editável) ───────────────────────────
    if custom_events_path:
        for ev in _load_custom_events(custom_events_path):
            try:
                date_str = ev.get("date", "")
                time_str = ev.get("time_utc", "00:00")
                name     = ev.get("name", "Evento")
                pre_min  = int(ev.get("pre_min",  DEFAULT_PRE_MINUTES))
                post_min = int(ev.get("post_min", DEFAULT_POST_MINUTES))

                event_dt = datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=timezone.utc)

                window_start = event_dt - timedelta(minutes=pre_min)
                window_end   = event_dt + timedelta(minutes=post_min)

                if window_start <= now <= window_end:
                    if now < event_dt:
                        mins_to = int((event_dt - now).total_seconds() / 60)
                        return True, f"NewsGuard: {name} em {mins_to}min"
                    else:
                        mins_after = int((now - event_dt).total_seconds() / 60)
                        return True, f"NewsGuard: {name} há {mins_after}min"
            except Exception:
                continue

    return False, ""


def next_event(now_ts: Optional[float] = None,
               custom_events_path: Optional[str] = None) -> Optional[dict]:
    """
    Retorna o próximo evento no calendário (dentro de 24h).
    Útil para exibir no dashboard.
    """
    if now_ts is None:
        now_ts = time.time()
    now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    horizon = now + timedelta(hours=24)

    upcoming = []
    for (yr, mo, dy, hr, mi, name) in ECONOMIC_CALENDAR_2026:
        event_dt = _event_to_utc(yr, mo, dy, hr, mi)
        if now <= event_dt <= horizon:
            upcoming.append({"name": name, "dt": event_dt,
                              "mins_to": int((event_dt - now).total_seconds() / 60)})

    if custom_events_path:
        for ev in _load_custom_events(custom_events_path):
            try:
                date_str = ev.get("date", "")
                time_str = ev.get("time_utc", "00:00")
                event_dt = datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=timezone.utc)
                if now <= event_dt <= horizon:
                    upcoming.append({"name": ev.get("name", "Evento"), "dt": event_dt,
                                     "mins_to": int((event_dt - now).total_seconds() / 60)})
            except Exception:
                continue

    if not upcoming:
        return None
    return min(upcoming, key=lambda x: x["dt"])
