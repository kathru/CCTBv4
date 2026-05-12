import os
import json
import logging
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_log_path():
    return os.path.join(LOG_DIR, f"trading_{datetime.now().strftime('%Y-%m-%d')}.log")


def setup_logger(name="trading_bot"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — rotates por dia
    fh = logging.FileHandler(get_log_path(), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_cycle(logger, cycle: int, pair: str, price: float, signals: dict, decision: str):
    parts = " | ".join(f"{k}={v}" for k, v in signals.items() if k not in ("Trend", "Vol Guard"))
    logger.info(f"CYCLE={cycle} | {pair} | price=${price:,.2f} | {parts} | DECISION={decision}")


def log_trade(logger, side: str, pair: str, qty: float, price: float, usd: float, strategy: str):
    logger.info(
        f"TRADE | {side} | {pair} | qty={qty:.6f} | price=${price:,.2f} | "
        f"usd=${usd:.2f} | strategy={strategy}"
    )


def log_portfolio(logger, usd: float, total: float, pnl: float, pnl_pct: float, holdings: dict):
    holdings_str = " | ".join(f"{k}={v:.6f}" for k, v in holdings.items()) or "none"
    logger.info(
        f"PORTFOLIO | usd=${usd:.2f} | total=${total:.2f} | "
        f"pnl=${pnl:+.2f} ({pnl_pct:+.2f}%) | holdings=[{holdings_str}]"
    )
