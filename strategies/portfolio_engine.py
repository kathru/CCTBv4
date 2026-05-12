"""
Portfolio Engine — Portfolio Layer
===================================
O bot pensa no portfolio inteiro antes de qualquer entrada.

Antes de abrir SOL, pergunta:
  1. Qual exposição atual ao beta cripto?
  2. Qual correlação atual entre pares abertos?
  3. Qual impacto no VaR?
  4. Qual concentração de volatilidade?

Risk Parity: limita risco agregado ajustado à correlação,
não número de posições.

BTC + ETH com correlação 0.92 = 1 posição efetiva duplicada.
"""

import math
from typing import Optional


# Beta de cada ativo vs mercado crypto geral
ASSET_BETA = {
    "BTC-USD": 1.0,
    "ETH-USD": 1.15,
    "SOL-USD": 1.45,
    "AVAX-USD": 1.50,
    "LINK-USD": 1.40,
}

# VaR limite do portfolio (perda máxima tolerada como % do portfolio)
PORTFOLIO_VAR_LIMIT = 0.05  # 5%


def calc_portfolio_beta(open_slots: dict, portfolio_value: float) -> float:
    """
    Beta ponderado do portfolio vs mercado crypto.

    Retorna o beta agregado (> 1 = mais agressivo que o mercado).
    """
    if not open_slots or portfolio_value <= 0:
        return 0.0

    total_exposure = 0.0
    weighted_beta  = 0.0

    for pair, slot in open_slots.items():
        qty   = slot.get("qty", 0)
        price = slot.get("current_price", slot.get("entry_price", 0))
        value = qty * price

        if value <= 0:
            continue

        beta = ASSET_BETA.get(pair, 1.2)
        weighted_beta  += beta * value
        total_exposure += value

    if total_exposure == 0:
        return 0.0

    return round(weighted_beta / total_exposure, 3)


def calc_vol_concentration(open_slots: dict, market_contexts: dict) -> dict:
    """
    Concentração de volatilidade por ativo no portfolio.

    Retorna % da volatilidade total que vem de cada ativo.
    Alta concentração em um único ativo = risco não diversificado.
    """
    vol_contributions = {}
    total_vol = 0.0

    for pair, slot in open_slots.items():
        qty   = slot.get("qty", 0)
        price = slot.get("current_price", slot.get("entry_price", 0))
        value = qty * price

        ctx  = market_contexts.get(pair, {})
        rvol = ctx.get("realized_vol", 0.30)

        vol_contrib = value * rvol
        vol_contributions[pair] = vol_contrib
        total_vol += vol_contrib

    if total_vol == 0:
        return {}

    return {
        pair: round(v / total_vol, 4)
        for pair, v in vol_contributions.items()
    }


def evaluate_new_entry(
    new_pair:          str,
    new_size_pct:      float,
    open_slots:        dict,
    correlation_matrix: dict,
    market_contexts:   dict,
    portfolio_value:   float,
    var_result:        dict,
) -> dict:
    """
    Avalia o impacto de uma nova posição no portfolio inteiro.

    Perguntas respondidas:
      1. Beta: portfolio ficará mais agressivo que o tolerado?
      2. Correlação: nova posição adiciona diversificação real?
      3. VaR: impacto no risco total é aceitável?
      4. Concentração: volatilidade ficará concentrada demais?

    Retorna:
      approved      — True/False
      size_adjusted — tamanho ajustado ao risco real
      reason        — explicação
      metrics       — métricas calculadas
    """
    new_size_usd = new_size_pct * portfolio_value

    # ── 1. Beta do portfolio atual ────────────────────────────────────────────
    current_beta = calc_portfolio_beta(open_slots, portfolio_value)
    new_beta     = ASSET_BETA.get(new_pair, 1.2)

    # Beta ponderado pós-entrada
    current_exposure = sum(
        slot.get("qty", 0) * slot.get("current_price", slot.get("entry_price", 0))
        for slot in open_slots.values()
    )
    total_exposure_after = current_exposure + new_size_usd
    beta_after = (current_beta * current_exposure + new_beta * new_size_usd) / total_exposure_after if total_exposure_after > 0 else new_beta

    beta_penalty = 0.0
    if beta_after > 1.30:
        beta_penalty = (beta_after - 1.30) * 0.5  # penaliza size

    # ── 2. Correlação com posições abertas ────────────────────────────────────
    avg_corr_with_open = 0.0
    n_open = len(open_slots)

    if n_open > 0:
        corr_sum = 0.0
        for open_pair in open_slots.keys():
            key = f"{new_pair}_{open_pair}" if f"{new_pair}_{open_pair}" in correlation_matrix else f"{open_pair}_{new_pair}"
            corr = correlation_matrix.get(key, 0.70)  # assume 0.70 se não disponível
            corr_sum += corr
        avg_corr_with_open = corr_sum / n_open

    # Penalidade por correlação alta
    corr_penalty = 0.0
    if avg_corr_with_open > 0.85:
        corr_penalty = (avg_corr_with_open - 0.85) * 2.0

    # ── 3. VaR impact ─────────────────────────────────────────────────────────
    var_current_pct = var_result.get("var_pct", 0.02)

    # VaR marginal da nova posição (simplificado)
    new_ctx     = market_contexts.get(new_pair, {})
    new_rvol    = new_ctx.get("realized_vol", 0.30)
    var_marginal = new_size_pct * new_rvol * 0.10  # VaR estimado da nova posição

    var_after = var_current_pct + var_marginal * (1 + avg_corr_with_open)
    var_headroom = max(0, (PORTFOLIO_VAR_LIMIT - var_after) / PORTFOLIO_VAR_LIMIT)

    # ── 4. Concentração de volatilidade ───────────────────────────────────────
    vol_conc = calc_vol_concentration(open_slots, market_contexts)
    max_conc  = max(vol_conc.values()) if vol_conc else 0.0

    # ── Decisão final ─────────────────────────────────────────────────────────
    size_penalty = beta_penalty + corr_penalty
    size_adjusted_pct = max(0.02, new_size_pct * (1 - size_penalty))

    blocked = False
    reasons = []

    if var_after > PORTFOLIO_VAR_LIMIT:
        blocked = True
        reasons.append(f"VaR pós-entrada {var_after:.2%} excede limite {PORTFOLIO_VAR_LIMIT:.2%}")

    if avg_corr_with_open > 0.92:
        blocked = True
        reasons.append(f"Correlação média com abertos {avg_corr_with_open:.2f} — sem diversificação real")

    if max_conc > 0.60:
        reasons.append(f"Concentração de vol em {max(vol_conc, key=vol_conc.get)}: {max_conc:.0%}")
        size_adjusted_pct *= 0.70

    # Clamp
    size_adjusted_pct = max(0.02, min(new_size_pct, size_adjusted_pct))

    return {
        "approved":        not blocked,
        "size_original":   round(new_size_pct, 4),
        "size_adjusted":   round(size_adjusted_pct, 4),
        "size_usd":        round(size_adjusted_pct * portfolio_value, 2),
        "reason":          "; ".join(reasons) if reasons else "OK",
        "metrics": {
            "portfolio_beta":        round(current_beta, 3),
            "beta_after_entry":      round(beta_after, 3),
            "avg_corr_with_open":    round(avg_corr_with_open, 3),
            "var_current_pct":       round(var_current_pct, 4),
            "var_after_entry":       round(var_after, 4),
            "var_headroom":          round(var_headroom, 3),
            "vol_concentration":     vol_conc,
            "effective_positions":   correlation_matrix.get("effective_positions", n_open),
        },
    }
