"""
Risk Prior — Distribuição Histórica de Risco
=============================================
Usa 8 anos de dados OKX para construir priors robustos para o Risk Engine.

Problema original:
  Com 10–20 trades reais, VaR/ES/P(ruína)/Sharpe são estatisticamente inválidos.
  Um VaR baseado em 10 observações tem IC de confiança enorme.

Solução — Priors Bayesianos:
  Prior  (8 anos histórico OKX)   → distribuição de retornos robusta
  Likelihood (trades reais)       → atualização incremental
  Posterior = blend(prior, real)  → risco calibrado

Blending por n_real:
  n_real <  30:  90% prior + 10% real  (prior domina — histórico real insuficiente)
  n_real <  50:  70% prior + 30% real
  n_real < 100:  50% prior + 50% real  (transição)
  n_real < 200:  30% prior + 70% real
  n_real >= 200:  0% prior + 100% real (histórico real suficiente)

Geração do prior:
  Roda simulação sobre dados históricos OKX (mesmo extract_features + label_outcome
  do validate.py) e salva a distribuição de retornos em data/risk_prior.json.
  Regenerado automaticamente quando validate.py completa.
"""

import os
import json
import math
import random
import statistics
from typing import Optional


PRIOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "risk_prior.json"
)

# Parâmetros de blending por n_real
BLEND_SCHEDULE = [
    (0,   0.95),   # 0–29 trades: 95% prior
    (30,  0.70),   # 30–49 trades: 70% prior
    (50,  0.50),   # 50–99 trades: 50% prior
    (100, 0.30),   # 100–199 trades: 30% prior
    (200, 0.00),   # 200+ trades: 100% real
]


# ── Prior default (fallback se não houver dados históricos) ───────────────────
# Calibrado com 28.315 amostras reais OKX 8 anos via calibrate.py (2026-05-13).
# WR real = 28-32% (todos os buckets de score) — não 52% como estimativa anterior.
# Com RR efetivo 3.1× (TP 2/3/4.5× sizing 30/40/30): EV = 0.30×3.1 - 0.70 - 0.005 = +0.225
DEFAULT_PRIOR = {
    "returns":          [-0.020, -0.016, -0.013, -0.010, -0.008, -0.006, -0.004,
                         -0.002,  0.001,  0.002,  0.004,  0.006,  0.009,  0.012,
                          0.016,  0.022,  0.030,  0.040,  0.055,  0.075],
    "win_rate":         0.30,   # calibrado: 28-32% histórico (era 0.52 — estimativa errada)
    "profit_factor":    1.30,   # RR=3.1×: 0.30×3.1/0.70 ≈ 1.33
    "avg_win_pct":      0.046,  # RR efetivo 3.1× × SL 1.5% = 4.6% ganho médio
    "avg_loss_pct":    -0.015,  # SL típico 1.5% do portfolio
    "sharpe":           0.60,   # AUC=0.579 → sinal marginal
    "max_drawdown":    -0.25,
    "n_samples":        28315,  # amostras reais do calibrate.py
    "source":           "calibration_platt_okx_8y_28k",
}


# ── Carrega/salva prior ───────────────────────────────────────────────────────

def load_prior() -> dict:
    """Carrega prior do arquivo gerado pelo validate.py, ou usa default."""
    if os.path.exists(PRIOR_PATH):
        try:
            with open(PRIOR_PATH) as f:
                prior = json.load(f)
            if prior.get("n_samples", 0) >= 100:
                return prior
        except Exception:
            pass
    return DEFAULT_PRIOR


def save_prior(prior: dict):
    os.makedirs(os.path.dirname(PRIOR_PATH), exist_ok=True)
    with open(PRIOR_PATH, "w") as f:
        json.dump(prior, f, indent=2)


def build_prior_from_validation(validation_result_path: str) -> dict:
    """
    Constrói prior a partir do resultado do validate.py.
    Extrai a distribuição de retornos de todas as janelas walk-forward.
    """
    if not os.path.exists(validation_result_path):
        return DEFAULT_PRIOR

    with open(validation_result_path) as f:
        result = json.load(f)

    all_returns = []
    all_wins    = []
    all_losses  = []

    # Coleta retornos de todas as janelas
    for window in result.get("windows", []):
        n  = window.get("n_trades", 0)
        wr = window.get("win_rate", 0.5)
        pf = window.get("profit_factor", 1.0)
        sl = 0.015   # SL típico em % do portfolio por trade

        if n < 5: continue

        # Reconstrói distribuição de retornos da janela
        avg_win  = sl * min(pf, 3.0)           # ganho médio estimado
        avg_loss = -sl                           # perda média = SL%
        size_pct = 0.08                          # tamanho médio de posição

        n_wins   = round(n * wr)
        n_losses = n - n_wins

        # Wins com variância realista
        for _ in range(n_wins):
            r = random.gauss(avg_win, avg_win * 0.4) * size_pct
            all_returns.append(r)
            all_wins.append(r)

        # Losses com variância realista
        for _ in range(n_losses):
            r = random.gauss(avg_loss, abs(avg_loss) * 0.3) * size_pct
            all_returns.append(r)
            all_losses.append(r)

    if not all_returns:
        return DEFAULT_PRIOR

    all_returns.sort()
    avg_win_r  = sum(all_wins)   / len(all_wins)   if all_wins   else 0
    avg_loss_r = sum(all_losses) / len(all_losses) if all_losses else 0
    wr_overall = len(all_wins) / len(all_returns)
    pf_overall = abs(sum(all_wins)) / abs(sum(all_losses)) if all_losses else 1.0

    # Sharpe dos retornos
    if len(all_returns) > 2:
        m = sum(all_returns) / len(all_returns)
        s = statistics.stdev(all_returns)
        sharpe = (m / s * math.sqrt(252)) if s > 0 else 0
    else:
        sharpe = 0.8

    # Drawdown máximo simulado
    portfolio = 1.0; peak = 1.0; max_dd = 0.0
    for r in all_returns:
        portfolio *= (1 + r)
        peak = max(peak, portfolio)
        dd = (portfolio - peak) / peak
        max_dd = min(max_dd, dd)

    prior = {
        "returns":       all_returns[-2000:],  # guarda até 2k amostras
        "win_rate":      round(wr_overall, 4),
        "profit_factor": round(pf_overall, 3),
        "avg_win_pct":   round(avg_win_r, 5),
        "avg_loss_pct":  round(avg_loss_r, 5),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd, 4),
        "n_samples":     len(all_returns),
        "source":        "okx_walkforward_8y",
    }
    save_prior(prior)
    return prior


# ── Blending ──────────────────────────────────────────────────────────────────

def prior_weight(n_real: int) -> float:
    """Retorna o peso do prior (0–1) dado n_real trades reais."""
    for threshold, weight in reversed(BLEND_SCHEDULE):
        if n_real >= threshold:
            return weight
    return 0.95


def blend_returns(real_returns: list, prior: dict, portfolio_value: float) -> list:
    """
    Combina retornos reais com prior histórico via blending ponderado.
    Retorna lista de retornos como % do portfolio.
    """
    n_real  = len(real_returns)
    w_prior = prior_weight(n_real)
    w_real  = 1.0 - w_prior

    prior_returns = prior.get("returns", DEFAULT_PRIOR["returns"])

    if not real_returns:
        return random.choices(prior_returns, k=200)

    # Normaliza real_returns para % do portfolio
    real_pct = [r / portfolio_value if abs(r) < portfolio_value else r / portfolio_value
                for r in real_returns]

    # Amostra proporcional
    n_blend  = max(200, n_real * 3)
    n_prior  = round(n_blend * w_prior)
    n_real_s = round(n_blend * w_real)

    blended = (
        random.choices(prior_returns, k=n_prior) +
        random.choices(real_pct, k=min(n_real_s, len(real_pct)))
    )
    random.shuffle(blended)
    return blended


# ── Métricas robustas ─────────────────────────────────────────────────────────

def calc_robust_var(
    real_trades: list,
    portfolio_value: float,
    prior: Optional[dict] = None,
    confidence: float = 0.95,
) -> dict:
    """
    VaR robusto usando blend de prior histórico + trades reais.

    Retorna:
      var_pct      — VaR como % do portfolio
      es_pct       — Expected Shortfall (CVaR)
      var_usd      — VaR em USD
      es_usd       — ES em USD
      n_real       — trades reais usados
      prior_weight — peso do prior (0 = só real, 1 = só prior)
      source       — 'blended' | 'real_only' | 'prior_only'
    """
    if prior is None:
        prior = load_prior()

    n_real = len(real_trades)

    # Extrai retornos reais por trade como % do portfolio
    real_returns = []
    for t in real_trades[-100:]:
        pnl  = t.get("pnl_usd", 0) or 0
        size = abs(t.get("usd", portfolio_value * 0.08) or (portfolio_value * 0.08))
        ret  = pnl / portfolio_value
        real_returns.append(ret)

    # Blend
    blended = blend_returns(real_returns, prior, portfolio_value)
    blended.sort()

    # VaR e ES
    cutoff_idx = int(len(blended) * (1 - confidence))
    var_pct    = abs(blended[cutoff_idx]) if cutoff_idx < len(blended) else 0.02
    tail       = blended[:cutoff_idx + 1]
    es_pct     = abs(sum(tail) / len(tail)) if tail else var_pct * 1.5

    # Clamp razoável
    var_pct = min(var_pct, 0.15)
    es_pct  = min(es_pct,  0.25)

    w = prior_weight(n_real)
    source = "prior_only" if n_real == 0 else ("blended" if w > 0 else "real_only")

    return {
        "var_pct":      round(var_pct, 5),
        "es_pct":       round(es_pct,  5),
        "var_usd":      round(var_pct * portfolio_value, 2),
        "es_usd":       round(es_pct  * portfolio_value, 2),
        "confidence":   confidence,
        "n_real":       n_real,
        "prior_weight": round(w, 3),
        "source":       source,
        "n_blended":    len(blended),
    }


def calc_robust_monte_carlo(
    real_trades: list,
    portfolio_value: float,
    prior: Optional[dict] = None,
    n_simulations: int = 5000,
    horizon: int = 50,
    ruin_threshold: float = 0.40,
) -> dict:
    """
    Monte Carlo robusto com prior histórico.
    Muito mais estável que o baseado apenas em trades reais.
    """
    if prior is None:
        prior = load_prior()

    n_real = len(real_trades)

    real_returns = []
    for t in real_trades[-100:]:
        pnl = t.get("pnl_usd", 0) or 0
        real_returns.append(pnl / portfolio_value)

    blended = blend_returns(real_returns, prior, portfolio_value)

    if not blended:
        blended = DEFAULT_PRIOR["returns"]

    mean_r = sum(blended) / len(blended)
    std_r  = statistics.stdev(blended) if len(blended) > 1 else abs(mean_r) + 0.01

    random.seed(42)
    ruin_count = 0
    max_dds    = []
    final_vals = []

    for _ in range(n_simulations):
        port = portfolio_value
        peak = portfolio_value
        max_dd = 0.0

        for _ in range(horizon):
            base   = random.choice(blended)
            noise  = random.gauss(0, std_r * 0.2)   # perturbação menor que no motor original
            r      = base + noise
            port  *= (1 + r)
            peak   = max(peak, port)
            dd     = (port - peak) / peak
            max_dd = min(max_dd, dd)

        if port < portfolio_value * (1 - ruin_threshold):
            ruin_count += 1
        max_dds.append(max_dd)
        final_vals.append(port)

    max_dds.sort()
    ruin_prob      = ruin_count / n_simulations
    expected_dd    = sum(max_dds) / len(max_dds)
    dd_95_idx      = int(len(max_dds) * 0.95)
    dd_95_worst    = max_dds[min(dd_95_idx, len(max_dds) - 1)]

    # Sharpe dos retornos blended (mais representativo que os 20 trades reais)
    if len(blended) > 5:
        m = sum(blended) / len(blended)
        s = statistics.stdev(blended)
        sharpe = max(-5.0, min(5.0, (m / s * math.sqrt(252)) if s > 0 else 0.0))
    else:
        sharpe = prior.get("sharpe", 0.8)

    # Win rate dos últimos trades reais (se disponíveis) vs prior
    if n_real >= 5:
        real_wr_7  = sum(1 for t in real_trades[-7:]  if (t.get("pnl_usd") or 0) > 0) / max(len(real_trades[-7:]),  1)
        real_wr_30 = sum(1 for t in real_trades[-30:] if (t.get("pnl_usd") or 0) > 0) / max(len(real_trades[-30:]), 1)
    else:
        real_wr_7  = prior.get("win_rate", 0.52)
        real_wr_30 = prior.get("win_rate", 0.52)

    # Win rate blended
    w           = prior_weight(n_real)
    prior_wr    = prior.get("win_rate", 0.52)
    blended_wr  = w * prior_wr + (1 - w) * real_wr_30

    # Tail risk
    tail_threshold = mean_r - 2 * std_r
    tail_risk      = sum(1 for r in blended if r < tail_threshold) / len(blended)

    # Expectancy
    wins_r   = [r for r in blended if r > 0]
    losses_r = [r for r in blended if r <= 0]
    expectancy = (
        (sum(wins_r) / len(wins_r) if wins_r else 0) * blended_wr +
        (sum(losses_r) / len(losses_r) if losses_r else 0) * (1 - blended_wr)
    ) * 100

    return {
        "ruin_probability":   round(ruin_prob, 4),
        "expected_max_dd":    round(expected_dd, 4),
        "dd_95pct_worst":     round(dd_95_worst, 4),
        "sharpe_rolling":     round(sharpe, 3),
        "expectancy_rolling": round(expectancy, 4),
        "winrate_7d":         round(real_wr_7,  3),
        "winrate_30d":        round(real_wr_30, 3),
        "winrate_blended":    round(blended_wr, 3),
        "tail_risk_next":     round(tail_risk, 4),
        "n_simulations":      n_simulations,
        "n_real_trades":      n_real,
        "prior_weight":       round(w, 3),
        "prior_source":       prior.get("source", "default"),
        "insufficient_data":  n_real < 10,
    }


def get_prior_summary(prior: Optional[dict] = None) -> dict:
    """Resumo do prior ativo para exibição no dashboard."""
    if prior is None:
        prior = load_prior()
    return {
        "source":        prior.get("source", "default"),
        "n_samples":     prior.get("n_samples", 0),
        "win_rate":      prior.get("win_rate", 0.52),
        "profit_factor": prior.get("profit_factor", 1.40),
        "sharpe":        prior.get("sharpe", 0.80),
        "max_drawdown":  prior.get("max_drawdown", -0.22),
        "is_historical": prior.get("source") == "okx_walkforward_8y",
    }
