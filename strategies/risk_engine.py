"""
Risk Engine — Camada 6
======================
Monitoramento contínuo de risco do portfolio inteiro.

Não é só stop loss. Instituições monitoram:
  - Portfolio VaR (Value at Risk 95%)
  - Expected Shortfall (tail risk)
  - Correlation exposure
  - Drawdown acceleration (detecta colapso antes do fundo)
  - Strategy decay (detecta edge morto)
  - Monte Carlo Engine (ruína, DD máximo provável, Sharpe rolling)

O V4 monitora risco CONTINUAMENTE — não só antes de entrar.
"""

import math
import random
import statistics
from typing import Optional


# ── VaR e Expected Shortfall ──────────────────────────────────────────────────

def calc_portfolio_var(
    trades: list,
    portfolio_value: float,
    confidence: float = 0.95,
    window: int = 50,
) -> dict:
    """
    VaR histórico do portfolio usando retornos dos últimos N trades.

    Retorna:
      var_pct   — perda máxima esperada como % do portfolio (nível de confiança)
      var_usd   — perda em USD
      es_pct    — Expected Shortfall (média dos piores cenários além do VaR)
      es_usd    — ES em USD
    """
    if len(trades) < 10:
        return {
            "var_pct": 0.02, "var_usd": portfolio_value * 0.02,
            "es_pct":  0.04, "es_usd":  portfolio_value * 0.04,
            "confidence": confidence, "n_trades": len(trades),
        }

    recent = trades[-window:]
    returns = []
    for t in recent:
        pnl = t.get("pnl_usd", 0) or 0
        cost = abs(t.get("usd", 1) or 1)
        returns.append(pnl / cost if cost > 0 else 0)

    if not returns:
        return {"var_pct": 0.02, "var_usd": portfolio_value * 0.02,
                "es_pct": 0.04, "es_usd": portfolio_value * 0.04,
                "confidence": confidence, "n_trades": 0}

    returns.sort()
    cutoff_idx = int(len(returns) * (1 - confidence))
    var_pct = abs(returns[cutoff_idx]) if cutoff_idx < len(returns) else 0.02

    tail = returns[:cutoff_idx + 1]
    es_pct = abs(sum(tail) / len(tail)) if tail else var_pct * 1.5

    # Escala para portfolio
    var_portfolio = var_pct * 0.10   # assume 10% médio por trade
    es_portfolio  = es_pct  * 0.10

    return {
        "var_pct":    round(var_portfolio, 4),
        "var_usd":    round(var_portfolio * portfolio_value, 2),
        "es_pct":     round(es_portfolio, 4),
        "es_usd":     round(es_portfolio  * portfolio_value, 2),
        "confidence": confidence,
        "n_trades":   len(recent),
    }


# ── Drawdown Acceleration ─────────────────────────────────────────────────────

def calc_drawdown_acceleration(portfolio_history: list, window: int = 6) -> dict:
    """
    Detecta aceleração do drawdown — muito mais importante que a magnitude.

    V3: "drawdown chegou em -15%? Para."
    V4: "drawdown acelerou 3× nas últimas 6h? Para ANTES de -15%."

    Retorna:
      current_dd     — drawdown atual vs peak
      dd_velocity    — velocidade de queda (DD/hora)
      acceleration   — dd_velocity atual / dd_velocity anterior
      alert          — True se aceleração detectada
    """
    if len(portfolio_history) < window * 2:
        return {"current_dd": 0.0, "dd_velocity": 0.0, "acceleration": 1.0, "alert": False}

    values = [h.get("portfolio_value", 0) for h in portfolio_history if h.get("portfolio_value")]
    if len(values) < window * 2:
        return {"current_dd": 0.0, "dd_velocity": 0.0, "acceleration": 1.0, "alert": False}

    peak = max(values)
    current = values[-1]
    current_dd = (current - peak) / peak if peak > 0 else 0.0

    # Velocidade nas últimas N vs N anteriores
    recent_values = values[-window:]
    prev_values   = values[-window * 2: -window]

    recent_change = (recent_values[-1] - recent_values[0]) / recent_values[0] if recent_values[0] > 0 else 0
    prev_change   = (prev_values[-1]   - prev_values[0])   / prev_values[0]   if prev_values[0]   > 0 else 0

    dd_velocity_recent = recent_change / window  # por unidade de tempo
    dd_velocity_prev   = prev_change   / window

    acceleration = (dd_velocity_recent / dd_velocity_prev) if dd_velocity_prev != 0 else 1.0

    # Alerta: drawdown acelerando mais de 2× e negativo
    alert = (
        current_dd < -0.05 and          # drawdown > 5%
        acceleration < -1.5 and         # acelerando para baixo
        dd_velocity_recent < -0.005     # queda significativa
    )

    return {
        "current_dd":   round(current_dd, 4),
        "dd_velocity":  round(dd_velocity_recent, 5),
        "acceleration": round(acceleration, 3),
        "alert":        alert,
        "peak_value":   round(peak, 2),
    }


# ── Strategy Decay Detection ──────────────────────────────────────────────────

def calc_strategy_decay(trades: list, strategy_name: str) -> dict:
    """
    Detecta se o edge de uma estratégia está decaindo.

    Compara performance em janelas diferentes:
      edge_30d  — últimos 30 trades da estratégia
      edge_7d   — últimos 7 trades
      edge_3d   — últimos 3 trades

    Status:
      HEALTHY    — edge estável ou melhorando
      DEGRADING  — edge caindo
      SUSPENDED  — edge negativo por 2+ janelas → suspende
      RECOVERING — foi suspenso mas mostra recuperação
    """
    strategy_trades = [t for t in trades if t.get("strategy") == strategy_name]

    if len(strategy_trades) < 5:
        return {"status": "INSUFFICIENT_DATA", "edge_7d": 0.0, "edge_30d": 0.0,
                "weight_adjustment": 1.0, "suspend": False}

    def _edge(subset: list) -> float:
        if not subset:
            return 0.0
        wins  = sum(1 for t in subset if (t.get("pnl_usd") or 0) > 0)
        total = len(subset)
        win_rate = wins / total
        avg_win  = sum((t.get("pnl_usd") or 0) for t in subset if (t.get("pnl_usd") or 0) > 0) / max(wins, 1)
        avg_loss = sum(abs(t.get("pnl_usd") or 0) for t in subset if (t.get("pnl_usd") or 0) <= 0) / max(total - wins, 1)
        payoff   = avg_win / avg_loss if avg_loss > 0 else 1.0
        return win_rate * payoff - (1 - win_rate)  # expectancy

    edge_30d = _edge(strategy_trades[-30:])
    edge_7d  = _edge(strategy_trades[-7:])
    edge_3d  = _edge(strategy_trades[-3:])

    # Determina status
    if edge_7d < 0 and edge_3d < 0:
        status = "SUSPENDED"
        suspend = True
        weight = 0.0
    elif edge_7d < edge_30d * 0.60:
        status = "DEGRADING"
        suspend = False
        weight = max(0.30, edge_7d / edge_30d) if edge_30d > 0 else 0.50
    elif edge_7d > edge_30d * 1.20:
        status = "EXPANDING"
        suspend = False
        weight = min(1.30, edge_7d / edge_30d) if edge_30d > 0 else 1.0
    else:
        status = "HEALTHY"
        suspend = False
        weight = 1.0

    return {
        "strategy":         strategy_name,
        "status":           status,
        "edge_30d":         round(edge_30d, 4),
        "edge_7d":          round(edge_7d, 4),
        "edge_3d":          round(edge_3d, 4),
        "weight_adjustment": round(weight, 3),
        "suspend":          suspend,
        "n_trades":         len(strategy_trades),
    }


# ── Monte Carlo Engine ────────────────────────────────────────────────────────

def run_monte_carlo(
    trades: list,
    portfolio_value: float,
    n_simulations: int = 5000,
    horizon: int = 50,
    ruin_threshold: float = 0.40,
) -> dict:
    """
    Simula N sequências de trades para estimar distribuição de retornos.

    Usa trades históricos como distribuição base.
    Retorna estatísticas de risco prospectivo.

    Parâmetros:
      n_simulations  — número de cenários (5000 para boa convergência)
      horizon        — próximos N trades a simular
      ruin_threshold — define "ruína" como perda > X% do portfolio
    """
    if len(trades) < 10:
        return {
            "ruin_probability":      0.05,
            "expected_max_dd":       -0.10,
            "dd_95pct_worst":        -0.20,
            "sharpe_rolling":        1.0,
            "expectancy_rolling":    0.01,
            "winrate_7d":            0.50,
            "winrate_30d":           0.50,
            "tail_risk_next":        0.05,
            "n_simulations":         0,
            "insufficient_data":     True,
        }

    # Extrai retornos dos trades reais
    returns = []
    for t in trades[-200:]:
        pnl = t.get("pnl_usd", 0) or 0
        size = abs(t.get("usd", portfolio_value * 0.08) or (portfolio_value * 0.08))
        ret = pnl / portfolio_value  # retorno como % do portfolio total
        returns.append(ret)

    if not returns:
        return {"ruin_probability": 0.05, "expected_max_dd": -0.10,
                "dd_95pct_worst": -0.20, "sharpe_rolling": 1.0,
                "expectancy_rolling": 0.0, "winrate_7d": 0.5, "winrate_30d": 0.5,
                "tail_risk_next": 0.05, "n_simulations": 0, "insufficient_data": True}

    mean_ret = sum(returns) / len(returns)
    std_ret  = statistics.stdev(returns) if len(returns) > 1 else abs(mean_ret) + 0.01

    # Simula N cenários
    ruin_count       = 0
    max_dds          = []
    final_values     = []

    random.seed(42)  # reproduzível

    for _ in range(n_simulations):
        portfolio = portfolio_value
        peak      = portfolio_value
        max_dd    = 0.0

        for _ in range(horizon):
            # Bootstrap com leve perturbação (não puro bootstrap para capturar tail)
            base_return = random.choice(returns)
            noise = random.gauss(0, std_ret * 0.3)
            trade_return = base_return + noise

            portfolio *= (1 + trade_return)
            peak = max(peak, portfolio)
            dd = (portfolio - peak) / peak
            max_dd = min(max_dd, dd)

        if portfolio < portfolio_value * (1 - ruin_threshold):
            ruin_count += 1
        max_dds.append(max_dd)
        final_values.append(portfolio)

    max_dds.sort()
    ruin_prob = ruin_count / n_simulations

    # DD esperado e 95% pior
    expected_max_dd = sum(max_dds) / len(max_dds)
    dd_95_idx = int(len(max_dds) * 0.95)
    dd_95_worst = max_dds[min(dd_95_idx, len(max_dds) - 1)]

    # Sharpe rolling (últimos 20 trades)
    recent_20 = returns[-20:]
    if len(recent_20) > 1:
        mean_20 = sum(recent_20) / len(recent_20)
        std_20  = statistics.stdev(recent_20)
        sharpe  = (mean_20 / std_20) * math.sqrt(8760 / 20) if std_20 > 0 else 0.0
    else:
        sharpe = 0.0

    # Expectancy rolling
    expectancy = mean_ret * 100  # em % por trade

    # Winrates
    recent_7  = [t for t in trades[-7:]  if (t.get("pnl_usd") or 0) != 0]
    recent_30 = [t for t in trades[-30:] if (t.get("pnl_usd") or 0) != 0]
    winrate_7  = sum(1 for t in recent_7  if (t.get("pnl_usd") or 0) > 0) / max(len(recent_7),  1)
    winrate_30 = sum(1 for t in recent_30 if (t.get("pnl_usd") or 0) > 0) / max(len(recent_30), 1)

    # Tail risk próximo ciclo: P(retorno < -2σ no próximo trade)
    threshold_tail = mean_ret - 2 * std_ret
    tail_risk = sum(1 for r in returns if r < threshold_tail) / len(returns)

    return {
        "ruin_probability":   round(ruin_prob, 4),
        "expected_max_dd":    round(expected_max_dd, 4),
        "dd_95pct_worst":     round(dd_95_worst, 4),
        "sharpe_rolling":     round(sharpe, 3),
        "expectancy_rolling": round(expectancy, 4),
        "winrate_7d":         round(winrate_7, 3),
        "winrate_30d":        round(winrate_30, 3),
        "tail_risk_next":     round(tail_risk, 4),
        "n_simulations":      n_simulations,
        "n_trades_used":      len(returns),
    }


# ── Risk Actions ──────────────────────────────────────────────────────────────

def evaluate_risk_actions(
    var_result:     dict,
    mc_result:      dict,
    dd_result:      dict,
    var_limit_pct:  float = 0.05,
    ruin_limit:     float = 0.05,
    tail_limit:     float = 0.08,
) -> dict:
    """
    Avalia os resultados de risco e determina ações automáticas.

    Retorna:
      action       — 'normal' | 'reduce_sizing' | 'suspend_entries' | 'close_all'
      alerts       — lista de alertas ativos
      sizing_mult  — multiplicador de sizing (1.0 = normal, 0.5 = reduzido)
    """
    alerts = []
    action = "normal"
    sizing_mult = 1.0

    # VaR excedendo limite
    if var_result.get("var_pct", 0) > var_limit_pct:
        alerts.append(f"VaR {var_result['var_pct']:.2%} excede limite {var_limit_pct:.2%}")
        sizing_mult = min(sizing_mult, 0.50)
        action = "reduce_sizing"

    # Probabilidade de ruína elevada
    if mc_result.get("ruin_probability", 0) > ruin_limit:
        alerts.append(f"P(ruína) {mc_result['ruin_probability']:.2%} > {ruin_limit:.2%}")
        sizing_mult = min(sizing_mult, 0.50)
        action = "reduce_sizing"

    # Tail risk extremo
    if mc_result.get("tail_risk_next", 0) > tail_limit:
        alerts.append(f"Tail risk {mc_result['tail_risk_next']:.2%} > {tail_limit:.2%}")
        sizing_mult = min(sizing_mult, 0.30)
        action = "suspend_entries"

    # DD worst case muito ruim
    if mc_result.get("dd_95pct_worst", 0) < -0.30:
        alerts.append(f"DD pior caso 95% = {mc_result['dd_95pct_worst']:.2%}")
        sizing_mult = min(sizing_mult, 0.50)
        if action == "normal":
            action = "reduce_sizing"

    # Drawdown acelerando
    if dd_result.get("alert", False):
        alerts.append(f"Drawdown acelerando: {dd_result['current_dd']:.2%} @ accel {dd_result['acceleration']:.2f}×")
        sizing_mult = 0.0
        action = "close_all"

    # Sharpe muito baixo
    if mc_result.get("sharpe_rolling", 1.0) < 0.50 and not mc_result.get("insufficient_data"):
        alerts.append(f"Sharpe rolling {mc_result['sharpe_rolling']:.2f} < 0.50")
        sizing_mult = min(sizing_mult, 0.70)
        if action == "normal":
            action = "reduce_sizing"

    return {
        "action":      action,
        "alerts":      alerts,
        "sizing_mult": round(sizing_mult, 2),
        "safe":        len(alerts) == 0,
    }
