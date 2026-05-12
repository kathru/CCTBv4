"""
V4 Orchestrator
===============
Integra todas as camadas V4 no ciclo de trading do app.py.

Substitui a lógica de decisão do V3 (regras discretas + score básico)
pela pipeline probabilística completa:

  Camada 1 → Data Engine (microestrutura)
  Camada 2 → Regime Engine (7 regimes)
  Camada 3 → Signal Engine (score probabilístico)
  Camada 4 → Sizing Engine (Edge × Confidence / Vol × Corr)
  Camada 5 → Execution Engine (staggered / passive limit)
  Camada 6 → Risk Engine (VaR / Monte Carlo / decay)
  Meta-Layer → Self-Optimizer (pesos adaptativos)
  Portfolio Layer → portfolio-level risk check

Uso no app.py:
  from dashboard.v4_orchestrator import V4Orchestrator
  v4 = V4Orchestrator()
  decision = v4.evaluate(pair, candles_1h, candles_6h, closes_map, engine)
"""

import os
import sys
import time
import asyncio
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data        import get_market_context
from strategies.regime_engine    import detect_regime, RegimeResult
from strategies.signal_engine    import compute_signal_score
from strategies.sizing_engine    import compute_position_size
from strategies.risk_engine      import (
    calc_portfolio_var, calc_drawdown_acceleration,
    calc_strategy_decay, run_monte_carlo, evaluate_risk_actions
)
from strategies.meta_layer       import MetaLayer
from strategies.portfolio_engine import evaluate_new_entry, calc_portfolio_beta
from exchange.execution_engine   import plan_entry, update_trailing_stop


class V4Orchestrator:
    """
    Orquestra todas as camadas V4 para cada ciclo de trading.
    Mantém estado entre ciclos (meta-layer, portfolio history, risk state).
    """

    def __init__(self, state_dir: str = "data"):
        self.state_dir       = state_dir
        self.meta_layer      = MetaLayer(os.path.join(state_dir, "meta_layer_state.json"))
        self.portfolio_history: list = []
        self.risk_state: dict        = {"action": "normal", "sizing_mult": 1.0, "alerts": []}
        self._last_mc_run: float     = 0.0
        self._last_context: dict     = {}  # cache de contextos por par

    # ── Avaliação completa por par ────────────────────────────────────────────

    def evaluate(
        self,
        pair:         str,
        candles_1h:   list,
        candles_6h:   list,
        closes_map:   dict,
        engine,                     # PaperTradingEngine
        open_slots:   dict = None,
        existing_slot: dict = None,
    ) -> dict:
        """
        Roda a pipeline V4 completa para um par.

        Retorna:
          decision     — 'BUY' | 'SELL' | 'HOLD'
          score        — score probabilístico 0-1
          size_pct     — tamanho sugerido como % do portfolio
          execution    — ExecutionPlan (tranches, SL, TP)
          regime       — RegimeResult
          risk_ok      — True se Risk Engine aprova
          reason       — texto explicativo
          full_context — todos os dados para dashboard
        """
        open_slots = open_slots or {}
        t0 = time.time()

        # ── Camada 1: Dados de microestrutura ────────────────────────────────
        try:
            market_ctx = get_market_context(pair, candles_1h, closes_map)
        except Exception as e:
            market_ctx = {
                "pair": pair, "timestamp": int(time.time()),
                "realized_vol": 0.30, "vol_percentile": 0.50,
                "atr_rate": {"expansion": 1.0},
                "funding": {"funding_rate": 0.0, "sentiment": "neutral"},
                "open_interest": {"expanding": False, "oi_change_pct": 0.0},
                "orderbook": {"imbalance": 0.0, "spread_pct": 0.0001},
                "taker_ratio": {"imbalance": 0.0},
                "volume_delta": {"delta_pct": 0.0, "aggressive": False},
                "correlation": {"avg_correlation": 0.70},
            }
        self._last_context[pair] = market_ctx

        # ── Camada 2: Regime ──────────────────────────────────────────────────
        try:
            regime_result = detect_regime(candles_1h, candles_6h, market_ctx, closes_map)
        except Exception:
            from strategies.regime_engine import RegimeResult
            regime_result = RegimeResult(
                regime="MEAN_REVERTING_CHOP",
                probabilities={},
                confidence=0.3,
                action="reduce",
            )

        regime = regime_result.regime

        # ── Camada 3: Signal Score ────────────────────────────────────────────
        try:
            signal = compute_signal_score(
                pair=pair,
                candles_1h=candles_1h,
                market_context=market_ctx,
                regime=regime,
                closes_map=closes_map,
                fee_rate=0.002,
                expected_rr=2.0,
            )
        except Exception:
            signal = {
                "score": 0.50, "expected_value": -0.01,
                "kelly_fraction": 0.0, "confidence": 0.3,
                "direction": "neutral", "factors": {}, "regime": regime,
            }

        # Decisão de SELL para posição aberta
        sell_decision = self._check_exit(
            existing_slot=existing_slot,
            pair=pair,
            candles_1h=candles_1h,
            market_ctx=market_ctx,
            regime_result=regime_result,
            signal=signal,
        )
        if sell_decision:
            return sell_decision

        # ── Risk Engine check (global — não por par) ──────────────────────────
        if self.risk_state.get("action") in ("close_all", "suspend_entries"):
            return {
                "decision": "HOLD",
                "score":    signal["score"],
                "size_pct": 0.0,
                "reason":   f"Risk Engine: {self.risk_state['action']}",
                "regime":   regime_result,
                "signal":   signal,
                "context":  market_ctx,
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # ── Sem entrada se EV negativo ────────────────────────────────────────
        if signal["expected_value"] <= 0 or signal["direction"] == "neutral":
            return {
                "decision": "HOLD",
                "score":    signal["score"],
                "size_pct": 0.0,
                "reason":   f"EV={signal['expected_value']:.4f} | dir={signal['direction']}",
                "regime":   regime_result,
                "signal":   signal,
                "context":  market_ctx,
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # ── Camada 4: Sizing ──────────────────────────────────────────────────
        portfolio_value = engine.portfolio_value()
        corr_data       = market_ctx.get("correlation", {})
        avg_corr        = corr_data.get("avg_correlation", 0.70)
        n_open          = sum(1 for s in open_slots.values() if s.get("qty", 0) > 0)

        portfolio_ctx = {
            "var_headroom":   1.0 - self.risk_state.get("sizing_mult", 1.0) * 0.5,
            "open_positions": n_open,
        }

        sizing = compute_position_size(
            signal_score=signal,
            regime=regime,
            portfolio_context=portfolio_ctx,
            vol_percentile=market_ctx.get("vol_percentile", 0.5),
            correlation_risk=avg_corr,
            portfolio_value=portfolio_value,
        )

        if sizing["blocked"]:
            return {
                "decision": "HOLD",
                "score":    signal["score"],
                "size_pct": 0.0,
                "reason":   f"Sizing bloqueado: {sizing['reason']}",
                "regime":   regime_result,
                "signal":   signal,
                "context":  market_ctx,
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # ── Portfolio Layer: impacto no portfolio ─────────────────────────────
        var_result = self.risk_state.get("var_result", {"var_pct": 0.02})
        portfolio_check = evaluate_new_entry(
            new_pair=pair,
            new_size_pct=sizing["size_pct"],
            open_slots=open_slots,
            correlation_matrix=corr_data,
            market_contexts={p: self._last_context.get(p, {}) for p in closes_map},
            portfolio_value=portfolio_value,
            var_result=var_result,
        )

        if not portfolio_check["approved"]:
            return {
                "decision": "HOLD",
                "score":    signal["score"],
                "size_pct": 0.0,
                "reason":   f"Portfolio: {portfolio_check['reason']}",
                "regime":   regime_result,
                "signal":   signal,
                "context":  market_ctx,
                "latency_ms": int((time.time() - t0) * 1000),
            }

        final_size = portfolio_check["size_adjusted"]

        # ── Camada 5: Execution Plan ──────────────────────────────────────────
        closes  = [float(c["close"]) for c in candles_1h]
        highs   = [float(c["high"])  for c in candles_1h]
        lows    = [float(c["low"])   for c in candles_1h]
        current_price = closes[-1] if closes else 0.0

        # ATR simples para execução
        atr = 0.0
        if len(closes) >= 14:
            trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
                   for i in range(-14, 0)]
            atr = sum(trs) / len(trs)

        exec_plan = plan_entry(
            direction=signal["direction"],
            current_price=current_price,
            size_usd=final_size * portfolio_value,
            market_context=market_ctx,
            regime=regime,
            signal_score=signal,
            atr_value=atr,
        )

        # ── Decisão final ─────────────────────────────────────────────────────
        # Threshold mínimo: score > 0.55 e EV > custo de execução
        min_score = 0.55
        if regime == "TREND_EXPANSION":
            min_score = 0.58
        elif regime == "VOLATILITY_COMPRESSION":
            min_score = 0.60

        if signal["score"] < min_score:
            return {
                "decision": "HOLD",
                "score":    signal["score"],
                "size_pct": 0.0,
                "reason":   f"Score {signal['score']:.3f} < min {min_score:.2f}",
                "regime":   regime_result,
                "signal":   signal,
                "context":  market_ctx,
                "latency_ms": int((time.time() - t0) * 1000),
            }

        return {
            "decision":        "BUY",
            "direction":       signal["direction"],
            "score":           signal["score"],
            "expected_value":  signal["expected_value"],
            "size_pct":        final_size,
            "size_usd":        round(final_size * portfolio_value, 2),
            "execution":       exec_plan,
            "regime":          regime_result,
            "signal":          signal,
            "sizing":          sizing,
            "portfolio_check": portfolio_check,
            "context":         market_ctx,
            "reason":          f"score={signal['score']:.3f} ev={signal['expected_value']:.4f} regime={regime}",
            "latency_ms":      int((time.time() - t0) * 1000),
        }

    def _check_exit(
        self,
        existing_slot: Optional[dict],
        pair: str,
        candles_1h: list,
        market_ctx: dict,
        regime_result,
        signal: dict,
    ) -> Optional[dict]:
        """Verifica se posição aberta deve ser fechada."""
        if not existing_slot or existing_slot.get("qty", 0) <= 0:
            return None

        closes = [float(c["close"]) for c in candles_1h]
        if not closes:
            return None

        current_price = closes[-1]
        entry_price   = existing_slot.get("entry", current_price)
        peak_price    = existing_slot.get("peak", current_price)
        sl_pct        = existing_slot.get("sl_pct", 0.03)
        current_sl    = existing_slot.get("sl_level", entry_price * (1 - sl_pct))

        # Regime de pânico → fecha imediatamente
        if regime_result.regime == "PANIC_LIQUIDATION":
            return {
                "decision": "SELL",
                "reason":   "PANIC_LIQUIDATION — fecha posição",
                "score":    signal["score"],
                "size_pct": 0.0,
                "regime":   regime_result,
                "signal":   signal,
                "context":  market_ctx,
            }

        # Sinal reverso forte → fecha APENAS se estiver no lucro
        # Regra institucional: sinal não pode forçar venda abaixo do preço de entrada.
        # Somente o SL é autorizado a fechar com prejuízo — o sinal só realiza lucro.
        if signal["direction"] == "short" and signal["score"] > 0.68:
            if current_price > entry_price:
                gain_pct = (current_price - entry_price) / entry_price * 100
                return {
                    "decision": "SELL",
                    "reason":   f"Sinal reverso com lucro: +{gain_pct:.2f}% | score={signal['score']:.3f}",
                    "score":    signal["score"],
                    "size_pct": 0.0,
                    "regime":   regime_result,
                    "signal":   signal,
                    "context":  market_ctx,
                }
            # Abaixo do entry: mantém posição, SL cuidará da saída
            existing_slot["sl_level"] = current_sl  # garante SL atualizado

        # Atualiza trailing stop
        sl_update = update_trailing_stop(
            current_price=current_price,
            entry_price=entry_price,
            peak_price=peak_price,
            current_sl=current_sl,
            sl_pct=sl_pct,
            regime=regime_result.regime,
        )

        # Stop atingido
        if current_price <= sl_update["sl"]:
            return {
                "decision": "SELL",
                "reason":   f"SL atingido: price={current_price:.2f} sl={sl_update['sl']:.2f} ({sl_update['action']})",
                "score":    signal["score"],
                "size_pct": 0.0,
                "sl_level": sl_update["sl"],
                "regime":   regime_result,
                "signal":   signal,
                "context":  market_ctx,
            }

        # Atualiza sl_level na slot (retorna None = não fecha, mas atualiza SL)
        existing_slot["sl_level"] = sl_update["sl"]
        existing_slot["peak"]     = max(peak_price, current_price)
        return None

    # ── Risk Engine global (roda a cada ciclo, não por par) ───────────────────

    def update_risk_state(self, engine, portfolio_history: list = None):
        """
        Atualiza o estado global de risco.
        Chamado uma vez por ciclo antes de avaliar os pares.
        """
        portfolio_history = portfolio_history or self.portfolio_history
        trades = list(engine.trades)
        portfolio_value = engine.portfolio_value()

        # Adiciona snapshot atual ao histórico
        self.portfolio_history.append({
            "portfolio_value": portfolio_value,
            "timestamp": time.time(),
        })
        if len(self.portfolio_history) > 200:
            self.portfolio_history = self.portfolio_history[-200:]

        # VaR
        var_result = calc_portfolio_var(trades, portfolio_value)

        # Monte Carlo (a cada 4h para não sobrecarregar)
        now = time.time()
        if now - self._last_mc_run > 14400 or not self.risk_state.get("mc_result"):
            mc_result = run_monte_carlo(trades, portfolio_value, n_simulations=2000, horizon=30)
            self._last_mc_run = now
        else:
            mc_result = self.risk_state.get("mc_result", {})

        # Drawdown acceleration
        dd_result = calc_drawdown_acceleration(self.portfolio_history)

        # Avalia ações
        risk_actions = evaluate_risk_actions(var_result, mc_result, dd_result)

        # Meta-layer
        meta_weights = self.meta_layer.update(trades)

        self.risk_state = {
            "action":       risk_actions["action"],
            "sizing_mult":  risk_actions["sizing_mult"],
            "alerts":       risk_actions["alerts"],
            "var_result":   var_result,
            "mc_result":    mc_result,
            "dd_result":    dd_result,
            "meta_weights": meta_weights,
            "meta_summary": self.meta_layer.get_summary(),
            "timestamp":    now,
        }

        return self.risk_state

    def get_v4_state(self) -> dict:
        """Estado completo V4 para o dashboard."""
        return {
            "risk": self.risk_state,
            "meta": self.meta_layer.get_summary(),
            "contexts": self._last_context,
        }
