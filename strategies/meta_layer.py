"""
Meta-Layer — Self-Optimizer
============================
O bot que analisa o próprio bot.

Monitora a performance real de cada modelo do Signal Engine
e redistribui pesos automaticamente conforme o edge evolui.

Mercado muda. Edge morre. Bots estáticos morrem junto.

Estados de uma estratégia/modelo:
  HEALTHY    — edge estável
  EXPANDING  — edge crescendo → aumenta peso
  DEGRADING  — edge caindo → reduz peso gradualmente
  SUSPENDED  — edge negativo por 2+ janelas → desativa
  RECOVERING — estava suspenso, mostra recuperação → retorna com peso mínimo

Ciclos de adaptação:
  Peso imediato    — ajuste automático a cada trade
  Parâmetros       — revisão a cada 30 dias (manual por agora)
  Reativação       — suspenso volta com peso mínimo após sinal positivo
"""

import time
import json
import os
from typing import Optional


MODEL_NAMES = [
    "volatility_expansion",
    "market_structure",
    "orderflow",
    "relative_strength",
]

# Pesos padrão (soma = 1.0)
DEFAULT_WEIGHTS = {
    "volatility_expansion": 0.25,
    "market_structure":     0.30,
    "orderflow":            0.25,
    "relative_strength":    0.20,
}

# Limites de peso
MIN_WEIGHT = 0.05   # nunca vai a zero (exceto SUSPENDED)
MAX_WEIGHT = 0.55   # evita concentração excessiva


class MetaLayer:
    """
    Gerencia pesos dos modelos do Signal Engine com base em performance real.
    Persiste estado em JSON para sobreviver a restarts.
    """

    def __init__(self, state_path: str = "data/meta_layer_state.json"):
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> dict:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return self._default_state()

    def _default_state(self) -> dict:
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "model_stats": {
                m: {
                    "status":      "HEALTHY",
                    "edge_30d":    0.0,
                    "edge_7d":     0.0,
                    "n_trades":    0,
                    "last_update": time.time(),
                }
                for m in MODEL_NAMES
            },
            "last_rebalance": time.time(),
        }

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    def update(self, trades: list) -> dict:
        """
        Recalcula pesos baseado nos trades recentes de cada modelo.
        Chamado a cada ciclo.

        Retorna o dict de pesos atualizado.
        """
        for model in MODEL_NAMES:
            model_trades = [t for t in trades if t.get("model") == model or
                            t.get("signal_source") == model]

            stats = self._calc_model_stats(model_trades)
            prev  = self.state["model_stats"].get(model, {})

            edge_7d  = stats["edge_7d"]
            edge_30d = stats["edge_30d"]
            prev_status = prev.get("status", "HEALTHY")

            # ── Determina novo status ─────────────────────────────────────────
            if edge_7d < 0 and stats["edge_3d"] < 0 and stats["n_trades"] >= 3:
                new_status = "SUSPENDED"
            elif prev_status == "SUSPENDED" and edge_7d > 0:
                new_status = "RECOVERING"
            elif edge_30d > 0 and edge_7d > edge_30d * 1.30:
                new_status = "EXPANDING"
            elif edge_30d > 0 and edge_7d < edge_30d * 0.60:
                new_status = "DEGRADING"
            else:
                new_status = "HEALTHY"

            self.state["model_stats"][model] = {
                "status":      new_status,
                "edge_30d":    round(edge_30d, 4),
                "edge_7d":     round(edge_7d, 4),
                "edge_3d":     round(stats["edge_3d"], 4),
                "n_trades":    stats["n_trades"],
                "last_update": time.time(),
            }

        # ── Rebalanceia pesos ─────────────────────────────────────────────────
        self._rebalance_weights()
        self._save_state()

        return self.get_weights()

    def _calc_model_stats(self, trades: list) -> dict:
        """Calcula edge em janelas múltiplas para um modelo."""
        def _edge(subset):
            if len(subset) < 2:
                return 0.0
            wins = sum(1 for t in subset if (t.get("pnl_usd") or 0) > 0)
            losses = len(subset) - wins
            avg_win  = sum((t.get("pnl_usd") or 0) for t in subset if (t.get("pnl_usd") or 0) > 0) / max(wins, 1)
            avg_loss = sum(abs(t.get("pnl_usd") or 0) for t in subset if (t.get("pnl_usd") or 0) <= 0) / max(losses, 1)
            wr = wins / len(subset)
            payoff = avg_win / avg_loss if avg_loss > 0 else 1.0
            return wr * payoff - (1 - wr)

        return {
            "edge_30d": _edge(trades[-30:]),
            "edge_7d":  _edge(trades[-7:]),
            "edge_3d":  _edge(trades[-3:]),
            "n_trades": len(trades),
        }

    def _rebalance_weights(self):
        """
        Redistribui pesos com base nos status dos modelos.
        Lógica:
          SUSPENDED  → peso 0
          RECOVERING → peso mínimo (MIN_WEIGHT)
          DEGRADING  → reduz 30%
          EXPANDING  → aumenta 20%
          HEALTHY    → mantém
        """
        current = self.state["weights"].copy()
        stats   = self.state["model_stats"]

        new_weights = {}
        for model in MODEL_NAMES:
            status = stats.get(model, {}).get("status", "HEALTHY")
            w = current.get(model, DEFAULT_WEIGHTS[model])

            if status == "SUSPENDED":
                w = 0.0
            elif status == "RECOVERING":
                w = MIN_WEIGHT
            elif status == "DEGRADING":
                w = max(MIN_WEIGHT, w * 0.70)
            elif status == "EXPANDING":
                w = min(MAX_WEIGHT, w * 1.20)
            # HEALTHY: mantém peso atual

            new_weights[model] = w

        # Normaliza para soma = 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {m: round(w / total, 4) for m, w in new_weights.items()}
        else:
            new_weights = DEFAULT_WEIGHTS.copy()

        self.state["weights"] = new_weights
        self.state["last_rebalance"] = time.time()

    def get_weights(self) -> dict:
        return self.state["weights"].copy()

    def get_model_stats(self) -> dict:
        return self.state["model_stats"].copy()

    def get_summary(self) -> dict:
        """Resumo para dashboard."""
        stats = self.state["model_stats"]
        weights = self.state["weights"]
        return {
            "weights": weights,
            "models": {
                m: {
                    "status":   stats[m]["status"],
                    "edge_7d":  stats[m]["edge_7d"],
                    "edge_30d": stats[m]["edge_30d"],
                    "weight":   weights.get(m, 0),
                }
                for m in MODEL_NAMES
            },
            "last_rebalance": self.state["last_rebalance"],
        }
