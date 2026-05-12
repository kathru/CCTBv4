"""
FeeModel — Fonte Única de Verdade para Custos de Execução
==========================================================
Centraliza todos os custos de execução para que Signal Engine, Paper Engine,
Backtest, Validate e Calibrate usem exatamente os mesmos números.

OKX Spot — Conta Regular (sem volume tier):
  Maker: 0.10%  (ordens limit que adicionam liquidez ao book)
  Taker: 0.40%  (ordens market ou limit que removem liquidez)

Round-trip típico para este bot:
  Entrada: limit passiva → maker (0.10%)
  Saída:   market/urgência → taker (0.40%)
  Total:   0.50% em fees
  + slippage market estimado: 0.05–0.15% (depende de ATR)
  = custo real médio: 0.55–0.65% round-trip

Por que isso importa:
  Com ciclo de 15 minutos, mesmo 10 trades/dia × 0.60% = 6%/dia em custos.
  EV calculado com fee errada invalida todo o sizing Kelly e threshold de score.

Uso:
  from strategies.fee_model import FEE

  # Fee round-trip para EV do signal engine
  cost = FEE.signal_ev_cost(rr=2.0, entry="limit", exit="market")

  # Fee por lado para paper engine / backtest
  fee_buy  = usd_amount * FEE.entry("limit")
  fee_sell = proceeds  * FEE.exit("market")

  # Slippage para market order
  slip = FEE.slippage_market(atr_pct=0.015)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _FeeModel:
    """
    Modelo de custos imutável. Instanciado uma vez como FEE.

    Parâmetros:
      maker              — fee para ordens limit passivas (adiciona liquidez)
      taker              — fee para ordens market / limit agressivas
      slippage_base      — slippage base para market orders (spread / 2)
      slippage_vol_coef  — fator adicional: atr_pct × coef = slippage extra
      slippage_limit     — residual de slippage em limit (normalmente 0)
    """
    exchange:           str   = "OKX"
    tier:               str   = "regular"   # sem volume discount
    maker:              float = 0.001       # 0.10%
    taker:              float = 0.004       # 0.40%
    slippage_base:      float = 0.0002      # 0.02% base (bid/ask spread/2)
    slippage_vol_coef:  float = 0.08        # atr_pct × 0.08 = vol slippage
    slippage_limit:     float = 0.0001      # 0.01% residual em limit

    # ── Fee por lado ──────────────────────────────────────────────────────────

    def entry(self, order_type: str = "limit") -> float:
        """Fee de entrada por lado (fração do notional)."""
        return self.maker if order_type == "limit" else self.taker

    def exit(self, order_type: str = "market") -> float:
        """Fee de saída por lado (fração do notional)."""
        return self.maker if order_type == "limit" else self.taker

    # ── Slippage ──────────────────────────────────────────────────────────────

    def slippage_market(self, atr_pct: float = 0.0, spread_pct: float = 0.0) -> float:
        """
        Slippage estimado para market order.
        Depende do ATR atual (volatilidade) e spread bid/ask.
        """
        base = max(self.slippage_base, spread_pct * 0.5)
        vol  = atr_pct * self.slippage_vol_coef
        return min(base + vol, 0.005)   # cap em 0.5% — acima disso é LIQUIDITY_VACUUM

    def slippage_limit(self, atr_pct: float = 0.0) -> float:
        """Residual de slippage em limit order (oportunidade de não fill)."""
        return self.slippage_limit   # constante baixa — custo real é o fill_prob

    # ── Round-trip ────────────────────────────────────────────────────────────

    def round_trip(
        self,
        entry_type: str = "limit",
        exit_type:  str = "market",
        atr_pct:    float = 0.0,
        spread_pct: float = 0.0,
    ) -> float:
        """
        Custo total round-trip como fração do notional.
        Inclui fees de entrada + saída + slippage de saída.

        Cenários típicos:
          limit/market (padrão):  0.001 + 0.004 + ~0.0005 = 0.0055 (0.55%)
          market/market (panic):  0.004 + 0.004 + ~0.0010 = 0.0090 (0.90%)
          limit/limit (ideal):    0.001 + 0.001 + ~0.0001 = 0.0021 (0.21%)
        """
        fee_entry = self.entry(entry_type)
        fee_exit  = self.exit(exit_type)
        slip_exit = (self.slippage_market(atr_pct, spread_pct)
                     if exit_type == "market"
                     else self.slippage_limit)
        return fee_entry + fee_exit + slip_exit

    # ── EV cost (para o Signal Engine) ───────────────────────────────────────

    def signal_ev_cost(
        self,
        rr:         float = 2.0,
        entry_type: str   = "limit",
        exit_type:  str   = "market",
        atr_pct:    float = 0.0,
        spread_pct: float = 0.0,
    ) -> float:
        """
        Custo para usar na fórmula EV do Signal Engine:
          ev = p × rr - (1-p) × 1.0 - fee_rate × (1 + rr)

        A fórmula multiplica fee_rate por (1 + rr) para capturar que
        a fee incide sobre a saída que pode ser maior (TP) ou menor (SL).
        Para que fee_rate × (1 + rr) == custo_real_round_trip:
          fee_rate = round_trip / (1 + rr)

        Com rr=2.0, entry=limit, exit=market (cenário padrão):
          round_trip ≈ 0.0055
          fee_rate   = 0.0055 / 3.0 ≈ 0.00183
        """
        rt = self.round_trip(entry_type, exit_type, atr_pct, spread_pct)
        return rt / (1 + rr) if rr > 0 else rt

    # ── Backtest / validate cost ──────────────────────────────────────────────

    def backtest_round_trip(self, entry_type: str = "limit", exit_type: str = "market") -> float:
        """
        Custo round-trip simplificado para backtest/validate/calibrate.
        Sem slippage variável — usa slippage conservador fixo.
        """
        return (self.entry(entry_type) + self.exit(exit_type) +
                self.slippage_base * 2)   # spread nos dois lados

    # ── Resumo ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "exchange":   self.exchange,
            "tier":       self.tier,
            "maker_pct":  f"{self.maker * 100:.3f}%",
            "taker_pct":  f"{self.taker * 100:.3f}%",
            "round_trip_typical_pct": f"{self.round_trip() * 100:.3f}%",
            "round_trip_panic_pct":   f"{self.round_trip('market','market') * 100:.3f}%",
            "signal_ev_cost_rr2":     round(self.signal_ev_cost(rr=2.0), 6),
            "backtest_rt_pct":        f"{self.backtest_round_trip() * 100:.3f}%",
        }


# ── Instância canônica — importe isso em vez de definir constantes locais ────
FEE = _FeeModel()

# Aliases de compatibilidade para módulos que importavam TAKER_FEE / MAKER_FEE
MAKER_FEE = FEE.maker   # 0.001
TAKER_FEE = FEE.taker   # 0.004
