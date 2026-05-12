"""
SimulatedExecutionEngine
========================
Paper trading realista que modela o que aconteceria em produção.

Problemas do paper engine simples que isso resolve:
  - Fill imediato a preço exato → probabilidade de fill por tipo de ordem
  - Fee fixa taker → maker vs taker real
  - Sem spread → compra no ask, vende no bid
  - Sem slippage → slippage proporcional à volatilidade (ATR)
  - Sem partial fills → fill parcial com resto em fila
  - Sem timeout → ordens limit expiram após N ciclos
  - Sem min notional → rejeita ordens abaixo do mínimo OKX
  - Sem lot size → arredonda qty para precisão do par

Modos de operação:
  'market'   — buy()/sell() resolvem imediatamente (simula market order)
  'limit'    — buy()/sell() criam ordens pendentes, resolvidas no tick()
  'adaptive' — escolhe o modo pelo score/regime (default recomendado)

Interface:
  Compatível com PaperTradingEngine (buy/sell/portfolio_value/etc.)
  Acrescenta:
    submit_order(...)  → enfileira ordem com todos os parâmetros
    tick(prices, vols) → processa pendentes, aplica timeouts
    pending_summary()  → lista ordens aguardando fill
    execution_stats()  → estatísticas de fill rate, slippage médio, etc.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from paper_trading.engine import PaperTradingEngine

# Import opcional — só usado se OKXTradingClient estiver disponível
try:
    from exchange.okx_trading import OKXTradingClient, InstrumentCache
    _HAS_TRADING_CLIENT = True
except ImportError:
    _HAS_TRADING_CLIENT = False


# ── Taxas OKX (conta regular, sem tier de volume) ────────────────────────────
MAKER_FEE = 0.001   # 0.10%
TAKER_FEE = 0.004   # 0.40%

# ── Parâmetros por símbolo (OKX lot size / min notional) ─────────────────────
SYMBOL_PARAMS = {
    "BTC-USDT": {"min_usd": 5.0,  "qty_decimals": 6,  "price_decimals": 1},
    "ETH-USDT": {"min_usd": 5.0,  "qty_decimals": 5,  "price_decimals": 2},
    "SOL-USDT": {"min_usd": 5.0,  "qty_decimals": 3,  "price_decimals": 3},
    "BNB-USDT": {"min_usd": 5.0,  "qty_decimals": 4,  "price_decimals": 2},
    "XRP-USDT": {"min_usd": 5.0,  "qty_decimals": 2,  "price_decimals": 5},
    "DEFAULT":  {"min_usd": 5.0,  "qty_decimals": 4,  "price_decimals": 4},
}

# ── Probabilidade de fill por tipo de ordem (por ciclo de 15 min) ─────────────
#
# Derivado de análise de microestrutura em crypto (BTC/ETH spot OKX):
#   - Market: quase sempre preenche, risco real é falha de API (~0.5%)
#   - Passive limit 2×spread abaixo do ask: ~65% de chance de fill em 15min
#     (o preço vem até nós ~65% das vezes dentro de 15min em condições normais)
#   - Staggered T1 (1×spread): ~78%, T2 (pullback 0.3%): ~52%, T3 (momentum): ~42%
#   - Em alta volatilidade: passive limit fill_prob cai (o mercado salta o nível)
#   - Em baixa volatilidade: passive limit fill_prob sobe (mercado consolida)
#
FILL_PROB = {
    "market":         0.995,
    "passive_limit":  0.65,
    "staggered_t1":   0.78,
    "staggered_t2":   0.52,
    "staggered_t3":   0.42,
}

# Multiplicador de fill_prob por volatilidade (atr_pct relativo ao par)
# atr_pct alto → mercado salta níveis → limit menos provável de ser atingido
def _vol_fill_adj(atr_pct: float) -> float:
    if atr_pct <= 0.005: return 1.15   # vol muito baixa: mais fill
    if atr_pct <= 0.010: return 1.00   # vol normal
    if atr_pct <= 0.020: return 0.85   # vol alta: limit pode ser pulado
    if atr_pct <= 0.040: return 0.70
    return 0.55                         # vol extrema: passive limit raramente preenche


# ── Slippage por volatilidade ─────────────────────────────────────────────────
#
# Market orders sofrem slippage porque consomem liquidez.
# Em alta volatilidade, o book se afasta antes do fill.
# Limit orders não têm slippage de preço (você definiu o preço),
# mas podem não preencher — o custo é opportunity cost, não slippage.
#
def _calc_slippage(order_type: str, atr_pct: float, spread_pct: float) -> float:
    if order_type == "limit":
        return 0.0  # sem slippage em limit — custo é o fill_prob
    # Market: slippage = f(spread, vol)
    base_slip = spread_pct * 0.5
    vol_slip  = atr_pct * 0.08   # 8% do ATR como slippage adicional em market
    return min(base_slip + vol_slip, 0.005)  # cap em 0.5%


# ── Partial fill ──────────────────────────────────────────────────────────────
#
# Quando uma ordem é executada, raramente preenche 100% em um tick.
# Modelamos com Beta(α, β) para capturar a assimetria realista:
#   - Fills pequenos (30–60%) são raros mas possíveis
#   - Fills grandes (80–100%) são os mais comuns
#
def _partial_fill_pct(order_type: str) -> float:
    if order_type == "market":
        # Market orders: fill quase completo (95–100%)
        return min(1.0, max(0.90, random.betavariate(20, 1)))
    else:
        # Limit: fill parcial mais provável (60–100%)
        return min(1.0, max(0.30, random.betavariate(5, 2)))


@dataclass
class PendingOrder:
    order_id:    str
    side:        str      # 'BUY' | 'SELL'
    symbol:      str
    usd_amount:  float    # USD alvo
    qty:         float    # qty alvo (para sells)
    limit_price: float    # preço limite
    order_type:  str      # 'market' | 'passive_limit' | 'staggered_t1' | ...
    strategy:    str
    atr_pct:     float    # ATR% do par no momento da ordem
    spread_pct:  float
    submitted_at: float   # timestamp unix
    expires_at:  float    # timestamp unix (timeout)
    cycles_alive: int = 0
    max_cycles:   int = 4   # passive_limit expira em 4 ciclos (~1h)
    # Fill parcial acumulado
    filled_usd:   float = 0.0
    filled_qty:   float = 0.0

    def is_expired(self) -> bool:
        return self.cycles_alive >= self.max_cycles or time.time() > self.expires_at

    def remaining_usd(self) -> float:
        return max(0.0, self.usd_amount - self.filled_usd)

    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)


class SimulatedExecutionEngine(PaperTradingEngine):
    """
    Drop-in substituto do PaperTradingEngine com simulação realista de execução.

    Compatibilidade total com a interface existente:
      buy(symbol, usd_amount, price, strategy) → bool
      sell(symbol, qty, price, strategy) → bool
      portfolio_value() → float
      update_price(symbol, price)
      print_status()

    Novos métodos:
      submit_order(...)     — enfileira ordem com parâmetros completos
      tick(prices, vols)    — processa pendentes a cada ciclo
      pending_summary()     — lista ordens aguardando
      execution_stats()     — estatísticas de fill/slippage
    """

    def __init__(
        self,
        initial_balance_usd: float = 10000.0,
        default_order_mode: str = "adaptive",
        default_spread_pct: float = 0.0002,
        seed: Optional[int] = None,
        trading_client=None,   # OKXTradingClient — fornece precisão real de instrumentos
    ):
        super().__init__(initial_balance_usd)
        self.default_order_mode  = default_order_mode
        self.default_spread_pct  = default_spread_pct
        # Cliente de trading real — usado apenas para ler dados de precisão de instrumento
        # Nenhuma ordem real é enviada; o bot continua em paper trading
        self._trading_client = trading_client
        self.pending_orders: list[PendingOrder] = []
        self._order_counter      = 0
        self._exec_stats = {
            "submitted": 0,
            "filled_full": 0,
            "filled_partial": 0,
            "rejected_min": 0,
            "expired_timeout": 0,
            "rejected_fill_prob": 0,
            "total_slippage_usd": 0.0,
            "total_maker_fee":    0.0,
            "total_taker_fee":    0.0,
            "fill_prices":        [],  # (ideal_price, actual_price) for slippage tracking
        }
        if seed is not None:
            random.seed(seed)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _sym_params(self, symbol: str) -> dict:
        return SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS["DEFAULT"])

    def _round_qty(self, symbol: str, qty: float) -> float:
        # Usa InstrumentCache do OKXTradingClient se disponível (lotSz real da OKX)
        if self._trading_client is not None and _HAS_TRADING_CLIENT:
            try:
                inst_id = symbol if "-USDT" in symbol else symbol + "T"  # BTC-USD → BTC-USDT
                self._trading_client._ensure_instrument(inst_id)
                return self._trading_client._instruments.round_qty(inst_id, qty)
            except Exception:
                pass
        decimals = self._sym_params(symbol)["qty_decimals"]
        factor   = 10 ** decimals
        return math.floor(qty * factor) / factor

    def _round_price(self, symbol: str, price: float) -> float:
        if self._trading_client is not None and _HAS_TRADING_CLIENT:
            try:
                inst_id = symbol if "-USDT" in symbol else symbol + "T"
                self._trading_client._ensure_instrument(inst_id)
                return self._trading_client._instruments.round_price(inst_id, price)
            except Exception:
                pass
        decimals = self._sym_params(symbol)["price_decimals"]
        return round(price, decimals)

    def _check_min_notional(self, symbol: str, usd_amount: float) -> bool:
        if self._trading_client is not None and _HAS_TRADING_CLIENT:
            try:
                inst_id = symbol if "-USDT" in symbol else symbol + "T"
                self._trading_client._ensure_instrument(inst_id)
                min_sz  = self._trading_client._instruments.min_size(inst_id)
                # min_sz é em unidade base (BTC); converte para USD via usd_amount/qty
                # Usamos min_usd hardcoded como fallback conservador
            except Exception:
                pass
        return usd_amount >= self._sym_params(symbol)["min_usd"]

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"SIM-{int(time.time())}-{self._order_counter:04d}"

    def _select_order_type(
        self,
        side: str,
        atr_pct: float,
        score: float = 0.5,
        regime: str = "",
    ) -> str:
        if self.default_order_mode == "market":
            return "market"
        if self.default_order_mode == "limit":
            return "passive_limit"
        # adaptive: escolha baseada em regime e vol
        if regime in ("PANIC_LIQUIDATION", "LIQUIDITY_VACUUM"):
            return "market"
        if atr_pct > 0.025:
            return "market"    # alta vol → market (limit pode ser saltado)
        if score >= 0.68:
            return "staggered_t1"
        return "passive_limit"

    def _calc_limit_price(
        self,
        side: str,
        mid_price: float,
        order_type: str,
        spread_pct: float,
    ) -> float:
        """Preço da ordem limit relativo ao mid. BUY abaixo do ask, SELL acima do bid."""
        if order_type == "market":
            # Market: paga o spread imediatamente
            if side == "BUY":
                return mid_price * (1 + spread_pct / 2)
            else:
                return mid_price * (1 - spread_pct / 2)

        if side == "BUY":
            offsets = {
                "passive_limit":  -spread_pct * 2.0,   # bem abaixo do ask
                "staggered_t1":   -spread_pct * 1.0,   # quasi-passivo
                "staggered_t2":   -0.003,               # pullback fixo 0.3%
                "staggered_t3":   +0.002,               # momentum acima
            }
        else:
            offsets = {
                "passive_limit":  +spread_pct * 2.0,
                "staggered_t1":   +spread_pct * 1.0,
                "staggered_t2":   +0.003,
                "staggered_t3":   -0.002,
            }
        offset = offsets.get(order_type, -spread_pct)
        return mid_price * (1 + offset)

    def _calc_actual_fill_price(
        self,
        side: str,
        limit_price: float,
        order_type: str,
        atr_pct: float,
        spread_pct: float,
    ) -> tuple[float, float, str]:
        """
        Retorna (fill_price, fee_rate, fee_type).
        Para market: aplica slippage.
        Para limit: sem slippage, mas verifica maker vs taker.
        """
        slippage = _calc_slippage(order_type, atr_pct, spread_pct)

        if order_type == "market":
            if side == "BUY":
                fill_price = limit_price * (1 + slippage)
            else:
                fill_price = limit_price * (1 - slippage)
            return fill_price, TAKER_FEE, "taker"

        # Limit: se o mercado veio até nós, pode ter virado taker se chegamos ao preço
        # Modelamos: 20% de chance de ser executado como taker (cruzou o spread)
        is_taker = random.random() < 0.20
        fee_rate = TAKER_FEE if is_taker else MAKER_FEE
        fee_type = "taker" if is_taker else "maker"
        return limit_price, fee_rate, fee_type

    # ── Interface principal ────────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        usd_amount: float,
        price: float,
        strategy: str,
        order_type: str = "auto",
        atr_pct: float = 0.0,
        spread_pct: float = 0.0,
        score: float = 0.5,
        regime: str = "",
    ) -> bool:
        """
        Compra com simulação realista.
        Compatível com PaperTradingEngine.buy() — parâmetros extras são opcionais.
        """
        if spread_pct <= 0:
            spread_pct = self.default_spread_pct
        if order_type == "auto":
            order_type = self._select_order_type("BUY", atr_pct, score, regime)

        # 1. Min notional
        if not self._check_min_notional(symbol, usd_amount):
            self._exec_stats["rejected_min"] += 1
            print(f"[SIM] BUY {symbol} rejeitado: ${usd_amount:.2f} abaixo min notional "
                  f"${self._sym_params(symbol)['min_usd']:.2f}")
            return False

        self._exec_stats["submitted"] += 1

        # 2. Fill probability
        fill_prob_base = FILL_PROB.get(order_type, 0.65)
        fill_prob      = min(0.999, fill_prob_base * _vol_fill_adj(atr_pct))

        if random.random() > fill_prob:
            # Ordem não preenchida neste ciclo — cria pendente se for limit
            if order_type != "market":
                self._enqueue_pending("BUY", symbol, usd_amount, 0.0, price, order_type,
                                      strategy, atr_pct, spread_pct)
                print(f"[SIM] BUY {symbol} ${usd_amount:.2f} → pendente "
                      f"({order_type}, fill_prob={fill_prob:.0%})")
            else:
                self._exec_stats["rejected_fill_prob"] += 1
                print(f"[SIM] BUY {symbol} market falhou (API timeout simulado)")
            return False

        # 3. Partial fill
        fill_pct    = _partial_fill_pct(order_type)
        filled_usd  = usd_amount * fill_pct
        remainder   = usd_amount - filled_usd

        # 4. Preço de fill com spread e slippage
        limit_price = self._calc_limit_price("BUY", price, order_type, spread_pct)
        fill_price, fee_rate, fee_type = self._calc_actual_fill_price(
            "BUY", limit_price, order_type, atr_pct, spread_pct
        )
        fill_price = self._round_price(symbol, fill_price)

        # 5. Qty com lot size rounding
        qty = self._round_qty(symbol, filled_usd / fill_price)
        if qty <= 0:
            self._exec_stats["rejected_min"] += 1
            return False

        actual_usd = qty * fill_price

        # 6. Executa no engine base (com fee real maker/taker)
        fee = actual_usd * fee_rate
        total_cost = actual_usd + fee
        if total_cost > self.balance_usd:
            print(f"[SIM] BUY {symbol} rejeitado: saldo insuficiente "
                  f"(need ${total_cost:.2f}, have ${self.balance_usd:.2f})")
            return False

        # Aplica estado diretamente (bypass do buy() do pai para usar fee correta)
        prev_qty    = self.holdings.get(symbol, 0)
        prev_entry  = self.entry_prices.get(symbol, 0)
        total_qty   = prev_qty + qty
        self.entry_prices[symbol] = (
            (prev_qty * prev_entry + qty * fill_price) / total_qty
        )
        self.balance_usd         -= total_cost
        self.total_fees_usd      += fee
        self.holdings[symbol]     = total_qty

        # Estatísticas
        slippage_usd = abs(fill_price - price) * qty
        self._exec_stats["total_slippage_usd"] += slippage_usd
        if fee_type == "maker":
            self._exec_stats["total_maker_fee"] += fee
        else:
            self._exec_stats["total_taker_fee"] += fee
        self._exec_stats["fill_prices"].append((price, fill_price))
        if fill_pct >= 0.99:
            self._exec_stats["filled_full"] += 1
        else:
            self._exec_stats["filled_partial"] += 1

        # Log trade
        self._log_trade("BUY", symbol, qty, fill_price, actual_usd, strategy, fee,
                        extra=f"| {fee_type} {fee_rate:.3%} | slip={slippage_usd:.3f} | fill={fill_pct:.0%}")

        # Resto como ordem pendente (partial fill)
        if remainder > self._sym_params(symbol)["min_usd"] and order_type != "market":
            self._enqueue_pending("BUY", symbol, remainder, 0.0, price, order_type,
                                  strategy, atr_pct, spread_pct)

        self._save_state()
        return True

    def sell(
        self,
        symbol: str,
        qty: float,
        price: float,
        strategy: str,
        order_type: str = "auto",
        atr_pct: float = 0.0,
        spread_pct: float = 0.0,
        score: float = 0.5,
        regime: str = "",
    ) -> bool:
        """
        Venda com simulação realista.
        Compatível com PaperTradingEngine.sell().
        """
        if spread_pct <= 0:
            spread_pct = self.default_spread_pct
        if order_type == "auto":
            order_type = self._select_order_type("SELL", atr_pct, score, regime)

        held = self.holdings.get(symbol, 0)
        qty  = min(qty, held)
        if qty <= 1e-10:
            print(f"[SIM] SELL {symbol} negado: sem posição (held={held:.8f})")
            return False

        usd_amount = qty * price
        if not self._check_min_notional(symbol, usd_amount):
            self._exec_stats["rejected_min"] += 1
            return False

        self._exec_stats["submitted"] += 1

        # Fill probability
        fill_prob_base = FILL_PROB.get(order_type, 0.65)
        fill_prob      = min(0.999, fill_prob_base * _vol_fill_adj(atr_pct))

        if random.random() > fill_prob:
            if order_type != "market":
                self._enqueue_pending("SELL", symbol, 0.0, qty, price, order_type,
                                      strategy, atr_pct, spread_pct)
                print(f"[SIM] SELL {symbol} {qty:.6f} → pendente "
                      f"({order_type}, fill_prob={fill_prob:.0%})")
            else:
                self._exec_stats["rejected_fill_prob"] += 1
            return False

        # Partial fill
        fill_pct    = _partial_fill_pct(order_type)
        filled_qty  = self._round_qty(symbol, qty * fill_pct)
        if filled_qty <= 0:
            return False

        # Preço de fill (sell a bid — spread vai contra nós)
        limit_price = self._calc_limit_price("SELL", price, order_type, spread_pct)
        fill_price, fee_rate, fee_type = self._calc_actual_fill_price(
            "SELL", limit_price, order_type, atr_pct, spread_pct
        )
        fill_price = self._round_price(symbol, fill_price)

        gross       = filled_qty * fill_price
        fee         = gross * fee_rate
        net         = gross - fee

        # Atualiza estado
        self.holdings[symbol] = held - filled_qty
        if self.holdings[symbol] < 1e-10:
            del self.holdings[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]
        self.balance_usd    += net
        self.total_fees_usd += fee

        # Estatísticas
        slippage_usd = abs(fill_price - price) * filled_qty
        self._exec_stats["total_slippage_usd"] += slippage_usd
        if fee_type == "maker":
            self._exec_stats["total_maker_fee"] += fee
        else:
            self._exec_stats["total_taker_fee"] += fee
        self._exec_stats["fill_prices"].append((price, fill_price))
        if fill_pct >= 0.99:
            self._exec_stats["filled_full"] += 1
        else:
            self._exec_stats["filled_partial"] += 1

        self._log_trade("SELL", symbol, filled_qty, fill_price, net, strategy, fee,
                        extra=f"| {fee_type} {fee_rate:.3%} | slip={slippage_usd:.3f} | fill={fill_pct:.0%}")

        # Resto como pendente
        remainder_qty = qty - filled_qty
        if remainder_qty > 1e-8 and order_type != "market":
            self._enqueue_pending("SELL", symbol, 0.0, remainder_qty, price,
                                  order_type, strategy, atr_pct, spread_pct)

        self._save_state()
        return True

    # ── Ordem pendente (limit deferred) ───────────────────────────────────────

    def _enqueue_pending(
        self,
        side: str,
        symbol: str,
        usd_amount: float,
        qty: float,
        limit_price: float,
        order_type: str,
        strategy: str,
        atr_pct: float,
        spread_pct: float,
        max_cycles: int = 4,
    ):
        order = PendingOrder(
            order_id      = self._next_order_id(),
            side          = side,
            symbol        = symbol,
            usd_amount    = usd_amount,
            qty           = qty,
            limit_price   = limit_price,
            order_type    = order_type,
            strategy      = strategy,
            atr_pct       = atr_pct,
            spread_pct    = spread_pct,
            submitted_at  = time.time(),
            expires_at    = time.time() + max_cycles * 900,   # 900s = 1 ciclo de 15min
            max_cycles    = max_cycles,
        )
        self.pending_orders.append(order)

    def tick(self, prices: dict, volatilities: dict = None) -> list:
        """
        Processa ordens pendentes contra os preços atuais.
        Deve ser chamado uma vez por ciclo (a cada 15 minutos).

        prices      — {symbol: mid_price}
        volatilities — {symbol: atr_pct} (opcional)

        Retorna lista de eventos: [{"type": "filled"|"expired", "order_id": ...}, ...]
        """
        if volatilities is None:
            volatilities = {}

        events       = []
        still_active = []

        for order in self.pending_orders:
            order.cycles_alive += 1
            symbol       = order.symbol
            current_mid  = prices.get(symbol)

            # Sem preço disponível: mantém pendente
            if current_mid is None or current_mid <= 0:
                still_active.append(order)
                continue

            # Timeout
            if order.is_expired():
                self._exec_stats["expired_timeout"] += 1
                events.append({
                    "type": "expired",
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": order.side,
                    "cycles": order.cycles_alive,
                })
                print(f"[SIM] ⏱ {order.side} {symbol} {order.order_id} expirou "
                      f"após {order.cycles_alive} ciclos")
                continue

            # Verifica se o preço chegou à ordem limit
            atr_pct = volatilities.get(symbol, order.atr_pct)
            hit = self._check_limit_hit(order, current_mid)
            if not hit:
                still_active.append(order)
                continue

            # Tenta preencher
            if order.side == "BUY":
                ok = self.buy(
                    symbol, order.usd_amount, current_mid, order.strategy,
                    order_type=order.order_type, atr_pct=atr_pct,
                    spread_pct=order.spread_pct,
                )
            else:
                ok = self.sell(
                    symbol, order.qty, current_mid, order.strategy,
                    order_type=order.order_type, atr_pct=atr_pct,
                    spread_pct=order.spread_pct,
                )

            if ok:
                events.append({
                    "type": "filled",
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": order.side,
                    "price": current_mid,
                    "cycles_waited": order.cycles_alive,
                })
            else:
                # Parcialmente executada ou rejeitada — descarta (já re-enfileirou se parcial)
                pass

        self.pending_orders = still_active
        return events

    def _check_limit_hit(self, order: PendingOrder, current_mid: float) -> bool:
        """Verifica se o mercado chegou ao preço limit da ordem."""
        lp = order.limit_price
        if order.side == "BUY":
            # BUY limit: queremos comprar abaixo do ask; preenche quando ask <= limit_price
            # Aproximação: mid <= limit_price + 0.5×spread
            return current_mid <= lp * (1 + order.spread_pct)
        else:
            # SELL limit: queremos vender acima do bid; preenche quando bid >= limit_price
            return current_mid >= lp * (1 - order.spread_pct)

    # ── Override do log ────────────────────────────────────────────────────────

    def _log_trade(
        self,
        side: str,
        symbol: str,
        qty: float,
        price: float,
        usd: float,
        strategy: str,
        fee: float = 0.0,
        extra: str = "",
    ):
        from colorama import Fore
        trade = {
            "time":     datetime.now().isoformat(),
            "side":     side,
            "symbol":   symbol,
            "qty":      qty,
            "price":    price,
            "usd":      usd,
            "fee":      fee,
            "strategy": strategy,
            "pnl_usd":  None,   # calculado pós-sell no app.py
        }
        # Calcula PnL para SELLs
        if side == "SELL" and symbol in self.entry_prices:
            entry = self.entry_prices.get(symbol, price)
            trade["pnl_usd"] = round((price - entry) * qty - fee, 4)

        self.trades.append(trade)
        color = Fore.GREEN if side == "BUY" else Fore.YELLOW
        print(color + f"[SIM] {side} {qty:.6f} {symbol} @ ${price:.2f} = ${usd:.2f} "
              f"fee=${fee:.4f} {extra} [{strategy}]")

    # ── Consultas ─────────────────────────────────────────────────────────────

    def pending_summary(self) -> list:
        """Lista ordens pendentes formatadas para o dashboard."""
        return [
            {
                "order_id":    o.order_id,
                "side":        o.side,
                "symbol":      o.symbol,
                "usd":         round(o.usd_amount, 2),
                "qty":         round(o.qty, 6),
                "limit_price": round(o.limit_price, 4),
                "order_type":  o.order_type,
                "cycles_alive": o.cycles_alive,
                "max_cycles":  o.max_cycles,
                "strategy":    o.strategy,
            }
            for o in self.pending_orders
        ]

    def execution_stats(self) -> dict:
        """Estatísticas acumuladas de qualidade de execução."""
        s = self._exec_stats
        fill_prices = s["fill_prices"]
        total_filled = s["filled_full"] + s["filled_partial"]
        fill_rate = total_filled / s["submitted"] if s["submitted"] > 0 else 0

        avg_slippage_bps = 0.0
        if fill_prices:
            slippages = [abs(fp - ip) / ip * 10000 for ip, fp in fill_prices]
            avg_slippage_bps = sum(slippages) / len(slippages)

        return {
            "submitted":          s["submitted"],
            "filled_full":        s["filled_full"],
            "filled_partial":     s["filled_partial"],
            "rejected_min":       s["rejected_min"],
            "expired_timeout":    s["expired_timeout"],
            "rejected_fill_prob": s["rejected_fill_prob"],
            "fill_rate":          round(fill_rate, 3),
            "avg_slippage_bps":   round(avg_slippage_bps, 2),
            "total_slippage_usd": round(s["total_slippage_usd"], 4),
            "total_maker_fee":    round(s["total_maker_fee"], 4),
            "total_taker_fee":    round(s["total_taker_fee"], 4),
            "pending_count":      len(self.pending_orders),
        }
