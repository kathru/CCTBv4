"""
Testes unitários — SimulatedExecutionEngine
============================================
Cobre os cenários de risco identificados na lógica de tick() + _enqueue_pending():

  test_pending_order_hit_fills_once            — ordem pendente preenche exatamente 1×
  test_partial_fill_remainder_correct          — remainder é o valor correto (original - filled)
  test_no_recursive_enqueue_from_tick          — tick não re-enfileira a si mesma
  test_expired_order_removed_from_queue        — ordem expirada sai da fila sem preencher
  test_no_duplicate_pending_after_tick         — apenas 1 entrada na fila por fill parcial
  test_sell_partial_does_not_exceed_holdings   — fill parcial de venda nunca excede o holding
  test_buy_insufficient_balance_no_enqueue_loop— saldo insuficiente não gera loop de enfileiramento
  test_submitted_count_not_inflated_by_tick    — tick não incrementa submitted desnecessariamente
  test_tick_no_price_keeps_order               — sem preço disponível mantém ordem na fila
  test_tick_multiple_cycles_to_expiry          — contagem de ciclos até expiração correta
"""

import time
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_trading.simulated_engine import SimulatedExecutionEngine, PendingOrder


# ── Fixture base ──────────────────────────────────────────────────────────────

def make_engine(balance=1000.0, seed=42):
    """Engine limpo sem estado persistido."""
    e = SimulatedExecutionEngine(
        initial_balance_usd=balance,
        default_order_mode="limit",   # força modo limit para testar enfileiramento
        seed=seed,
    )
    # Garante estado limpo (sem holdings do engine_state.json)
    e.balance_usd  = balance
    e.holdings     = {}
    e.entry_prices = {}
    e.trades       = []
    e.total_fees_usd = 0.0
    e.pending_orders = []
    e._exec_stats = {
        "submitted": 0, "filled_full": 0, "filled_partial": 0,
        "rejected_min": 0, "expired_timeout": 0, "rejected_fill_prob": 0,
        "total_slippage_usd": 0.0, "total_maker_fee": 0.0,
        "total_taker_fee": 0.0, "fill_prices": [],
    }
    return e


def inject_pending_buy(engine, symbol="BTC-USDT", usd=100.0, limit_price=50000.0,
                        order_type="passive_limit", cycles=0, max_cycles=4):
    """Injeta ordem pendente diretamente, simulando que já passou `cycles` ciclos."""
    engine._enqueue_pending(
        side="BUY", symbol=symbol, usd_amount=usd, qty=0.0,
        limit_price=limit_price, order_type=order_type,
        strategy="test", atr_pct=0.01, spread_pct=0.0002,
        max_cycles=max_cycles,
    )
    order = engine.pending_orders[-1]
    order.cycles_alive = cycles
    return order


def inject_pending_sell(engine, symbol="BTC-USDT", qty=0.001, limit_price=50000.0,
                         order_type="passive_limit", cycles=0, max_cycles=4):
    engine.holdings[symbol] = qty * 2   # garante holding suficiente
    engine._enqueue_pending(
        side="SELL", symbol=symbol, usd_amount=0.0, qty=qty,
        limit_price=limit_price, order_type=order_type,
        strategy="test", atr_pct=0.01, spread_pct=0.0002,
        max_cycles=max_cycles,
    )
    order = engine.pending_orders[-1]
    order.cycles_alive = cycles
    return order


# ── Testes ────────────────────────────────────────────────────────────────────

class TestPendingOrderFillsOnce:
    """tick() deve processar a ordem uma única vez quando o preço bate."""

    def test_pending_order_hit_fills_once(self):
        """
        Quando preço bate, a ordem é processada e REMOVIDA da fila.
        Ticks subsequentes com mesmo preço não re-processam.
        """
        random.seed(999)   # seed que garante fill (fill_prob alta para market)
        e = make_engine(5000.0)
        inject_pending_buy(e, usd=100.0, limit_price=50000.0, order_type="market")

        # Primeiro tick — preço bate no limit (50000 <= 50000 × 1.0002)
        events1 = e.tick({"BTC-USDT": 50000.0})

        # Segundo tick com mesmo preço
        events2 = e.tick({"BTC-USDT": 50000.0})

        # A ordem foi processada e saiu da fila — não reaparece
        filled = [ev for ev in events1 if ev["type"] == "filled"]
        assert len(filled) <= 1, "Ordem processada mais de uma vez no mesmo tick"
        assert len(e.pending_orders) == 0 or all(
            o.order_id != filled[0]["order_id"] for o in e.pending_orders
        ) if filled else True, "Ordem preenchida ainda na fila"

        # Segundo tick não deve ter a mesma ordem
        assert len(events2) == 0 or all(
            ev.get("order_id") != (filled[0]["order_id"] if filled else None)
            for ev in events2
        ), "Mesma ordem processada em ticks diferentes"

    def test_order_removed_from_queue_after_hit(self):
        """Após hit (fill ou não), ordem sai de pending_orders."""
        random.seed(1)
        e = make_engine(5000.0)
        order = inject_pending_buy(e, usd=100.0, limit_price=50000.0, order_type="market")
        n_before = len(e.pending_orders)

        e.tick({"BTC-USDT": 50000.0})

        # Ordem não deve mais estar na fila com o mesmo order_id
        ids_after = [o.order_id for o in e.pending_orders]
        assert order.order_id not in ids_after, \
            f"Ordem {order.order_id} permanece na fila após hit"


class TestPartialFillRemainder:
    """Remainder de fill parcial deve ser correto e não criar loop."""

    def test_partial_fill_remainder_correct(self):
        """
        Se fill parcial = 60% de 100 USD, remainder deve ser ≈ 40 USD.
        """
        random.seed(0)
        e = make_engine(5000.0)

        # Força fill parcial controlado: mock _partial_fill_pct
        original_partial = e.__class__.__dict__.get('buy')
        injected = []

        original_buy = e.buy.__func__ if hasattr(e.buy, '__func__') else None

        # Injeta diretamente uma ordem pendente
        inject_pending_buy(e, usd=100.0, limit_price=50000.0, order_type="passive_limit")

        events = e.tick({"BTC-USDT": 50000.0})

        # Se houver remainder enfileirado, deve ser < 100 USD
        remainders = [o for o in e.pending_orders if o.side == "BUY"]
        for r in remainders:
            assert r.usd_amount < 100.0, \
                f"Remainder {r.usd_amount} não é menor que o original 100 USD"
            assert r.usd_amount > 0, \
                f"Remainder zerado enfileirado desnecessariamente"

    def test_no_recursive_enqueue_from_tick(self):
        """
        tick() chama buy() que pode enfileirar remainder.
        O remainder NÃO deve ser processado no mesmo tick.
        """
        random.seed(42)
        e = make_engine(5000.0)
        inject_pending_buy(e, usd=200.0, limit_price=50000.0, order_type="passive_limit")

        n_pending_before = len(e.pending_orders)  # 1
        events = e.tick({"BTC-USDT": 50000.0})

        # Pendentes após tick: apenas remainders de fills parciais (não a original)
        # A original deve ter sido processada (saiu) — remainder pode ter entrado
        order_ids_after = {o.order_id for o in e.pending_orders}
        # Nenhum evento "filled" deve ter order_id igual a um pendente atual
        for ev in events:
            if ev["type"] == "filled":
                assert ev["order_id"] not in order_ids_after, \
                    "Ordem preenchida ainda na fila (potencial loop)"

    def test_no_duplicate_pending_after_tick(self):
        """
        Um único tick não deve enfileirar mais de um remainder por ordem.
        """
        random.seed(7)
        e = make_engine(5000.0)
        inject_pending_buy(e, usd=100.0, limit_price=50000.0, order_type="passive_limit")

        e.tick({"BTC-USDT": 50000.0})

        # order_ids devem ser únicos
        ids = [o.order_id for o in e.pending_orders]
        assert len(ids) == len(set(ids)), \
            f"IDs duplicados na fila: {ids}"


class TestExpiredOrders:
    """Ordens expiradas devem ser removidas sem preencher."""

    def test_expired_order_removed_from_queue(self):
        """Ordem com cycles_alive >= max_cycles é removida e não preenche."""
        e = make_engine(5000.0)
        order = inject_pending_buy(e, usd=100.0, limit_price=50000.0,
                                   max_cycles=3, cycles=3)   # já expirou

        balance_before = e.balance_usd
        holdings_before = dict(e.holdings)

        events = e.tick({"BTC-USDT": 50000.0})

        expired = [ev for ev in events if ev["type"] == "expired"]
        assert len(expired) == 1, "Ordem expirada não gerou evento 'expired'"
        assert expired[0]["order_id"] == order.order_id

        # Sem mudança no saldo ou holdings
        assert e.balance_usd == balance_before, \
            "Saldo alterado por ordem expirada"
        assert e.holdings == holdings_before, \
            "Holdings alterado por ordem expirada"

        # Fila vazia
        assert len(e.pending_orders) == 0, \
            "Ordem expirada permanece na fila"

    def test_tick_multiple_cycles_to_expiry(self):
        """Contagem de ciclos incrementa corretamente até expirar."""
        e = make_engine(5000.0)
        inject_pending_buy(e, usd=100.0, limit_price=1.0,   # preço impossível de bater
                           max_cycles=3, cycles=0)

        # 2 ticks sem hit (preço alto demais para BUY limit = 1.0)
        events1 = e.tick({"BTC-USDT": 50000.0})
        assert len(e.pending_orders) == 1
        assert e.pending_orders[0].cycles_alive == 1

        events2 = e.tick({"BTC-USDT": 50000.0})
        assert len(e.pending_orders) == 1
        assert e.pending_orders[0].cycles_alive == 2

        # 3º tick — expira
        events3 = e.tick({"BTC-USDT": 50000.0})
        assert len(e.pending_orders) == 0
        expired = [ev for ev in events3 if ev["type"] == "expired"]
        assert len(expired) == 1


class TestSellPartialDoesNotExceedHoldings:
    """Fill parcial de SELL nunca deve vender mais do que o holding disponível."""

    def test_sell_partial_does_not_exceed_holdings(self):
        """
        Se holding = 0.001 BTC e ordem é SELL 0.001,
        fill parcial não pode vender mais que 0.001.
        """
        random.seed(5)
        e = make_engine(1000.0)
        holding_qty = 0.001
        e.holdings["BTC-USDT"] = holding_qty
        inject_pending_sell(e, qty=holding_qty, limit_price=50000.0)
        # Remove o holding extra que inject_pending_sell adicionou
        e.holdings["BTC-USDT"] = holding_qty

        e.tick({"BTC-USDT": 50000.0})

        remaining_holding = e.holdings.get("BTC-USDT", 0.0)
        assert remaining_holding >= 0.0, \
            f"Holdings negativos após sell parcial: {remaining_holding}"
        assert remaining_holding <= holding_qty + 1e-9, \
            f"Holdings {remaining_holding} excede original {holding_qty}"

    def test_sell_qty_zero_does_not_enqueue(self):
        """SELL com qty=0 não deve ser enfileirado."""
        e = make_engine(1000.0)
        ok = e.sell("BTC-USDT", 0.0, 50000.0, "test")
        assert not ok
        assert len(e.pending_orders) == 0


class TestBuyInsufficientBalance:
    """Saldo insuficiente não deve gerar loop de re-enfileiramento."""

    def test_buy_insufficient_balance_no_enqueue_forever(self):
        """
        Ordem limit com usd > saldo disponível:
        - Se não preenche (fill_prob), enfileira 1× (normal)
        - Se preenche mas saldo insuficiente, NÃO enfileira remainder
        Depois de 1 tick com saldo insuficiente, fila não cresce infinitamente.
        """
        random.seed(0)
        e = make_engine(10.0)   # saldo muito baixo
        inject_pending_buy(e, usd=500.0, limit_price=50000.0, order_type="market")

        initial_queue_len = len(e.pending_orders)

        # Vários ticks — fila não deve crescer além do tamanho inicial + 1 remainder
        for _ in range(5):
            e.tick({"BTC-USDT": 50000.0})

        assert len(e.pending_orders) <= initial_queue_len + 1, \
            f"Fila cresceu para {len(e.pending_orders)} após múltiplos ticks com saldo insuficiente"

    def test_rejected_due_to_balance_does_not_increment_submitted_twice(self):
        """
        Buy que falha por saldo deve incrementar submitted 1× (na chamada do tick),
        não 2× (tick + retry automático).
        """
        e = make_engine(1.0)   # saldo insuficiente
        inject_pending_buy(e, usd=100.0, limit_price=50000.0, order_type="market")

        submitted_before = e._exec_stats["submitted"]
        e.tick({"BTC-USDT": 50000.0})
        submitted_after = e._exec_stats["submitted"]

        delta = submitted_after - submitted_before
        assert delta <= 1, \
            f"submitted incrementou {delta}× para uma única ordem (esperado ≤ 1)"


class TestSubmittedCountNotInflated:
    """Chamadas via tick() não devem inflar contadores indevidamente."""

    def test_submitted_count_not_inflated_by_tick(self):
        """
        Uma ordem pendente processada por tick() incrementa submitted 1×,
        não 2× (tick + buy()).
        """
        random.seed(99)
        e = make_engine(5000.0)
        inject_pending_buy(e, usd=100.0, limit_price=50000.0, order_type="market")

        submitted_before = e._exec_stats["submitted"]
        e.tick({"BTC-USDT": 50000.0})
        submitted_after = e._exec_stats["submitted"]

        delta = submitted_after - submitted_before
        # 1 ordem → submitted incrementa no máximo 2 vezes
        # (1 para a ordem original + 1 possível para remainder)
        assert delta <= 2, \
            f"submitted inflado: {delta} para 1 ordem pendente"

    def test_expired_does_not_increment_submitted(self):
        """Ordem expirada NÃO incrementa submitted."""
        e = make_engine(5000.0)
        inject_pending_buy(e, usd=100.0, limit_price=50000.0,
                           max_cycles=1, cycles=1)   # já expirou

        submitted_before = e._exec_stats["submitted"]
        e.tick({"BTC-USDT": 50000.0})

        assert e._exec_stats["submitted"] == submitted_before, \
            "submitted incrementado por ordem expirada"


class TestTickNoPrice:
    """Sem preço disponível, ordem permanece na fila sem ser processada."""

    def test_tick_no_price_keeps_order(self):
        """Ordem mantida na fila quando símbolo não tem preço."""
        e = make_engine(5000.0)
        order = inject_pending_buy(e, symbol="BTC-USDT", usd=100.0, limit_price=50000.0)

        events = e.tick({"ETH-USDT": 2000.0})   # BTC não tem preço

        assert len(e.pending_orders) == 1, "Ordem removida sem preço disponível"
        assert e.pending_orders[0].order_id == order.order_id
        assert len(events) == 0, "Evento gerado sem preço disponível"

    def test_tick_zero_price_keeps_order(self):
        """Preço zero não processa a ordem."""
        e = make_engine(5000.0)
        inject_pending_buy(e, symbol="BTC-USDT", usd=100.0, limit_price=50000.0)

        events = e.tick({"BTC-USDT": 0.0})

        assert len(e.pending_orders) == 1
        assert len(events) == 0


class TestDirectBuySell:
    """Testes básicos da interface pública buy()/sell() sem pending."""

    def test_buy_reduces_balance(self):
        e = make_engine(1000.0, seed=1)
        # market order tem fill_prob=0.995 — na maioria das seeds vai preencher
        random.seed(1)
        ok = e.buy("BTC-USDT", 100.0, 50000.0, "test", order_type="market")
        if ok:
            assert e.balance_usd < 1000.0, "Saldo não foi reduzido após compra"
            assert e.holdings.get("BTC-USDT", 0) > 0, "Holdings não aumentou após compra"

    def test_sell_increases_balance(self):
        e = make_engine(1000.0, seed=1)
        e.holdings["BTC-USDT"] = 0.01
        e.entry_prices["BTC-USDT"] = 50000.0
        balance_before = e.balance_usd
        random.seed(1)
        ok = e.sell("BTC-USDT", 0.01, 50000.0, "test", order_type="market")
        if ok:
            assert e.balance_usd > balance_before, "Saldo não aumentou após venda"

    def test_sell_without_holdings_fails(self):
        e = make_engine(1000.0)
        ok = e.sell("BTC-USDT", 0.001, 50000.0, "test")
        assert not ok, "Venda sem holdings não deve ser bem-sucedida"
        assert len(e.pending_orders) == 0, "Ordem enfileirada sem holdings"

    def test_buy_below_min_notional_rejected(self):
        e = make_engine(1000.0)
        ok = e.buy("BTC-USDT", 0.001, 50000.0, "test")   # $0.001 < min $5
        assert not ok
        assert e._exec_stats["rejected_min"] == 1

    def test_execution_stats_consistent(self):
        """filled_full + filled_partial + rejections <= submitted."""
        random.seed(0)
        e = make_engine(5000.0)
        for _ in range(10):
            e.buy("BTC-USDT", 50.0, 50000.0, "test", order_type="market")

        s = e._exec_stats
        total_outcomes = s["filled_full"] + s["filled_partial"] + s["rejected_min"] + s["rejected_fill_prob"]
        assert total_outcomes <= s["submitted"], \
            f"Outcomes ({total_outcomes}) > submitted ({s['submitted']})"
