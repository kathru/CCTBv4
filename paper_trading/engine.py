import os
import json
import time
from datetime import datetime
from colorama import Fore, init

init(autoreset=True)

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "engine_state.json")


TAKER_FEE = 0.004   # 0.40% — OKX Regular taker fee


class PaperTradingEngine:
    def __init__(self, initial_balance_usd: float = 10000.0):
        self.initial_balance = initial_balance_usd
        self.balance_usd = initial_balance_usd
        self.holdings = {}
        self.entry_prices = {}
        self.trades = []
        self.prices = {}
        self.total_fees_usd = 0.0   # total de taxas pagas
        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    s = json.load(f)
                self.initial_balance  = s.get("initial_balance", s.get("balance_usd", self.initial_balance))
                self.balance_usd     = s.get("balance_usd", self.initial_balance)
                self.holdings        = s.get("holdings", {})
                self.entry_prices    = s.get("entry_prices", {})
                self.trades          = s.get("trades", [])
                self.total_fees_usd  = s.get("total_fees_usd", 0.0)
                # Recalcula entry_prices faltantes a partir do histórico de trades
                self._recalc_missing_entry_prices()
                print(Fore.CYAN + f"[PAPER] Estado restaurado — USD: ${self.balance_usd:.2f} | Holdings: {self.holdings}")
        except Exception as e:
            print(Fore.YELLOW + f"[PAPER] Sem estado salvo, iniciando do zero. ({e})")

    def _recalc_missing_entry_prices(self):
        """Reconstrói preço médio de entrada para posições sem entry_price salvo."""
        for symbol, qty_held in self.holdings.items():
            if symbol in self.entry_prices:
                continue
            total_qty = 0.0
            total_cost = 0.0
            for t in self.trades:
                if t.get("symbol") != symbol:
                    continue
                if t.get("side") == "BUY":
                    total_qty   += t.get("qty", 0)
                    total_cost  += t.get("usd", 0)
                elif t.get("side") == "SELL":
                    total_qty   -= t.get("qty", 0)
                    total_cost  -= t.get("usd", 0)
            if total_qty > 1e-10:
                self.entry_prices[symbol] = total_cost / total_qty
        self._save_state()

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "initial_balance": self.initial_balance,
                    "balance_usd":     self.balance_usd,
                    "holdings":        self.holdings,
                    "entry_prices":    self.entry_prices,
                    "trades":          self.trades[-200:],
                    "total_fees_usd":  self.total_fees_usd,
                    "saved_at":        datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            print(Fore.RED + f"[PAPER] Erro ao salvar estado: {e}")

    def update_price(self, symbol: str, price: float):
        self.prices[symbol] = price

    def buy(self, symbol: str, usd_amount: float, price: float, strategy: str) -> bool:
        fee = usd_amount * TAKER_FEE
        total_cost = usd_amount + fee
        if total_cost > self.balance_usd:
            print(Fore.RED + f"[PAPER] COMPRA negada: saldo insuficiente (${self.balance_usd:.2f})")
            return False
        qty = usd_amount / price
        prev_qty = self.holdings.get(symbol, 0)
        prev_entry = self.entry_prices.get(symbol, 0)
        total_qty = prev_qty + qty
        self.entry_prices[symbol] = ((prev_qty * prev_entry) + (qty * price)) / total_qty
        self.balance_usd -= total_cost
        self.total_fees_usd += fee
        self.holdings[symbol] = total_qty
        self._log_trade("BUY", symbol, qty, price, usd_amount, strategy, fee)
        self._save_state()
        return True

    def sell(self, symbol: str, qty: float, price: float, strategy: str) -> bool:
        held = self.holdings.get(symbol, 0)
        # Tolera epsilon de ponto flutuante (acumulação entre múltiplos slots)
        qty = min(qty, held)
        if qty <= 1e-10:
            print(Fore.RED + f"[PAPER] VENDA negada: sem {symbol} disponível (held={held:.10f})")
            return False
        gross = qty * price
        fee = gross * TAKER_FEE
        net_received = gross - fee
        self.holdings[symbol] = held - qty
        if self.holdings[symbol] < 1e-10:
            del self.holdings[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]
        self.balance_usd += net_received
        self.total_fees_usd += fee
        self._log_trade("SELL", symbol, qty, price, net_received, strategy, fee)
        self._save_state()
        return True

    def _log_trade(self, side: str, symbol: str, qty: float, price: float, usd: float, strategy: str, fee: float = 0.0):
        trade = {
            "time": datetime.now().isoformat(),
            "side": side, "symbol": symbol,
            "qty": qty, "price": price, "usd": usd,
            "fee": fee, "strategy": strategy,
        }
        self.trades.append(trade)
        color = Fore.GREEN if side == "BUY" else Fore.YELLOW
        print(color + f"[PAPER] {side} {qty:.6f} {symbol} @ ${price:.2f} = ${usd:.2f} | taxa: ${fee:.4f} [{strategy}]")

    def portfolio_value(self) -> float:
        total = self.balance_usd
        for symbol, qty in self.holdings.items():
            total += qty * self.prices.get(symbol, 0)
        return total

    def print_status(self):
        pnl = self.portfolio_value() - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100
        color = Fore.GREEN if pnl >= 0 else Fore.RED
        print(f"\n{'='*50}")
        print(f"  Saldo USD:      ${self.balance_usd:.2f}")
        for symbol, qty in self.holdings.items():
            price = self.prices.get(symbol, 0)
            print(f"  {symbol}:          {qty:.6f} (${qty * price:.2f})")
        print(color + f"  Portfolio:      ${self.portfolio_value():.2f}")
        print(color + f"  P&L:            ${pnl:.2f} ({pnl_pct:+.2f}%)")
        print(f"  Trades:         {len(self.trades)}")
        print(f"{'='*50}\n")
