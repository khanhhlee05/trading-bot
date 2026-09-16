"""Broker abstraction used by the live daemon. Alpaca implements it; FakeBroker is for tests
and dry runs. Kept deliberately small: quotes, one entry, one exit, positions, flatten."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    time: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else max(self.bid, self.ask)

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)


@dataclass(frozen=True)
class ContractQuote:
    """One option contract as seen in a chain snapshot."""

    symbol: str            # OCC symbol, e.g. SPY240607C00530000
    underlying: str
    expiry: date
    strike: float
    is_call: bool
    bid: float
    ask: float
    delta: float | None
    iv: float | None
    open_interest: int | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else max(self.bid, self.ask)

    @property
    def spread_frac(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else float("inf")


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str              # "buy" | "sell"
    qty: int
    status: str            # alpaca status string
    filled_qty: int = 0
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None
    raw: dict = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status == "filled" and self.filled_qty == self.qty


@dataclass
class BrokerPosition:
    symbol: str
    qty: int
    avg_entry_price: float
    asset_class: str
    market_value: float | None = None


class Broker(Protocol):
    def is_paper(self) -> bool: ...
    def equity(self) -> float: ...
    def market_open(self) -> bool: ...
    def underlying_quote(self, symbol: str) -> Quote: ...
    def option_chain(self, underlying: str, is_call: bool, expiry_from: date, expiry_to: date,
                     strike_lo: float, strike_hi: float) -> list[ContractQuote]: ...
    def submit_option_market(self, contract: str, qty: int, side: str, position_intent: str, client_order_id: str) -> OrderResult: ...
    def submit_stock_market(self, symbol: str, qty: int, side: str, client_order_id: str) -> OrderResult: ...
    def order_status(self, order_id: str) -> OrderResult: ...
    def cancel_all(self) -> None: ...
    def positions(self) -> list[BrokerPosition]: ...
    def close_all_positions(self) -> None: ...
    def today_minute_bars(self, symbol: str, feed: str) -> pd.DataFrame: ...


class FakeBroker:
    """Deterministic broker for tests and dry runs: fills instantly at the quote's touch."""

    def __init__(self, equity: float = 100_000.0, paper: bool = True) -> None:
        self._equity = equity
        self._paper = paper
        self.quotes: dict[str, Quote] = {}
        self.chains: dict[str, list[ContractQuote]] = {}
        self.orders: list[OrderResult] = []
        self._positions: dict[str, BrokerPosition] = {}
        self.bars: dict[str, pd.DataFrame] = {}
        self.open = True
        self._n = 0

    def is_paper(self) -> bool:
        return self._paper

    def equity(self) -> float:
        return self._equity

    def market_open(self) -> bool:
        return self.open

    def underlying_quote(self, symbol: str) -> Quote:
        return self.quotes[symbol]

    def option_chain(self, underlying, is_call, expiry_from, expiry_to, strike_lo, strike_hi):
        return [c for c in self.chains.get(underlying, []) if c.is_call == is_call and expiry_from <= c.expiry <= expiry_to
                and strike_lo <= c.strike <= strike_hi]

    def _fill(self, symbol: str, qty: int, side: str, price: float, asset_class: str) -> OrderResult:
        self._n += 1
        res = OrderResult(f"fake-{self._n}", symbol, side, qty, "filled", qty, price, datetime.now())
        self.orders.append(res)
        pos = self._positions.get(symbol)
        signed = qty if side == "buy" else -qty
        if pos is None:
            self._positions[symbol] = BrokerPosition(symbol, signed, price, asset_class)
        else:
            pos.qty += signed
            if pos.qty == 0:
                del self._positions[symbol]
        return res

    def submit_option_market(self, contract, qty, side, position_intent, client_order_id):
        cq = next(c for chain in self.chains.values() for c in chain if c.symbol == contract)
        price = cq.ask if side == "buy" else cq.bid
        return self._fill(contract, qty, side, price, "us_option")

    def submit_stock_market(self, symbol, qty, side, client_order_id):
        q = self.quotes[symbol]
        return self._fill(symbol, qty, side, q.ask if side == "buy" else q.bid, "us_equity")

    def order_status(self, order_id):
        return next(o for o in self.orders if o.order_id == order_id)

    def cancel_all(self):
        pass

    def positions(self):
        return list(self._positions.values())

    def close_all_positions(self):
        for sym, pos in list(self._positions.items()):
            side = "sell" if pos.qty > 0 else "buy"
            if pos.asset_class == "us_option":
                self.submit_option_market(sym, abs(pos.qty), side, "sell_to_close", "flatten")
            else:
                self.submit_stock_market(sym, abs(pos.qty), side, "flatten")

    def today_minute_bars(self, symbol, feed):
        return self.bars.get(symbol, pd.DataFrame())
