"""Alpaca implementation of the Broker protocol, paper-only by construction.

Facts baked in (verified against alpaca-py 0.44 and Alpaca's documented behaviour):
  - options: bracket / OCO / OTO order classes are rejected ("complex orders not supported for
    options trading"), so exits are managed by the daemon, not by resting broker orders
  - options: time_in_force must be DAY
  - free data plan: real-time stock quotes/bars are IEX-only; option chain is the 15-minute
    delayed "indicative" feed unless the account has an OPRA subscription
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from ..core.bars import NY_TZ, ensure_bars
from ..data.alpaca_hist import load_credentials
from .broker import BrokerPosition, ContractQuote, OrderResult, Quote

_OCC = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ(symbol: str) -> tuple[str, date, bool, float]:
    m = _OCC.match(symbol)
    if not m:
        raise ValueError(f"not an OCC option symbol: {symbol}")
    ymd = m.group("ymd")
    expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    return m.group("root"), expiry, m.group("cp") == "C", int(m.group("strike")) / 1000.0


def snapshot_to_contract(symbol: str, snap: Any, underlying: str) -> ContractQuote | None:
    root, expiry, is_call, strike = parse_occ(symbol)
    q = getattr(snap, "latest_quote", None)
    if q is None:
        return None
    greeks = getattr(snap, "greeks", None)
    return ContractQuote(
        symbol=symbol,
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        is_call=is_call,
        bid=float(q.bid_price or 0.0),
        ask=float(q.ask_price or 0.0),
        delta=float(greeks.delta) if greeks is not None and greeks.delta is not None else None,
        iv=float(snap.implied_volatility) if getattr(snap, "implied_volatility", None) is not None else None,
    )


def _order_result(o: Any) -> OrderResult:
    status = o.status.value if hasattr(o.status, "value") else str(o.status)
    side = o.side.value if hasattr(o.side, "value") else str(o.side)
    return OrderResult(
        order_id=str(o.id),
        symbol=o.symbol,
        side=side,
        qty=int(float(o.qty or 0)),
        status=status,
        filled_qty=int(float(o.filled_qty or 0)),
        filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price is not None else None,
        submitted_at=o.submitted_at,
    )


class AlpacaBroker:
    def __init__(self, trading_client: Any | None = None, stock_data: Any | None = None, option_data: Any | None = None,
                 options_feed: str | None = None) -> None:
        if trading_client is None or stock_data is None or option_data is None:
            from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
            from alpaca.trading.client import TradingClient

            key, secret, paper = load_credentials()
            if not paper:
                raise RuntimeError("ALPACA_PAPER must be true; this project never trades live")
            trading_client = trading_client or TradingClient(key, secret, paper=True)
            stock_data = stock_data or StockHistoricalDataClient(key, secret)
            option_data = option_data or OptionHistoricalDataClient(key, secret)
        self.trading = trading_client
        self.stock_data = stock_data
        self.option_data = option_data
        self.options_feed = options_feed
        self._paper = True

    # -------------------------------------------------------------- account

    def is_paper(self) -> bool:
        return self._paper

    def equity(self) -> float:
        return float(self.trading.get_account().equity)

    def market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    # -------------------------------------------------------------- data

    def underlying_quote(self, symbol: str) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest

        q = self.stock_data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        return Quote(symbol, float(q.bid_price), float(q.ask_price), q.timestamp)

    def today_minute_bars(self, symbol: str, feed: str) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        now_ny = pd.Timestamp.now(tz=NY_TZ)
        start = now_ny.normalize() + pd.Timedelta(hours=9, minutes=30)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                               start=start.to_pydatetime(), feed=DataFeed(feed))
        bars = self.stock_data.get_stock_bars(req).data.get(symbol, [])
        if not bars:
            return ensure_bars(pd.DataFrame())
        df = pd.DataFrame({"time": [b.timestamp for b in bars], "open": [b.open for b in bars], "high": [b.high for b in bars],
                           "low": [b.low for b in bars], "close": [b.close for b in bars], "volume": [b.volume for b in bars]})
        return ensure_bars(df)

    def option_chain(self, underlying: str, is_call: bool, expiry_from: date, expiry_to: date,
                     strike_lo: float, strike_hi: float) -> list[ContractQuote]:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionChainRequest
        from alpaca.trading.enums import ContractType

        req = OptionChainRequest(
            underlying_symbol=underlying,
            type=ContractType.CALL if is_call else ContractType.PUT,
            expiration_date_gte=expiry_from,
            expiration_date_lte=expiry_to,
            strike_price_gte=round(strike_lo, 2),
            strike_price_lte=round(strike_hi, 2),
            feed=OptionsFeed(self.options_feed) if self.options_feed else None,
        )
        snaps = self.option_data.get_option_chain(req)
        out = []
        for sym, snap in snaps.items():
            c = snapshot_to_contract(sym, snap, underlying)
            if c is not None:
                out.append(c)
        return out

    # -------------------------------------------------------------- orders

    def submit_option_market(self, contract: str, qty: int, side: str, position_intent: str, client_order_id: str) -> OrderResult:
        from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(symbol=contract, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY,
                                 position_intent=PositionIntent(position_intent), client_order_id=client_order_id)
        return _order_result(self.trading.submit_order(req))

    def submit_stock_market(self, symbol: str, qty: int, side: str, client_order_id: str) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY,
                                 client_order_id=client_order_id)
        return _order_result(self.trading.submit_order(req))

    def order_status(self, order_id: str) -> OrderResult:
        return _order_result(self.trading.get_order_by_id(order_id))

    def cancel_all(self) -> None:
        self.trading.cancel_orders()

    def positions(self) -> list[BrokerPosition]:
        out = []
        for p in self.trading.get_all_positions():
            qty = int(float(p.qty))
            if p.side.value == "short":
                qty = -abs(qty)
            out.append(BrokerPosition(p.symbol, qty, float(p.avg_entry_price), p.asset_class.value,
                                      float(p.market_value) if p.market_value is not None else None))
        return out

    def close_all_positions(self) -> None:
        self.trading.close_all_positions(cancel_orders=True)
