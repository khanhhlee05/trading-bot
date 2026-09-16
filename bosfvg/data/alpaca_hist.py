"""Historical minute bars from Alpaca's Market Data API, with the free-tier rule baked in.

Free (Basic) plan: SIP-consolidated *historical* bars are available as long as the request's
`end` is at least 15 minutes in the past. That is enough for every backtest in this project.
Real-time bars on the free plan are IEX-only, which is a different tape (see `alpaca_live`).

Pulled as 1-minute bars and cached per month; all coarser timeframes are built locally with
session-aligned resampling so HTF/LTF bars agree bar-for-bar with the live daemon's.
"""

from __future__ import annotations

import os
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..core.bars import ensure_bars, rth_only
from .cache import BarCache

FREE_TIER_DELAY = timedelta(minutes=16)   # 15 minutes plus a margin


def load_credentials() -> tuple[str, str, bool]:
    """Read ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_PAPER from the environment (or .env)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        pass
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() in ("1", "true", "yes")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required (see .env.example)")
    return key, secret, paper


def clamp_end_for_free_tier(end: datetime, now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    latest = now - FREE_TIER_DELAY
    return min(end, latest)


def _bars_to_frame(barset: Any, symbol: str) -> pd.DataFrame:
    rows = barset.data.get(symbol, []) if hasattr(barset, "data") else barset.get(symbol, [])
    if not rows:
        return ensure_bars(pd.DataFrame())
    df = pd.DataFrame(
        {
            "time": [b.timestamp for b in rows],
            "open": [b.open for b in rows],
            "high": [b.high for b in rows],
            "low": [b.low for b in rows],
            "close": [b.close for b in rows],
            "volume": [b.volume for b in rows],
        }
    )
    return ensure_bars(df)


class AlpacaHistory:
    def __init__(self, client: Any | None = None, cache: BarCache | None = None, feed: str = "sip",
                 sleep_between_calls: float = 0.35) -> None:
        """`client` is an alpaca.data.historical.StockHistoricalDataClient (or a test double)."""
        if client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            key, secret, _ = load_credentials()
            client = StockHistoricalDataClient(key, secret)
        self.client = client
        self.cache = cache or BarCache()
        self.feed = feed
        self.sleep = sleep_between_calls

    def fetch_month(self, symbol: str, month: str) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        start = pd.Timestamp(month + "-01", tz="UTC")
        end = (start + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59)
        end = pd.Timestamp(clamp_end_for_free_tier(end.to_pydatetime()))
        if end <= start:
            return ensure_bars(pd.DataFrame())
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            adjustment=Adjustment.SPLIT,
            feed=DataFeed(self.feed),
            limit=None,
        )
        barset = self.client.get_stock_bars(req)
        _time.sleep(self.sleep)
        return _bars_to_frame(barset, symbol)

    def get_minute_bars(self, symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp,
                        rth: bool = True, refresh_current_month: bool = True) -> pd.DataFrame:
        start_ts = pd.Timestamp(start, tz="America/New_York") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        end_ts = pd.Timestamp(end, tz="America/New_York") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        this_month = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m")
        parts = []
        for month in BarCache.months_between(start_ts, end_ts):
            is_current = month == this_month
            if self.cache.has(symbol, self.feed, month) and not (is_current and refresh_current_month):
                df = self.cache.load(symbol, self.feed, month)
            else:
                df = self.fetch_month(symbol, month)
                if len(df) and not is_current:
                    self.cache.store(symbol, self.feed, month, df)
                elif len(df) and is_current:
                    self.cache.store(symbol, self.feed, month, df)
            parts.append(df)
        if not parts:
            return ensure_bars(pd.DataFrame())
        df = ensure_bars(pd.concat(parts))
        df = df[(df.index >= start_ts) & (df.index <= end_ts + pd.Timedelta(days=1))]
        return rth_only(df) if rth else df
