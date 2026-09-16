"""Bar series conventions shared by every driver (backtest, replay, live).

A bar frame is a pandas DataFrame with:
  - a tz-aware DatetimeIndex in America/New_York, named "time", holding the bar OPEN time
  - float columns open, high, low, close, volume
  - sorted ascending, no duplicate timestamps

Everything downstream assumes these invariants; `ensure_bars` enforces them once at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from functools import lru_cache

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"
COLUMNS = ["open", "high", "low", "close", "volume"]

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


def ensure_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a bar frame. Returns a new frame; never mutates the input."""
    if df is None or len(df) == 0:
        out = pd.DataFrame(columns=COLUMNS, dtype="float64")
        out.index = pd.DatetimeIndex([], tz=NY_TZ, name="time")
        return out
    out = df.copy()
    if "time" in out.columns:
        out = out.set_index("time")
    elif "timestamp" in out.columns:
        out = out.set_index("timestamp")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert(NY_TZ)
    out.index.name = "time"
    missing = [c for c in COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"bar frame missing columns: {missing}")
    out = out[COLUMNS].astype("float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    bad = (out["high"] < out["low"]) | (out["high"] < out[["open", "close"]].max(axis=1)) | (
        out["low"] > out[["open", "close"]].min(axis=1)
    )
    if bad.any():
        raise ValueError(f"{int(bad.sum())} bars violate OHLC ordering, first at {out.index[bad][0]}")
    return out


def rth_only(df: pd.DataFrame, open_t: time = RTH_OPEN, close_t: time = RTH_CLOSE) -> pd.DataFrame:
    """Keep bars whose OPEN time is within [open_t, close_t)."""
    t = df.index.time
    mask = (t >= open_t) & (t < close_t)
    return df[mask]


def resample_bars(df: pd.DataFrame, minutes: int, session_open: time = RTH_OPEN) -> pd.DataFrame:
    """Aggregate finer bars into `minutes`-wide bars aligned to the session open.

    Alignment matters: a 60-minute bar must run 9:30-10:30, not 9:00-10:00, or the first
    hour's structure is wrong. Bars never span a session boundary because bins are computed
    per day from that day's session open.
    """
    if len(df) == 0:
        return ensure_bars(df)
    day = df.index.normalize()
    open_offset = pd.Timedelta(hours=session_open.hour, minutes=session_open.minute)
    since_open = (df.index - (day + open_offset)).total_seconds() // 60
    bin_no = np.floor(since_open / minutes).astype("int64")
    bin_start = day + open_offset + pd.to_timedelta(bin_no * minutes, unit="m")
    agg = df.groupby(bin_start, sort=True).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg.index.name = "time"
    return ensure_bars(agg)


@lru_cache(maxsize=65536)
def bar_close_time(open_time: pd.Timestamp, minutes: int, session_close: time = RTH_CLOSE) -> pd.Timestamp:
    """When a bar that opened at `open_time` is known to be complete (capped at the session close)."""
    close = open_time + _MINUTE_DELTAS.setdefault(minutes, pd.Timedelta(minutes=minutes))
    if close.time() > session_close or close.date() != open_time.date():
        return open_time.replace(hour=session_close.hour, minute=session_close.minute, second=0, microsecond=0, nanosecond=0)
    return close


_MINUTE_DELTAS: dict[int, pd.Timedelta] = {}


@dataclass(frozen=True)
class Candle:
    """A single bar with its positional index; used by detectors that need both."""

    index: int
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


def candle_at(df: pd.DataFrame, i: int) -> Candle:
    row = df.iloc[i]
    return Candle(i, df.index[i], float(row.open), float(row.high), float(row.low), float(row.close))


def as_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
    )
