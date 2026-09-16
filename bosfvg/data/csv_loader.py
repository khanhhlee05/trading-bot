"""Load bars from CSV. Accepts `time|timestamp|date, open, high, low, close[, volume]`.
Timestamps without a timezone are assumed to be America/New_York."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.bars import NY_TZ, ensure_bars


def load_csv_bars(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("time", "timestamp", "date", "datetime") if c in cols), None)
    if tcol is None:
        raise ValueError("csv needs a time/timestamp/date column")
    ts = pd.to_datetime(df[tcol])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ)
    df = df.rename(columns={cols.get(k, k): k for k in ("open", "high", "low", "close", "volume")})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df.index = pd.DatetimeIndex(ts, name="time")
    return ensure_bars(df[["open", "high", "low", "close", "volume"]])
