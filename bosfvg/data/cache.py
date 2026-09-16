"""Parquet cache for minute bars, one file per symbol per month. Never re-pull a month twice."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.bars import ensure_bars

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data_cache"


class BarCache:
    def __init__(self, root: str | Path = DEFAULT_CACHE) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str, feed: str, month: str) -> Path:
        return self.root / f"{symbol.upper()}_{feed}_1min_{month}.parquet"

    def has(self, symbol: str, feed: str, month: str) -> bool:
        return self.path(symbol, feed, month).exists()

    def load(self, symbol: str, feed: str, month: str) -> pd.DataFrame:
        return ensure_bars(pd.read_parquet(self.path(symbol, feed, month)))

    def store(self, symbol: str, feed: str, month: str, df: pd.DataFrame) -> None:
        ensure_bars(df).to_parquet(self.path(symbol, feed, month))

    @staticmethod
    def months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        s = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)
        e = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)
        return [p.strftime("%Y-%m") for p in pd.period_range(s.to_period("M"), e.to_period("M"), freq="M")]
