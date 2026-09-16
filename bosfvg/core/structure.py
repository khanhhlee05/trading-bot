"""Swing detection and Break of Structure / Change of Character.

Causality: a swing high at index j is only *known* at index j + lookback (the right-hand
confirmation candles must exist). Every function here exposes that confirmation index and
never uses a swing before it is confirmed, so running the detector on bars[:i] gives the same
events up to i as running it on the whole series.

Break rule: a level is broken by a CLOSE beyond it, not a wick.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from .bars import as_arrays


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.BULLISH else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.BEARISH if self is Direction.BULLISH else Direction.BULLISH


class Trend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    UNKNOWN = "unknown"


class EventKind(str, Enum):
    BOS = "bos"      # break in the direction of the current trend
    CHOCH = "choch"  # break against the current trend (trend flips)


@dataclass(frozen=True)
class Swing:
    index: int            # candle where the extreme printed
    confirmed_index: int  # first candle at which the swing is known
    price: float
    is_high: bool

    @property
    def time_known(self) -> int:
        return self.confirmed_index


@dataclass(frozen=True)
class StructureEvent:
    index: int              # candle whose close broke the level
    kind: EventKind
    direction: Direction
    level: float            # the swing price that was broken
    swing_index: int        # index of the broken swing
    origin_index: int       # index of the opposite swing the impulse started from (or -1)
    trend_after: Trend


def find_swings(high: np.ndarray, low: np.ndarray, lookback: int) -> list[Swing]:
    """Fractal swings: high[j] is a swing high if it is the strict max of high[j-lb : j+lb+1].

    Ties on the left side are allowed (>=) so a flat double-top still registers once, but a
    later equal high on the right side cancels it so the swing sits on the last extreme.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    n = len(high)
    swings: list[Swing] = []
    for j in range(lookback, n - lookback):
        h, l = high[j], low[j]
        left_h = high[j - lookback : j]
        right_h = high[j + 1 : j + lookback + 1]
        if (h >= left_h).all() and (h > right_h).all():
            swings.append(Swing(j, j + lookback, float(h), True))
        left_l = low[j - lookback : j]
        right_l = low[j + 1 : j + lookback + 1]
        if (l <= left_l).all() and (l < right_l).all():
            swings.append(Swing(j, j + lookback, float(l), False))
    swings.sort(key=lambda s: (s.confirmed_index, s.index, not s.is_high))
    return swings


@dataclass
class StructureState:
    """Incremental structure tracker. Feed candles in order via `step`."""

    lookback: int
    trend: Trend = Trend.UNKNOWN
    last_high: Swing | None = None   # most recent confirmed, unbroken swing high
    last_low: Swing | None = None
    prev_high: Swing | None = None   # the swing high before last_high (impulse origin for bearish)
    prev_low: Swing | None = None
    events: list[StructureEvent] | None = None
    swings: list[Swing] | None = None

    def __post_init__(self) -> None:
        self.events = [] if self.events is None else self.events
        self.swings = [] if self.swings is None else self.swings

    def _confirm_swings(self, high: Sequence[float], low: Sequence[float], i: int) -> None:
        """Register swings whose confirmation index is exactly i. Only touches a 2*lookback+1 window."""
        j = i - self.lookback
        if j < self.lookback:
            return
        lb = self.lookback
        h, l = high[j], low[j]
        left_h, right_h = high[j - lb : j], high[j + 1 : j + lb + 1]
        if all(h >= x for x in left_h) and all(h > x for x in right_h):
            s = Swing(j, i, float(h), True)
            self.swings.append(s)
            self.prev_high, self.last_high = self.last_high, s
        left_l, right_l = low[j - lb : j], low[j + 1 : j + lb + 1]
        if all(l <= x for x in left_l) and all(l < x for x in right_l):
            s = Swing(j, i, float(l), False)
            self.swings.append(s)
            self.prev_low, self.last_low = self.last_low, s

    def step(self, high: Sequence[float], low: Sequence[float], close: Sequence[float], i: int) -> StructureEvent | None:
        """Process candle i. Returns a structure event if candle i's close broke a level.

        `high/low/close` may be lists or arrays; only indices <= i are ever read."""
        self._confirm_swings(high, low, i)
        c = close[i]
        event: StructureEvent | None = None
        if self.last_high is not None and c > self.last_high.price:
            kind = EventKind.BOS if self.trend is Trend.BULLISH else EventKind.CHOCH
            origin = self.last_low.index if self.last_low is not None else -1
            self.trend = Trend.BULLISH
            event = StructureEvent(i, kind, Direction.BULLISH, self.last_high.price, self.last_high.index, origin, self.trend)
            self.last_high = None  # consumed: needs a fresh swing high to break again
        elif self.last_low is not None and c < self.last_low.price:
            kind = EventKind.BOS if self.trend is Trend.BEARISH else EventKind.CHOCH
            origin = self.last_high.index if self.last_high is not None else -1
            self.trend = Trend.BEARISH
            event = StructureEvent(i, kind, Direction.BEARISH, self.last_low.price, self.last_low.index, origin, self.trend)
            self.last_low = None
        if event is not None:
            self.events.append(event)
        return event

    def key_levels_above(self, price: float) -> list[float]:
        """Confirmed swing highs above `price`, ascending. Used for 'next key level' targets."""
        return sorted({s.price for s in self.swings if s.is_high and s.price > price})

    def key_levels_below(self, price: float) -> list[float]:
        return sorted({s.price for s in self.swings if not s.is_high and s.price < price}, reverse=True)


def detect_structure(df: pd.DataFrame, lookback: int = 3) -> tuple[list[StructureEvent], list[Swing], Trend]:
    """Run the incremental tracker over a whole frame."""
    _, high, low, close = as_arrays(df)
    state = StructureState(lookback=lookback)
    for i in range(len(df)):
        state.step(high, low, close, i)
    return state.events, state.swings, state.trend
