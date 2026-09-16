"""Fair Value Gap detection with causal mitigation tracking.

A bullish FVG is a 3-candle sequence where candle1.high < candle3.low; the gap zone is
[candle1.high, candle3.low]. Bearish is the mirror: candle1.low > candle3.high, zone
[candle3.high, candle1.low]. The gap is *known* at the close of candle 3 (index i).

Mitigation states, in order:
  touched_index   first later candle whose range enters the zone
  filled_index    first later candle that trades through the zone's far edge
  invalid_index   first later candle that CLOSES beyond the far edge (gap is dead for entries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .bars import as_arrays
from .structure import Direction


@dataclass
class FVG:
    index: int               # candle 3 index (gap becomes known at its close)
    direction: Direction
    top: float
    bottom: float
    mid_high: float          # middle candle's high / low: reference stop placement
    mid_low: float
    touched_index: int | None = None
    filled_index: int | None = None
    invalid_index: int | None = None
    expired_index: int | None = None   # dropped from tracking for age, not by price action
    id: int = field(default=-1)

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def near_edge(self) -> float:
        """Edge price first reaches on a pullback."""
        return self.top if self.direction is Direction.BULLISH else self.bottom

    @property
    def far_edge(self) -> float:
        return self.bottom if self.direction is Direction.BULLISH else self.top

    def is_unmitigated_as_of(self, i: int) -> bool:
        """True if no candle with index <= i has touched the zone."""
        return self.touched_index is None or self.touched_index > i

    def is_live_as_of(self, i: int) -> bool:
        """Known, not invalidated by candle i (touching is fine; closing through is not), not expired."""
        return (self.index <= i and (self.invalid_index is None or self.invalid_index > i)
                and (self.expired_index is None or self.expired_index > i))


def _gap_from_candles(i: int, high: Sequence[float], low: Sequence[float]) -> FVG | None:
    h1, l1 = high[i - 2], low[i - 2]
    h2, l2 = high[i - 1], low[i - 1]
    h3, l3 = high[i], low[i]
    if h1 < l3:
        return FVG(i, Direction.BULLISH, top=float(l3), bottom=float(h1), mid_high=float(h2), mid_low=float(l2))
    if l1 > h3:
        return FVG(i, Direction.BEARISH, top=float(l1), bottom=float(h3), mid_high=float(h2), mid_low=float(l2))
    return None


def _update_mitigation(gap: FVG, k: int, high: Sequence[float], low: Sequence[float], close: Sequence[float]) -> None:
    """Apply candle k (> gap.index) to the gap's mitigation state."""
    h, l, c = high[k], low[k], close[k]
    if gap.direction is Direction.BULLISH:
        touched = l <= gap.top
        filled = l <= gap.bottom
        invalid = c < gap.bottom
    else:
        touched = h >= gap.bottom
        filled = h >= gap.top
        invalid = c > gap.top
    if touched and gap.touched_index is None:
        gap.touched_index = k
    if filled and gap.filled_index is None:
        gap.filled_index = k
    if invalid and gap.invalid_index is None:
        gap.invalid_index = k


class FVGState:
    """Incremental FVG tracker. Feed candles in order via `step`."""

    def __init__(self, min_size: float = 0.0, max_age: int | None = None) -> None:
        """`max_age`: a gap older than this many candles stops being tracked. Bounds the work per
        candle on long series; the signal engine never trades gaps outside its arm window anyway."""
        self.gaps: list[FVG] = []
        self._live: list[FVG] = []
        self.min_size = min_size
        self.max_age = max_age
        self._next_id = 0

    def step(self, high: Sequence[float], low: Sequence[float], close: Sequence[float], i: int) -> FVG | None:
        """Update mitigation for existing gaps with candle i, then detect a new gap ending at i.

        Only the live (not yet invalidated) gaps are scanned; dead gaps are moved out of the
        hot list so long series stay linear."""
        still_live = []
        for gap in self._live:
            if self.max_age is not None and i - gap.index > self.max_age:
                gap.expired_index = i
                continue
            if gap.index < i:
                _update_mitigation(gap, i, high, low, close)
            if gap.invalid_index is None:
                still_live.append(gap)
        self._live = still_live
        if i < 2:
            return None
        gap = _gap_from_candles(i, high, low)
        if gap is None or gap.size < self.min_size:
            return None
        gap.id = self._next_id
        self._next_id += 1
        self.gaps.append(gap)
        self._live.append(gap)
        return gap

    def live_gaps(self, i: int, direction: Direction | None = None) -> list[FVG]:
        return [g for g in self._live if g.is_live_as_of(i) and (direction is None or g.direction is direction)]


def detect_fvgs(df: pd.DataFrame, min_size: float = 0.0) -> list[FVG]:
    _, high, low, close = as_arrays(df)
    state = FVGState(min_size=min_size)
    for i in range(len(df)):
        state.step(high, low, close, i)
    return state.gaps


def unmitigated_gaps_as_of(gaps: list[FVG], i: int) -> list[FVG]:
    """Gaps known at i whose zone no candle <= i has touched."""
    return [g for g in gaps if g.index <= i and g.is_unmitigated_as_of(i)]
