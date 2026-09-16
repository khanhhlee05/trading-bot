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
        """Known and not invalidated by candle i (touching is fine; closing through is not)."""
        return self.index <= i and (self.invalid_index is None or self.invalid_index > i)


def _gap_from_candles(i: int, high: np.ndarray, low: np.ndarray) -> FVG | None:
    h1, l1 = high[i - 2], low[i - 2]
    h2, l2 = high[i - 1], low[i - 1]
    h3, l3 = high[i], low[i]
    if h1 < l3:
        return FVG(i, Direction.BULLISH, top=float(l3), bottom=float(h1), mid_high=float(h2), mid_low=float(l2))
    if l1 > h3:
        return FVG(i, Direction.BEARISH, top=float(l1), bottom=float(h3), mid_high=float(h2), mid_low=float(l2))
    return None


def _update_mitigation(gap: FVG, k: int, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> None:
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

    def __init__(self, min_size: float = 0.0) -> None:
        self.gaps: list[FVG] = []
        self.min_size = min_size
        self._next_id = 0

    def step(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int) -> FVG | None:
        """Update mitigation for existing gaps with candle i, then detect a new gap ending at i."""
        for gap in self.gaps:
            if gap.invalid_index is None and gap.index < i:
                _update_mitigation(gap, i, high, low, close)
        if i < 2:
            return None
        gap = _gap_from_candles(i, high, low)
        if gap is None or gap.size < self.min_size:
            return None
        gap.id = self._next_id
        self._next_id += 1
        self.gaps.append(gap)
        return gap

    def live_gaps(self, i: int, direction: Direction | None = None) -> list[FVG]:
        out = [g for g in self.gaps if g.is_live_as_of(i) and (direction is None or g.direction is direction)]
        return out


def detect_fvgs(df: pd.DataFrame, min_size: float = 0.0) -> list[FVG]:
    _, high, low, close = as_arrays(df)
    state = FVGState(min_size=min_size)
    for i in range(len(df)):
        state.step(high, low, close, i)
    return state.gaps


def unmitigated_gaps_as_of(gaps: list[FVG], i: int) -> list[FVG]:
    """Gaps known at i whose zone no candle <= i has touched."""
    return [g for g in gaps if g.index <= i and g.is_unmitigated_as_of(i)]
