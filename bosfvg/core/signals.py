"""Multi-timeframe BOS/FVG signal engine.

HTF (e.g. 1H): swings, BOS/CHoCH, and (by default) the FVGs formed by the impulse that broke
structure. A break *arms* a directional bias for `arm_window_htf_bars` HTF closes.
LTF (e.g. 5m): while armed, a candle that pulls back into a live, same-direction gap and prints
a rejection wick produces a Signal.

The engine is incremental and driver-agnostic: the backtester and the live daemon both call
`on_htf_bar` / `on_ltf_bar` in wall-clock order and read the same Signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from .bars import Candle, bar_close_time
from .config import StrategyConfig
from .fvg import FVG, FVGState
from .structure import Direction, EventKind, StructureEvent, StructureState, Trend


@dataclass(frozen=True)
class Signal:
    symbol: str
    time: pd.Timestamp            # close time of the LTF candle that triggered
    direction: Direction
    entry: float
    stop: float
    target: float
    risk: float                   # entry - stop, in price units (>0)
    reward_r: float               # (target - entry) / risk
    gap_top: float
    gap_bottom: float
    gap_source: str
    gap_time: pd.Timestamp
    bos_time: pd.Timestamp
    bos_kind: EventKind
    bos_level: float
    htf_trend: Trend
    rejection_wick_ratio: float
    rejection_close_position: float
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "signal_time": self.time.isoformat(),
            "direction": self.direction.value,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "risk": self.risk,
            "reward_r": self.reward_r,
            "gap_top": self.gap_top,
            "gap_bottom": self.gap_bottom,
            "gap_source": self.gap_source,
            "gap_time": self.gap_time.isoformat(),
            "bos_time": self.bos_time.isoformat(),
            "bos_kind": self.bos_kind.value,
            "bos_level": self.bos_level,
            "htf_trend": self.htf_trend.value,
            "rejection_wick_ratio": self.rejection_wick_ratio,
            "rejection_close_position": self.rejection_close_position,
            "notes": self.notes,
        }
        return d


@dataclass
class Arm:
    event: StructureEvent
    event_time: pd.Timestamp
    htf_index: int
    expires_after_htf_index: int
    origin_index: int


@dataclass
class _Pending:
    """A rejection candle waiting for next-candle confirmation."""

    gap: FVG
    arm: Arm
    rejection: Candle
    wick_ratio: float
    close_pos: float


@dataclass
class EngineState:
    htf_open: list[float] = field(default_factory=list)
    htf_high: list[float] = field(default_factory=list)
    htf_low: list[float] = field(default_factory=list)
    htf_close: list[float] = field(default_factory=list)
    htf_time: list[pd.Timestamp] = field(default_factory=list)
    ltf_high: list[float] = field(default_factory=list)
    ltf_low: list[float] = field(default_factory=list)
    ltf_close: list[float] = field(default_factory=list)
    ltf_time: list[pd.Timestamp] = field(default_factory=list)


class SignalEngine:
    def __init__(self, cfg: StrategyConfig, symbol: str) -> None:
        cfg.validate()
        self.cfg = cfg
        self.symbol = symbol
        self.st = EngineState()
        self.structure = StructureState(lookback=cfg.swing_lookback)
        self.htf_fvg = FVGState(max_age=cfg.fvg_max_age_bars)
        self.ltf_fvg = FVGState(max_age=cfg.fvg_max_age_bars)
        self.arms: list[Arm] = []
        self.signals: list[Signal] = []
        self.pending: _Pending | None = None
        self._ltf_touched: set[tuple[str, int]] = set()   # (source, gap id) touched by an LTF candle
        self._gap_known_time: dict[tuple[str, int], pd.Timestamp] = {}
        self._trades_by_day: dict[pd.Timestamp, int] = {}
        self._used_gaps: set[tuple[str, int]] = set()

    # ---------------------------------------------------------------- HTF

    def on_htf_bar(self, t: pd.Timestamp, o: float, h: float, l: float, c: float) -> StructureEvent | None:
        st = self.st
        st.htf_time.append(t)
        st.htf_open.append(o)
        st.htf_high.append(h)
        st.htf_low.append(l)
        st.htf_close.append(c)
        i = len(st.htf_close) - 1
        event = self.structure.step(st.htf_high, st.htf_low, st.htf_close, i)
        gap = self.htf_fvg.step(st.htf_high, st.htf_low, st.htf_close, i)
        if gap is not None:
            self._gap_known_time[("htf", gap.id)] = bar_close_time(t, self.cfg.htf_minutes)
        self.arms = [a for a in self.arms if a.expires_after_htf_index >= i]
        if event is not None and (event.kind is EventKind.BOS or self.cfg.allow_choch):
            arm = Arm(
                event=event,
                event_time=bar_close_time(t, self.cfg.htf_minutes),
                htf_index=i,
                expires_after_htf_index=i + self.cfg.arm_window_htf_bars,
                origin_index=event.origin_index if event.origin_index >= 0 else max(0, i - self.cfg.swing_lookback * 2),
            )
            # a new arm in the opposite direction cancels older opposite arms
            self.arms = [a for a in self.arms if a.event.direction is event.direction]
            self.arms.append(arm)
            self.pending = None
        return event

    # ---------------------------------------------------------------- LTF

    def on_ltf_bar(self, t: pd.Timestamp, o: float, h: float, l: float, c: float) -> Signal | None:
        st = self.st
        st.ltf_time.append(t)
        st.ltf_high.append(h)
        st.ltf_low.append(l)
        st.ltf_close.append(c)
        i = len(st.ltf_close) - 1
        candle = Candle(i, t, o, h, l, c)
        close_time = bar_close_time(t, self.cfg.ltf_minutes)

        if self.cfg.fvg_source == "ltf":
            gap = self.ltf_fvg.step(st.ltf_high, st.ltf_low, st.ltf_close, i)
            if gap is not None:
                self._gap_known_time[("ltf", gap.id)] = close_time

        # confirmation of a pending rejection candle takes priority
        if self.pending is not None:
            p = self.pending
            self.pending = None
            if self._confirms(p, candle):
                sig = self._build_signal(p.gap, p.arm, candle, close_time, p.wick_ratio, p.close_pos, entry=c)
                if sig is not None:
                    return self._emit(sig)
            # a failed confirmation falls through to normal processing of this candle

        if not self.arms:
            return None
        day = t.normalize()
        # touches are tracked on every candle so "first touch" means first touch, not first
        # touch inside the entry window; emission is gated further down
        eligible = self._within_entry_window(close_time) and self._trades_by_day.get(day, 0) < self.cfg.max_trades_per_day

        best: tuple[FVG, Arm, float, float, float] | None = None
        for arm in self.arms:
            d = arm.event.direction
            for gap in self._candidate_gaps(arm):
                key = (self.cfg.fvg_source, gap.id)
                known = self._gap_known_time.get(key)
                if known is None or t < known:
                    continue  # the gap did not exist when this candle opened
                if key in self._used_gaps:
                    continue
                touched = self._touches(gap, candle)
                if not touched:
                    continue
                first_touch = key not in self._ltf_touched
                self._ltf_touched.add(key)
                if not eligible:
                    continue
                if self.cfg.first_touch_only and not first_touch:
                    continue
                wick_ratio, close_pos = self._rejection_stats(gap.direction, candle)
                if wick_ratio < self.cfg.wick_ratio or close_pos < self.cfg.close_position:
                    continue
                if d is Direction.BULLISH and c <= gap.bottom:
                    continue
                if d is Direction.BEARISH and c >= gap.top:
                    continue
                # prefer the gap whose near edge is deepest into the wick (the one price actually reacted at)
                depth = (gap.top - l) if d is Direction.BULLISH else (h - gap.bottom)
                if best is None or depth > best[2]:
                    best = (gap, arm, depth, wick_ratio, close_pos)
        if best is None:
            return None
        gap, arm, _, wick_ratio, best_close_pos = best
        if self.cfg.confirmation == "next_close_beyond":
            self.pending = _Pending(gap, arm, candle, wick_ratio, best_close_pos)
            return None
        sig = self._build_signal(gap, arm, candle, close_time, wick_ratio, best_close_pos, entry=c)
        return self._emit(sig) if sig is not None else None

    # ---------------------------------------------------------------- helpers

    def _emit(self, sig: Signal) -> Signal:
        day = sig.time.normalize()
        self._trades_by_day[day] = self._trades_by_day.get(day, 0) + 1
        self.signals.append(sig)
        return sig

    def _within_entry_window(self, close_time: pd.Timestamp) -> bool:
        tt: time = close_time.time()
        return self.cfg.entry_window_start < tt <= self.cfg.entry_window_end

    def _candidate_gaps(self, arm: Arm) -> list[FVG]:
        d = arm.event.direction
        htf_i = len(self.st.htf_close) - 1
        if self.cfg.fvg_source == "htf":
            lo = arm.origin_index
            hi = arm.htf_index + self.cfg.fvg_impulse_forward_bars
            gaps = [g for g in self.htf_fvg.live_gaps(htf_i, d) if lo <= g.index <= hi]
        else:
            ltf_i = len(self.st.ltf_close) - 1
            gaps = [g for g in self.ltf_fvg.live_gaps(ltf_i, d)
                    if self._gap_known_time[("ltf", g.id)] >= arm.event_time]
        price = self.st.ltf_close[-1] if self.st.ltf_close else self.st.htf_close[-1]
        min_size = self.cfg.fvg_min_size_frac * price
        return [g for g in gaps if g.size >= min_size]

    @staticmethod
    def _touches(gap: FVG, c: Candle) -> bool:
        if gap.direction is Direction.BULLISH:
            return c.low <= gap.top and c.high > gap.top
        return c.high >= gap.bottom and c.low < gap.bottom

    @staticmethod
    def _rejection_stats(d: Direction, c: Candle) -> tuple[float, float]:
        rng = c.range
        if rng <= 0:
            return 0.0, 0.0
        if d is Direction.BULLISH:
            return c.lower_wick / rng, (c.close - c.low) / rng
        return c.upper_wick / rng, (c.high - c.close) / rng

    @staticmethod
    def _confirms(p: _Pending, c: Candle) -> bool:
        if p.gap.direction is Direction.BULLISH:
            return c.close > p.rejection.high
        return c.close < p.rejection.low

    def _stop_price(self, gap: FVG, rej: Candle, entry: float) -> float:
        d = gap.direction
        buf = self.cfg.stop_buffer_frac * entry
        mode = self.cfg.stop_mode
        if mode == "mid_candle":
            ref = gap.mid_low if d is Direction.BULLISH else gap.mid_high
        elif mode == "gap_far_edge":
            ref = gap.far_edge
        else:
            ref = rej.low if d is Direction.BULLISH else rej.high
        return ref - buf if d is Direction.BULLISH else ref + buf

    def _target_price(self, d: Direction, entry: float, risk: float) -> tuple[float, str]:
        fixed = entry + d.sign * self.cfg.reward_r * risk
        if self.cfg.target_mode == "fixed_r":
            return fixed, "fixed_r"
        levels = self.structure.key_levels_above(entry) if d is Direction.BULLISH else self.structure.key_levels_below(entry)
        for lvl in levels:
            if abs(lvl - entry) / risk >= self.cfg.next_level_min_r:
                return lvl, "next_level"
        if self.cfg.next_level_fallback_fixed_r:
            return fixed, "fixed_r_fallback"
        return float("nan"), "none"

    def _build_signal(self, gap: FVG, arm: Arm, rej: Candle, close_time: pd.Timestamp,
                      wick_ratio: float, close_pos: float, entry: float) -> Signal | None:
        d = gap.direction
        stop = self._stop_price(gap, rej, entry)
        risk = (entry - stop) * d.sign
        if risk <= 0:
            return None
        frac = risk / entry
        if frac < self.cfg.min_risk_frac or frac > self.cfg.max_risk_frac:
            return None
        target, tmode = self._target_price(d, entry, risk)
        if not np.isfinite(target):
            return None
        key = (self.cfg.fvg_source, gap.id)
        self._used_gaps.add(key)
        return Signal(
            symbol=self.symbol,
            time=close_time,
            direction=d,
            entry=float(entry),
            stop=float(stop),
            target=float(target),
            risk=float(risk),
            reward_r=float(abs(target - entry) / risk),
            gap_top=gap.top,
            gap_bottom=gap.bottom,
            gap_source=self.cfg.fvg_source,
            gap_time=self._gap_known_time[key],
            bos_time=arm.event_time,
            bos_kind=arm.event.kind,
            bos_level=arm.event.level,
            htf_trend=self.structure.trend,
            rejection_wick_ratio=float(wick_ratio),
            rejection_close_position=float(close_pos),
            notes=f"target={tmode};stop={self.cfg.stop_mode}",
        )


def run_engine(cfg: StrategyConfig, symbol: str, htf: pd.DataFrame, ltf: pd.DataFrame) -> list[Signal]:
    """Drive the engine over aligned HTF and LTF frames in wall-clock order.

    An HTF bar is fed once its close time is <= the close time of the LTF bar about to be fed,
    and it is fed *before* that LTF bar. This is the only place ordering is decided; the live
    daemon reproduces it by construction (it only ever sees closed bars).
    """
    eng = SignalEngine(cfg, symbol)
    h_times = [bar_close_time(t, cfg.htf_minutes) for t in htf.index]
    h_rows = htf[["open", "high", "low", "close"]].to_numpy()
    l_rows = ltf[["open", "high", "low", "close"]].to_numpy()
    hi = 0
    for li, t in enumerate(ltf.index):
        lc = bar_close_time(t, cfg.ltf_minutes)
        while hi < len(h_times) and h_times[hi] <= lc:
            o, h, l, c = h_rows[hi]
            eng.on_htf_bar(htf.index[hi], float(o), float(h), float(l), float(c))
            hi += 1
        o, h, l, c = l_rows[li]
        eng.on_ltf_bar(t, float(o), float(h), float(l), float(c))
    return eng.signals
