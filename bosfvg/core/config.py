"""Strategy configuration. Every tunable lives here, named, with its default.

Rule of thumb from the plan: if the edge needs more than ~3 of these tuned away from default
to appear, it is fit, not edge. `walkforward` reports sensitivity per parameter for that reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import time
from typing import Any, Literal

StopMode = Literal["mid_candle", "gap_far_edge", "rejection_extreme"]
TargetMode = Literal["fixed_r", "next_level"]
FVGSource = Literal["htf", "ltf"]
Confirmation = Literal["none", "next_close_beyond"]


@dataclass
class StrategyConfig:
    # timeframes (minutes). HTF bars are built from LTF bars, session-aligned.
    htf_minutes: int = 60
    ltf_minutes: int = 5

    # structure
    swing_lookback: int = 3          # fractal confirmation candles each side, on HTF
    allow_choch: bool = True         # arm on CHoCH as well as BOS
    arm_window_htf_bars: int = 6     # an arm expires after this many HTF closes

    # fair value gaps
    fvg_source: FVGSource = "htf"    # where the gap that we pull back into is detected
    fvg_impulse_forward_bars: int = 2  # gaps formed up to N HTF bars after the break still count as "the impulse"
    fvg_min_size_frac: float = 0.0005  # gap height as a fraction of price; below this it is noise
    first_touch_only: bool = True      # only the first LTF pullback into a gap can trigger
    fvg_max_age_bars: int = 400        # stop tracking a gap after this many bars of its own timeframe

    # rejection candle (LTF)
    wick_ratio: float = 0.4          # entry-side wick / candle range must be >= this
    close_position: float = 0.6      # close must sit at least this far up (bull) / down (bear) the range
    confirmation: Confirmation = "none"

    # stop / target
    stop_mode: StopMode = "mid_candle"
    stop_buffer_frac: float = 0.0    # extra distance beyond the stop reference, fraction of price
    min_risk_frac: float = 0.0008    # skip setups whose stop is tighter than this (noise fills)
    max_risk_frac: float = 0.01      # skip setups whose stop is wider than this
    target_mode: TargetMode = "fixed_r"
    reward_r: float = 2.0
    next_level_min_r: float = 1.0    # for next_level targets, skip if the level offers less than this
    next_level_fallback_fixed_r: bool = True

    # session / discipline filters (times are America/New_York)
    session_start: time = time(9, 30)
    entry_window_start: time = time(9, 30)
    entry_window_end: time = time(11, 0)
    flat_time: time = time(15, 45)   # hard time stop for every position
    max_trades_per_day: int = 1
    max_hold_minutes: int | None = None
    symbols_allowed: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")

    def validate(self) -> None:
        if self.htf_minutes % self.ltf_minutes != 0:
            raise ValueError("htf_minutes must be a multiple of ltf_minutes")
        if not (0 <= self.wick_ratio <= 1 and 0 <= self.close_position <= 1):
            raise ValueError("wick_ratio and close_position must be within [0, 1]")
        if self.reward_r <= 0:
            raise ValueError("reward_r must be positive")
        if self.entry_window_end <= self.entry_window_start:
            raise ValueError("entry window is empty")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, time):
                d[k] = v.strftime("%H:%M")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyConfig":
        kwargs: dict[str, Any] = {}
        names = {f.name: f for f in fields(cls)}
        for k, v in d.items():
            if k not in names:
                raise KeyError(f"unknown config key: {k}")
            if isinstance(getattr(cls, k, None), time) or names[k].type in ("time", time):
                if isinstance(v, str):
                    hh, mm = v.split(":")
                    v = time(int(hh), int(mm))
            if k == "symbols_allowed" and isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def replace(self, **changes: Any) -> "StrategyConfig":
        d = asdict(self)
        d.update(changes)
        cfg = StrategyConfig(**d)
        cfg.validate()
        return cfg


@dataclass
class RiskConfig:
    equity: float = 100_000.0
    risk_pct: float = 0.005            # fraction of equity risked per trade to the stop
    max_positions: int = 1
    daily_loss_limit_pct: float = 0.02  # stop trading for the day past this realized loss
    instrument: Literal["underlying", "option"] = "underlying"
    # option contract selection
    target_delta: float = 0.5
    min_dte: int = 1
    max_dte: int = 7
    flat_iv: float = 0.18              # used only by the Black-Scholes offline mode
    contract_multiplier: int = 100
    max_contracts: int = 50
    max_shares: int = 5_000
    extra: dict[str, Any] = field(default_factory=dict)
