"""Regime-switching synthetic intraday bars. For wiring and sanity checks ONLY.

Numbers produced from this data say nothing about edge; they say the code runs and the
plumbing is causal. Real bars come from `alpaca_hist`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.bars import NY_TZ, ensure_bars


def synthetic_minute_bars(
    days: int = 60,
    start: str = "2024-01-02",
    price: float = 450.0,
    seed: int = 7,
    minutes_per_day: int = 390,
) -> pd.DataFrame:
    """1-minute RTH bars with alternating trend / chop regimes and occasional impulse moves.

    Impulses (3-4 bars of one-sided movement) exist so that FVGs and structure breaks appear
    at a realistic rate; without them a pure random walk almost never prints a 3-candle gap.
    """
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start=start, periods=days)
    rows = []
    p = price
    regime_drift = 0.0
    for d in bdays:
        if rng.random() < 0.3:
            regime_drift = rng.choice([-1.0, 0.0, 1.0]) * 0.00004
        vol = rng.uniform(0.00035, 0.0009)
        t0 = pd.Timestamp(d.date(), tz=NY_TZ) + pd.Timedelta(hours=9, minutes=30)
        impulse_left = 0
        impulse_dir = 0.0
        for m in range(minutes_per_day):
            if impulse_left == 0 and rng.random() < 0.012:
                impulse_left = int(rng.integers(3, 6))
                impulse_dir = rng.choice([-1.0, 1.0]) * rng.uniform(0.0012, 0.0025)
            if impulse_left > 0:
                ret = impulse_dir + rng.normal(0, vol * 0.4)
                impulse_left -= 1
            else:
                ret = regime_drift + rng.normal(0, vol)
            o = p
            c = p * (1 + ret)
            wick = abs(rng.normal(0, vol * 0.8)) * p
            h = max(o, c) + wick * rng.uniform(0, 1)
            l = min(o, c) - wick * rng.uniform(0, 1)
            v = float(rng.integers(20_000, 200_000))
            rows.append((t0 + pd.Timedelta(minutes=m), o, h, l, c, v))
            p = c
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]).set_index("time")
    return ensure_bars(df)
