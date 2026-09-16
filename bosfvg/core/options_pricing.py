"""Flat-IV Black-Scholes. Offline sanity-check mode for option P&L when no chain data exists.

Known simplification: real premiums carry a spread, a skew, and an IV that moves with the
underlying. `execution/costs.py` layers the spread on; skew and IV dynamics are not modelled.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def bs_price(spot: float, strike: float, t_years: float, iv: float, is_call: bool, r: float = 0.05) -> float:
    if t_years <= 0 or iv <= 0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    if is_call:
        return spot * norm.cdf(d1) - strike * math.exp(-r * t_years) * norm.cdf(d2)
    return strike * math.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_delta(spot: float, strike: float, t_years: float, iv: float, is_call: bool, r: float = 0.05) -> float:
    if t_years <= 0 or iv <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def strike_for_delta(spot: float, t_years: float, iv: float, is_call: bool, target_delta: float,
                     strike_step: float = 1.0, r: float = 0.05) -> float:
    """Nearest listed strike (multiple of strike_step) whose |delta| is closest to target."""
    lo = math.floor(spot * 0.9 / strike_step) * strike_step
    hi = math.ceil(spot * 1.1 / strike_step) * strike_step
    best, best_err = spot, float("inf")
    k = lo
    while k <= hi + 1e-9:
        err = abs(abs(bs_delta(spot, k, t_years, iv, is_call, r)) - target_delta)
        if err < best_err:
            best, best_err = k, err
        k += strike_step
    return round(best, 4)
