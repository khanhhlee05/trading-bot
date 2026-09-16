"""Option contract selection. Pure function over chain quotes so it is testable offline.

Picks the contract closest to `target_delta` inside [min_dte, max_dte], rejecting anything
with a dead quote or a spread wider than `max_spread_frac` of mid. When the chain has no
greeks (indicative feed), falls back to the nearest strike to the delta-implied Black-Scholes
strike using a flat IV.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from ..core.options_pricing import strike_for_delta
from ..execution.broker import ContractQuote


@dataclass(frozen=True)
class SelectionCriteria:
    target_delta: float = 0.5
    min_dte: int = 1
    max_dte: int = 7
    max_spread_frac: float = 0.10     # reject contracts quoted wider than 10% of mid
    min_bid: float = 0.05
    min_open_interest: int = 0
    flat_iv: float = 0.18             # fallback when the chain carries no greeks


def expiry_window(today: date, crit: SelectionCriteria) -> tuple[date, date]:
    return today + timedelta(days=crit.min_dte), today + timedelta(days=crit.max_dte)


def strike_window(spot: float, width_frac: float = 0.04) -> tuple[float, float]:
    return spot * (1 - width_frac), spot * (1 + width_frac)


def select_contract(chain: list[ContractQuote], spot: float, is_call: bool, today: date,
                    crit: SelectionCriteria) -> tuple[ContractQuote | None, str]:
    lo, hi = expiry_window(today, crit)
    pool = [c for c in chain if c.is_call == is_call and lo <= c.expiry <= hi]
    if not pool:
        return None, "no contracts in expiry window"
    live = [c for c in pool if c.bid >= crit.min_bid and c.ask > c.bid and c.spread_frac <= crit.max_spread_frac
            and (c.open_interest is None or c.open_interest >= crit.min_open_interest)]
    if not live:
        return None, "no contract passed the liquidity filter"
    # nearest expiry first: less theta surprise, and the reference strategy is intraday
    nearest_expiry = min(c.expiry for c in live)
    live = [c for c in live if c.expiry == nearest_expiry]
    with_delta = [c for c in live if c.delta is not None]
    if with_delta:
        best = min(with_delta, key=lambda c: abs(abs(c.delta) - crit.target_delta))
        return best, f"delta {best.delta:+.2f} vs target {crit.target_delta:.2f}, exp {best.expiry}"
    t_years = max((nearest_expiry - today).days, 0.5) / 365.0
    k = strike_for_delta(spot, t_years, crit.flat_iv, is_call, crit.target_delta, strike_step=1.0)
    best = min(live, key=lambda c: abs(c.strike - k))
    return best, f"no greeks in chain; nearest strike to BS delta strike {k:g}: {best.strike:g}, exp {best.expiry}"


def loss_per_contract_at_stop(contract: ContractQuote, spot: float, stop: float, multiplier: int = 100) -> float:
    """Dollar loss per contract if the underlying goes to the stop right now (delta approximation,
    floored at a quarter of the premium so a tiny delta can't produce absurd size)."""
    delta = abs(contract.delta) if contract.delta is not None else 0.5
    move = abs(spot - stop) * delta
    premium_loss = min(move, contract.ask)
    premium_loss = max(premium_loss, contract.ask * 0.25)
    return premium_loss * multiplier
