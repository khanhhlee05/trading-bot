"""Position sizing and portfolio-level guards. Checked before every order, in every driver."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RiskConfig


@dataclass(frozen=True)
class SizeDecision:
    qty: int
    risk_amount: float      # dollars at risk to the stop
    loss_per_unit: float    # dollars lost per share/contract if the stop is hit
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.qty > 0


def size_underlying(equity: float, risk_pct: float, risk_per_share: float, max_shares: int) -> SizeDecision:
    if risk_per_share <= 0:
        return SizeDecision(0, 0.0, 0.0, "non-positive risk per share")
    risk_amount = equity * risk_pct
    qty = int(risk_amount // risk_per_share)
    qty = min(qty, max_shares)
    if qty <= 0:
        return SizeDecision(0, risk_amount, risk_per_share, "risk budget smaller than one share's risk")
    return SizeDecision(qty, qty * risk_per_share, risk_per_share)


def size_option(equity: float, risk_pct: float, loss_per_contract: float, max_contracts: int) -> SizeDecision:
    """`loss_per_contract` = (premium now - premium if the underlying were at the stop) * multiplier."""
    if loss_per_contract <= 0:
        return SizeDecision(0, 0.0, 0.0, "non-positive loss per contract")
    risk_amount = equity * risk_pct
    qty = int(risk_amount // loss_per_contract)
    qty = min(qty, max_contracts)
    if qty <= 0:
        return SizeDecision(0, risk_amount, loss_per_contract, "risk budget smaller than one contract's risk")
    return SizeDecision(qty, qty * loss_per_contract, loss_per_contract)


@dataclass
class DayGuard:
    """Daily loss limit and position count. Reset at the start of each session."""

    cfg: RiskConfig
    start_equity: float
    realized_today: float = 0.0
    open_positions: int = 0
    halted: bool = False
    halt_reason: str = ""

    def can_open(self) -> tuple[bool, str]:
        if self.halted:
            return False, self.halt_reason
        if self.open_positions >= self.cfg.max_positions:
            return False, "max positions open"
        limit = -self.cfg.daily_loss_limit_pct * self.start_equity
        if self.realized_today <= limit:
            self.halted = True
            self.halt_reason = f"daily loss limit hit ({self.realized_today:.2f} <= {limit:.2f})"
            return False, self.halt_reason
        return True, ""

    def on_open(self) -> None:
        self.open_positions += 1

    def on_close(self, pnl: float) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.realized_today += pnl
