"""One cost model for everything. The backtester prices friction with it, and the live
journal re-marks every paper fill with it so "backtest said X, paper gave Y" is attributable.

Convention: `mid` is the fair price; a buy fills at mid + half_spread + slippage, a sell at
mid - half_spread - slippage. Commissions are added per unit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    # underlying (per share, dollars)
    stock_spread: float = 0.01
    stock_slippage: float = 0.0
    stock_commission: float = 0.0
    # options (per share of premium, dollars; one contract = 100 shares)
    option_spread: float = 0.02
    option_slippage: float = 0.01
    option_commission_per_contract: float = 0.65
    contract_multiplier: int = 100

    def stock_fill(self, mid: float, is_buy: bool) -> float:
        adj = self.stock_spread / 2 + self.stock_slippage
        return mid + adj if is_buy else max(mid - adj, 0.0)

    def option_fill(self, mid: float, is_buy: bool) -> float:
        adj = self.option_spread / 2 + self.option_slippage
        return mid + adj if is_buy else max(mid - adj, 0.0)

    def stock_round_trip_commission(self, qty: int) -> float:
        return 2 * qty * self.stock_commission

    def option_round_trip_commission(self, contracts: int) -> float:
        return 2 * contracts * self.option_commission_per_contract


ZERO_COSTS = CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
