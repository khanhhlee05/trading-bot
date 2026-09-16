import math

from bosfvg.core.options_pricing import bs_delta, bs_price, strike_for_delta
from bosfvg.core.risk import DayGuard, size_option, size_underlying
from bosfvg.core.config import RiskConfig
from bosfvg.execution.costs import CostModel


def test_bs_put_call_parity():
    s, k, t, iv, r = 100.0, 100.0, 0.25, 0.2, 0.05
    c = bs_price(s, k, t, iv, True, r)
    p = bs_price(s, k, t, iv, False, r)
    assert math.isclose(c - p, s - k * math.exp(-r * t), rel_tol=1e-9)


def test_bs_expiry_is_intrinsic():
    assert bs_price(105, 100, 0, 0.2, True) == 5
    assert bs_price(95, 100, 0, 0.2, False) == 5


def test_delta_bounds_and_strike_selection():
    assert 0.45 < bs_delta(100, 100, 0.02, 0.2, True) < 0.6
    k = strike_for_delta(450.0, 3 / 365, 0.18, True, 0.5, strike_step=1.0)
    assert 448 <= k <= 452
    kp = strike_for_delta(450.0, 3 / 365, 0.18, False, 0.5, strike_step=1.0)
    assert 448 <= kp <= 452


def test_cost_model_fills():
    cm = CostModel(stock_spread=0.02, stock_slippage=0.01, option_spread=0.04, option_slippage=0.01, option_commission_per_contract=0.65)
    assert math.isclose(cm.stock_fill(100.0, True), 100.02)
    assert math.isclose(cm.stock_fill(100.0, False), 99.98)
    assert math.isclose(cm.option_fill(1.00, True), 1.03)
    assert math.isclose(cm.option_fill(0.01, False), 0.0)  # never negative
    assert math.isclose(cm.option_round_trip_commission(3), 3.9)


def test_sizing():
    s = size_underlying(100_000, 0.005, 0.5, max_shares=5000)
    assert s.qty == 1000 and s.risk_amount == 500
    assert size_underlying(100_000, 0.005, 0.0, 5000).qty == 0
    o = size_option(100_000, 0.005, 120.0, max_contracts=50)
    assert o.qty == 4 and o.risk_amount == 480
    assert size_option(1_000, 0.005, 120.0, 50).qty == 0


def test_day_guard_halts_on_loss_limit():
    g = DayGuard(RiskConfig(daily_loss_limit_pct=0.01, max_positions=1), start_equity=100_000)
    assert g.can_open()[0]
    g.on_open()
    assert not g.can_open()[0]
    g.on_close(-1500)
    ok, why = g.can_open()
    assert not ok and "daily loss" in why
