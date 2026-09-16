from datetime import date, datetime
from types import SimpleNamespace

import pytest

from bosfvg.execution.alpaca_broker import parse_occ, snapshot_to_contract
from bosfvg.execution.broker import ContractQuote, FakeBroker, Quote
from bosfvg.live.contracts import SelectionCriteria, loss_per_contract_at_stop, select_contract


def _c(sym, expiry, strike, is_call, bid, ask, delta):
    return ContractQuote(sym, "SPY", expiry, strike, is_call, bid, ask, delta, 0.18)


def test_parse_occ():
    root, exp, is_call, strike = parse_occ("SPY240607C00530000")
    assert (root, exp, is_call, strike) == ("SPY", date(2024, 6, 7), True, 530.0)
    assert parse_occ("QQQ251219P00450500")[3] == 450.5
    with pytest.raises(ValueError):
        parse_occ("SPY")


def test_snapshot_to_contract_with_and_without_greeks():
    snap = SimpleNamespace(latest_quote=SimpleNamespace(bid_price=1.2, ask_price=1.3), greeks=SimpleNamespace(delta=0.52), implied_volatility=0.17)
    c = snapshot_to_contract("SPY240607C00530000", snap, "SPY")
    assert c.delta == 0.52 and c.bid == 1.2 and c.iv == 0.17
    snap2 = SimpleNamespace(latest_quote=SimpleNamespace(bid_price=1.2, ask_price=1.3), greeks=None, implied_volatility=None)
    c2 = snapshot_to_contract("SPY240607C00530000", snap2, "SPY")
    assert c2.delta is None and c2.iv is None
    assert snapshot_to_contract("SPY240607C00530000", SimpleNamespace(latest_quote=None), "SPY") is None


def test_select_nearest_delta_nearest_expiry():
    today = date(2024, 6, 3)
    chain = [
        _c("a", date(2024, 6, 4), 528, True, 2.0, 2.05, 0.62),
        _c("b", date(2024, 6, 4), 530, True, 1.2, 1.25, 0.50),
        _c("c", date(2024, 6, 4), 532, True, 0.6, 0.65, 0.38),
        _c("d", date(2024, 6, 7), 530, True, 2.5, 2.55, 0.51),   # later expiry, ignored
        _c("e", date(2024, 6, 3), 530, True, 1.0, 1.05, 0.50),   # 0DTE, outside min_dte=1
    ]
    best, why = select_contract(chain, 530.0, True, today, SelectionCriteria(target_delta=0.5, min_dte=1, max_dte=7))
    assert best.symbol == "b"


def test_select_rejects_wide_or_dead_quotes_and_falls_back_without_greeks():
    today = date(2024, 6, 3)
    chain = [
        _c("wide", date(2024, 6, 4), 530, True, 1.0, 1.5, 0.50),     # 40% spread
        _c("dead", date(2024, 6, 4), 531, True, 0.0, 0.05, 0.45),
        ContractQuote("ok", "SPY", date(2024, 6, 4), 529, True, 1.6, 1.65, None, None),
        ContractQuote("ok2", "SPY", date(2024, 6, 4), 533, True, 0.4, 0.45, None, None),
    ]
    best, why = select_contract(chain, 530.0, True, today, SelectionCriteria())
    assert best.symbol == "ok" and "no greeks" in why
    none, why = select_contract(chain, 530.0, False, today, SelectionCriteria())
    assert none is None


def test_loss_per_contract_bounds():
    c = _c("x", date(2024, 6, 4), 530, True, 1.2, 1.3, 0.5)
    assert loss_per_contract_at_stop(c, 530.0, 529.0) == pytest.approx(0.5 * 100)
    # a stop 10 points away cannot lose more than the premium
    assert loss_per_contract_at_stop(c, 530.0, 520.0) == pytest.approx(1.3 * 100)
    # a stop 0.01 away still risks at least a quarter of the premium
    assert loss_per_contract_at_stop(c, 530.0, 529.99) == pytest.approx(0.325 * 100)


def test_fake_broker_round_trip():
    b = FakeBroker()
    b.chains["SPY"] = [_c("SPY240604C00530000", date(2024, 6, 4), 530, True, 1.2, 1.3, 0.5)]
    o = b.submit_option_market("SPY240604C00530000", 2, "buy", "buy_to_open", "t1")
    assert o.is_filled and o.filled_avg_price == 1.3
    assert b.positions()[0].qty == 2
    b.close_all_positions()
    assert b.positions() == []
    assert b.orders[-1].filled_avg_price == 1.2
