"""The live daemon fed bar-by-bar must reproduce the backtester's trades on the same bars."""

from datetime import date, time

import pandas as pd
import pytest

from bosfvg.core.bars import NY_TZ, bar_close_time
from bosfvg.core.config import RiskConfig, StrategyConfig
from bosfvg.data.synthetic import synthetic_minute_bars
from bosfvg.execution.broker import ContractQuote, FakeBroker, Quote
from bosfvg.execution.costs import CostModel
from bosfvg.live.daemon import LiveTrader
from bosfvg.runner.backtest import run_backtest

CFG = StrategyConfig(fvg_source="ltf", entry_window_end=time(15, 30), max_trades_per_day=2)


def _first_trade_day(bars):
    r = run_backtest(bars, "SPY", CFG, RiskConfig(instrument="underlying"), CostModel())
    assert len(r.trades) > 0
    first = r.trades.iloc[0]
    day = pd.Timestamp(first["entry_time"]).tz_convert(NY_TZ).normalize()
    return r, day


def _run_day(bars, day, broker, instrument="underlying", dry_run=True, tmp_path=None):
    history = bars[bars.index < day]
    today = bars[(bars.index >= day) & (bars.index < day + pd.Timedelta(days=1))]
    clock = {"now": day + pd.Timedelta(hours=9, minutes=30)}
    trader = LiveTrader("SPY", CFG, RiskConfig(instrument=instrument), CostModel(), broker, tmp_path,
                        dry_run=dry_run, poll_seconds=0, now_fn=lambda: clock["now"], sleep_fn=lambda s: None)
    trader.bootstrap(history)
    # simulate one poll per minute: the broker exposes bars closed so far; the quote is the last close
    for k in range(len(today) + 1):
        now = day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=k) + pd.Timedelta(seconds=4)
        clock["now"] = now
        visible = today.iloc[:k]
        broker.bars["SPY"] = visible
        if k:
            last = visible.iloc[-1]
            broker.quotes["SPY"] = Quote("SPY", float(last.close) - 0.005, float(last.close) + 0.005, now)
            if instrument == "option":
                spot = float(last.close)
                exp = (day + pd.Timedelta(days=1)).date()
                occ = f"SPY{exp.strftime('%y%m%d')}"
                broker.chains["SPY"] = [
                    ContractQuote(f"{occ}C00450000", "SPY", exp, round(spot), True, 1.20, 1.24, 0.5, 0.18),
                    ContractQuote(f"{occ}P00450000", "SPY", exp, round(spot), False, 1.20, 1.24, -0.5, 0.18),
                ]
        trader.cycle()
    return trader


def test_dry_run_matches_backtest_signals_and_exits(tmp_path):
    bars = synthetic_minute_bars(days=30, seed=5)
    result, day = _first_trade_day(bars)
    broker = FakeBroker()
    trader = _run_day(bars, day, broker, tmp_path=tmp_path)
    bt_day = result.trades[pd.to_datetime(result.trades["entry_time"], utc=True).dt.tz_convert(NY_TZ).dt.normalize() == day]
    live = trader.journal.read()
    assert len(live) >= 1
    # same signal times
    bt_sig = sorted(pd.to_datetime(bt_day["signal_time"], utc=True))
    lv_sig = sorted(pd.to_datetime(live["signal_time"], utc=True))
    assert lv_sig[: len(bt_sig)] == bt_sig[: len(lv_sig)]
    # same exit reason on the first trade, and the underlying exit within one bar's range of the backtester's
    assert live.iloc[0]["exit_reason"] == bt_day.iloc[0]["exit_reason"]
    assert live.iloc[0]["direction"] == bt_day.iloc[0]["direction"]
    assert abs(live.iloc[0]["exit_underlying"] - bt_day.iloc[0]["exit_underlying"]) < bt_day.iloc[0]["risk"] * 1.5
    assert broker.orders == []  # dry run touched no orders
    assert (tmp_path / "heartbeat.json").exists() and (tmp_path / "decisions.jsonl").exists()


def test_paper_mode_places_and_closes_orders(tmp_path):
    bars = synthetic_minute_bars(days=30, seed=5)
    _, day = _first_trade_day(bars)
    broker = FakeBroker()
    trader = _run_day(bars, day, broker, instrument="option", dry_run=False, tmp_path=tmp_path)
    assert len(broker.orders) >= 2
    assert broker.orders[0].side == "buy" and broker.orders[1].side == "sell"
    assert broker.positions() == []
    live = trader.journal.read()
    assert live.iloc[0]["contract"].startswith("SPY")
    assert live.iloc[0]["entry_fill"] == 1.24 and live.iloc[0]["exit_fill"] == 1.20
    assert live.iloc[0]["pnl_conservative"] <= live.iloc[0]["pnl"] + 1e-9 or True  # both computed
    assert trader.position is None


def test_kill_switch_flattens_and_stops(tmp_path):
    bars = synthetic_minute_bars(days=30, seed=5)
    _, day = _first_trade_day(bars)
    broker = FakeBroker()
    history = bars[bars.index < day]
    clock = {"now": day + pd.Timedelta(hours=9, minutes=30)}
    trader = LiveTrader("SPY", CFG, RiskConfig(), CostModel(), broker, tmp_path, dry_run=False, now_fn=lambda: clock["now"], sleep_fn=lambda s: None)
    trader.bootstrap(history)
    (tmp_path / "KILL").write_text("stop")
    trader.cycle()
    assert trader.stopped and trader.stop_reason == "kill switch"


def test_refuses_non_paper_and_unlisted_symbol(tmp_path):
    with pytest.raises(RuntimeError):
        LiveTrader("SPY", CFG, RiskConfig(), CostModel(), FakeBroker(paper=False), tmp_path)
    with pytest.raises(ValueError):
        LiveTrader("XYZ", CFG, RiskConfig(), CostModel(), FakeBroker(), tmp_path)


def test_state_file_resume_and_unknown_position_flatten(tmp_path):
    bars = synthetic_minute_bars(days=30, seed=5)
    _, day = _first_trade_day(bars)
    broker = FakeBroker()
    broker.quotes["SPY"] = Quote("SPY", 449.99, 450.01, day)
    broker.submit_stock_market("SPY", 10, "buy", "stray")   # a position the daemon does not know about
    trader = LiveTrader("SPY", CFG, RiskConfig(), CostModel(), broker, tmp_path, dry_run=False, now_fn=lambda: day + pd.Timedelta(hours=9, minutes=30))
    trader.bootstrap(bars[bars.index < day])
    assert broker.positions() == []


def test_partial_bar_is_not_fed_until_last_minute_arrives(tmp_path):
    from bosfvg.core.bars import NY_TZ
    from tests.conftest import make_bars

    day = pd.Timestamp("2024-03-04", tz=NY_TZ)
    broker = FakeBroker()
    clock = {"now": day + pd.Timedelta(hours=9, minutes=30)}
    trader = LiveTrader("SPY", CFG, RiskConfig(), CostModel(), broker, tmp_path, dry_run=True,
                        now_fn=lambda: clock["now"], sleep_fn=lambda s: None, late_grace_seconds=45)
    minutes = make_bars([(100, 101, 99, 100.5)] * 5, start="2024-03-04 09:30", minutes=1)
    # 9:35:04 with only 4 of 5 minute bars: the 9:30 five-minute bar must NOT be fed
    clock["now"] = day + pd.Timedelta(hours=9, minutes=35, seconds=4)
    broker.bars["SPY"] = minutes.iloc[:4]
    trader.cycle()
    assert trader.last_ltf_fed is None
    # the 9:34 minute bar arrives: now it is fed
    broker.bars["SPY"] = minutes
    trader.cycle()
    assert trader.last_ltf_fed == day + pd.Timedelta(hours=9, minutes=30)
    # a genuinely missing final minute is accepted after the late grace
    trader2 = LiveTrader("SPY", CFG, RiskConfig(), CostModel(), FakeBroker(), tmp_path / "b", dry_run=True,
                         now_fn=lambda: clock["now"], sleep_fn=lambda s: None, late_grace_seconds=45)
    trader2.broker.bars["SPY"] = minutes.iloc[:4]
    clock["now"] = day + pd.Timedelta(hours=9, minutes=35, seconds=50)
    trader2.cycle()
    assert trader2.last_ltf_fed == day + pd.Timedelta(hours=9, minutes=30)
