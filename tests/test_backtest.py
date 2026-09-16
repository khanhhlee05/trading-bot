from datetime import time

import pandas as pd

from bosfvg.core.bars import NY_TZ
from bosfvg.core.config import RiskConfig, StrategyConfig
from bosfvg.core.signals import Signal
from bosfvg.core.structure import Direction, EventKind, Trend
from bosfvg.data.synthetic import synthetic_minute_bars
from bosfvg.execution.costs import ZERO_COSTS, CostModel
from bosfvg.runner.backtest import _walk_exit, run_backtest
from tests.conftest import make_bars


def _sig(direction=Direction.BULLISH, entry=100.0, stop=99.0, target=102.0, t="2024-03-04 09:40"):
    ts = pd.Timestamp(t, tz=NY_TZ)
    return Signal("SPY", ts, direction, entry, stop, target, abs(entry - stop), abs(target - entry) / abs(entry - stop),
                  101, 99.5, "htf", ts, ts, EventKind.BOS, 100.5, Trend.BULLISH, 0.5, 0.7)


def test_stop_wins_when_both_hit_in_one_bar():
    ltf = make_bars([(100, 100.5, 99.8, 100.2), (100.2, 103, 98.5, 101)], start="2024-03-04 09:40")
    cfg = StrategyConfig(entry_window_end=time(15, 0))
    pos, px, reason = _walk_exit(ltf, 0, _sig(), cfg)
    assert (pos, px, reason) == (1, 99.0, "stop")


def test_gap_through_stop_fills_at_open():
    ltf = make_bars([(98.0, 98.5, 97.5, 98.2)], start="2024-03-04 09:40")
    pos, px, reason = _walk_exit(ltf, 0, _sig(), StrategyConfig())
    assert (px, reason) == (98.0, "gap_stop")


def test_target_hit():
    ltf = make_bars([(100, 101, 99.5, 100.8), (100.8, 102.5, 100.5, 102.2)], start="2024-03-04 09:40")
    pos, px, reason = _walk_exit(ltf, 0, _sig(), StrategyConfig())
    assert (pos, px, reason) == (1, 102.0, "target")


def test_bearish_exits_mirror():
    sig = _sig(Direction.BEARISH, entry=100, stop=101, target=98)
    ltf = make_bars([(100, 100.4, 99.6, 100.1), (100.1, 100.9, 97.9, 98.5)], start="2024-03-04 09:40")
    assert _walk_exit(ltf, 0, sig, StrategyConfig())[1:] == (98.0, "target")
    ltf = make_bars([(100, 100.4, 99.6, 100.1), (100.1, 101.2, 97.9, 98.5)], start="2024-03-04 09:40")
    assert _walk_exit(ltf, 0, sig, StrategyConfig())[1:] == (101.0, "stop")


def test_time_stop_at_flat_time():
    rows = [(100, 100.3, 99.7, 100.1)] * 80  # 80 five-minute bars from 09:40 runs past 15:45
    ltf = make_bars(rows, start="2024-03-04 09:40")
    cfg = StrategyConfig(flat_time=time(15, 45))
    pos, px, reason = _walk_exit(ltf, 0, _sig(), cfg)
    assert reason == "time"
    assert ltf.index[pos] == pd.Timestamp("2024-03-04 15:40", tz=NY_TZ)


def test_max_hold_minutes():
    ltf = make_bars([(100, 100.3, 99.7, 100.1)] * 20, start="2024-03-04 09:40")
    cfg = StrategyConfig(max_hold_minutes=30)
    pos, px, reason = _walk_exit(ltf, 0, _sig(), cfg)
    assert reason == "time" and pos == 5  # 6 bars * 5 min = 30 min


def test_backtest_runs_and_costs_reduce_pnl():
    bars = synthetic_minute_bars(days=120, seed=11)
    cfg = StrategyConfig(fvg_source="ltf", entry_window_end=time(15, 30), max_trades_per_day=3)
    free = run_backtest(bars, "SPY", cfg, RiskConfig(instrument="underlying"), ZERO_COSTS)
    paid = run_backtest(bars, "SPY", cfg, RiskConfig(instrument="underlying"), CostModel(stock_spread=0.02, stock_commission=0.005))
    assert len(free.trades) > 5
    assert len(free.trades) == len(paid.trades)
    assert paid.stats["total_pnl"] < free.stats["total_pnl"]
    assert paid.stats["commission_total"] > 0
    assert set(free.trades["exit_reason"]).issubset({"stop", "target", "time", "gap_stop", "eod", "data_end"})
    # no overlapping positions: each entry is at or after the previous exit
    t = free.trades
    entries = pd.to_datetime(t["entry_time"], utc=True)
    exits = pd.to_datetime(t["exit_time"], utc=True)
    assert (entries.iloc[1:].to_numpy() >= exits.iloc[:-1].to_numpy()).all()


def test_option_mode_sizes_by_premium_loss():
    bars = synthetic_minute_bars(days=120, seed=11)
    cfg = StrategyConfig(fvg_source="ltf", entry_window_end=time(15, 30), max_trades_per_day=3)
    r = run_backtest(bars, "SPY", cfg, RiskConfig(instrument="option", risk_pct=0.005), CostModel())
    assert len(r.trades) > 5
    t = r.trades
    assert (t["qty"] >= 1).all()
    equity_before = t["equity_after"].shift(1).fillna(100_000.0)
    assert (t["risk_amount"] <= equity_before * 0.005 + 1e-6).all()
    assert t["contract"].str.contains(" C| P").all()
    # risk_amount recorded is what would be lost at the stop, so a stopped trade loses about -1R (theta aside)
    stopped = t[t["exit_reason"] == "stop"]
    if len(stopped):
        assert (stopped["pnl_r"].between(-1.6, -0.4)).all()
