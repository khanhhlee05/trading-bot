from datetime import time

import pandas as pd

from bosfvg.core.bars import NY_TZ, resample_bars
from bosfvg.core.config import StrategyConfig
from bosfvg.core.signals import SignalEngine, run_engine
from bosfvg.core.structure import Direction
from bosfvg.data.synthetic import synthetic_minute_bars


def _cfg(**kw):
    base = dict(htf_minutes=60, ltf_minutes=5, swing_lookback=2, entry_window_end=time(15, 30),
                max_trades_per_day=5, min_risk_frac=0.0, max_risk_frac=1.0, fvg_min_size_frac=0.0)
    base.update(kw)
    return StrategyConfig(**base)


def _feed(engine, day, htf_rows, ltf_rows):
    """Feed HTF bars (hourly, 7 per session, on the sessions BEFORE `day`) then LTF bars on
    `day` at the given (HH:MM, o, h, l, c)."""
    sessions = pd.bdate_range(end=pd.Timestamp(day) - pd.Timedelta(days=1), periods=(len(htf_rows) + 6) // 7)
    times = [pd.Timestamp(f"{d.date()} 09:30", tz=NY_TZ) + pd.Timedelta(hours=k) for d in sessions for k in range(7)]
    times = times[len(times) - len(htf_rows):]
    for t, (o, h, l, c) in zip(times, htf_rows):
        engine.on_htf_bar(t, o, h, l, c)
    out = []
    for hhmm, o, h, l, c in ltf_rows:
        t = pd.Timestamp(f"{day} {hhmm}", tz=NY_TZ)
        out.append(engine.on_ltf_bar(t, o, h, l, c))
    return out


def test_bullish_setup_fires_with_rejection_and_stop_at_mid_candle():
    eng = SignalEngine(_cfg(), "SPY")
    # HTF: swing high 105 at bar 2 (lookback 2 -> confirmed at bar 4); bar 6 gap up + close above 105
    htf = [
        (100, 102, 99, 101), (101, 103, 100, 102), (102, 105, 101, 103), (103, 104, 101, 102),
        (102, 103, 100.5, 101.5), (101.5, 102.5, 100, 102),          # c1 high 102.5
        (102, 108, 101.8, 107.5),                                    # c2: BOS candle (close 107.5 > 105), mid candle
        (107.5, 110, 106.0, 109.0),                                  # c3 low 106 > 102.5 -> gap [102.5, 106.0]
    ]
    # LTF the next day: pullback into the gap with a hammer (low 105.5 inside gap, close near high)
    ltf = [
        ("09:30", 108.5, 108.8, 107.9, 108.0),
        ("09:35", 108.0, 108.2, 105.5, 108.1),   # lower wick 105.5..108.0 -> wick 2.5 / range 2.7, close 108.1
    ]
    sigs = _feed(eng, "2024-03-05", htf, ltf)
    assert sigs[0] is None
    s = sigs[1]
    assert s is not None and s.direction is Direction.BULLISH
    assert s.entry == 108.1
    assert s.stop == 101.8                       # mid candle low
    assert s.target == 108.1 + 2.0 * (108.1 - 101.8)
    assert s.gap_bottom == 102.5 and s.gap_top == 106.0
    assert s.time == pd.Timestamp("2024-03-05 09:40", tz=NY_TZ)


def test_touch_without_rejection_does_not_fire():
    eng = SignalEngine(_cfg(), "SPY")
    htf = [
        (100, 102, 99, 101), (101, 103, 100, 102), (102, 105, 101, 103), (103, 104, 101, 102),
        (102, 103, 100.5, 101.5), (101.5, 102.5, 100, 102), (102, 108, 101.8, 107.5), (107.5, 110, 106.0, 109.0),
    ]
    ltf = [("09:30", 108.5, 108.8, 107.9, 108.0), ("09:35", 108.0, 108.2, 105.5, 105.7)]  # closes at the low
    sigs = _feed(eng, "2024-03-05", htf, ltf)
    assert all(s is None for s in sigs)
    # first touch consumed: a later perfect hammer on the same gap is ignored under first_touch_only
    later = eng.on_ltf_bar(pd.Timestamp("2024-03-05 09:40", tz=NY_TZ), 108.0, 108.5, 105.6, 108.4)
    assert later is None


def test_first_touch_only_false_allows_later_rejection():
    eng = SignalEngine(_cfg(first_touch_only=False), "SPY")
    htf = [
        (100, 102, 99, 101), (101, 103, 100, 102), (102, 105, 101, 103), (103, 104, 101, 102),
        (102, 103, 100.5, 101.5), (101.5, 102.5, 100, 102), (102, 108, 101.8, 107.5), (107.5, 110, 106.0, 109.0),
    ]
    ltf = [("09:30", 108.5, 108.8, 107.9, 108.0), ("09:35", 108.0, 108.2, 105.5, 105.7)]
    _feed(eng, "2024-03-05", htf, ltf)
    later = eng.on_ltf_bar(pd.Timestamp("2024-03-05 09:40", tz=NY_TZ), 108.0, 108.5, 105.6, 108.4)
    assert later is not None


def test_entry_window_blocks_late_entries():
    eng = SignalEngine(_cfg(entry_window_end=time(11, 0)), "SPY")
    htf = [
        (100, 102, 99, 101), (101, 103, 100, 102), (102, 105, 101, 103), (103, 104, 101, 102),
        (102, 103, 100.5, 101.5), (101.5, 102.5, 100, 102), (102, 108, 101.8, 107.5), (107.5, 110, 106.0, 109.0),
    ]
    ltf = [("13:30", 108.5, 108.8, 107.9, 108.0), ("13:35", 108.0, 108.2, 105.5, 108.1)]
    sigs = _feed(eng, "2024-03-05", htf, ltf)
    assert all(s is None for s in sigs)


def test_max_trades_per_day():
    cfg = _cfg(max_trades_per_day=1)
    eng = SignalEngine(cfg, "SPY")
    htf = [
        (100, 102, 99, 101), (101, 103, 100, 102), (102, 105, 101, 103), (103, 104, 101, 102),
        (102, 103, 100.5, 101.5), (101.5, 102.5, 100, 102), (102, 108, 101.8, 107.5), (107.5, 110, 106.0, 109.0),
        (109, 112, 108.0, 111.0),                     # c3 of a second gap [110, 108]? no: c1 high 110 > low 108 -> none
        (111, 115, 110.5, 114.5), (114.5, 118, 113.0, 117.0),  # gap [112, 113.0] (c1 high 112 < c3 low 113)
    ]
    ltf = [
        ("09:30", 108.5, 108.8, 107.9, 108.0),
        ("09:35", 108.0, 108.2, 105.5, 108.1),   # fires on gap 1
        ("09:40", 116, 116.5, 112.5, 116.2),     # would fire on gap 2 but daily limit reached
    ]
    sigs = _feed(eng, "2024-03-05", htf, ltf)
    assert sigs[1] is not None and sigs[2] is None


def test_bearish_setup_with_gap_far_edge_stop_and_next_level_target():
    cfg = _cfg(stop_mode="gap_far_edge", target_mode="next_level", next_level_min_r=0.5)
    eng = SignalEngine(cfg, "SPY")
    htf = [
        (100, 101, 98, 99), (99, 100, 97, 98), (98, 99, 95, 96), (96, 98, 95.5, 97),      # swing low 95 at bar 2, confirmed bar 4
        (97, 99, 96.5, 98.5), (98.5, 100, 98, 99.5),                                     # c1 low 98
        (99.5, 100, 93, 93.5),                                                           # break: close 93.5 < 95; mid candle high 100
        (93.5, 96.5, 91, 91.5),                                                          # c3 high 96.5 < 98 -> bearish gap [96.5, 98]
    ]
    ltf = [("09:30", 92, 92.5, 91.5, 92.2), ("09:35", 92.2, 97.0, 92.0, 92.1)]         # upper wick into the gap, close near low
    sigs = _feed(eng, "2024-03-05", htf, ltf)
    s = sigs[1]
    assert s is not None and s.direction is Direction.BEARISH
    assert s.stop == 98.0                                   # gap far edge
    # next key level below 92.1: none exists (only swing low 95 above) -> fallback fixed R
    assert "fixed_r_fallback" in s.notes
    assert s.target == 92.1 - 2.0 * (98.0 - 92.1)


def test_confirmation_mode_waits_for_next_candle():
    eng = SignalEngine(_cfg(confirmation="next_close_beyond"), "SPY")
    htf = [
        (100, 102, 99, 101), (101, 103, 100, 102), (102, 105, 101, 103), (103, 104, 101, 102),
        (102, 103, 100.5, 101.5), (101.5, 102.5, 100, 102), (102, 108, 101.8, 107.5), (107.5, 110, 106.0, 109.0),
    ]
    ltf = [
        ("09:30", 108.5, 108.8, 107.9, 108.0),
        ("09:35", 108.0, 108.2, 105.5, 108.1),   # rejection candle, high 108.2
        ("09:40", 108.1, 108.9, 107.8, 108.7),   # closes above 108.2 -> confirmed, entry 108.7
    ]
    sigs = _feed(eng, "2024-03-05", htf, ltf)
    assert sigs[1] is None and sigs[2] is not None and sigs[2].entry == 108.7


def test_gap_unknown_when_ltf_candle_opened_is_ignored():
    """The HTF bar that creates a gap closes at 10:30; the 10:25 LTF candle opened before that."""
    cfg = _cfg()
    bars = synthetic_minute_bars(days=10)
    ltf = resample_bars(bars, 5)
    htf = resample_bars(bars, 60)
    sigs = run_engine(cfg, "SPY", htf, ltf)
    for s in sigs:
        assert s.time - pd.Timedelta(minutes=cfg.ltf_minutes) >= s.gap_time


def test_prefix_invariance_both_gap_sources():
    """Signals up to time T must not change when future bars are appended."""
    bars = synthetic_minute_bars(days=40, seed=3)
    for src in ("htf", "ltf"):
        cfg = _cfg(fvg_source=src)
        ltf = resample_bars(bars, 5)
        htf = resample_bars(bars, 60)
        full = run_engine(cfg, "SPY", htf, ltf)
        assert len(full) > 0, f"fixture produced no signals for {src}"
        cut_time = ltf.index[len(ltf) * 2 // 3]
        ltf_p = ltf[ltf.index < cut_time]
        htf_p = htf[htf.index < cut_time]
        part = run_engine(cfg, "SPY", htf_p, ltf_p)
        expected = [s for s in full if s.time <= cut_time]
        assert part == expected
