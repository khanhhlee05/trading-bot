from bosfvg.core.structure import Direction, EventKind, Trend, detect_structure, find_swings
import numpy as np


def test_find_swings_confirmation_index():
    high = np.array([10, 11, 12, 15, 12, 11, 10, 9, 8, 7, 8.0])
    low = high - 1
    swings = find_swings(high, low, lookback=2)
    highs = [s for s in swings if s.is_high]
    assert len(highs) == 1
    assert highs[0].index == 3 and highs[0].confirmed_index == 5 and highs[0].price == 15


def test_close_beyond_not_wick_breaks(make):
    # swing high 15 at bar 3 (confirmed at bar 5 with lookback 2), then a wick above but close below: no event
    rows = [
        (10, 11, 9, 10), (10, 12, 9, 11), (11, 13, 10, 12), (12, 15, 11, 13), (13, 14, 12, 12.5),
        (12.5, 13, 11, 12), (12, 16, 11.5, 14.5),  # wick to 16, close 14.5 < 15: no break
        (14.5, 15.5, 14, 15.2),                    # close 15.2 > 15: break
    ]
    df = make(rows)
    events, swings, trend = detect_structure(df, lookback=2)
    assert len(events) == 1
    e = events[0]
    assert e.index == 7 and e.direction is Direction.BULLISH and e.level == 15
    assert e.kind is EventKind.CHOCH  # trend was unknown; first break is a change of character
    assert trend is Trend.BULLISH


def test_bos_then_choch(make):
    # up-leg break (CHoCH from unknown), then another up break (BOS), then a break of a swing low (CHoCH bearish)
    rows = [
        (10, 11, 9, 10), (10, 12, 9, 11), (11, 13, 10, 12), (12, 15, 11, 13), (13, 14, 12, 12.5),
        (12.5, 13, 11, 12), (12, 13, 11.5, 12.8), (12.8, 15.5, 12.5, 15.2),  # bar 7: break 15 -> CHoCH bull
        (15.2, 17, 15, 16.5), (16.5, 18, 16, 17), (17, 17.5, 16, 16.2), (16.2, 16.8, 15.8, 16),  # swing high 18 at bar 9 confirmed at 11
        (16, 16.5, 15.5, 15.7), (15.7, 18.5, 15.6, 18.3),  # bar 13: close 18.3 > 18 -> BOS bull
        (18.3, 18.6, 17, 17.2), (17.2, 17.4, 15.9, 16.0), (16.0, 16.2, 15.0, 15.1),  # lows: swing low 15.5 at bar 12 confirmed at 14; bar 15 closes 16.0 > 15.5; bar 16 closes 15.1 < 15.5 -> CHoCH bear
    ]
    df = make(rows)
    events, swings, trend = detect_structure(df, lookback=2)
    kinds = [(e.kind, e.direction) for e in events]
    assert kinds[0] == (EventKind.CHOCH, Direction.BULLISH)
    assert kinds[1] == (EventKind.BOS, Direction.BULLISH)
    assert (EventKind.CHOCH, Direction.BEARISH) in kinds
    assert trend is Trend.BEARISH


def test_structure_is_causal():
    from bosfvg.data.synthetic import synthetic_minute_bars
    from bosfvg.core.bars import resample_bars

    htf = resample_bars(synthetic_minute_bars(days=15), 60)
    full, _, _ = detect_structure(htf, 3)
    n = len(htf)
    for cut in (n // 3, n // 2, n - 5):
        part, _, _ = detect_structure(htf.iloc[:cut], 3)
        assert part == [e for e in full if e.index < cut]
