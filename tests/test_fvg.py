from bosfvg.core.fvg import detect_fvgs, unmitigated_gaps_as_of
from bosfvg.core.structure import Direction


def test_bullish_gap_and_mitigation(make):
    rows = [
        (10, 10.5, 9.5, 10.2),   # c1 high 10.5
        (10.2, 12, 10.1, 11.8),  # c2 (middle)
        (11.8, 12.5, 11.0, 12.2),  # c3 low 11.0 > 10.5 -> bullish gap [10.5, 11.0]
        (12.2, 12.6, 11.5, 12.0),  # no touch
        (12.0, 12.1, 10.9, 11.7),  # low 10.9 <= 11.0: touched
        (11.7, 11.8, 10.4, 11.2),  # low 10.4 <= 10.5: filled, close 11.2 > 10.5: still not invalid
        (11.2, 11.3, 10.0, 10.2),  # close 10.2 < 10.5: invalid
    ]
    gaps = detect_fvgs(make(rows))
    assert len(gaps) == 1
    g = gaps[0]
    assert g.direction is Direction.BULLISH and g.index == 2
    assert (g.bottom, g.top) == (10.5, 11.0)
    assert (g.mid_high, g.mid_low) == (12.0, 10.1)
    assert (g.touched_index, g.filled_index, g.invalid_index) == (4, 5, 6)
    assert g.is_unmitigated_as_of(3) and not g.is_unmitigated_as_of(4)
    assert g.is_live_as_of(5) and not g.is_live_as_of(6)
    assert unmitigated_gaps_as_of(gaps, 1) == []  # not known yet
    assert unmitigated_gaps_as_of(gaps, 3) == [g]


def test_bearish_gap(make):
    rows = [
        (20, 20.5, 19.8, 20.0),  # c1 low 19.8
        (20.0, 20.1, 18.0, 18.2),
        (18.2, 19.0, 17.5, 17.8),  # c3 high 19.0 < 19.8 -> bearish gap [19.0, 19.8]
        (17.8, 19.2, 17.6, 18.0),  # high 19.2 >= 19.0: touched
    ]
    gaps = detect_fvgs(make(rows))
    assert len(gaps) == 1
    g = gaps[0]
    assert g.direction is Direction.BEARISH and (g.bottom, g.top) == (19.0, 19.8)
    assert g.near_edge == 19.0 and g.far_edge == 19.8
    assert g.touched_index == 3


def test_min_size_filter(make):
    rows = [(10, 10.5, 9.5, 10.2), (10.2, 12, 10.1, 11.8), (11.8, 12.5, 10.6, 12.2)]
    assert len(detect_fvgs(make(rows), min_size=0.5)) == 0
    assert len(detect_fvgs(make(rows), min_size=0.05)) == 1
