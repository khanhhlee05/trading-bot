import pandas as pd
import pytest

from bosfvg.core.bars import NY_TZ, bar_close_time, ensure_bars, resample_bars, rth_only
from bosfvg.data.synthetic import synthetic_minute_bars


def test_ensure_bars_normalizes_tz_and_sorts():
    idx = pd.to_datetime(["2024-03-04 15:30:00", "2024-03-04 14:30:00"], utc=True)
    df = pd.DataFrame({"open": [1, 1], "high": [2, 2], "low": [0.5, 0.5], "close": [1.5, 1.5], "volume": [1, 1]}, index=idx)
    out = ensure_bars(df)
    assert str(out.index.tz) == NY_TZ
    assert out.index[0] < out.index[1]
    assert out.index[0].hour == 9 and out.index[0].minute == 30


def test_ensure_bars_rejects_bad_ohlc():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-03-04 09:30", tz=NY_TZ)])
    df = pd.DataFrame({"open": [10], "high": [9], "low": [8], "close": [9.5], "volume": [1]}, index=idx)
    with pytest.raises(ValueError):
        ensure_bars(df)


def test_resample_is_session_aligned():
    bars = synthetic_minute_bars(days=2)
    h = resample_bars(bars, 60)
    times = h.index[:7]
    assert [t.strftime("%H:%M") for t in times] == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    # the last hour bar of the day only spans 15:30-16:00; it must not bleed into the next day
    assert h.index[7].day != h.index[6].day
    # OHLC aggregation is exact
    first_hour = bars.loc[bars.index[0] : bars.index[0] + pd.Timedelta(minutes=59)]
    assert h.iloc[0].open == first_hour.iloc[0].open
    assert h.iloc[0].high == first_hour.high.max()
    assert h.iloc[0].low == first_hour.low.min()
    assert h.iloc[0].close == first_hour.iloc[-1].close
    assert h.iloc[0].volume == first_hour.volume.sum()


def test_resample_five_minute_count():
    bars = synthetic_minute_bars(days=1)
    assert len(resample_bars(bars, 5)) == 78


def test_bar_close_time_caps_at_session_close():
    t = pd.Timestamp("2024-03-04 15:30", tz=NY_TZ)
    assert bar_close_time(t, 60) == pd.Timestamp("2024-03-04 16:00", tz=NY_TZ)
    t = pd.Timestamp("2024-03-04 09:30", tz=NY_TZ)
    assert bar_close_time(t, 5) == pd.Timestamp("2024-03-04 09:35", tz=NY_TZ)


def test_rth_only_drops_premarket():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-03-04 09:00", tz=NY_TZ), pd.Timestamp("2024-03-04 09:30", tz=NY_TZ), pd.Timestamp("2024-03-04 16:00", tz=NY_TZ)])
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    out = rth_only(ensure_bars(df))
    assert len(out) == 1
