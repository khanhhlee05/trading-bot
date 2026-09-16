import pandas as pd
import pytest

from bosfvg.core.bars import NY_TZ, ensure_bars


def make_bars(rows, start="2024-03-04 09:30", minutes=5):
    """rows: list of (open, high, low, close). Builds consecutive RTH bars."""
    t0 = pd.Timestamp(start, tz=NY_TZ)
    idx = [t0 + pd.Timedelta(minutes=minutes * i) for i in range(len(rows))]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx, name="time"))
    df["volume"] = 1000.0
    return ensure_bars(df)


@pytest.fixture
def make():
    return make_bars
