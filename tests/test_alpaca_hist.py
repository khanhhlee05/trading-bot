from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from bosfvg.data.alpaca_hist import AlpacaHistory, clamp_end_for_free_tier
from bosfvg.data.cache import BarCache


def test_clamp_end_respects_15_minute_rule():
    now = datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc)
    assert clamp_end_for_free_tier(datetime(2024, 6, 3, 16, 0, tzinfo=timezone.utc), now) == now - timedelta(minutes=16)
    old = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert clamp_end_for_free_tier(old, now) == old


class FakeClient:
    def __init__(self):
        self.requests = []

    def get_stock_bars(self, req):
        self.requests.append(req)
        # the SDK stores start/end as tz-naive UTC
        assert req.start.tzinfo is None
        # month start in UTC is the previous evening in New York; bars on the first NY session of the month
        first_ny_day = (pd.Timestamp(req.start, tz="UTC") + pd.Timedelta(hours=12)).tz_convert("America/New_York").normalize()
        t0 = first_ny_day + pd.Timedelta(hours=9, minutes=30)
        bars = [SimpleNamespace(timestamp=t0 + pd.Timedelta(minutes=i), open=100 + i, high=101 + i, low=99 + i, close=100.5 + i, volume=10)
                for i in range(5)]
        return SimpleNamespace(data={req.symbol_or_symbols: bars})


def test_fetch_builds_request_and_caches(tmp_path):
    client = FakeClient()
    hist = AlpacaHistory(client=client, cache=BarCache(tmp_path), feed="sip", sleep_between_calls=0)
    df = hist.get_minute_bars("SPY", "2024-02-01", "2024-02-29")
    assert len(df) == 5
    req = client.requests[0]
    assert req.symbol_or_symbols == "SPY"
    assert req.feed.value == "sip"
    assert req.timeframe.amount == 1 and req.timeframe.unit.value == "Min"
    assert req.start == datetime(2024, 2, 1, 0, 0)
    # second call served from the cache
    hist.get_minute_bars("SPY", "2024-02-01", "2024-02-29")
    assert len(client.requests) == 1
    assert BarCache(tmp_path).has("SPY", "sip", "2024-02")
