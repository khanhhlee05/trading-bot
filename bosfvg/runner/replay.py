"""Replay: run the backtester over bars a live session recorded, so any live decision can be
reproduced offline. Given identical bars, replay signals must equal the live daemon's."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.bars import ensure_bars
from ..core.config import RiskConfig, StrategyConfig
from ..execution.costs import CostModel
from .backtest import BacktestResult, run_backtest


def load_recorded_bars(run_dir: str | Path, symbol: str) -> pd.DataFrame:
    files = sorted(Path(run_dir).glob(f"bars_{symbol}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no recorded bars for {symbol} in {run_dir}")
    return ensure_bars(pd.concat(pd.read_parquet(f) for f in files))


def replay(history: pd.DataFrame, recorded: pd.DataFrame, symbol: str, cfg: StrategyConfig, risk: RiskConfig,
           costs: CostModel) -> BacktestResult:
    bars = ensure_bars(pd.concat([history, recorded]))
    return run_backtest(bars, symbol, cfg, risk, costs, driver_name="replay")
