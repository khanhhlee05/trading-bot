"""Statistical honesty: walk-forward splits, parameter sensitivity, and a null comparison.

None of this proves an edge. It is designed to make a fake one hard to keep.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..core.config import RiskConfig, StrategyConfig
from ..execution.costs import CostModel
from ..runner.backtest import BacktestResult, run_backtest


def _slice(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return bars[(bars.index >= start) & (bars.index < end)]


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any] = field(default_factory=dict)
    train_stats: dict = field(default_factory=dict)
    test_stats: dict = field(default_factory=dict)


def sensitivity(bars: pd.DataFrame, symbol: str, base: StrategyConfig, risk: RiskConfig, costs: CostModel,
                grid: dict[str, Iterable[Any]]) -> pd.DataFrame:
    """One-at-a-time sweep: vary each parameter alone, hold the rest at `base`.

    A real edge degrades smoothly across a parameter; a fitted one has a spike at one value.
    """
    rows = []
    for name, values in grid.items():
        for v in values:
            cfg = base.replace(**{name: v})
            r = run_backtest(bars, symbol, cfg, risk, costs)
            s = r.stats
            rows.append({"param": name, "value": v, "trades": s.get("trades", 0), "avg_r": s.get("avg_r", np.nan),
                         "t_stat": s.get("t_stat_r", np.nan), "profit_factor": s.get("profit_factor", np.nan),
                         "max_dd_pct": s.get("max_drawdown_pct", np.nan)})
    return pd.DataFrame(rows)


def grid_search(bars: pd.DataFrame, symbol: str, base: StrategyConfig, risk: RiskConfig, costs: CostModel,
                grid: dict[str, Iterable[Any]], objective: str = "t_stat_r", min_trades: int = 20) -> tuple[dict, dict]:
    names = list(grid)
    best_params: dict[str, Any] = {}
    best_stats: dict = {}
    best_val = -np.inf
    for combo in itertools.product(*(list(grid[n]) for n in names)):
        params = dict(zip(names, combo))
        r = run_backtest(bars, symbol, base.replace(**params), risk, costs)
        s = r.stats
        if s.get("trades", 0) < min_trades:
            continue
        val = s.get(objective, np.nan)
        if np.isfinite(val) and val > best_val:
            best_val, best_params, best_stats = val, params, s
    return best_params, best_stats


def walk_forward(bars: pd.DataFrame, symbol: str, base: StrategyConfig, risk: RiskConfig, costs: CostModel,
                 grid: dict[str, Iterable[Any]], train_months: int = 12, test_months: int = 3,
                 objective: str = "t_stat_r", min_trades: int = 20) -> tuple[list[Fold], pd.DataFrame]:
    """Rolling-window walk-forward: fit on `train_months`, evaluate untouched on the following
    `test_months`, then roll both windows forward by `test_months`.

    Returns the folds and the concatenated out-of-sample trade frame.
    """
    start = bars.index[0].normalize()
    end = bars.index[-1]
    folds: list[Fold] = []
    oos_trades = []
    t0 = start
    while True:
        tr_end = t0 + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if tr_end >= end:
            break
        fold = Fold(t0, tr_end, tr_end, min(te_end, end + pd.Timedelta(minutes=1)))
        train = _slice(bars, fold.train_start, fold.train_end)
        test = _slice(bars, fold.test_start, fold.test_end)
        params, tstats = grid_search(train, symbol, base, risk, costs, grid, objective, min_trades)
        fold.best_params, fold.train_stats = params, tstats
        if len(test):
            r = run_backtest(test, symbol, base.replace(**params) if params else base, risk, costs)
            fold.test_stats = r.stats
            if len(r.trades):
                df = r.trades.copy()
                df["fold"] = len(folds)
                oos_trades.append(df)
        folds.append(fold)
        t0 = t0 + pd.DateOffset(months=test_months)
    oos = pd.concat(oos_trades) if oos_trades else pd.DataFrame()
    return folds, oos


def null_comparison(result: BacktestResult, bars: pd.DataFrame, n: int = 200, seed: int = 0) -> dict:
    """Same entries, same stops-in-R, coin-flip direction. Where does the real avg R sit
    among random-direction re-runs? A percentile near 50 means the direction call adds nothing.

    Implemented by re-simulating each real trade with its direction flipped at random, using the
    same LTF walk as the backtester (stop-first precedence, same time stop).
    """
    from ..core.bars import resample_bars, rth_only
    from ..core.signals import Signal
    from ..core.structure import Direction
    from ..runner.backtest import _walk_exit

    if len(result.signals) == 0:
        return {"note": "no signals"}
    cfg = result.cfg
    ltf = resample_bars(rth_only(bars, cfg.session_start), cfg.ltf_minutes, cfg.session_start)
    rng = np.random.default_rng(seed)
    real_r = float(result.trades["pnl_r"].mean()) if len(result.trades) else float("nan")
    sims = []
    for _ in range(n):
        rs = []
        for sig in result.signals:
            flip = rng.random() < 0.5
            d = sig.direction.opposite if flip else sig.direction
            entry = sig.entry
            stop = entry - d.sign * sig.risk
            target = entry + d.sign * sig.reward_r * sig.risk
            s2 = Signal(**{**sig.__dict__, "direction": d, "stop": stop, "target": target})
            pos = int(ltf.index.searchsorted(sig.time, side="left"))
            if pos >= len(ltf):
                continue
            _, exit_px, _ = _walk_exit(ltf, pos, s2, cfg)
            rs.append((exit_px - entry) * d.sign / sig.risk)
        sims.append(float(np.mean(rs)) if rs else float("nan"))
    sims_arr = np.array(sims)
    pct = float((sims_arr < real_r).mean() * 100) if np.isfinite(real_r) else float("nan")
    return {
        "real_avg_r_gross": float(np.mean([((t.exit_underlying - t.entry_underlying) * (1 if t.direction == "bullish" else -1)) / t.risk
                                           for t in result.trades.itertuples()])) if len(result.trades) else float("nan"),
        "real_avg_r_net": real_r,
        "null_mean_r": float(np.nanmean(sims_arr)),
        "null_std_r": float(np.nanstd(sims_arr)),
        "percentile_of_real": pct,
        "n_sims": n,
    }
