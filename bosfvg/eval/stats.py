"""Trade statistics. Everything is computed from the shared trade schema so backtest, replay
and paper journals are compared with the same code."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Returns (max drawdown in dollars, max drawdown as a fraction of the running peak)."""
    if len(equity) == 0:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = equity - peak
    frac = (dd / peak).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min()), float(frac.min())


def summarize(trades: pd.DataFrame, equity: pd.Series, start_equity: float) -> dict:
    n = int(len(trades))
    if n == 0:
        return {"trades": 0, "note": "no trades"}
    pnl = trades["pnl"].astype(float)
    r = trades["pnl_r"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else math.inf
    dd_abs, dd_frac = max_drawdown(equity)
    # daily P&L series for a Sharpe that is not inflated by intraday trade count
    days = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("America/New_York").dt.normalize()
    daily = pnl.groupby(days).sum()
    daily_ret = daily / start_equity
    sharpe = float(daily_ret.mean() / daily_ret.std(ddof=1) * math.sqrt(252)) if len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0 else float("nan")
    # standard error of expectancy in R: how much of the average is noise
    se_r = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    t_stat = float(r.mean() / se_r) if se_r and se_r > 0 else float("nan")
    out = {
        "trades": n,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": pf,
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
        "se_r": se_r,
        "t_stat_r": t_stat,
        "total_pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / start_equity),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "max_drawdown": dd_abs,
        "max_drawdown_pct": dd_frac,
        "sharpe_daily": sharpe,
        "commission_total": float(trades["commission"].astype(float).sum()),
        "trading_days": int(daily.shape[0]),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }
    return out


def breakdown(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group expectancy. `by` is a column name or 'year' / 'month' / 'weekday' / 'hour'."""
    if len(trades) == 0:
        return pd.DataFrame()
    t = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert("America/New_York")
    if by == "year":
        key = t.dt.year
    elif by == "month":
        key = t.dt.to_period("M").astype(str)
    elif by == "weekday":
        key = t.dt.day_name()
    elif by == "hour":
        key = t.dt.hour
    else:
        key = trades[by]
    g = trades.groupby(key)
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate": g["pnl"].apply(lambda s: float((s > 0).mean())),
        "avg_r": g["pnl_r"].mean(),
        "total_pnl": g["pnl"].sum(),
    })
    return out


def format_stats(stats: dict) -> str:
    if stats.get("trades", 0) == 0:
        return "no trades"
    lines = [
        f"trades            {stats['trades']}  over {stats['trading_days']} trading days",
        f"win rate          {stats['win_rate']:.1%}",
        f"profit factor     {stats['profit_factor']:.2f}",
        f"avg R             {stats['avg_r']:+.3f}  (se {stats['se_r']:.3f}, t {stats['t_stat_r']:+.2f})",
        f"median R          {stats['median_r']:+.3f}",
        f"total P&L         {stats['total_pnl']:+,.2f}  ({stats['return_pct']:+.2%})",
        f"max drawdown      {stats['max_drawdown']:,.2f}  ({stats['max_drawdown_pct']:.2%})",
        f"sharpe (daily)    {stats['sharpe_daily']:.2f}",
        f"commissions       {stats['commission_total']:,.2f}",
        f"exit reasons      {stats['exit_reasons']}",
    ]
    return "\n".join(lines)
