"""Backtest driver: signals from the shared engine, fills from the shared cost model.

Exit precedence on a single LTF bar, in order: gap through the stop at the open, stop, target,
time stop. Stop always wins when stop and target are both inside one bar (conservative).
One position at a time; signals that arrive while in a position are skipped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from ..core.bars import bar_close_time, ensure_bars, resample_bars, rth_only
from ..core.config import RiskConfig, StrategyConfig
from ..core.options_pricing import bs_price, strike_for_delta
from ..core.risk import size_option, size_underlying
from ..core.signals import Signal, run_engine
from ..core.structure import Direction
from ..execution.costs import CostModel
from .journal import TradeRecord, trades_to_frame

ExitReason = Literal["stop", "target", "time", "eod", "data_end", "gap_stop"]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    signals: list[Signal]
    skipped_in_position: int
    skipped_sizing: int
    cfg: StrategyConfig
    risk: RiskConfig
    costs: CostModel
    symbol: str
    stats: dict = field(default_factory=dict)


def _walk_exit(ltf: pd.DataFrame, start_pos: int, sig: Signal, cfg: StrategyConfig) -> tuple[int, float, str]:
    """Return (bar position of exit, underlying exit price, reason)."""
    d = sig.direction
    highs = ltf["high"].to_numpy()
    lows = ltf["low"].to_numpy()
    opens = ltf["open"].to_numpy()
    closes = ltf["close"].to_numpy()
    idx = ltf.index
    day = sig.time.normalize()
    flat_dt = day + pd.Timedelta(hours=cfg.flat_time.hour, minutes=cfg.flat_time.minute)
    max_hold_dt = sig.time + pd.Timedelta(minutes=cfg.max_hold_minutes) if cfg.max_hold_minutes else None
    for k in range(start_pos, len(ltf)):
        t = idx[k]
        if t.normalize() != day:
            # position carried past the session: close at the previous bar's close (should not happen with flat_time)
            return k - 1, float(closes[k - 1]), "eod"
        o, h, l, c = opens[k], highs[k], lows[k], closes[k]
        if d is Direction.BULLISH:
            if o <= sig.stop:
                return k, float(o), "gap_stop"
            if l <= sig.stop:
                return k, float(sig.stop), "stop"
            if h >= sig.target:
                return k, float(sig.target), "target"
        else:
            if o >= sig.stop:
                return k, float(o), "gap_stop"
            if h >= sig.stop:
                return k, float(sig.stop), "stop"
            if l <= sig.target:
                return k, float(sig.target), "target"
        close_t = bar_close_time(t, cfg.ltf_minutes)
        if close_t >= flat_dt or (max_hold_dt is not None and close_t >= max_hold_dt):
            return k, float(c), "time"
    return len(ltf) - 1, float(closes[-1]), "data_end"


def _expiry_for(entry_time: pd.Timestamp, risk: RiskConfig) -> pd.Timestamp:
    """First weekday at least `min_dte` calendar days out, at the 16:00 ET close.

    SPY/QQQ list daily expiries, so a weekday is always a listed expiry for the liquid names
    this project is scoped to. Holidays are ignored in the offline model.
    """
    d = entry_time.normalize() + pd.Timedelta(days=risk.min_dte)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d + pd.Timedelta(hours=16)


def _years_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return max((b - a).total_seconds() / (365.0 * 24 * 3600), 1e-6)


def run_backtest(
    base_bars: pd.DataFrame,
    symbol: str,
    cfg: StrategyConfig,
    risk: RiskConfig,
    costs: CostModel,
    driver_name: str = "backtest",
) -> BacktestResult:
    base = rth_only(ensure_bars(base_bars), cfg.session_start)
    ltf = resample_bars(base, cfg.ltf_minutes, cfg.session_start)
    htf = resample_bars(base, cfg.htf_minutes, cfg.session_start)
    signals = run_engine(cfg, symbol, htf, ltf)

    equity = risk.equity
    records: list[TradeRecord] = []
    eq_points: list[tuple[pd.Timestamp, float]] = [(ltf.index[0] if len(ltf) else pd.Timestamp.now(tz="America/New_York"), equity)]
    in_position_until = -1
    skipped_pos = 0
    skipped_size = 0
    ltf_index = ltf.index

    for n, sig in enumerate(signals):
        # first LTF bar that opens at/after the signal's close time
        start_pos = int(ltf_index.searchsorted(sig.time, side="left"))
        if start_pos >= len(ltf):
            break
        if start_pos <= in_position_until:
            skipped_pos += 1
            continue
        exit_pos, exit_under, reason = _walk_exit(ltf, start_pos, sig, cfg)
        in_position_until = exit_pos
        d = sig.direction
        entry_time = sig.time
        exit_time = bar_close_time(ltf_index[exit_pos], cfg.ltf_minutes) if reason in ("time", "eod", "data_end") else ltf_index[exit_pos]
        is_buy_entry = d is Direction.BULLISH

        if risk.instrument == "underlying":
            entry_fill = costs.stock_fill(sig.entry, is_buy_entry)
            exit_fill = costs.stock_fill(exit_under, not is_buy_entry)
            # risk per share includes the entry-side friction we already paid
            risk_per_share = abs(entry_fill - sig.stop)
            size = size_underlying(equity, risk.risk_pct, risk_per_share, risk.max_shares)
            if not size.ok:
                skipped_size += 1
                continue
            qty = size.qty
            commission = costs.stock_round_trip_commission(qty)
            pnl = (exit_fill - entry_fill) * qty * d.sign - commission
            pnl_cons = pnl
            contract = ""
            entry_mid, exit_mid = sig.entry, exit_under
        else:
            is_call = d is Direction.BULLISH
            expiry = _expiry_for(entry_time, risk)
            t_entry = _years_between(entry_time, expiry)
            t_exit = _years_between(exit_time, expiry)
            strike = strike_for_delta(sig.entry, t_entry, risk.flat_iv, is_call, risk.target_delta)
            prem_entry_mid = bs_price(sig.entry, strike, t_entry, risk.flat_iv, is_call)
            prem_at_stop = bs_price(sig.stop, strike, t_entry, risk.flat_iv, is_call)
            prem_exit_mid = bs_price(exit_under, strike, t_exit, risk.flat_iv, is_call)
            entry_fill = costs.option_fill(prem_entry_mid, True)
            exit_fill = costs.option_fill(prem_exit_mid, False)
            loss_per_contract = (entry_fill - costs.option_fill(prem_at_stop, False)) * risk.contract_multiplier
            size = size_option(equity, risk.risk_pct, loss_per_contract, risk.max_contracts)
            if not size.ok:
                skipped_size += 1
                continue
            qty = size.qty
            commission = costs.option_round_trip_commission(qty)
            pnl = (exit_fill - entry_fill) * qty * risk.contract_multiplier - commission
            pnl_cons = pnl
            contract = f"{symbol} {expiry.date()} {strike:g} {'C' if is_call else 'P'}"
            entry_mid, exit_mid = prem_entry_mid, prem_exit_mid

        equity += pnl
        pnl_r = pnl / size.risk_amount if size.risk_amount > 0 else float("nan")
        rec = TradeRecord(
            trade_id=f"{driver_name}-{n:05d}",
            driver=driver_name,
            symbol=symbol,
            instrument=risk.instrument,
            direction=d.value,
            signal_time=sig.time.isoformat(),
            entry_time=entry_time.isoformat(),
            exit_time=exit_time.isoformat(),
            exit_reason=reason,
            entry_underlying=sig.entry,
            exit_underlying=exit_under,
            stop=sig.stop,
            target=sig.target,
            risk=sig.risk,
            reward_r=sig.reward_r,
            qty=qty,
            contract=contract,
            entry_fill=float(entry_fill),
            exit_fill=float(exit_fill),
            entry_mid=float(entry_mid),
            exit_mid=float(exit_mid),
            pnl=float(pnl),
            pnl_conservative=float(pnl_cons),
            pnl_r=float(pnl_r),
            commission=float(commission),
            risk_amount=float(size.risk_amount),
            equity_after=float(equity),
            bos_time=sig.bos_time.isoformat(),
            bos_kind=sig.bos_kind.value,
            bos_level=sig.bos_level,
            gap_top=sig.gap_top,
            gap_bottom=sig.gap_bottom,
            gap_source=sig.gap_source,
            gap_time=sig.gap_time.isoformat(),
            htf_trend=sig.htf_trend.value,
            rejection_wick_ratio=sig.rejection_wick_ratio,
            rejection_close_position=sig.rejection_close_position,
            notes=sig.notes,
        )
        records.append(rec)
        eq_points.append((exit_time, equity))

    trades = trades_to_frame(records)
    eq = pd.Series([e for _, e in eq_points], index=pd.DatetimeIndex([t for t, _ in eq_points]), name="equity")
    from ..eval.stats import summarize  # local import: eval depends on nothing here, avoids cycle

    result = BacktestResult(trades, eq, signals, skipped_pos, skipped_size, cfg, risk, costs, symbol)
    result.stats = summarize(trades, eq, risk.equity)
    return result
