"""Command line entry points.

  bosfvg fetch       pull and cache 1-minute SIP bars from Alpaca (free tier: end >= 15 min ago)
  bosfvg backtest    run the strategy over cached/CSV/synthetic bars
  bosfvg sensitivity one-at-a-time parameter sweep
  bosfvg walkforward anchored walk-forward with a small grid
  bosfvg null        random-direction null comparison for a backtest
  bosfvg live        paper daemon (dry-run by default; --execute places paper orders)
  bosfvg replay      re-run the backtester over bars a live session recorded
  bosfvg check       verify credentials, paper endpoint, clock, and data feed access
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import time
from pathlib import Path

import pandas as pd

from .core.config import RiskConfig, StrategyConfig
from .eval.stats import breakdown, format_stats
from .execution.costs import CostModel


def _hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def _add_strategy_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("strategy")
    g.add_argument("--config", help="JSON file of StrategyConfig overrides")
    g.add_argument("--htf", type=int, help="structure timeframe in minutes (default 60)")
    g.add_argument("--ltf", type=int, help="entry timeframe in minutes (default 5)")
    g.add_argument("--lookback", type=int, help="swing lookback on HTF")
    g.add_argument("--arm-window", type=int, help="HTF bars an arm stays valid")
    g.add_argument("--fvg-source", choices=["htf", "ltf"])
    g.add_argument("--wick-ratio", type=float)
    g.add_argument("--close-position", type=float)
    g.add_argument("--stop-mode", choices=["mid_candle", "gap_far_edge", "rejection_extreme"])
    g.add_argument("--target-mode", choices=["fixed_r", "next_level"])
    g.add_argument("--reward-r", type=float)
    g.add_argument("--entry-start", type=_hhmm, help="HH:MM America/New_York")
    g.add_argument("--entry-end", type=_hhmm)
    g.add_argument("--flat-time", type=_hhmm)
    g.add_argument("--max-trades-per-day", type=int)
    g.add_argument("--confirmation", choices=["none", "next_close_beyond"])
    g.add_argument("--no-first-touch-only", action="store_true")
    r = p.add_argument_group("risk & costs")
    r.add_argument("--instrument", choices=["underlying", "option"], default="underlying")
    r.add_argument("--equity", type=float, default=100_000)
    r.add_argument("--risk-pct", type=float, default=0.005)
    r.add_argument("--delta", type=float, default=0.5)
    r.add_argument("--min-dte", type=int, default=1)
    r.add_argument("--max-dte", type=int, default=7)
    r.add_argument("--iv", type=float, default=0.18, help="flat IV for the offline option model")
    r.add_argument("--stock-spread", type=float, default=0.01)
    r.add_argument("--stock-slippage", type=float, default=0.0)
    r.add_argument("--option-spread", type=float, default=0.02)
    r.add_argument("--option-slippage", type=float, default=0.01)
    r.add_argument("--commission-per-contract", type=float, default=0.65)


def _add_data_args(p: argparse.ArgumentParser) -> None:
    d = p.add_argument_group("data")
    d.add_argument("--symbol", default="SPY")
    d.add_argument("--start", help="YYYY-MM-DD (cached Alpaca bars)")
    d.add_argument("--end", help="YYYY-MM-DD")
    d.add_argument("--csv", help="minute bars CSV instead of Alpaca cache")
    d.add_argument("--synthetic-days", type=int, help="use synthetic bars (wiring check only)")
    d.add_argument("--feed", default="sip", help="historical feed (sip on any plan when end >= 15 min ago)")


def build_strategy(args: argparse.Namespace) -> StrategyConfig:
    cfg = StrategyConfig()
    if getattr(args, "config", None):
        cfg = StrategyConfig.from_dict({**cfg.to_dict(), **json.loads(Path(args.config).read_text())})
    mapping = {
        "htf": "htf_minutes", "ltf": "ltf_minutes", "lookback": "swing_lookback", "arm_window": "arm_window_htf_bars",
        "fvg_source": "fvg_source", "wick_ratio": "wick_ratio", "close_position": "close_position", "stop_mode": "stop_mode",
        "target_mode": "target_mode", "reward_r": "reward_r", "entry_start": "entry_window_start", "entry_end": "entry_window_end",
        "flat_time": "flat_time", "max_trades_per_day": "max_trades_per_day", "confirmation": "confirmation",
    }
    changes = {v: getattr(args, k) for k, v in mapping.items() if getattr(args, k, None) is not None}
    if getattr(args, "no_first_touch_only", False):
        changes["first_touch_only"] = False
    return cfg.replace(**changes) if changes else cfg


def build_risk(args: argparse.Namespace) -> RiskConfig:
    return RiskConfig(equity=args.equity, risk_pct=args.risk_pct, instrument=args.instrument, target_delta=args.delta,
                      min_dte=args.min_dte, max_dte=args.max_dte, flat_iv=args.iv)


def build_costs(args: argparse.Namespace) -> CostModel:
    return CostModel(stock_spread=args.stock_spread, stock_slippage=args.stock_slippage, option_spread=args.option_spread,
                     option_slippage=args.option_slippage, option_commission_per_contract=args.commission_per_contract)


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic_days:
        from .data.synthetic import synthetic_minute_bars

        print(f"[synthetic] {args.synthetic_days} days; results are a wiring check, not evidence", file=sys.stderr)
        return synthetic_minute_bars(days=args.synthetic_days)
    if args.csv:
        from .data.csv_loader import load_csv_bars

        return load_csv_bars(args.csv)
    if not (args.start and args.end):
        raise SystemExit("need --start and --end (Alpaca cache), or --csv, or --synthetic-days")
    from .data.alpaca_hist import AlpacaHistory

    return AlpacaHistory(feed=args.feed).get_minute_bars(args.symbol, args.start, args.end)


def _print_result(r, out_dir: Path | None) -> None:
    print(format_stats(r.stats))
    print(f"signals {len(r.signals)}  skipped in-position {r.skipped_in_position}  skipped sizing {r.skipped_sizing}")
    if len(r.trades):
        print("\nby year\n" + breakdown(r.trades, "year").to_string())
        print("\nby exit\n" + breakdown(r.trades, "exit_reason").to_string())
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        r.trades.to_csv(out_dir / "trade_log.csv", index=False)
        r.equity.to_csv(out_dir / "equity_curve.csv", header=True)
        (out_dir / "stats.json").write_text(json.dumps({**r.stats, "config": r.cfg.to_dict()}, indent=2, default=str))
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            r.equity.plot(ax=ax)
            ax.set_title(f"{r.symbol} equity ({r.risk.instrument})")
            fig.tight_layout()
            fig.savefig(out_dir / "equity_curve.png", dpi=120)
        except ImportError:
            pass
        print(f"\nwrote {out_dir}/trade_log.csv, equity_curve.csv, stats.json")


# ------------------------------------------------------------------ commands

def cmd_fetch(args: argparse.Namespace) -> None:
    from .data.alpaca_hist import AlpacaHistory

    hist = AlpacaHistory(feed=args.feed)
    df = hist.get_minute_bars(args.symbol, args.start, args.end)
    print(f"{args.symbol}: {len(df)} RTH minute bars, {df.index[0] if len(df) else '-'} .. {df.index[-1] if len(df) else '-'}")
    print(f"cache dir: {hist.cache.root}")


def cmd_backtest(args: argparse.Namespace) -> None:
    from .runner.backtest import run_backtest

    bars = load_bars(args)
    r = run_backtest(bars, args.symbol, build_strategy(args), build_risk(args), build_costs(args))
    _print_result(r, Path(args.out) if args.out else None)


def cmd_sensitivity(args: argparse.Namespace) -> None:
    from .eval.walkforward import sensitivity

    bars = load_bars(args)
    grid = json.loads(args.grid) if args.grid else {
        "wick_ratio": [0.2, 0.3, 0.4, 0.5, 0.6],
        "close_position": [0.5, 0.6, 0.7, 0.8],
        "reward_r": [1.0, 1.5, 2.0, 2.5, 3.0],
        "swing_lookback": [2, 3, 4, 5],
        "arm_window_htf_bars": [3, 6, 9, 12],
    }
    df = sensitivity(bars, args.symbol, build_strategy(args), build_risk(args), build_costs(args), grid)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(args.out) / "sensitivity.csv", index=False)


def cmd_walkforward(args: argparse.Namespace) -> None:
    from .eval.stats import summarize
    from .eval.walkforward import walk_forward

    bars = load_bars(args)
    grid = json.loads(args.grid) if args.grid else {"wick_ratio": [0.3, 0.4, 0.5], "reward_r": [1.5, 2.0, 2.5]}
    folds, oos = walk_forward(bars, args.symbol, build_strategy(args), build_risk(args), build_costs(args), grid,
                              train_months=args.train_months, test_months=args.test_months)
    for i, f in enumerate(folds):
        tr, te = f.train_stats, f.test_stats
        print(f"fold {i}: train {f.train_start.date()}..{f.train_end.date()} best={f.best_params} "
              f"train_avg_r={tr.get('avg_r', float('nan')):+.3f} (n={tr.get('trades', 0)})  "
              f"test {f.test_start.date()}..{f.test_end.date()} avg_r={te.get('avg_r', float('nan')):+.3f} (n={te.get('trades', 0)})")
    if len(oos):
        eq = pd.Series(oos["pnl"].cumsum().to_numpy() + args.equity, index=pd.to_datetime(oos["exit_time"], utc=True))
        print("\nOUT-OF-SAMPLE, all folds concatenated\n" + format_stats(summarize(oos.reset_index(drop=True), eq, args.equity)))
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            oos.to_csv(Path(args.out) / "oos_trades.csv", index=False)
    else:
        print("no out-of-sample trades")


def cmd_null(args: argparse.Namespace) -> None:
    from .eval.walkforward import null_comparison
    from .runner.backtest import run_backtest

    bars = load_bars(args)
    r = run_backtest(bars, args.symbol, build_strategy(args), build_risk(args), build_costs(args))
    print(format_stats(r.stats))
    res = null_comparison(r, bars, n=args.sims)
    print("\nnull comparison (same entries, coin-flip direction)")
    for k, v in res.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")


def cmd_live(args: argparse.Namespace) -> None:
    from .data.alpaca_hist import AlpacaHistory
    from .execution.alpaca_broker import AlpacaBroker
    from .live.daemon import LiveTrader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg, risk, costs = build_strategy(args), build_risk(args), build_costs(args)
    broker = AlpacaBroker(options_feed=args.options_feed)
    run_dir = Path(args.run_dir) / args.symbol
    trader = LiveTrader(args.symbol, cfg, risk, costs, broker, run_dir, dry_run=not args.execute,
                        live_feed=args.live_feed, poll_seconds=args.poll_seconds)
    end = pd.Timestamp.now(tz="America/New_York").normalize() - pd.Timedelta(minutes=1)
    start = end - pd.Timedelta(days=args.bootstrap_days)
    history = AlpacaHistory(feed="sip").get_minute_bars(args.symbol, start, end)
    trader.bootstrap(history)
    print(f"{'DRY RUN' if not args.execute else 'PAPER EXECUTION'} for {args.symbol}; run dir {run_dir}; touch {run_dir}/KILL to stop", file=sys.stderr)
    trader.run()


def cmd_replay(args: argparse.Namespace) -> None:
    from .data.alpaca_hist import AlpacaHistory
    from .runner.replay import load_recorded_bars, replay

    recorded = load_recorded_bars(Path(args.run_dir) / args.symbol, args.symbol)
    first = recorded.index[0].normalize()
    history = AlpacaHistory(feed="sip").get_minute_bars(args.symbol, first - pd.Timedelta(days=args.bootstrap_days), first - pd.Timedelta(minutes=1))
    r = replay(history, recorded, args.symbol, build_strategy(args), build_risk(args), build_costs(args))
    _print_result(r, Path(args.out) if args.out else None)


def cmd_check(args: argparse.Namespace) -> None:
    from .data.alpaca_hist import load_credentials

    key, secret, paper = load_credentials()
    print(f"credentials: present, paper={paper}")
    if not paper:
        raise SystemExit("ALPACA_PAPER is not true; refusing")
    from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest, StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient

    tc = TradingClient(key, secret, paper=True)
    acct = tc.get_account()
    clock = tc.get_clock()
    print(f"account: status={acct.status} equity={acct.equity} options_approved_level={getattr(acct, 'options_approved_level', '?')}")
    print(f"clock: open={clock.is_open} next_open={clock.next_open} next_close={clock.next_close}")
    q = StockHistoricalDataClient(key, secret).get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=args.symbol))[args.symbol]
    print(f"{args.symbol} latest quote: bid={q.bid_price} ask={q.ask_price} at {q.timestamp}")
    chain = OptionHistoricalDataClient(key, secret).get_option_chain(OptionChainRequest(underlying_symbol=args.symbol))
    sample = next(iter(chain.items()), None)
    if sample:
        sym, snap = sample
        print(f"option chain: {len(chain)} contracts; e.g. {sym} greeks={'yes' if snap.greeks else 'no'} quote={snap.latest_quote.bid_price if snap.latest_quote else None}/{snap.latest_quote.ask_price if snap.latest_quote else None}")
    else:
        print("option chain: empty")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bosfvg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch", help="pull and cache minute bars from Alpaca")
    _add_data_args(s)
    s.set_defaults(fn=cmd_fetch)

    for name, fn, help_ in [("backtest", cmd_backtest, "run a backtest"), ("sensitivity", cmd_sensitivity, "parameter sweep"),
                            ("walkforward", cmd_walkforward, "walk-forward evaluation"), ("null", cmd_null, "null comparison")]:
        s = sub.add_parser(name, help=help_)
        _add_data_args(s)
        _add_strategy_args(s)
        s.add_argument("--out", help="output directory")
        if name in ("sensitivity", "walkforward"):
            s.add_argument("--grid", help="JSON {param: [values]}")
        if name == "walkforward":
            s.add_argument("--train-months", type=int, default=12)
            s.add_argument("--test-months", type=int, default=3)
        if name == "null":
            s.add_argument("--sims", type=int, default=200)
        s.set_defaults(fn=fn)

    s = sub.add_parser("live", help="paper daemon")
    s.add_argument("--symbol", default="SPY")
    _add_strategy_args(s)
    s.add_argument("--execute", action="store_true", help="place PAPER orders (default: dry run, log only)")
    s.add_argument("--live-feed", default="iex", help="today's bar feed: iex (free) or sip (Algo Trader Plus)")
    s.add_argument("--options-feed", default=None, help="opra (subscription) or indicative; default lets Alpaca pick")
    s.add_argument("--poll-seconds", type=float, default=5.0)
    s.add_argument("--bootstrap-days", type=int, default=10)
    s.add_argument("--run-dir", default="runs")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("replay", help="replay recorded live bars through the backtester")
    s.add_argument("--symbol", default="SPY")
    _add_strategy_args(s)
    s.add_argument("--run-dir", default="runs")
    s.add_argument("--bootstrap-days", type=int, default=10)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("check", help="verify Alpaca paper access")
    s.add_argument("--symbol", default="SPY")
    s.set_defaults(fn=cmd_check)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
