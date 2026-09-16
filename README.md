# trading-bot: BOS/FVG multi-timeframe options strategy on Alpaca

Research and engineering project. It backtests a Break-of-Structure + Fair-Value-Gap strategy,
evaluates it with tooling designed to expose fake edges, and paper-trades it on Alpaca.

**Paper trading only.** The broker adapter refuses non-paper credentials. Nothing here is
financial advice. Options can lose the full premium paid.

## What is in the box

```
bosfvg/
  core/        pure, deterministic, no I/O
    bars.py            bar-frame conventions, session-aligned resampling
    structure.py       fractal swings, BOS / CHoCH (close beyond a level, not a wick)
    fvg.py             3-candle gaps with touched / filled / invalidated tracking
    signals.py         the multi-timeframe engine: HTF break arms a bias, LTF rejection into a gap fires
    config.py          every tunable, named, with defaults
    risk.py            sizing and daily guards
    options_pricing.py flat-IV Black-Scholes (offline option mode)
  data/        alpaca_hist.py (SIP minute bars, free-tier clamp, parquet cache), csv_loader.py, synthetic.py
  execution/   costs.py (ONE cost model shared by backtest and paper), broker.py (protocol + fake), alpaca_broker.py
  runner/      backtest.py, replay.py, journal.py (one trade schema for every driver)
  eval/        stats.py, walkforward.py (sensitivity, walk-forward, null comparison)
  live/        contracts.py (option selection), daemon.py (paper loop with two clocks)
  cli.py
tests/         50+ tests incl. prefix-invariance (no look-ahead) and daemon-vs-backtester consistency
```

The same `SignalEngine` produces signals in backtest, replay and live. A test feeds a day's
bars to the daemon minute by minute and asserts it makes the backtester's trades.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # paste PAPER keys from app.alpaca.markets
pytest -q
```

## Commands

```bash
bosfvg check                                   # credentials, paper endpoint, clock, chain access
bosfvg fetch --symbol SPY --start 2018-01-01 --end 2026-08-31     # SIP 1-minute bars -> data_cache/
bosfvg backtest --symbol SPY --start 2018-01-01 --end 2026-08-31 --out runs/bt_spy
bosfvg backtest ... --instrument option --delta 0.5 --min-dte 1   # Black-Scholes option overlay
bosfvg sensitivity --symbol SPY --start ... --end ...             # one-at-a-time parameter sweep
bosfvg walkforward --symbol SPY --start ... --end ... --train-months 12 --test-months 3
bosfvg null --symbol SPY --start ... --end ... --sims 500          # coin-flip direction comparison
bosfvg live --symbol SPY                       # DRY RUN: logs signals/decisions, no orders
bosfvg live --symbol SPY --execute             # paper orders
bosfvg replay --symbol SPY --run-dir runs      # re-run recorded live bars through the backtester
bosfvg backtest --synthetic-days 120           # wiring check only; numbers mean nothing
```

Strategy overrides: any `StrategyConfig` field via `--config file.json` or flags
(`--fvg-source ltf --wick-ratio 0.5 --entry-end 11:00 --stop-mode gap_far_edge ...`).

## Strategy as implemented

1. **HTF (default 60m, built from 1m with 9:30-aligned bins):** fractal swings confirmed by
   `swing_lookback` candles on each side. A candle *closing* beyond the last unbroken swing is a
   BOS (with trend) or CHoCH (against trend, flips it). A break arms that direction for
   `arm_window_htf_bars`; an opposite break cancels earlier arms.
2. **Gap:** by default the HTF FVGs that formed in the impulse (between the origin swing and
   `fvg_impulse_forward_bars` after the break). `--fvg-source ltf` uses LTF gaps formed after
   the break instead.
3. **LTF (default 5m) trigger:** a candle whose wick enters a live gap, with
   `wick_ratio` of its range as entry-side wick and its close in the top/bottom
   `close_position` of the range, and which does not close through the far edge. First touch
   only by default. If several gaps are touched, the one the wick reached deepest wins.
   Optional `confirmation=next_close_beyond` waits for the next candle to close past the
   rejection candle's extreme.
4. **Stop:** beyond the gap's middle candle (`mid_candle`, the reference rule), or the gap's far
   edge, or the rejection candle's extreme. Setups with a stop tighter than `min_risk_frac` or
   wider than `max_risk_frac` are skipped.
5. **Target:** `fixed_r` (default 2R) or `next_level` (nearest confirmed HTF swing beyond
   entry with at least `next_level_min_r`, falling back to fixed R).
6. **Filters:** entries only between `entry_window_start` and `entry_window_end`
   (9:30-11:00 ET), `max_trades_per_day` (1), symbol allowlist, hard flat at `flat_time` (15:45).

## Alpaca facts this design is built around

| Fact | Consequence in code |
|---|---|
| Bracket / OCO / OTO are rejected for options (`complex orders not supported`) | The daemon is the risk manager: exits are evaluated every poll against the underlying quote (`live/daemon.py`). Stops are on the *underlying*, never on the option premium. |
| Options `time_in_force` must be `day` | Hard-coded in `alpaca_broker.py`. Every position is flat by `flat_time`. |
| Free plan: SIP historical bars if `end` is >= 15 minutes old | `alpaca_hist.py` clamps `end`; every backtest is free. |
| Free plan: real-time stock data is IEX-only; options are the 15-min delayed indicative feed | `--live-feed iex` is the default; expect IEX highs/lows to differ from SIP. Compare `replay` (SIP) with the dry-run journal (IEX) before trusting live signals. `--live-feed sip` and `--options-feed opra` need Algo Trader Plus. |
| Paper fills at NBBO without checking size | Every trade carries `pnl` (broker fill) and `pnl_conservative` (marked at bid/ask with the shared cost model). Report both. |
| Option history starts Feb 2024 | Signal research runs on the underlying over long history; the option layer is a cost overlay. |
| PDT: 4 day trades / 5 days needs $25k in a margin account | Irrelevant in paper (default $100k), decisive for any future live discussion. |

## Live daemon

`bosfvg live` bootstraps the engine with the previous `--bootstrap-days` of SIP bars, then every
`--poll-seconds`:

1. kill switch (`runs/SPY/KILL`) -> flatten and stop
2. market closed -> flatten if needed
3. **exit check** on the open position against the live underlying mid: stop, target, flat time, max hold
4. fetch today's minute bars, feed every newly *closed* LTF/HTF bar in wall-clock order
5. on a signal: daily guard (loss limit, max positions), quote, option chain -> contract nearest
   `--delta` inside `[min_dte, max_dte]` with a live quote and spread <= 10% of mid, size by
   delta-approximated loss to the stop, submit a market buy-to-open, wait for the fill
6. heartbeat `runs/SPY/heartbeat.json`, decisions `decisions.jsonl`, trades `trades.csv`,
   today's bars `bars_SPY_<feed>_<date>.parquet` (for `replay`), `state.json` for restart

On restart it resumes a recorded position and flattens any position it does not recognise.

## Evaluation discipline

Gates from the plan (`docs/PLAN.md`), in order. Decide the number before running the test.

1. Real data + the prefix-invariance test passing. Record the baseline.
2. Underlying, with costs: positive out-of-sample expectancy with >= 100 trades, and no more
   than ~3 parameters moved from default.
3. `walkforward` and `null`: edge survives out of sample, is not concentrated in one year, and
   the real average R sits in the upper tail of the coin-flip distribution.
4. Option overlay keeps a meaningful share of the underlying edge after friction.
5. One week of `live` dry run: its signals must equal `replay` on the same bars.
6. >= 8 weeks of paper, >= 100 trades, multiple regimes, before any other conversation.

Nothing in this repository has passed gate 1 yet: the historical pull needs API keys and
network access to `data.alpaca.markets`.
