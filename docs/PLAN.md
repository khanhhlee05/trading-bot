# BOS/FVG Options Bot on Alpaca — Research + Build Proposal

Research/engineering plan. Not financial advice. Options can lose 100% of premium.

---

## 0. Repo mismatch (blocking a first commit, not the planning)

`/home/user/nova-agent` is the Villanova Brightspace Chrome extension (TypeScript/React).
There is no `structure.py`, `fvg.py`, `backtester.py` here — zero `.py` files in the tree.
The backtesting engine described in the handoff lives somewhere else (local machine or another repo).

Decision needed before any code lands: new repo (recommended, e.g. `bos-fvg-bot`), or push the
existing Python engine into this session so it can be extended here.

---

## 1. Alpaca research — the facts that actually change the design

### 1.1 No bracket / OCO orders on options

Alpaca rejects bracket and OCO order classes for options: `complex orders not supported for
options trading`. Single-leg options accept `market`, `limit`, `stop`, `stop_limit`;
`time_in_force` must be `day`.

**Consequence:** Section 6 of the handoff ("let the broker hold the stop") is not available.
The bot is the risk manager. Two follow-ons:

- A resting `stop` on the *option premium* is the wrong instrument anyway — premium moves with
  IV and the spread, so a 2-cent quote blip on a $1.20 contract fires a stop the underlying never
  justified. The strategy's invalidation is defined on the **underlying** (far edge of the FVG).
- So the exit monitor must poll the **underlying** every few seconds (5–15s), not once per closed
  candle. Entry logic stays candle-close driven; exit logic cannot be. Budget for a always-on
  process with a hard "flatten everything" path (EOD, error, disconnect, kill switch).

### 1.2 Data tiers decide what is honestly testable

Free (Basic) plan:
- **Historical stock bars: full SIP coverage, free**, as long as `end` is ≥15 minutes old.
  Backtesting SPY/QQQ at 1H and 1–5m on real consolidated data costs $0. 200 req/min.
- **Real-time stocks: IEX only** (~2–3% of consolidated volume).
- **Options: Indicative feed, 15-minute delayed.** Real-time OPRA needs Algo Trader Plus.

Algo Trader Plus: $99/mo — full real-time SIP + OPRA, 10,000 req/min.

**Consequence:** a wick/gap strategy is defined by exact highs and lows. IEX-only 5-minute bars
have *different* highs, lows and gap geometry than SIP bars — the FVG that the backtest saw on SIP
data may not exist in the live IEX feed. That is a silent backtest/live divergence, and it is
exactly the class of bug that makes a paper run uninterpretable.

Plan: phases 1–3 on free historical SIP. Subscribe to Algo Trader Plus only at the point of
running live paper execution, and before that, quantify the damage — re-run the signal engine over
IEX-only bars vs SIP bars for the same week and count how many signals survive. If the answer is
"most signals change", the $99/mo is mandatory, not optional.

### 1.3 Options historical data starts Feb 2024

Real option-chain history is ~2.5 years deep. Underlying history is decades.
So: run structural-edge research on the underlying over a long window; use the 2024+ option data
only to calibrate a realistic cost/fill model and to validate contract selection.

### 1.4 Paper fills are optimistic

Paper matches against real-time NBBO, but **order quantity is not checked against NBBO size** —
you can get filled on size that doesn't exist. Partial fills are simulated randomly (~10%).
Mid-price-ish fills with no slippage on small size.

**Consequence:** never record a paper fill as the truth. Log the quote at signal time and mark
every trade at the **conservative side** (buy at ask, sell at bid) in a parallel "realistic" P&L
column. Report both. A strategy that is only profitable on the paper engine's fill price is not
profitable.

### 1.5 Misc that matters

- Options Level 3 is on by default in paper; multi-leg (`order_class: "mleg"`, `legs[]`) is
  supported in paper — spreads, condors, straddles available when you want to cut vega/theta risk.
- Index options (SPX, SPXW, XSP, VIX) are available **in paper** via the Trading API. Cash-settled,
  no assignment, no early-exercise noise — a cleaner test bed than SPY equity options, but paper-only
  today, so don't build the live path around them.
- PDT: paper accounts default to $100k equity, so the 4-day-trades rule won't bite in paper. Live
  intraday options = day trades → $25k minimum in a margin account. Alpaca rejects the order with
  403 rather than letting you drift into a flag. Know this now; it affects whether "live" is even
  reachable for you later.
- No Alpaca MCP connector in this workspace — `alpaca-py` directly, as the handoff already assumed.
  Note `docs.alpaca.markets` and `forum.alpaca.markets` are blocked by this session's egress proxy,
  so API details get verified against the SDK and live paper responses, not by reading the docs here.

---

## 2. Strategy research — the honest read on BOS/FVG

Published evidence is mixed and mostly unflattering to the raw pattern:

- A statistical sweep of ICT/SMC core entries found **no statistically significant forward-return
  edge** (best t-stat ≈ +1.22, 0 of 648 backtests beat buy-and-hold). On SPY, 1,122 bullish-FVG
  retrace events produced ≈ **−0.005% edge at 5 days** — i.e. nothing.
- Vendor/practitioner backtests that report 60%+ win rates get there only after aggressive
  filtering (one EUR/USD study: 1,247 setups → 341 kept, 27%, then 66.5% win rate, +1.14R).
  That filtering is where overfitting lives, and it's also where any real edge would live.
- One consistently reported non-trivial fact: **session matters.** FVGs formed in the NY open
  window are revisited more often than Asian-session FVGs. Your 9:30–11:00 ET filter is the
  single most evidence-supported rule in the whole spec — implement it early, it's cheap.
- Also reported: ~64% of FVGs never fill. "Wait for the pullback" therefore selects a minority
  of setups — which is fine (that's what a filter does) but it means your sample size per month is
  small, and small samples are how you end up believing in noise.

**Position to build from:** BOS/FVG is a hypothesis, not a known edge. The deliverable of this
project is not "a profitable bot"; it's **a harness that can tell you, with a straight face,
whether this thing has an edge** — and that stays useful even if this particular signal dies.
Design every phase around falsification, with an explicit kill criterion written *before* you see
the number.

### 2.1 Options make it strictly harder — prove the signal first

Buying options to express a 1–3R intraday directional move adds three taxes the backtest currently
ignores: the spread (SPY near-ATM is $0.01–0.03 wide in calm tape, $0.05–0.20 on stressed 0DTE),
theta (brutal on same-day expiry after ~noon), and vega (a correct directional call can lose money
into an IV crush).

So the gate is: **if the signal has no edge trading SPY shares, options cannot rescue it — they can
only amplify a negative expectancy.** Test on the underlying first. Only after the signal clears
that bar do you ask whether the convexity is worth the friction.

When you do get there:
- Delta 0.45–0.60 (ATM/slightly ITM). Cheap far-OTM contracts look attractive per-contract and have
  the worst spread-as-%-of-premium and the worst fill quality.
- Expiry: start at 1–7 DTE, **not 0DTE**. 0DTE gamma punishes infrastructure bugs in minutes.
  Move to 0DTE only after the loop has run clean for weeks.
- SPY/QQQ only at first — penny-wide strikes, deep books.
- Time-stop every position (the handoff has no time-based exit; theta means a trade that is
  "still valid but going nowhere" bleeds). Hard flat by a fixed time (e.g. 15:45 ET).

---

## 3. Gaps in the existing code, re-prioritized

The handoff's Section 5 ordering is roughly right; I'd re-rank on "what invalidates results" vs
"what is polish":

| # | Gap | Priority | Why |
|---|-----|----------|-----|
| 7 | Synthetic data | **P0** | Every current number is meaningless until real bars are in. Cheapest fix, free tier. |
| 4 | Session/liquidity filter | **P0** | Best-evidenced rule in the spec, ~20 lines. |
| 1 | Multi-timeframe | **P1** | Structural: single-TF results don't transfer to the reference strategy at all. |
| 2 | Rejection-wick confirmation | **P1** | Touch-triggers fire on every wick-through; this is the difference between "FVG strategy" and "mean-revert into any gap". |
| 3 | Real options pricing | **P2** | Needed for *P&L* realism, not for *signal* validity. Keep flat-IV BS as the fast offline mode. |
| 5 | Target = next key level | **P2** | Keep fixed-R as a config option and as the control arm. Fixed-R is easier to evaluate statistically; "next key level" is another free parameter to overfit. |
| — | **Cost & fill model** | **P1 (new)** | Not in the handoff. Without spread + slippage modeling, backtest expectancy is fiction. |
| — | **Statistical harness** | **P1 (new)** | Not in the handoff. Walk-forward split, out-of-sample holdout, per-parameter sensitivity, null-hypothesis comparison. Without this you will overfit; it's not optional. |
| 6 | CRT | **P3** | Separate module, separate phase, only after BOS/FVG passes or fails cleanly. |

Two more that aren't in the handoff at all:

- **Look-ahead audit.** `structure.py` claims causal correctness. Prove it: a test that runs the
  signal engine on `bars[:i]` for each `i` and asserts identical signals to the full-series run.
  This is the #1 source of fake backtest edge and it is cheap to rule out permanently.
- **Regime labeling.** Tag each trade with VIX bucket and trend/chop regime. A strategy that only
  works in one regime is fine — if you know which one.

---

## 4. Proposed architecture

Guiding rule: **one signal engine, three drivers.** The exact same code produces signals in
backtest, in replay, and live. Any divergence between backtest and paper is then an execution or
data issue, never a logic issue — which is the only way to debug a live bot at all.

```
core/                    pure, no I/O, no network, deterministic
  bars.py                BarSeries: tz-aware, typed, resample 1m -> 5m/1H
  structure.py           swings, BOS/CHoCH        (existing, extended)
  fvg.py                 gaps, mitigation         (existing, extended)
  signals.py             MTF state machine: HTF bias + LTF trigger -> Signal
  filters.py             session, liquidity, regime, one-trade-per-day
  risk.py                position sizing, stop/target/time-stop resolution

data/
  alpaca_hist.py         historical SIP bars -> BarSeries (free tier, end >= now-15m)
  alpaca_live.py         live bar poller (IEX free / SIP paid) -> same BarSeries
  cache.py               parquet on disk; never re-pull the same day twice

execution/
  broker.py              Broker protocol: quote(), submit(), positions(), flatten()
  alpaca_broker.py       alpaca-py paper impl: chain resolution, entry, managed exit
  sim_broker.py          backtest impl with explicit cost model
  costs.py               spread + slippage + commission; ONE place, shared by both

runner/
  backtest.py            drive core over historical bars
  replay.py              drive core over recorded live bars (bug repro, no network)
  live.py                the daemon: candle-close entry loop + fast exit monitor
  journal.py             one trade schema for all three; backtest and paper are diffable

eval/
  stats.py               expectancy, PF, Sharpe/Sortino, DD, per-regime breakdown
  walkforward.py         train/test splits, parameter sensitivity, null comparison
  report.py              the go/no-go artifact
```

Non-obvious commitments:
- **`costs.py` is shared.** The backtester and the live trade journal price friction with the same
  function, so "backtest said +0.4R, paper gave +0.1R" is attributable.
- **The live daemon has two clocks.** Entry: fires on LTF candle close. Exit: 5–15s underlying
  poll, because Alpaca won't hold the stop for you (§1.1).
- **Every live bar is persisted**, so `replay.py` can reproduce any live session offline. You will
  need this the first time the bot does something inexplicable at 10:14 ET.
- **Detection stays deterministic code.** No LLM in the per-candle path — the handoff is right.
- **Kill switch + daily loss limit + max concurrent positions** enforced in `risk.py`, checked
  before every submit, plus a heartbeat file so you can tell "running and idle" from "crashed".

---

## 5. Phased plan with explicit gates

Each gate has a number you commit to *before* running it. Failing a gate means stop and rethink,
not re-tune until it passes.

**Phase 0 — Foundation (½ day)**
Repo decision, Alpaca paper keys, `alpaca-py` installed, `.env` handling, existing Python engine
imported and its tests green.

**Phase 1 — Real data + look-ahead audit (1–2 days)**
Pull SPY + QQQ 1H and 5m from Alpaca back to 2016 (free tier, SIP). Cache to parquet.
Replace the synthetic generator. Add the causality test from §3.
**Gate:** causality test passes; the existing single-TF strategy's stats on real data are computed
and *recorded as the baseline you must beat*. (Expect them to be much worse than the synthetic run.)

**Phase 2 — Multi-timeframe + rejection + filters (3–5 days)**
Gaps #1, #2, #4. MTF state machine: BOS/FVG on 1H, trigger on 5m, session filter 9:30–11:00 ET,
one-setup-per-day discipline. Rejection-wick rule with a *named, tunable* threshold, not a magic
number buried in code.
**Gate:** on the underlying, with costs, positive expectancy out-of-sample with ≥100 trades.
If it needs more than ~3 tuned parameters to get there, it's fit, not edge.

**Phase 3 — Statistical honesty (2–3 days)**
Walk-forward, parameter sensitivity heat maps, per-regime and per-year breakdown, and a null
comparison: the same entry timing with a *random* direction, plus a plain "buy the 5m breakout"
control. The signal must beat its own null.
**Gate:** edge survives out-of-sample and is not concentrated in one year or one regime.
**This is the real go/no-go for the whole project.** If it fails here, you've saved yourself weeks
of paper trading a coin flip — and you still keep the harness.

**Phase 4 — Options layer (2–3 days)**
Real chain data (2024+), contract selection (delta 0.45–0.60, 1–7 DTE), realistic fill at
bid/ask, theta/vega drag. Re-run Phase 3's winners *as option trades*.
**Gate:** the option expression keeps a meaningful share of the underlying edge after friction.
If friction eats it, trade shares instead and say so — that's a legitimate, useful finding.

**Phase 5 — Live paper loop (3–5 days)**
The daemon against Alpaca paper: entry on candle close, exit monitor, chain resolution, journal,
kill switch, heartbeat, EOD flatten. Run it for a week **in dry-run** (signals logged, no orders)
and diff its signals against the backtester's signals over the same days.
**Gate:** dry-run signals match backtest signals on the same bars. Any mismatch is a bug, and the
IEX-vs-SIP question from §1.2 gets answered here.

**Phase 6 — Paper trading for real (4–8 weeks)**
Orders live in paper. Weekly report: paper fills vs backtest expectancy vs the conservative
bid/ask-marked P&L. Do not touch parameters mid-run — that resets your sample.
**Gate for even *discussing* live money:** ≥8 weeks, multiple regimes, ≥100 trades, paper
expectancy within a credible band of backtest expectancy, and no unexplained execution incidents.

Phase 7 (CRT as a second, independent module) only after 6, and only if you still want it.

---

## 6. What I'd start on right now

Phase 1, in one sitting: `data/alpaca_hist.py` + parquet cache + the look-ahead causality test, then
re-run the existing engine on real SPY bars and publish the honest baseline. It's the cheapest step,
it's free-tier, and its output (a much worse number than the synthetic run) is the thing that makes
every later decision truthful.

Blocked on: repo decision (§0) and Alpaca paper API keys.
