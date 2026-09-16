"""Live paper-trading daemon.

Two clocks, one thread:
  - entries fire when an LTF candle closes (the engine only ever sees closed bars, exactly
    like the backtester), and
  - exits are checked every poll against the live underlying quote, because Alpaca will not
    hold a bracket/OCO on an option leg for us.

Safety rails, checked every cycle: paper-only broker, kill-switch file, daily loss limit,
max positions, hard flat time, market-closed flatten, heartbeat file, state file for restart
reconciliation. Dry-run mode logs every decision and touches no orders.
"""

from __future__ import annotations

import json
import logging
import time as _time
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from ..core.bars import NY_TZ, bar_close_time, ensure_bars, resample_bars, rth_only
from ..core.config import RiskConfig, StrategyConfig
from ..core.risk import DayGuard, size_option, size_underlying
from ..core.signals import Signal, SignalEngine
from ..core.structure import Direction
from ..execution.broker import Broker, ContractQuote, Quote
from ..execution.costs import CostModel
from ..runner.journal import Journal, TradeRecord
from .contracts import SelectionCriteria, expiry_window, loss_per_contract_at_stop, select_contract, strike_window

log = logging.getLogger("bosfvg.live")


@dataclass
class OpenPosition:
    trade_id: str
    signal: dict
    instrument: str
    symbol: str            # what we hold: the OCC symbol or the stock ticker
    qty: int
    direction: str
    entry_time: str
    entry_fill: float
    entry_mid: float       # instrument mid at decision time
    entry_underlying: float
    stop: float
    target: float
    risk_amount: float
    contract_delta: float | None = None
    order_id: str = ""


class LiveTrader:
    def __init__(
        self,
        symbol: str,
        cfg: StrategyConfig,
        risk: RiskConfig,
        costs: CostModel,
        broker: Broker,
        run_dir: str | Path,
        dry_run: bool = True,
        live_feed: str = "iex",
        poll_seconds: float = 5.0,
        bar_grace_seconds: float = 3.0,
        now_fn: Callable[[], pd.Timestamp] | None = None,
        sleep_fn: Callable[[float], None] = _time.sleep,
        selection: SelectionCriteria | None = None,
        fill_timeout_seconds: float = 20.0,
    ) -> None:
        if not broker.is_paper():
            raise RuntimeError("refusing to run against a non-paper broker")
        if symbol not in cfg.symbols_allowed:
            raise ValueError(f"{symbol} is not in the liquidity allowlist {cfg.symbols_allowed}")
        self.symbol = symbol
        self.cfg = cfg
        self.risk = risk
        self.costs = costs
        self.broker = broker
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.live_feed = live_feed
        self.poll_seconds = poll_seconds
        self.grace = pd.Timedelta(seconds=bar_grace_seconds)
        self.now_fn = now_fn or (lambda: pd.Timestamp.now(tz=NY_TZ))
        self.sleep_fn = sleep_fn
        self.selection = selection or SelectionCriteria(target_delta=risk.target_delta, min_dte=risk.min_dte,
                                                        max_dte=risk.max_dte, flat_iv=risk.flat_iv)
        self.fill_timeout = fill_timeout_seconds

        self.engine = SignalEngine(cfg, symbol)
        self.journal = Journal(self.run_dir / "trades.csv")
        self.position: OpenPosition | None = None
        self.guard: DayGuard | None = None
        self.guard_day: date | None = None
        self.last_ltf_fed: pd.Timestamp | None = None
        self.last_htf_fed: pd.Timestamp | None = None
        self.stopped = False
        self.stop_reason = ""
        self.decisions: list[dict] = []
        self._state_path = self.run_dir / "state.json"
        self._heartbeat_path = self.run_dir / "heartbeat.json"
        self._kill_path = self.run_dir / "KILL"
        self._driver = "dryrun" if dry_run else "paper"

    # ------------------------------------------------------------------ bootstrap

    def bootstrap(self, history_bars: pd.DataFrame) -> None:
        """Feed prior sessions' closed bars so structure/gaps carry into today, then reconcile."""
        bars = rth_only(ensure_bars(history_bars), self.cfg.session_start)
        today = self.now_fn().normalize()
        bars = bars[bars.index < today]
        if len(bars):
            self._feed_closed_bars(bars, up_to=today)
        self._reconcile()

    def _reconcile(self) -> None:
        """Restart safety: resume a position we recorded, flatten anything we did not."""
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text())
                if data.get("position"):
                    self.position = OpenPosition(**data["position"])
                    log.warning("resumed open position from state file: %s x%d", self.position.symbol, self.position.qty)
            except (json.JSONDecodeError, TypeError) as exc:
                log.error("state file unreadable (%s); ignoring", exc)
        if self.dry_run:
            return
        held = {p.symbol: p for p in self.broker.positions()}
        if self.position is not None and self.position.symbol not in held:
            log.warning("recorded position %s is not at the broker; dropping it", self.position.symbol)
            self.position = None
        unknown = [p for s, p in held.items() if self.position is None or s != self.position.symbol]
        if unknown:
            log.error("unknown positions at broker %s; flattening for safety", [p.symbol for p in unknown])
            self.broker.close_all_positions()

    # ------------------------------------------------------------------ bar feeding

    def _feed_closed_bars(self, minute_bars: pd.DataFrame, up_to: pd.Timestamp) -> list[Signal]:
        """Feed every LTF/HTF bar that has closed by `up_to` and hasn't been fed, in wall-clock order."""
        ltf = resample_bars(minute_bars, self.cfg.ltf_minutes, self.cfg.session_start)
        htf = resample_bars(minute_bars, self.cfg.htf_minutes, self.cfg.session_start)
        ltf = ltf[[bar_close_time(t, self.cfg.ltf_minutes) <= up_to for t in ltf.index]]
        htf = htf[[bar_close_time(t, self.cfg.htf_minutes) <= up_to for t in htf.index]]
        if self.last_ltf_fed is not None:
            ltf = ltf[ltf.index > self.last_ltf_fed]
        if self.last_htf_fed is not None:
            htf = htf[htf.index > self.last_htf_fed]
        h_close = [bar_close_time(t, self.cfg.htf_minutes) for t in htf.index]
        hi = 0
        signals: list[Signal] = []
        for t, row in ltf.iterrows():
            lc = bar_close_time(t, self.cfg.ltf_minutes)
            while hi < len(h_close) and h_close[hi] <= lc:
                hr = htf.iloc[hi]
                self.engine.on_htf_bar(htf.index[hi], float(hr.open), float(hr.high), float(hr.low), float(hr.close))
                self.last_htf_fed = htf.index[hi]
                hi += 1
            sig = self.engine.on_ltf_bar(t, float(row.open), float(row.high), float(row.low), float(row.close))
            self.last_ltf_fed = t
            if sig is not None:
                signals.append(sig)
        # trailing HTF bars that closed by up_to but after the last LTF bar (only at session end)
        while hi < len(h_close) and h_close[hi] <= up_to:
            hr = htf.iloc[hi]
            self.engine.on_htf_bar(htf.index[hi], float(hr.open), float(hr.high), float(hr.low), float(hr.close))
            self.last_htf_fed = htf.index[hi]
            hi += 1
        return signals

    # ------------------------------------------------------------------ cycle

    def _ensure_guard(self, now: pd.Timestamp) -> DayGuard:
        if self.guard is None or self.guard_day != now.date():
            self.guard = DayGuard(self.risk, start_equity=self._equity())
            self.guard_day = now.date()
            if self.position is not None:
                self.guard.on_open()
        return self.guard

    def _equity(self) -> float:
        try:
            return float(self.broker.equity())
        except Exception as exc:  # broker hiccup must not kill the loop; fall back to configured equity
            log.warning("equity lookup failed: %s", exc)
            return self.risk.equity

    def cycle(self) -> None:
        now = self.now_fn()
        if self._kill_path.exists():
            self._flatten("kill switch", now)
            self.stopped, self.stop_reason = True, "kill switch"
            return
        guard = self._ensure_guard(now)
        if not self.broker.market_open():
            if self.position is not None:
                self._flatten("market closed", now)
            self._heartbeat(now, "market closed")
            return

        # exit check first: an open position is the riskiest thing we own
        if self.position is not None:
            self._check_exit(now)

        bars = self.broker.today_minute_bars(self.symbol, self.live_feed)
        if len(bars):
            bars = rth_only(bars, self.cfg.session_start)
            self._record_bars(bars, now)
            signals = self._feed_closed_bars(bars, up_to=now - self.grace)
            for sig in signals:
                self._on_signal(sig, now, guard)
        self._heartbeat(now, "ok")

    def run(self) -> None:
        log.info("starting %s daemon for %s (feed=%s, instrument=%s)", self._driver, self.symbol, self.live_feed, self.risk.instrument)
        while not self.stopped:
            try:
                self.cycle()
            except Exception:
                log.exception("cycle failed; continuing")
            self.sleep_fn(self.poll_seconds)

    # ------------------------------------------------------------------ entries

    def _on_signal(self, sig: Signal, now: pd.Timestamp, guard: DayGuard) -> None:
        ok, why = guard.can_open()
        rec = {"time": now.isoformat(), "signal": sig.to_record(), "action": "skip", "reason": why}
        if not ok:
            self._decide(rec)
            return
        if self.position is not None:
            rec["reason"] = "position already open"
            self._decide(rec)
            return
        quote = self.broker.underlying_quote(self.symbol)
        equity = self._equity()
        if self.risk.instrument == "option":
            contract, why = self._pick_contract(sig, quote, now)
            if contract is None:
                rec["reason"] = why
                self._decide(rec)
                return
            loss = loss_per_contract_at_stop(contract, quote.mid, sig.stop, self.risk.contract_multiplier)
            size = size_option(equity, self.risk.risk_pct, loss, self.risk.max_contracts)
            if not size.ok:
                rec["reason"] = size.reason
                self._decide(rec)
                return
            held_symbol, side, intent, mid, touch = contract.symbol, "buy", "buy_to_open", contract.mid, contract.ask
            delta = contract.delta
        else:
            risk_per_share = abs(quote.ask - sig.stop) if sig.direction is Direction.BULLISH else abs(sig.stop - quote.bid)
            size = size_underlying(equity, self.risk.risk_pct, risk_per_share, self.risk.max_shares)
            if not size.ok:
                rec["reason"] = size.reason
                self._decide(rec)
                return
            held_symbol = self.symbol
            side = "buy" if sig.direction is Direction.BULLISH else "sell"
            intent, mid, touch = "", quote.mid, (quote.ask if side == "buy" else quote.bid)
            delta = None

        trade_id = f"{self._driver}-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        rec.update({"action": "enter", "instrument": self.risk.instrument, "held": held_symbol, "qty": size.qty, "side": side,
                    "expected_fill": touch, "risk_amount": size.risk_amount, "reason": why if self.risk.instrument == "option" else ""})
        if self.dry_run:
            fill = touch
        else:
            result = self._submit(held_symbol, size.qty, side, intent, trade_id + "-in")
            if result is None or not result.is_filled:
                rec["action"] = "entry_failed"
                rec["reason"] = f"order status {result.status if result else 'none'}"
                self._decide(rec)
                return
            fill = float(result.filled_avg_price)
        self.position = OpenPosition(
            trade_id=trade_id, signal=sig.to_record(), instrument=self.risk.instrument, symbol=held_symbol, qty=size.qty,
            direction=sig.direction.value, entry_time=now.isoformat(), entry_fill=fill, entry_mid=mid,
            entry_underlying=quote.mid, stop=sig.stop, target=sig.target, risk_amount=size.risk_amount, contract_delta=delta,
        )
        guard.on_open()
        self._save_state()
        self._decide(rec)
        log.info("ENTER %s %s x%d @ %.4f (stop %.2f target %.2f)", side, held_symbol, size.qty, fill, sig.stop, sig.target)

    def _pick_contract(self, sig: Signal, quote: Quote, now: pd.Timestamp) -> tuple[ContractQuote | None, str]:
        is_call = sig.direction is Direction.BULLISH
        lo, hi = expiry_window(now.date(), self.selection)
        klo, khi = strike_window(quote.mid)
        try:
            chain = self.broker.option_chain(self.symbol, is_call, lo, hi, klo, khi)
        except Exception as exc:
            return None, f"chain lookup failed: {exc}"
        return select_contract(chain, quote.mid, is_call, now.date(), self.selection)

    def _submit(self, symbol: str, qty: int, side: str, intent: str, client_order_id: str):
        try:
            if self.risk.instrument == "option":
                res = self.broker.submit_option_market(symbol, qty, side, intent, client_order_id)
            else:
                res = self.broker.submit_stock_market(symbol, qty, side, client_order_id)
        except Exception as exc:
            log.error("order submit failed: %s", exc)
            return None
        deadline = _time.monotonic() + self.fill_timeout
        while not res.is_filled and _time.monotonic() < deadline and res.status not in ("canceled", "rejected", "expired"):
            self.sleep_fn(0.5)
            res = self.broker.order_status(res.order_id)
        if not res.is_filled:
            log.error("order %s not filled (status %s); cancelling open orders", res.order_id, res.status)
            try:
                self.broker.cancel_all()
            except Exception as exc:
                log.error("cancel failed: %s", exc)
        return res

    # ------------------------------------------------------------------ exits

    def _check_exit(self, now: pd.Timestamp) -> None:
        pos = self.position
        assert pos is not None
        quote = self.broker.underlying_quote(self.symbol)
        px = quote.mid
        d = 1 if pos.direction == "bullish" else -1
        day = now.normalize()
        flat_dt = day + pd.Timedelta(hours=self.cfg.flat_time.hour, minutes=self.cfg.flat_time.minute)
        reason = None
        if d == 1 and px <= pos.stop or d == -1 and px >= pos.stop:
            reason = "stop"
        elif d == 1 and px >= pos.target or d == -1 and px <= pos.target:
            reason = "target"
        elif now >= flat_dt:
            reason = "time"
        elif self.cfg.max_hold_minutes and now >= pd.Timestamp(pos.entry_time) + pd.Timedelta(minutes=self.cfg.max_hold_minutes):
            reason = "time"
        if reason:
            self._close_position(reason, now, quote)

    def _close_position(self, reason: str, now: pd.Timestamp, quote: Quote | None = None) -> None:
        pos = self.position
        assert pos is not None
        quote = quote or self.broker.underlying_quote(self.symbol)
        if pos.instrument == "option":
            side, intent = "sell", "sell_to_close"
            exit_mid, touch = self._option_mid(pos.symbol)
        else:
            side, intent = ("sell", "") if pos.direction == "bullish" else ("buy", "")
            exit_mid = quote.mid
            touch = quote.bid if side == "sell" else quote.ask
        if self.dry_run:
            fill = touch
        else:
            res = self._submit(pos.symbol, pos.qty, side, intent, pos.trade_id + "-out")
            if res is None or not res.is_filled:
                log.critical("EXIT ORDER NOT FILLED for %s; retrying via close_all_positions", pos.symbol)
                try:
                    self.broker.close_all_positions()
                except Exception as exc:
                    log.critical("close_all_positions failed: %s", exc)
                fill = touch
            else:
                fill = float(res.filled_avg_price)
        mult = self.risk.contract_multiplier if pos.instrument == "option" else 1
        sign = 1 if (pos.instrument == "option" or pos.direction == "bullish") else -1
        commission = (self.costs.option_round_trip_commission(pos.qty) if pos.instrument == "option"
                      else self.costs.stock_round_trip_commission(pos.qty))
        pnl = (fill - pos.entry_fill) * pos.qty * mult * sign - commission
        # conservative mark: what the shared cost model says the round trip should have cost
        if pos.instrument == "option":
            cons_entry, cons_exit = self.costs.option_fill(pos.entry_mid, True), self.costs.option_fill(exit_mid, False)
        else:
            is_buy = pos.direction == "bullish"
            cons_entry, cons_exit = self.costs.stock_fill(pos.entry_mid, is_buy), self.costs.stock_fill(exit_mid, not is_buy)
        pnl_cons = (cons_exit - cons_entry) * pos.qty * mult * sign - commission
        equity = self._equity()
        s = pos.signal
        rec = TradeRecord(
            trade_id=pos.trade_id, driver=self._driver, symbol=self.symbol, instrument=pos.instrument, direction=pos.direction,
            signal_time=s["signal_time"], entry_time=pos.entry_time, exit_time=now.isoformat(), exit_reason=reason,
            entry_underlying=pos.entry_underlying, exit_underlying=quote.mid, stop=pos.stop, target=pos.target,
            risk=s["risk"], reward_r=s["reward_r"], qty=pos.qty, contract=pos.symbol if pos.instrument == "option" else "",
            entry_fill=pos.entry_fill, exit_fill=fill, entry_mid=pos.entry_mid, exit_mid=exit_mid, pnl=pnl,
            pnl_conservative=pnl_cons, pnl_r=pnl / pos.risk_amount if pos.risk_amount else float("nan"),
            commission=commission, risk_amount=pos.risk_amount, equity_after=equity, bos_time=s["bos_time"],
            bos_kind=s["bos_kind"], bos_level=s["bos_level"], gap_top=s["gap_top"], gap_bottom=s["gap_bottom"],
            gap_source=s["gap_source"], gap_time=s["gap_time"], htf_trend=s["htf_trend"],
            rejection_wick_ratio=s["rejection_wick_ratio"], rejection_close_position=s["rejection_close_position"],
            notes=s["notes"], extra={"live_feed": self.live_feed, "contract_delta": pos.contract_delta},
        )
        self.journal.append(rec)
        if self.guard is not None:
            self.guard.on_close(pnl)
        self.position = None
        self._save_state()
        log.info("EXIT %s %s x%d @ %.4f reason=%s pnl=%.2f", side, pos.symbol, pos.qty, fill, reason, pnl)

    def _option_mid(self, contract_symbol: str) -> tuple[float, float]:
        """(mid, bid) for the held contract; falls back to the entry mid if the chain is unavailable."""
        try:
            _, expiry, is_call, strike = _parse(contract_symbol)
            chain = self.broker.option_chain(self.symbol, is_call, expiry, expiry, strike - 0.01, strike + 0.01)
            for c in chain:
                if c.symbol == contract_symbol:
                    return c.mid, c.bid
        except Exception as exc:
            log.warning("option quote lookup failed: %s", exc)
        pos = self.position
        return (pos.entry_mid, pos.entry_mid) if pos else (0.0, 0.0)

    def _flatten(self, reason: str, now: pd.Timestamp) -> None:
        if self.position is not None:
            self._close_position(reason, now)
        if not self.dry_run:
            try:
                self.broker.cancel_all()
                if self.broker.positions():
                    self.broker.close_all_positions()
            except Exception as exc:
                log.critical("flatten failed: %s", exc)

    # ------------------------------------------------------------------ persistence

    def _decide(self, rec: dict) -> None:
        self.decisions.append(rec)
        with (self.run_dir / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps({"position": asdict(self.position) if self.position else None}, default=str))

    def _heartbeat(self, now: pd.Timestamp, status: str) -> None:
        self._heartbeat_path.write_text(json.dumps({
            "time": now.isoformat(), "status": status, "driver": self._driver, "symbol": self.symbol,
            "position": self.position.symbol if self.position else None,
            "last_ltf_bar": self.last_ltf_fed.isoformat() if self.last_ltf_fed is not None else None,
            "signals_today": sum(1 for s in self.engine.signals if s.time.normalize() == now.normalize()),
            "halted": self.guard.halted if self.guard else False,
        }))

    def _record_bars(self, bars: pd.DataFrame, now: pd.Timestamp) -> None:
        """Persist today's live bars so `replay` can reproduce the session offline."""
        path = self.run_dir / f"bars_{self.symbol}_{self.live_feed}_{now.strftime('%Y-%m-%d')}.parquet"
        try:
            bars.to_parquet(path)
        except Exception as exc:
            log.warning("bar recording failed: %s", exc)


def _parse(contract_symbol: str):
    from ..execution.alpaca_broker import parse_occ

    return parse_occ(contract_symbol)
