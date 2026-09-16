"""One trade schema for backtest, replay and paper. Diffable across drivers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

TRADE_COLUMNS = [
    "trade_id", "driver", "symbol", "instrument", "direction",
    "signal_time", "entry_time", "exit_time", "exit_reason",
    "entry_underlying", "exit_underlying", "stop", "target", "risk", "reward_r",
    "qty", "contract", "entry_fill", "exit_fill", "entry_mid", "exit_mid",
    "pnl", "pnl_conservative", "pnl_r", "commission", "risk_amount", "equity_after",
    "bos_time", "bos_kind", "bos_level", "gap_top", "gap_bottom", "gap_source", "gap_time",
    "htf_trend", "rejection_wick_ratio", "rejection_close_position", "notes",
]


@dataclass
class TradeRecord:
    trade_id: str
    driver: str
    symbol: str
    instrument: str
    direction: str
    signal_time: str
    entry_time: str
    exit_time: str
    exit_reason: str
    entry_underlying: float
    exit_underlying: float
    stop: float
    target: float
    risk: float
    reward_r: float
    qty: int
    contract: str
    entry_fill: float
    exit_fill: float
    entry_mid: float
    exit_mid: float
    pnl: float
    pnl_conservative: float
    pnl_r: float
    commission: float
    risk_amount: float
    equity_after: float
    bos_time: str = ""
    bos_kind: str = ""
    bos_level: float = float("nan")
    gap_top: float = float("nan")
    gap_bottom: float = float("nan")
    gap_source: str = ""
    gap_time: str = ""
    htf_trend: str = ""
    rejection_wick_ratio: float = float("nan")
    rejection_close_position: float = float("nan")
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra")
        return d


class Journal:
    """Append-only CSV + JSONL writer. Safe to reopen; header written once."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.path.with_suffix(".jsonl")

    def append(self, rec: TradeRecord) -> None:
        row = rec.to_row()
        new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        with self.jsonl.open("a") as f:
            f.write(json.dumps({**row, "extra": rec.extra}, default=str) + "\n")

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=TRADE_COLUMNS)
        return pd.read_csv(self.path)


def trades_to_frame(records: list[TradeRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame([r.to_row() for r in records], columns=TRADE_COLUMNS)
