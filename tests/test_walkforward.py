from datetime import time

from bosfvg.core.config import RiskConfig, StrategyConfig
from bosfvg.data.synthetic import synthetic_minute_bars
from bosfvg.eval.walkforward import null_comparison, sensitivity, walk_forward
from bosfvg.execution.costs import CostModel
from bosfvg.runner.backtest import run_backtest

CFG = StrategyConfig(fvg_source="ltf", entry_window_end=time(15, 30), max_trades_per_day=3)


def test_sensitivity_table_shape():
    bars = synthetic_minute_bars(days=60, seed=2)
    df = sensitivity(bars, "SPY", CFG, RiskConfig(), CostModel(), {"reward_r": [1.0, 2.0], "wick_ratio": [0.3, 0.5]})
    assert list(df["param"]) == ["reward_r", "reward_r", "wick_ratio", "wick_ratio"]
    assert set(df.columns) >= {"avg_r", "t_stat", "trades"}


def test_walk_forward_produces_folds_and_oos():
    bars = synthetic_minute_bars(days=200, seed=2)
    folds, oos = walk_forward(bars, "SPY", CFG, RiskConfig(), CostModel(), {"reward_r": [1.5, 2.0]},
                              train_months=4, test_months=2, min_trades=5)
    assert len(folds) >= 2
    assert all(f.test_start == f.train_end for f in folds)
    assert len(oos) > 0 and "fold" in oos.columns


def test_null_comparison_reports_percentile():
    bars = synthetic_minute_bars(days=90, seed=2)
    r = run_backtest(bars, "SPY", CFG, RiskConfig(), CostModel())
    res = null_comparison(r, bars, n=20, seed=1)
    assert 0 <= res["percentile_of_real"] <= 100
    assert res["n_sims"] == 20
