import json

from bosfvg.cli import main


def test_backtest_cli_synthetic(tmp_path, capsys):
    main(["backtest", "--synthetic-days", "60", "--fvg-source", "ltf", "--entry-end", "15:30", "--max-trades-per-day", "3",
          "--out", str(tmp_path / "bt")])
    out = capsys.readouterr().out
    assert "trades" in out
    assert (tmp_path / "bt" / "trade_log.csv").exists()
    stats = json.loads((tmp_path / "bt" / "stats.json").read_text())
    assert stats["config"]["fvg_source"] == "ltf"


def test_config_file_and_overrides(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"reward_r": 3.0, "entry_window_end": "15:30", "fvg_source": "ltf", "max_trades_per_day": 3}))
    main(["backtest", "--synthetic-days", "40", "--config", str(cfg), "--reward-r", "1.5", "--out", str(tmp_path / "o")])
    stats = json.loads((tmp_path / "o" / "stats.json").read_text())
    assert stats["config"]["reward_r"] == 1.5 and stats["config"]["entry_window_end"] == "15:30"
