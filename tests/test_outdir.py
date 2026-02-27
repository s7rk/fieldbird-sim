from pathlib import Path

import pytest

from fieldbird_sim.cli import load_run_config


def test_load_run_config_reads_values(tmp_path):
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "run:\n"
        "  n_fields: 12\n"
        "  run_name: smoke\n"
        "seeds:\n"
        "  sim: 777\n",
        encoding="utf-8",
    )

    cfg = load_run_config(config_path)
    assert cfg.run.n_fields == 12
    assert cfg.run.run_name == "smoke"
    assert cfg.seeds.sim == 777


def test_load_run_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_run_config(Path("does-not-exist.yml"))
