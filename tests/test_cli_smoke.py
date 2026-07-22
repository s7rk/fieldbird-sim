import sys
import types

import pytest

from fieldbird_sim import cli


def test_cli_import_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    assert "usage: fieldbird-sim" in capsys.readouterr().out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("fieldbird-sim ")


def test_cli_run_dispatches_to_legacy_runner(monkeypatch, tmp_path):
    config_path = tmp_path / "run.yml"
    config_path.write_text("run:\n  n_fields: 3\n  n_years: 1\n", encoding="utf-8")

    observed = {}

    fake_kpis = types.ModuleType("fieldbird_sim.legacy.kpis")

    def _fake_run(*, cfg):
        observed["cfg"] = cfg

    fake_kpis.run = _fake_run
    monkeypatch.setitem(sys.modules, "fieldbird_sim.legacy.kpis", fake_kpis)

    exit_code = cli.main(["run", "--config", str(config_path)])
    assert exit_code == 0
    assert observed["cfg"].run.n_fields == 3
