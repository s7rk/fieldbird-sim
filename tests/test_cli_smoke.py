import pytest

def test_cli_import_and_help(capsys):
    from fieldbird_sim import cli

    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])

    assert e.value.code == 0

    out = capsys.readouterr().out
    assert "usage: fieldbird-sim" in out
