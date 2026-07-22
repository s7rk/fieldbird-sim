from pathlib import Path

import pytest

from fieldbird_sim import cli
from fieldbird_sim.resources import (
    bundled_calendar_path,
    bundled_species_dir,
    create_starter_project,
    resolve_calendar_path,
    resolve_species_dir,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config_path",
    sorted((ROOT / "configs").glob("run_*.yml")),
    ids=lambda path: path.name,
)
def test_repository_config_presets_load(config_path):
    cfg = cli.load_run_config(config_path)
    assert cfg.run.n_fields > 0
    assert cfg.run.n_years > 0
    assert cfg.config_dir == config_path.parent.resolve()


@pytest.mark.parametrize(
    "name",
    [
        "species_Feldlerche.csv",
        "species_Feldlerche_pref.csv",
        "species_Rebhuhn.csv",
        "species_Rebhuhn_pref.csv",
    ],
)
def test_bundled_species_files_are_available(name):
    path = bundled_species_dir() / name
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip()


def test_bundled_calendar_matches_editable_repository_default():
    assert bundled_calendar_path().read_bytes() == (ROOT / "configs/crop_calendar.csv").read_bytes()


def test_resolvers_fall_back_to_bundled_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert resolve_calendar_path("configs/crop_calendar.csv") == bundled_calendar_path()
    assert resolve_species_dir("Feldlerche") == bundled_species_dir()


def test_resolvers_prefer_paths_relative_to_config(tmp_path):
    starter = tmp_path / "starter"
    create_starter_project(starter)

    assert resolve_calendar_path("crop_calendar.csv", base_dir=starter) == (
        starter / "crop_calendar.csv"
    )
    assert resolve_species_dir(
        "Rebhuhn",
        configured_dir=".",
        base_dir=starter,
    ) == starter


def test_cli_init_creates_editable_starter_project(tmp_path, capsys):
    target = tmp_path / "my-simulation"

    assert cli.main(["init", str(target)]) == 0
    assert (target / "README.md").is_file()
    assert (target / "run.yml").is_file()
    assert (target / "crop_calendar.csv").is_file()
    assert (target / "species_Feldlerche.csv").is_file()
    assert (target / "species_Rebhuhn_pref.csv").is_file()
    assert "fieldbird-sim run --config run.yml" in capsys.readouterr().out
    assert "First run" in (target / "README.md").read_text(encoding="utf-8")

    assert cli.main(["init", str(target)]) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_starter_creation_checks_all_conflicts_before_writing(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    existing = target / "species_Rebhuhn.csv"
    existing.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="species_Rebhuhn.csv"):
        create_starter_project(target)

    assert existing.read_text(encoding="utf-8") == "keep me"
    assert not (target / "run.yml").exists()
