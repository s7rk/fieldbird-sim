from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from fieldbird_sim import cli


def _write_config(path: Path, *, outdir_prefix: str, run_name: str, weather_block: str = "") -> None:
    path.write_text(
        (
            "run:\n"
            "  use_synthetic: true\n"
            "  n_fields: 10\n"
            "  n_years: 1\n"
            "  year_to_show: 1\n"
            "  day_to_show: 160\n"
            f"  outdir_prefix: \"{outdir_prefix}\"\n"
            f"  run_name: \"{run_name}\"\n"
            "  species: [\"Feldlerche\"]\n"
            "\n"
            "plots:\n"
            "  do_maps: false\n"
            "  do_heatmaps: false\n"
            "\n"
            "scenarios:\n"
            "  A:\n"
            "    enabled: false\n"
            "  B:\n"
            "    enabled: false\n"
            "  C:\n"
            "    enabled: false\n"
            "  D:\n"
            "    enabled: false\n"
            f"{weather_block}"
        ),
        encoding="utf-8",
    )


def _daily_csv_path(outdir_prefix: Path, run_name: str) -> Path:
    return outdir_prefix / run_name / "Baseline" / "Feldlerche" / "birdsim_daily_full.csv"


def _weather_log_path(outdir_prefix: Path, run_name: str) -> Path:
    return outdir_prefix / run_name / "Baseline" / "Feldlerche" / "weather_impact_log.csv"


def test_weather_flag_off_matches_baseline_without_weather_section(tmp_path):
    outdir = tmp_path / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_without_weather = tmp_path / "run_no_weather.yml"
    cfg_with_weather_disabled = tmp_path / "run_weather_disabled.yml"

    _write_config(
        cfg_without_weather,
        outdir_prefix=str(outdir),
        run_name="baseline_no_weather_block",
    )
    _write_config(
        cfg_with_weather_disabled,
        outdir_prefix=str(outdir),
        run_name="baseline_weather_disabled",
        weather_block=(
            "\n"
            "weather_module:\n"
            "  enabled: false\n"
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert cli.main(["run", "--config", str(cfg_without_weather)]) == 0
        assert cli.main(["run", "--config", str(cfg_with_weather_disabled)]) == 0

    df_no_weather = pd.read_csv(_daily_csv_path(outdir, "baseline_no_weather_block"))
    df_weather_off = pd.read_csv(_daily_csv_path(outdir, "baseline_weather_disabled"))

    pd.testing.assert_frame_equal(df_no_weather, df_weather_off, check_dtype=False, check_exact=True)
    assert not _weather_log_path(outdir, "baseline_no_weather_block").exists()
    assert not _weather_log_path(outdir, "baseline_weather_disabled").exists()
