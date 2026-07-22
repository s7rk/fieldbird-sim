from pathlib import Path

import pandas as pd

from fieldbird_sim import cli


def test_no_feedback_run_writes_reproduction_and_debug_logs(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    config = tmp_path / "run.yml"
    config.write_text(
        "run:\n"
        "  use_synthetic: true\n"
        "  n_fields: 5\n"
        "  n_years: 1\n"
        "  year_to_show: 1\n"
        "  day_to_show: 160\n"
        f"  outdir_prefix: \"{output_root}\"\n"
        "  run_name: no_feedback\n"
        "  species: [Feldlerche]\n"
        "plots:\n"
        "  do_maps: false\n"
        "  do_heatmaps: false\n"
        "winter:\n"
        "  use_year_feedback: false\n"
        "scenarios:\n"
        "  A: {enabled: false}\n"
        "  B: {enabled: false}\n"
        "  C: {enabled: false}\n"
        "  D: {enabled: false}\n",
        encoding="utf-8",
    )

    assert cli.main(["run", "--config", str(config)]) == 0

    year_dir = output_root / "no_feedback" / "Baseline" / "Feldlerche" / "year_1"
    reproduction = pd.read_csv(year_dir / "repro_log.csv")
    debug = pd.read_csv(year_dir / "debug_daily.csv")
    assert len(reproduction) == 211
    assert reproduction["day"].iloc[[0, -1]].tolist() == [90, 300]
    assert set(debug["doy"]) == {110, 130, 160, 200}
    assert {"mean_nest_surv", "total_fledglings"}.issubset(reproduction.columns)
