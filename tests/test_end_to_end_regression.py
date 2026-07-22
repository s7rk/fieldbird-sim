from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from fieldbird_sim.resources import create_starter_project


EXPECTED_KPIS = {
    "Feldlerche": {
        "adults_arrival": [100, 95],
        "fledglings_total": [37, 60],
        "lambda_year_2": 0.95,
    },
    "Rebhuhn": {
        "adults_arrival": [100, 256],
        "fledglings_total": [222, 323],
        "lambda_year_2": 2.56,
    },
}


def test_starter_project_produces_deterministic_species_kpis(tmp_path: Path) -> None:
    project_dir = tmp_path / "starter"
    create_starter_project(project_dir)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [sys.executable, "-m", "fieldbird_sim.cli", "run", "--config", "run.yml"],
        cwd=project_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    for species, expected in EXPECTED_KPIS.items():
        kpi_path = (
            project_dir
            / "outputs"
            / "my_simulation"
            / "Baseline"
            / species
            / "analysis"
            / "kpi_yearly.csv"
        )
        kpis = pd.read_csv(kpi_path)
        assert kpis["adults_arrival"].tolist() == expected["adults_arrival"]
        assert kpis["fledglings_total"].tolist() == expected["fledglings_total"]
        assert kpis.loc[kpis["year"] == 2, "lambda_prev"].item() == pytest.approx(
            expected["lambda_year_2"]
        )
    assert not (project_dir / "analysis" / "kpi_yearly.csv").exists()
