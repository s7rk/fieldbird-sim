from dataclasses import replace

import numpy as np
import pytest

from fieldbird_sim.legacy.debug_utils import DebugTracer, ReproLogger, sanity_check_species
from fieldbird_sim.legacy.species import load_species_from_csv
from fieldbird_sim.resources import bundled_species_dir


def test_species_validation_accepts_bundled_parameters_and_reports_all_failures() -> None:
    cfg = load_species_from_csv("Feldlerche", bundled_species_dir())
    sanity_check_species(cfg)

    invalid = replace(
        cfg,
        arrival_day=400,
        departure_day=0,
        breeding_window=(220, 100),
        fledge_rate=1.5,
        nest_cycle_days=0,
        juvenile_free_age=-1,
    )
    with pytest.raises(ValueError) as exc_info:
        sanity_check_species(invalid)

    message = str(exc_info.value)
    for field in (
        "arrival_day",
        "departure_day",
        "breeding_window",
        "fledge_rate",
        "nest_cycle_days",
        "juvenile_free_age",
    ):
        assert field in message


def test_debug_tracer_captures_numeric_daily_summary() -> None:
    tracer = DebugTracer()
    tracer.capture_day(
        12,
        U=np.array([0.2, 0.8]),
        today_m=np.array([2, 1]),
        today_f=np.array([3, 1]),
        breeding_f=np.array([1, 1]),
        new_nests=np.array([1, 0]),
        fledglings=np.array([0, 2]),
        density_factor=np.array([0.4, 0.9]),
        nest_surv_day=np.array([0.7, 0.95]),
    )

    row = tracer.to_frames()["daily"].iloc[0]
    assert row["doy"] == 12
    assert row["U_mean"] == pytest.approx(0.5)
    assert row["M_sum"] == 3
    assert row["F_sum"] == 4
    assert row["fledglings_sum"] == 2
    assert row["nest_surv_max"] == pytest.approx(0.95)


def test_reproduction_logger_returns_tidy_aggregates() -> None:
    logger = ReproLogger()
    logger.log(
        day=101,
        in_breed=True,
        U=np.array([0.5, 1.0]),
        p_start_vec=np.array([0.1, 0.3]),
        density_factor=np.array([0.8, 1.0]),
        nest_surv_day=np.array([0.7, 0.9]),
        new_nests_vec=np.array([1, 2]),
        fledglings_vec=np.array([3, 4]),
        A_m_today=np.array([5, 6]),
        A_f_today=np.array([7, 8]),
        J_tied_today=np.array([9, 10]),
        J_free_today=np.array([11, 12]),
    )

    row = logger.to_df().iloc[0]
    assert bool(row["in_breed"]) is True
    assert row["mean_U"] == pytest.approx(0.75)
    assert row["total_new_nests"] == 3
    assert row["total_fledglings"] == 7
    assert row["A_f_total"] == 15
    assert row["J_free_total"] == 23
