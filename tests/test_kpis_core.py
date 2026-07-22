from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fieldbird_sim.legacy import kpis


def _result(*, occupied_first_field: bool = False) -> dict[str, np.ndarray | pd.DataFrame]:
    first = [1, 1, 1, 1] if occupied_first_field else [0, 0, 0, 0]
    adult_females = np.column_stack([first, [2, 2, 1, 1]])
    return {
        "A_m": np.array([[2, 0], [1, 0], [0, 0], [0, 0]]),
        "A_f": adult_females,
        "A_f_breeding": adult_females,
        "daily_fledglings": np.array([[0, 0], [1, 2], [0, 0], [0, 0]]),
        "J_tied_total": np.zeros((4, 2), dtype=int),
        "J_free": np.array([[0, 0], [0, 1], [0, 1], [0, 0]]),
        "debug_repro": pd.DataFrame(
            {"day": [1, 2, 3, 4], "mean_nest_surv": [0.9, 0.8, 0.7, 0.1]}
        ),
    }


def test_kpi_helpers_cover_boundaries_and_colonization() -> None:
    np.testing.assert_array_equal(kpis._breed_mask(5, (9, -2)), [True] * 5)
    assert np.isnan(kpis.mayfield_daily_from_log(pd.DataFrame(), (1, 2)))
    assert np.isnan(
        kpis.mayfield_daily_from_log(
            pd.DataFrame({"day": [20], "mean_nest_surv": [0.9]}), (1, 2)
        )
    )

    prev = _result(occupied_first_field=False)
    curr = _result(occupied_first_field=True)
    assert kpis.colonization_rate(prev, curr, (1, 4)) == pytest.approx(1.0)
    assert kpis.colonization_rate(curr, curr, (1, 4)) == pytest.approx(0.0)
    np.testing.assert_allclose(kpis.field_occupancy(prev["A_f"], (1, 4)), [0.0, 1.0])


def test_compute_kpis_and_yearly_output(tmp_path) -> None:
    cfg = SimpleNamespace(arrival_day=1, breeding_window=(1, 3), nest_cycle_days=2)
    one_year = kpis.compute_kpis_one_year(_result(), cfg)
    assert one_year["adults_arrival"] == 4
    assert one_year["births_total"] == 3
    assert one_year["adult_deaths"] == 3
    assert one_year["juvenile_deaths"] == 4
    assert one_year["female_breeding_equiv"] == pytest.approx(2.5)
    assert one_year["fledglings_per_female_equiv"] == pytest.approx(1.2)
    assert one_year["occupancy_mean"] == pytest.approx(0.5)
    assert one_year["mayfield_daily"] == pytest.approx(0.8)

    output = tmp_path / "analysis" / "kpi_yearly.csv"
    frame = kpis.compute_kpis_from_results({2: _result(), 1: _result()}, cfg, output_path=output)
    assert output.is_file()
    assert frame["year"].tolist() == [1, 2]
    assert frame.loc[1, "lambda_prev"] == pytest.approx(1.0)
    assert frame.loc[1, "colonization_rate"] == pytest.approx(0.0)


def test_latin_hypercube_and_prcc_are_reproducible() -> None:
    vary = {"alpha": (0.0, 1.0), "beta": (10.0, 20.0)}
    first = kpis.lhs_sample(20, vary, seed=11)
    second = kpis.lhs_sample(20, vary, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert first["alpha"].between(0.0, 1.0).all()
    assert first["beta"].between(10.0, 20.0).all()
    assert first["alpha"].nunique() == 20

    result = kpis.compute_prcc(first, 3.0 * first["alpha"] - 0.2 * first["beta"])
    assert set(result["param"]) == {"alpha", "beta"}
    assert result["prcc"].abs().le(1.0).all()
    assert result["p_approx"].between(0.0, 1.0).all()


@dataclass
class _MonteCarloConfig:
    arrival_day: int = 1
    breeding_window: tuple[int, int] = (1, 2)
    nest_cycle_days: int = 1
    fec_scale: float = 0.3


def test_lhs_monte_carlo_restores_globals(monkeypatch) -> None:
    def fake_run(_fields, _rotation, _management, cfg, *, years, **_kwargs):
        results = {}
        for year in years:
            adults = int(round(cfg.fec_scale * 10)) + int(year)
            results[year] = {
                "A_m": np.full((2, 1), adults, dtype=int),
                "A_f": np.ones((2, 1), dtype=int),
                "daily_fledglings": np.ones((2, 1), dtype=int),
                "J_tied_total": np.zeros((2, 1), dtype=int),
                "J_free": np.zeros((2, 1), dtype=int),
            }
        return results

    monkeypatch.setattr(kpis, "run_years_with_feedback", fake_run)
    original_gamma = kpis.CAPACITY_GAMMA
    result = kpis.run_mc_with_lhs(
        None,
        None,
        None,
        _MonteCarloConfig(),
        years=[1, 2],
        n_reps=3,
        seed=5,
        vary={"fec_scale": (0.2, 0.5), "CAPACITY_GAMMA": (0.1, 0.2)},
    )

    assert len(result) == 3
    assert result["rep"].tolist() == [0, 1, 2]
    assert result["lambda_mean"].notna().all()
    assert kpis.CAPACITY_GAMMA == original_gamma


def test_active_scenario_callbacks_modify_copies_and_validate_inputs() -> None:
    management = pd.DataFrame(
        {"harvest_date": [2, 360, None], "harvest_win": ["2-5", "350-360", "bad"]}
    )
    shifted = kpis.apply_scenario_A_shift_management(management, 10)
    assert shifted["harvest_date"].iloc[:2].tolist() == [12.0, 365.0]
    assert shifted["harvest_win"].tolist() == ["12-15", "360-365", "bad"]
    assert management.loc[0, "harvest_date"] == 2

    fields = pd.DataFrame(
        {"field_id": [1, 2, 3], "area": [5.0, 3.0, 2.0], "flower_strip_share": [0.2, 0.0, 0.1]}
    )
    flowers = kpis.apply_scenario_B_flower_strips(
        fields, share=2 / 3, fraction=0.4, select="no_flower_first", seed=1
    )
    assert flowers["flower_strip_share"].tolist() == [0.2, 0.4, 0.4]
    with pytest.raises(ValueError, match="flower_strip_share"):
        kpis.apply_scenario_B_flower_strips(fields.drop(columns="flower_strip_share"), share=1, fraction=0.1)

    rotation = pd.DataFrame(
        {"year": [1, 1, 1, 2, 2, 2], "field_id": [1, 2, 3] * 2, "crop": ["Mais"] * 6}
    )
    fallow = kpis.apply_scenario_C_fallow(
        fields_gdf_in=fields,
        rotation_in=rotation,
        fallow_area_share=0.5,
        years_to_apply=[2],
    )
    assert fallow[(fallow["year"] == 1)]["crop"].eq("Mais").all()
    assert fallow[(fallow["year"] == 2) & (fallow["field_id"] == 1)]["crop"].item() == "Fallow"

    skylark = kpis.apply_scenario_D_skylark(
        fields_gdf_in=fields,
        rotation_in=rotation,
        years_to_sim=[1, 2],
        add_skylark_plots=2,
        skylark_plot_area=0.02,
        target_fraction=0.5,
        min_eligible_years=2,
        seed=4,
    )
    assert skylark.loc[skylark["field_id"] == 2, "skylark_plots"].item() == 2
    assert fields.columns.tolist() == ["field_id", "area", "flower_strip_share"]


def test_field_subset_matching_handles_exact_and_large_inputs() -> None:
    exact = kpis._choose_fields_best_match(np.array([1, 2, 3]), np.array([5.0, 3.0, 2.0]), 5.0)
    assert exact == [1]

    many_ids = np.arange(1, 28)
    chosen = kpis._choose_fields_best_match(many_ids, np.ones(27), 3.0)
    assert len(chosen) == 3
