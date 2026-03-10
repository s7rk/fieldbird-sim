from __future__ import annotations

from types import SimpleNamespace

from fieldbird_sim.legacy.scenario_plan import apply_scenario_to_inputs, build_scenarios_to_run


def test_build_scenarios_to_run_includes_enabled_blocks():
    cfg = SimpleNamespace(
        scenarios=SimpleNamespace(
            A=SimpleNamespace(enabled=True, shifts=[7]),
            B=SimpleNamespace(enabled=True, shares=[0.10], fractions=[0.50], select="no_flower_first", seed=5),
            C=SimpleNamespace(enabled=True, fallow_area_shares=[0.05], apply_years="all", selection="best_match", seed=6),
            D=SimpleNamespace(
                enabled=True,
                add_skylark_plots=2,
                skylark_plot_area=0.02,
                target_fraction=0.20,
                min_eligible_years=1,
                only_without_flower=True,
            ),
        )
    )

    out = build_scenarios_to_run(cfg)

    assert [s["kind"] for s in out] == ["baseline", "A", "B", "C", "D"]
    assert out[0]["name"] == "Baseline"
    assert out[1]["name"] == "Scenario_A_shift_7"
    assert out[2]["name"] == "Scenario_B_share_0.10_frac_0.50"
    assert out[3]["name"] == "Scenario_C_fallow_0.05"
    assert out[4]["name"] == "Scenario_D_skylark"


def test_apply_scenario_to_inputs_dispatches_c_with_year_selector():
    calls = {}

    def _a(mgmt, *, shift):
        calls["a_shift"] = shift
        return "M_A"

    def _b(fields, *, share, fraction, select, seed):
        calls["b"] = (share, fraction, select, seed)
        return "F_B"

    def _c(*, fields_gdf_in, rotation_in, fallow_area_share, years_to_apply, selection, seed):
        calls["c"] = (fields_gdf_in, rotation_in, fallow_area_share, years_to_apply, selection, seed)
        return "R_C"

    def _d(
        *,
        fields_gdf_in,
        rotation_in,
        years_to_sim,
        add_skylark_plots,
        skylark_plot_area,
        target_fraction,
        min_eligible_years,
        only_without_flower,
        seed,
    ):
        calls["d"] = (
            fields_gdf_in,
            rotation_in,
            years_to_sim,
            add_skylark_plots,
            skylark_plot_area,
            target_fraction,
            min_eligible_years,
            only_without_flower,
            seed,
        )
        return "F_D"

    scen_c = {
        "kind": "C",
        "payload": {
            "fallow_area_share": 0.1,
            "apply_years": "all",
            "selection": "best_match",
            "seed": 9,
        },
    }

    out_c = apply_scenario_to_inputs(
        scen=scen_c,
        fields_gdf_in="F",
        rotation_in="R",
        management_in="M",
        years_to_sim=[1, 2, 3],
        apply_A_shift_management=_a,
        apply_B_flower_strips=_b,
        apply_C_fallow=_c,
        apply_D_skylark=_d,
    )

    assert out_c == ("F", "R_C", "M")
    assert calls["c"] == ("F", "R", 0.1, [1, 2, 3], "best_match", 9)
