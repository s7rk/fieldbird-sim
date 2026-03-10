from __future__ import annotations

from types import SimpleNamespace

from fieldbird_sim.legacy.plot_controls import get_plot_controls, parse_year_selector


def test_parse_year_selector_variants():
    years = [1, 2, 3]
    assert parse_year_selector(None, years_to_sim=years, year_to_show=2) == [1, 2, 3]
    assert parse_year_selector("all", years_to_sim=years, year_to_show=2) == [1, 2, 3]
    assert parse_year_selector("year_to_show", years_to_sim=years, year_to_show=2) == [2]
    assert parse_year_selector([3, 99, "x"], years_to_sim=years, year_to_show=2) == [3]


def test_get_plot_controls_normalizes_settings():
    cfg = SimpleNamespace(
        plots=SimpleNamespace(
            maps_years="year_to_show",
            heatmaps_years=[1, 3, 42],
            usage_map=False,
            heatmap_vmax_mode="bad_value",
        )
    )

    out = get_plot_controls(cfg, years_to_sim=[1, 2, 3], year_to_show=2)

    assert out["maps_years"] == [2]
    assert out["heatmaps_years"] == [1, 3]
    assert out["usage_map"] is False
    assert out["heatmap_vmax_mode"] == "global"
