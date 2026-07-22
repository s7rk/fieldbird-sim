from __future__ import annotations

import math

import pandas as pd
import pytest

from fieldbird_sim.legacy.scenarios import (
    apply_fallow_scenario,
    apply_management_scenario,
    choose_flower_target_fields,
    choose_skylark_target_fields,
    harvest_overlap_share,
    shift_window_str,
)


def test_flower_selection_handles_missing_optional_share_column() -> None:
    fields = pd.DataFrame({"field_id": [1, 2, 3, 4]})

    chosen = choose_flower_target_fields(
        fields,
        fraction=0.5,
        mode="no_flower_first",
        seed=17,
    )

    assert len(chosen) == 2
    assert set(chosen) <= {1, 2, 3, 4}
    assert chosen == choose_flower_target_fields(
        fields,
        fraction=0.5,
        mode="no_flower_first",
        seed=17,
    )
    assert choose_flower_target_fields(fields, fraction=0.0) == []


def test_window_shift_and_harvest_overlap_are_clamped_and_validated() -> None:
    assert shift_window_str("5-20", -10) == "1-10"
    assert shift_window_str("350-360", 20) == "365-365"
    assert shift_window_str("not-a-window", 10) == "not-a-window"

    management = pd.DataFrame({"harvest_date": [100, 150, 220, None, 999]})
    assert harvest_overlap_share(management, (90, 180)) == pytest.approx(2 / 3)
    assert math.isnan(harvest_overlap_share(pd.DataFrame({"other": [1]}), (90, 180)))


def test_skylark_selection_prefers_eligible_fields_without_flower_strips() -> None:
    fields = pd.DataFrame({"field_id": [1, 2, 3, 4]})
    rotation = pd.DataFrame(
        {
            "field_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "year": [1, 2] * 4,
            "crop": [
                "Winterweizen",
                "Winterweizen",
                "Winterweizen",
                "Mais",
                "Mais",
                "Mais",
                "Sommergerste",
                "Sommergerste",
            ],
        }
    )

    chosen = choose_skylark_target_fields(
        fields,
        rotation,
        eligible_crops={"Winterweizen", "Sommergerste"},
        min_years=2,
        top_fraction=0.25,
    )

    assert chosen == [1]


def test_management_scenario_updates_copies_and_clamps_days() -> None:
    fields = pd.DataFrame(
        {"field_id": [1, 2, 3], "flower_strip_share": [0.0, 0.2, 0.1]}
    )
    management = pd.DataFrame({"field_id": [1, 2, 3], "harvest_date": [360, 20, 100]})

    changed_fields, changed_management = apply_management_scenario(
        fields_df=fields,
        management_df=management,
        harvest_shift_days=10,
        add_flower_strip=0.1,
        target_fraction=0.2,
        selection="no_flower_first",
        skylark_plots=2,
        skylark_plot_area=0.02,
        skylark_target_field_ids=[2],
    )

    assert fields.columns.tolist() == ["field_id", "flower_strip_share"]
    assert management["harvest_date"].tolist() == [360, 20, 100]
    assert changed_management["harvest_date"].tolist() == [365, 30, 110]
    assert changed_fields.set_index("field_id").loc[1, "flower_strip_share"] == pytest.approx(0.1)
    assert changed_fields.set_index("field_id").loc[2, "skylark_plots"] == 2
    assert changed_fields.set_index("field_id").loc[2, "skylark_plot_area"] == pytest.approx(0.02)


def test_fallow_scenario_changes_only_selected_year_and_quiets_management() -> None:
    fields = pd.DataFrame({"field_id": [1, 2, 3], "area": [6.0, 3.0, 1.0]})
    rotation = pd.DataFrame(
        {
            "field_id": [1, 2, 3, 1, 2, 3],
            "year": [1, 1, 1, 2, 2, 2],
            "crop": ["Mais"] * 6,
        }
    )
    management = pd.DataFrame(
        {
            "field_id": [1, 2, 3, 1, 2, 3],
            "year": [1, 1, 1, 2, 2, 2],
            "harvest_date": [200] * 6,
            "insecticide_date": [150] * 6,
            "mow_dates": ["120"] * 6,
        }
    )

    changed_rotation, changed_management, chosen, actual_share = apply_fallow_scenario(
        fields,
        rotation,
        management,
        share=0.4,
        years=[2],
        mode="greedy",
        seed=9,
    )

    assert set(chosen) == {2, 3}
    assert actual_share == pytest.approx(0.4)
    year_one = changed_rotation[changed_rotation["year"] == 1]
    assert set(year_one["crop"]) == {"Mais"}
    year_two = changed_rotation[changed_rotation["year"] == 2].set_index("field_id")
    assert year_two.loc[1, "crop"] == "Mais"
    assert year_two.loc[2, "crop"] == "Brache"
    quiet = changed_management[
        (changed_management["year"] == 2) & changed_management["field_id"].isin(chosen)
    ]
    assert quiet["harvest_date"].isna().all()
    assert quiet["insecticide_date"].isna().all()
    assert (quiet["mow_dates"] == "").all()
