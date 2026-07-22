from __future__ import annotations

import numpy as np
import pytest

from fieldbird_sim.legacy.habitat import (
    cover_index,
    disturbance_components,
    food_index,
    load_skylark_windows_config,
    measure_modifiers,
    softmax,
)


def test_crop_aliases_and_indices_are_bounded() -> None:
    assert food_index("WW", 160) == pytest.approx(food_index("Winterweizen", 160))
    assert food_index("FALLOW", 170) == pytest.approx(food_index("Brache", 170))
    assert 0.0 <= cover_index("Mais", 170) <= 1.0
    assert cover_index("Brache", 100) == pytest.approx(0.6)
    assert food_index("unknown crop", -100) > 0.0


def test_disturbance_combines_event_pulses_and_windows() -> None:
    pulse, window = disturbance_components(
        150,
        np.array([149]),
        harvest_day=150,
        insecticide_day=150,
        sow_win=(140, 155),
        harv_win=(145, 160),
        other_win=None,
        field_area_ha=10.0,
    )

    assert pulse < -2.9
    assert window < -1.7


def test_measure_modifiers_normalizes_refuge_area() -> None:
    food, cover, disturbance, refuge_area, refuge_share = measure_modifiers(
        flower_strip_share=0.2,
        skylark_plots=2,
        skylark_plot_area=0.05,
        perimeter_m=400,
        median_perimeter_m=200,
        field_area_ha=2.0,
    )

    assert food == pytest.approx(0.12)
    assert cover == pytest.approx(0.10)
    assert refuge_area == pytest.approx(0.10)
    assert refuge_share == pytest.approx(0.05)
    assert disturbance == pytest.approx(0.985)


def test_softmax_handles_empty_and_nonfinite_inputs() -> None:
    assert softmax([]).size == 0
    np.testing.assert_allclose(softmax([np.nan, np.inf]), [0.5, 0.5])
    values = softmax([1.0, 2.0, 3.0])
    assert values.sum() == pytest.approx(1.0)
    assert values[2] > values[1] > values[0]


def test_skylark_window_config_parses_localized_values(tmp_path) -> None:
    config = tmp_path / "skylark_windows.csv"
    config.write_text(
        "key,value\n"
        "size_m2,25\n"
        "target_per_ha,2.5\n"
        "effect_crops,Winterweizen;Sommergerste\n"
        "effect_only_second_half,false\n",
        encoding="utf-8",
    )

    loaded = load_skylark_windows_config(config)

    assert loaded.size_m2 == pytest.approx(25.0)
    assert loaded.target_per_ha == pytest.approx(2.5)
    assert loaded.effect_only_second_half is False
