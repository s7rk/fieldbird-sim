from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from fieldbird_sim.legacy.weather_builder import build_weather_module_from_cfg


def test_build_weather_module_from_cfg_disabled_returns_none():
    cfg = SimpleNamespace(weather_module=SimpleNamespace(enabled=False))
    module, start_date = build_weather_module_from_cfg(cfg)
    assert module is None
    assert start_date is None


def test_build_weather_module_from_cfg_enabled_uses_defaults_and_csv(tmp_path):
    csv_path = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "temp_min_C": [1.0, 2.0],
            "temp_max_C": [8.0, 9.0],
            "precip_mm": [0.0, 1.0],
            "wind_gust_max_ms": [4.0, 5.0],
        }
    ).to_csv(csv_path, index=False)

    cfg = SimpleNamespace(
        weather_module=SimpleNamespace(
            enabled=True,
            weather_csv=str(csv_path),
            seed=123,
            missing_date_policy="bfill",
        )
    )

    module, start_date = build_weather_module_from_cfg(cfg)

    assert module is not None
    assert module.cfg.weather_csv == str(csv_path)
    assert module.cfg.seed == 123
    assert module.cfg.missing_date_policy == "bfill"
    assert str(start_date.date()) == "2020-01-01"
