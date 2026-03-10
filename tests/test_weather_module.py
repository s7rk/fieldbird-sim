from __future__ import annotations

import pandas as pd
import pytest

from fieldbird_sim.cli import load_run_config
from fieldbird_sim.weather.weather_data import WeatherDataStore
from fieldbird_sim.weather.weather_impact import (
    FactorConfig,
    StressConfig,
    WeatherEventsConfig,
    WeatherEventTypeConfig,
    WeatherImpactConfig,
    WeatherImpactModule,
)


def test_weather_data_lookup_policies(tmp_path):
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,temp_min_C,temp_max_C,precip_mm,wind_gust_max_ms\n"
        "2020-01-01,1,10,2,5\n"
        "2020-01-03,2,11,3,6\n",
        encoding="utf-8",
    )

    store = WeatherDataStore.from_csv(csv_path)
    row = store.get_row("2020-01-01")
    assert float(row["temp_max_C"]) == 10.0

    with pytest.raises(KeyError):
        store.get_row("2020-01-02", missing_date_policy="raise")

    row_ffill = store.get_row("2020-01-02", missing_date_policy="ffill")
    assert float(row_ffill["temp_max_C"]) == 10.0

    row_bfill = store.get_row("2020-01-02", missing_date_policy="bfill")
    assert float(row_bfill["temp_max_C"]) == 11.0


def test_weather_impact_stress_and_factors_bounds(tmp_path):
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,temp_min_C,temp_max_C,precip_mm,wind_gust_max_ms,temp_mean_C,wind_mean_ms,sunshine_duration_h\n"
        "2020-01-01,12,20,1,5,16,2,6\n"
        "2020-01-02,-5,39,40,25,12,9,1\n",
        encoding="utf-8",
    )

    cfg = WeatherImpactConfig(
        enabled=True,
        weather_csv=str(csv_path),
        missing_date_policy="raise",
        reproduction=FactorConfig(
            intercept=1.0,
            stress_weight=1.2,
            food_weight=0.0,
            habitat_weight=0.0,
            min=0.2,
            max=1.5,
        ),
        juvenile_survival=FactorConfig(
            intercept=0.8,
            stress_weight=1.5,
            food_weight=0.0,
            habitat_weight=0.0,
            min=0.1,
            max=1.3,
        ),
        adult_survival=FactorConfig(
            intercept=1.2,
            stress_weight=0.8,
            food_weight=0.0,
            habitat_weight=0.0,
            min=0.3,
            max=1.2,
        ),
    )
    module = WeatherImpactModule.from_csv(cfg)
    day1 = module.step("2020-01-01")
    day2 = module.step("2020-01-02")

    assert float(day2["stress_state"]) > float(day1["stress_state"])
    assert float(day2["reproduction_factor"]) < float(day1["reproduction_factor"])
    assert float(day2["juvenile_survival_factor"]) < float(day1["juvenile_survival_factor"])
    assert float(day2["adult_survival_factor"]) < float(day1["adult_survival_factor"])

    for key, lo, hi in [
        ("reproduction_factor", cfg.reproduction.min, cfg.reproduction.max),
        ("juvenile_survival_factor", cfg.juvenile_survival.min, cfg.juvenile_survival.max),
        ("adult_survival_factor", cfg.adult_survival.min, cfg.adult_survival.max),
    ]:
        assert lo <= float(day2[key]) <= hi


def test_weather_module_steps_365_days_without_error(tmp_path):
    csv_path = tmp_path / "weather_365.csv"
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temp_min_C": 6.0,
            "temp_max_C": 18.0,
            "precip_mm": 2.0,
            "wind_gust_max_ms": 4.0,
            "temp_mean_C": 12.0,
            "wind_mean_ms": 2.0,
            "sunshine_duration_h": 5.0,
        }
    )
    df.to_csv(csv_path, index=False)

    module = WeatherImpactModule.from_csv(
        WeatherImpactConfig(enabled=True, weather_csv=str(csv_path), missing_date_policy="raise")
    )
    for d in dates:
        module.step(d)
    assert len(module.to_frame()) == 365


def test_weather_module_phase2_states_and_factors(tmp_path):
    csv_path = tmp_path / "weather_phase2.csv"
    csv_path.write_text(
        "date,temp_min_C,temp_max_C,precip_mm,wind_gust_max_ms,sunshine_duration_h,precip_30d_sum\n"
        "2020-01-01,10,20,1,3,0,90\n"
        "2020-01-02,10,20,4,3,10,90\n"
        "2020-01-03,10,20,30,35,1,220\n",
        encoding="utf-8",
    )

    cfg = WeatherImpactConfig(
        enabled=True,
        weather_csv=str(csv_path),
        missing_date_policy="raise",
        stress=StressConfig(
            decay=0.95,
            heat_threshold=100.0,
            heat_scale=10.0,
            cold_threshold=-100.0,
            cold_scale=10.0,
            wet_threshold=100.0,
            wet_scale=20.0,
            wind_threshold=100.0,
            wind_scale=10.0,
            w_heat=0.25,
            w_cold=0.20,
            w_wet=0.15,
            w_wind=0.10,
            state_clip=5.0,
        ),
        reproduction=FactorConfig(
            intercept=0.0,
            stress_weight=0.0,
            food_weight=1.5,
            habitat_weight=1.2,
            min=0.1,
            max=2.0,
        ),
    )
    module = WeatherImpactModule.from_csv(cfg)
    day1 = module.step("2020-01-01")
    day2 = module.step("2020-01-02")
    day3 = module.step("2020-01-03")

    assert float(day2["food_state"]) > float(day1["food_state"])
    assert float(day3["habitat_state"]) < float(day2["habitat_state"])
    assert float(day2["reproduction_factor"]) > float(day1["reproduction_factor"])
    assert float(day3["reproduction_factor"]) < float(day2["reproduction_factor"])


def test_weather_module_phase3_interactions_and_season_weighting(tmp_path):
    csv_path = tmp_path / "weather_phase3.csv"
    csv_path.write_text(
        "date,dayofyear,temp_min_C,temp_max_C,precip_mm,wind_gust_max_ms,sunshine_duration_h,precip_30d_sum\n"
        "2020-01-20,20,0,42,16,18,5,5\n"
        "2020-06-08,160,0,42,16,18,5,5\n",
        encoding="utf-8",
    )

    module = WeatherImpactModule.from_csv(
        WeatherImpactConfig(
            enabled=True,
            weather_csv=str(csv_path),
            missing_date_policy="raise",
        )
    )
    day_winter = module.step("2020-01-20")
    day_breeding = module.step("2020-06-08")

    assert float(day_winter["drought"]) > 0.0
    assert float(day_winter["heat_drought"]) > 0.0
    assert abs(float(day_winter["heat_drought"]) - float(day_breeding["heat_drought"])) < 1e-9
    assert float(day_breeding["season_weight"]) > float(day_winter["season_weight"])
    assert float(day_breeding["interaction_mix"]) > float(day_winter["interaction_mix"])


def test_load_run_config_parses_weather_module(tmp_path):
    cfg_path = tmp_path / "run.yml"
    cfg_path.write_text(
        "run:\n"
        "  n_fields: 5\n"
        "weather_module:\n"
        "  enabled: true\n"
        "  weather_csv: weather.csv\n"
        "  seed: 123\n"
        "  missing_date_policy: ffill\n"
        "  stress:\n"
        "    heat_threshold: 30\n"
        "  food:\n"
        "    sunshine_threshold_h: 5\n"
        "  habitat:\n"
        "    storm_threshold: 20\n"
        "  interactions:\n"
        "    season_mode: trig\n"
        "    drought_threshold_30d_mm: 40\n"
        "  events:\n"
        "    heatwave:\n"
        "      threshold: 31\n"
        "      min_consecutive_days: 2\n"
        "      effect_duration_days: 5\n"
        "  factors:\n"
        "    reproduction:\n"
        "      intercept: 1.4\n",
        encoding="utf-8",
    )

    cfg = load_run_config(cfg_path)
    assert cfg.weather_module.enabled is True
    assert cfg.weather_module.weather_csv == "weather.csv"
    assert cfg.weather_module.seed == 123
    assert cfg.weather_module.missing_date_policy == "ffill"
    assert cfg.weather_module.stress.heat_threshold == 30.0
    assert cfg.weather_module.food.sunshine_threshold_h == 5.0
    assert cfg.weather_module.habitat.storm_threshold == 20.0
    assert cfg.weather_module.interactions.season_mode == "trig"
    assert cfg.weather_module.interactions.drought_threshold_30d_mm == 40.0
    assert cfg.weather_module.events.heatwave.threshold == 31.0
    assert cfg.weather_module.events.heatwave.min_consecutive_days == 2
    assert cfg.weather_module.events.heatwave.effect_duration_days == 5
    assert cfg.weather_module.factors.reproduction.intercept == 1.4


def test_weather_module_phase4_event_shocks_and_duration(tmp_path):
    csv_path = tmp_path / "weather_phase4.csv"
    csv_path.write_text(
        "date,temp_min_C,temp_max_C,precip_mm,wind_gust_max_ms,sunshine_duration_h,precip_30d_sum\n"
        "2020-07-01,12,25,0,4,6,80\n"
        "2020-07-02,12,35,0,4,6,80\n"
        "2020-07-03,12,35,0,4,6,80\n"
        "2020-07-04,12,25,0,4,6,80\n",
        encoding="utf-8",
    )

    cfg = WeatherImpactConfig(
        enabled=True,
        weather_csv=str(csv_path),
        missing_date_policy="raise",
        stress=StressConfig(
            decay=0.95,
            heat_threshold=100.0,
            heat_scale=10.0,
            cold_threshold=-100.0,
            cold_scale=10.0,
            wet_threshold=100.0,
            wet_scale=20.0,
            wind_threshold=100.0,
            wind_scale=10.0,
            w_heat=0.25,
            w_cold=0.20,
            w_wet=0.15,
            w_wind=0.10,
            state_clip=5.0,
        ),
        events=WeatherEventsConfig(
            heatwave=WeatherEventTypeConfig(
                enabled=True,
                threshold=30.0,
                min_consecutive_days=2,
                effect_duration_days=2,
                trigger_probability=1.0,
                shock_reproduction_mult=0.5,
                shock_juvenile_survival_mult=1.0,
                shock_adult_survival_mult=1.0,
            ),
            heavy_rain=WeatherEventTypeConfig(enabled=False),
            storm=WeatherEventTypeConfig(enabled=False),
            cold_snap=WeatherEventTypeConfig(enabled=False),
        ),
        reproduction=FactorConfig(
            intercept=0.0,
            stress_weight=0.0,
            food_weight=0.0,
            habitat_weight=0.0,
            min=0.5,
            max=1.5,
        ),
    )
    module = WeatherImpactModule.from_csv(cfg)

    day1 = module.step("2020-07-01")
    day2 = module.step("2020-07-02")
    day3 = module.step("2020-07-03")
    day4 = module.step("2020-07-04")

    assert day1["event_trigger_heatwave"] == 0
    assert day2["event_trigger_heatwave"] == 0
    assert day3["event_trigger_heatwave"] == 1
    assert day4["event_trigger_heatwave"] == 0

    assert day1["event_active_heatwave"] == 0
    assert day2["event_active_heatwave"] == 0
    assert day3["event_active_heatwave"] == 1
    assert day4["event_active_heatwave"] == 1

    assert float(day1["reproduction_factor"]) == pytest.approx(1.0)
    assert float(day2["reproduction_factor"]) == pytest.approx(1.0)
    assert float(day3["reproduction_factor"]) == pytest.approx(0.5)
    assert float(day4["reproduction_factor"]) == pytest.approx(0.5)


def test_weather_module_seed_reproducible_event_triggers(tmp_path):
    csv_path = tmp_path / "weather_seed_repro.csv"
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temp_min_C": 12.0,
            "temp_max_C": 20.0,
            "precip_mm": 12.0,
            "wind_gust_max_ms": 3.0,
            "sunshine_duration_h": 6.0,
            "precip_30d_sum": 100.0,
        }
    )
    df.to_csv(csv_path, index=False)

    def run_with_seed(seed: int) -> pd.DataFrame:
        cfg = WeatherImpactConfig(
            enabled=True,
            weather_csv=str(csv_path),
            seed=seed,
            missing_date_policy="raise",
            stress=StressConfig(
                decay=0.95,
                heat_threshold=100.0,
                heat_scale=10.0,
                cold_threshold=-100.0,
                cold_scale=10.0,
                wet_threshold=100.0,
                wet_scale=20.0,
                wind_threshold=100.0,
                wind_scale=10.0,
                w_heat=0.25,
                w_cold=0.20,
                w_wet=0.15,
                w_wind=0.10,
                state_clip=5.0,
            ),
            events=WeatherEventsConfig(
                heavy_rain=WeatherEventTypeConfig(
                    enabled=True,
                    threshold=1.0,
                    min_consecutive_days=1,
                    effect_duration_days=1,
                    trigger_probability=0.5,
                    shock_reproduction_mult=0.7,
                    shock_juvenile_survival_mult=1.0,
                    shock_adult_survival_mult=1.0,
                ),
                heatwave=WeatherEventTypeConfig(enabled=False),
                storm=WeatherEventTypeConfig(enabled=False),
                cold_snap=WeatherEventTypeConfig(enabled=False),
            ),
            reproduction=FactorConfig(
                intercept=0.0,
                stress_weight=0.0,
                food_weight=0.0,
                habitat_weight=0.0,
                min=0.5,
                max=1.5,
            ),
        )
        module = WeatherImpactModule.from_csv(cfg)
        for d in dates:
            module.step(d)
        return module.to_frame()

    run_a = run_with_seed(12345)
    run_b = run_with_seed(12345)
    run_c = run_with_seed(54321)

    pd.testing.assert_frame_equal(run_a, run_b, check_dtype=False, check_exact=True)
    assert run_a["event_trigger_heavy_rain"].tolist() != run_c["event_trigger_heavy_rain"].tolist()
