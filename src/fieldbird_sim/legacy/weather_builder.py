from __future__ import annotations

from fieldbird_sim.weather.weather_impact import (
    FactorConfig,
    FoodConfig,
    HabitatConfig,
    InteractionConfig,
    StressConfig,
    WeatherEventTypeConfig,
    WeatherEventsConfig,
    WeatherImpactConfig,
    WeatherImpactModule,
)


def build_weather_module_from_cfg(cfg):
    weather = getattr(cfg, "weather_module", None)
    if weather is None or not bool(getattr(weather, "enabled", False)):
        return None, None

    stress = getattr(weather, "stress", None)
    food = getattr(weather, "food", None)
    habitat = getattr(weather, "habitat", None)
    interactions = getattr(weather, "interactions", None)
    events = getattr(weather, "events", None)
    heatwave = getattr(events, "heatwave", None)
    heavy_rain = getattr(events, "heavy_rain", None)
    storm = getattr(events, "storm", None)
    cold_snap = getattr(events, "cold_snap", None)
    factors = getattr(weather, "factors", None)
    rep = getattr(factors, "reproduction", None)
    juv = getattr(factors, "juvenile_survival", None)
    adult = getattr(factors, "adult_survival", None)

    weather_cfg = WeatherImpactConfig(
        enabled=True,
        weather_csv=str(getattr(weather, "weather_csv", "dwd_hoexter_out/hoexter_kl_final_for_birdsim.csv")),
        seed=int(getattr(weather, "seed", 42)),
        start_date=getattr(weather, "start_date", None),
        missing_date_policy=str(getattr(weather, "missing_date_policy", "raise")),
        stress=StressConfig(
            decay=float(getattr(stress, "decay", 0.95)),
            heat_threshold=float(getattr(stress, "heat_threshold", 28.0)),
            heat_scale=float(getattr(stress, "heat_scale", 10.0)),
            cold_threshold=float(getattr(stress, "cold_threshold", 5.0)),
            cold_scale=float(getattr(stress, "cold_scale", 10.0)),
            wet_threshold=float(getattr(stress, "wet_threshold", 12.0)),
            wet_scale=float(getattr(stress, "wet_scale", 20.0)),
            wind_threshold=float(getattr(stress, "wind_threshold", 12.0)),
            wind_scale=float(getattr(stress, "wind_scale", 10.0)),
            w_heat=float(getattr(stress, "w_heat", 0.25)),
            w_cold=float(getattr(stress, "w_cold", 0.20)),
            w_wet=float(getattr(stress, "w_wet", 0.15)),
            w_wind=float(getattr(stress, "w_wind", 0.10)),
            state_clip=float(getattr(stress, "state_clip", 5.0)),
        ),
        food=FoodConfig(
            decay=float(getattr(food, "decay", 0.90)),
            sunshine_threshold_h=float(getattr(food, "sunshine_threshold_h", 4.0)),
            sunshine_scale_h=float(getattr(food, "sunshine_scale_h", 6.0)),
            rain_optimal_mm=float(getattr(food, "rain_optimal_mm", 4.0)),
            rain_tolerance_mm=float(getattr(food, "rain_tolerance_mm", 8.0)),
            w_sunshine=float(getattr(food, "w_sunshine", 0.20)),
            w_moderate_rain=float(getattr(food, "w_moderate_rain", 0.15)),
            heat_penalty_weight=float(getattr(food, "heat_penalty_weight", 0.45)),
            wet_penalty_weight=float(getattr(food, "wet_penalty_weight", 0.40)),
            state_clip=float(getattr(food, "state_clip", 5.0)),
        ),
        habitat=HabitatConfig(
            decay=float(getattr(habitat, "decay", 0.94)),
            wetness_optimal_30d_mm=float(getattr(habitat, "wetness_optimal_30d_mm", 90.0)),
            wetness_tolerance_30d_mm=float(getattr(habitat, "wetness_tolerance_30d_mm", 80.0)),
            storm_threshold=float(getattr(habitat, "storm_threshold", 16.0)),
            storm_scale=float(getattr(habitat, "storm_scale", 8.0)),
            w_wet_balance=float(getattr(habitat, "w_wet_balance", 0.20)),
            w_stability=float(getattr(habitat, "w_stability", 0.15)),
            storm_penalty_weight=float(getattr(habitat, "storm_penalty_weight", 0.45)),
            state_clip=float(getattr(habitat, "state_clip", 5.0)),
        ),
        interactions=InteractionConfig(
            drought_threshold_30d_mm=float(
                getattr(interactions, "drought_threshold_30d_mm", 50.0)
            ),
            drought_scale_30d_mm=float(getattr(interactions, "drought_scale_30d_mm", 35.0)),
            heat_drought_weight=float(getattr(interactions, "heat_drought_weight", 0.35)),
            cold_wet_weight=float(getattr(interactions, "cold_wet_weight", 0.20)),
            season_mode=str(getattr(interactions, "season_mode", "window")),
            breeding_start_doy=int(getattr(interactions, "breeding_start_doy", 95)),
            breeding_end_doy=int(getattr(interactions, "breeding_end_doy", 205)),
            season_strength=float(getattr(interactions, "season_strength", 0.85)),
            add_to_stress_weight=float(getattr(interactions, "add_to_stress_weight", 0.60)),
            add_to_food_weight=float(getattr(interactions, "add_to_food_weight", 0.25)),
            add_to_habitat_weight=float(getattr(interactions, "add_to_habitat_weight", 0.30)),
        ),
        events=WeatherEventsConfig(
            heatwave=WeatherEventTypeConfig(
                enabled=bool(getattr(heatwave, "enabled", True)),
                threshold=float(getattr(heatwave, "threshold", 32.0)),
                min_consecutive_days=int(getattr(heatwave, "min_consecutive_days", 3)),
                effect_duration_days=int(getattr(heatwave, "effect_duration_days", 4)),
                trigger_probability=float(getattr(heatwave, "trigger_probability", 1.0)),
                shock_reproduction_mult=float(getattr(heatwave, "shock_reproduction_mult", 0.80)),
                shock_juvenile_survival_mult=float(
                    getattr(heatwave, "shock_juvenile_survival_mult", 0.95)
                ),
                shock_adult_survival_mult=float(getattr(heatwave, "shock_adult_survival_mult", 0.99)),
            ),
            heavy_rain=WeatherEventTypeConfig(
                enabled=bool(getattr(heavy_rain, "enabled", True)),
                threshold=float(getattr(heavy_rain, "threshold", 35.0)),
                min_consecutive_days=int(getattr(heavy_rain, "min_consecutive_days", 1)),
                effect_duration_days=int(getattr(heavy_rain, "effect_duration_days", 2)),
                trigger_probability=float(getattr(heavy_rain, "trigger_probability", 1.0)),
                shock_reproduction_mult=float(getattr(heavy_rain, "shock_reproduction_mult", 0.88)),
                shock_juvenile_survival_mult=float(
                    getattr(heavy_rain, "shock_juvenile_survival_mult", 0.96)
                ),
                shock_adult_survival_mult=float(
                    getattr(heavy_rain, "shock_adult_survival_mult", 0.995)
                ),
            ),
            storm=WeatherEventTypeConfig(
                enabled=bool(getattr(storm, "enabled", True)),
                threshold=float(getattr(storm, "threshold", 22.0)),
                min_consecutive_days=int(getattr(storm, "min_consecutive_days", 1)),
                effect_duration_days=int(getattr(storm, "effect_duration_days", 3)),
                trigger_probability=float(getattr(storm, "trigger_probability", 1.0)),
                shock_reproduction_mult=float(getattr(storm, "shock_reproduction_mult", 0.85)),
                shock_juvenile_survival_mult=float(
                    getattr(storm, "shock_juvenile_survival_mult", 0.94)
                ),
                shock_adult_survival_mult=float(getattr(storm, "shock_adult_survival_mult", 0.992)),
            ),
            cold_snap=WeatherEventTypeConfig(
                enabled=bool(getattr(cold_snap, "enabled", True)),
                threshold=float(getattr(cold_snap, "threshold", -2.0)),
                min_consecutive_days=int(getattr(cold_snap, "min_consecutive_days", 2)),
                effect_duration_days=int(getattr(cold_snap, "effect_duration_days", 3)),
                trigger_probability=float(getattr(cold_snap, "trigger_probability", 1.0)),
                shock_reproduction_mult=float(getattr(cold_snap, "shock_reproduction_mult", 0.86)),
                shock_juvenile_survival_mult=float(
                    getattr(cold_snap, "shock_juvenile_survival_mult", 0.95)
                ),
                shock_adult_survival_mult=float(getattr(cold_snap, "shock_adult_survival_mult", 0.993)),
            ),
        ),
        reproduction=FactorConfig(
            intercept=float(getattr(rep, "intercept", 0.35)),
            stress_weight=float(getattr(rep, "stress_weight", 0.75)),
            food_weight=float(getattr(rep, "food_weight", 0.25)),
            habitat_weight=float(getattr(rep, "habitat_weight", 0.15)),
            min=float(getattr(rep, "min", 0.90)),
            max=float(getattr(rep, "max", 1.10)),
        ),
        juvenile_survival=FactorConfig(
            intercept=float(getattr(juv, "intercept", 0.35)),
            stress_weight=float(getattr(juv, "stress_weight", 0.55)),
            food_weight=float(getattr(juv, "food_weight", 0.12)),
            habitat_weight=float(getattr(juv, "habitat_weight", 0.10)),
            min=float(getattr(juv, "min", 0.97)),
            max=float(getattr(juv, "max", 1.03)),
        ),
        adult_survival=FactorConfig(
            intercept=float(getattr(adult, "intercept", 0.40)),
            stress_weight=float(getattr(adult, "stress_weight", 0.45)),
            food_weight=float(getattr(adult, "food_weight", 0.08)),
            habitat_weight=float(getattr(adult, "habitat_weight", 0.08)),
            min=float(getattr(adult, "min", 0.995)),
            max=float(getattr(adult, "max", 1.005)),
        ),
    )
    module = WeatherImpactModule.from_csv(weather_cfg)
    start_date = weather_cfg.start_date or module.default_start_date
    print(
        f"[weather] enabled=True | csv={weather_cfg.weather_csv} | "
        f"start_date={start_date} | missing_date_policy={weather_cfg.missing_date_policy}"
    )
    return module, start_date
