from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .weather_data import WeatherDataStore


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return float(f)


def _sigmoid(x: float) -> float:
    z = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(-z))


def _clamp(v: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return float(min(max(v, lo), hi))


@dataclass(frozen=True)
class StressConfig:
    decay: float = 0.95
    heat_threshold: float = 28.0
    heat_scale: float = 10.0
    cold_threshold: float = 5.0
    cold_scale: float = 10.0
    wet_threshold: float = 12.0
    wet_scale: float = 20.0
    wind_threshold: float = 12.0
    wind_scale: float = 10.0
    w_heat: float = 0.25
    w_cold: float = 0.20
    w_wet: float = 0.15
    w_wind: float = 0.10
    state_clip: float = 5.0


@dataclass(frozen=True)
class FactorConfig:
    intercept: float = 1.0
    stress_weight: float = 1.0
    food_weight: float = 0.0
    habitat_weight: float = 0.0
    min: float = 0.2
    max: float = 1.5


@dataclass(frozen=True)
class FoodConfig:
    decay: float = 0.90
    sunshine_threshold_h: float = 4.0
    sunshine_scale_h: float = 6.0
    rain_optimal_mm: float = 4.0
    rain_tolerance_mm: float = 8.0
    w_sunshine: float = 0.20
    w_moderate_rain: float = 0.15
    heat_penalty_weight: float = 0.45
    wet_penalty_weight: float = 0.40
    state_clip: float = 5.0


@dataclass(frozen=True)
class HabitatConfig:
    decay: float = 0.94
    wetness_optimal_30d_mm: float = 90.0
    wetness_tolerance_30d_mm: float = 80.0
    storm_threshold: float = 16.0
    storm_scale: float = 8.0
    w_wet_balance: float = 0.20
    w_stability: float = 0.15
    storm_penalty_weight: float = 0.45
    state_clip: float = 5.0


@dataclass(frozen=True)
class InteractionConfig:
    drought_threshold_30d_mm: float = 50.0
    drought_scale_30d_mm: float = 35.0
    heat_drought_weight: float = 0.35
    cold_wet_weight: float = 0.20
    season_mode: str = "window"  # window | trig
    breeding_start_doy: int = 95
    breeding_end_doy: int = 205
    season_strength: float = 0.85
    add_to_stress_weight: float = 0.60
    add_to_food_weight: float = 0.25
    add_to_habitat_weight: float = 0.30


@dataclass(frozen=True)
class WeatherEventTypeConfig:
    enabled: bool = True
    threshold: float = 0.0
    min_consecutive_days: int = 1
    effect_duration_days: int = 1
    trigger_probability: float = 1.0
    shock_reproduction_mult: float = 1.0
    shock_juvenile_survival_mult: float = 1.0
    shock_adult_survival_mult: float = 1.0


@dataclass(frozen=True)
class WeatherEventsConfig:
    heatwave: WeatherEventTypeConfig = WeatherEventTypeConfig(
        threshold=32.0,
        min_consecutive_days=3,
        effect_duration_days=4,
        trigger_probability=1.0,
        shock_reproduction_mult=0.80,
        shock_juvenile_survival_mult=0.95,
        shock_adult_survival_mult=0.99,
    )
    heavy_rain: WeatherEventTypeConfig = WeatherEventTypeConfig(
        threshold=35.0,
        min_consecutive_days=1,
        effect_duration_days=2,
        trigger_probability=1.0,
        shock_reproduction_mult=0.88,
        shock_juvenile_survival_mult=0.96,
        shock_adult_survival_mult=0.995,
    )
    storm: WeatherEventTypeConfig = WeatherEventTypeConfig(
        threshold=22.0,
        min_consecutive_days=1,
        effect_duration_days=3,
        trigger_probability=1.0,
        shock_reproduction_mult=0.85,
        shock_juvenile_survival_mult=0.94,
        shock_adult_survival_mult=0.992,
    )
    cold_snap: WeatherEventTypeConfig = WeatherEventTypeConfig(
        threshold=-2.0,
        min_consecutive_days=2,
        effect_duration_days=3,
        trigger_probability=1.0,
        shock_reproduction_mult=0.86,
        shock_juvenile_survival_mult=0.95,
        shock_adult_survival_mult=0.993,
    )


@dataclass(frozen=True)
class WeatherImpactConfig:
    enabled: bool = False
    weather_csv: str = "dwd_hoexter_out/hoexter_kl_final_for_birdsim.csv"
    seed: int = 42
    start_date: str | None = None
    missing_date_policy: str = "raise"
    stress: StressConfig = StressConfig()
    food: FoodConfig = FoodConfig()
    habitat: HabitatConfig = HabitatConfig()
    interactions: InteractionConfig = InteractionConfig()
    events: WeatherEventsConfig = WeatherEventsConfig()
    reproduction: FactorConfig = FactorConfig(
        intercept=0.35,
        stress_weight=0.75,
        food_weight=0.25,
        habitat_weight=0.15,
        min=0.90,
        max=1.10,
    )
    juvenile_survival: FactorConfig = FactorConfig(
        intercept=0.35,
        stress_weight=0.55,
        food_weight=0.12,
        habitat_weight=0.10,
        min=0.97,
        max=1.03,
    )
    adult_survival: FactorConfig = FactorConfig(
        intercept=0.40,
        stress_weight=0.45,
        food_weight=0.08,
        habitat_weight=0.08,
        min=0.995,
        max=1.005,
    )


class WeatherImpactModule:
    def __init__(self, data: WeatherDataStore, cfg: WeatherImpactConfig):
        self.data = data
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))
        self.stress_state = 0.0
        self.food_state = 0.0
        self.habitat_state = 0.0
        self._precip_history: list[float] = []
        self._heatwave_streak = 0
        self._cold_snap_streak = 0
        self._active_events = {
            "heatwave": 0,
            "heavy_rain": 0,
            "storm": 0,
            "cold_snap": 0,
        }
        self._rows: list[dict[str, Any]] = []

    @classmethod
    def from_csv(cls, cfg: WeatherImpactConfig) -> "WeatherImpactModule":
        data = WeatherDataStore.from_csv(cfg.weather_csv)
        return cls(data=data, cfg=cfg)

    @property
    def default_start_date(self) -> pd.Timestamp:
        if self.cfg.start_date:
            return pd.Timestamp(self.cfg.start_date).normalize()
        return self.data.start_date

    def _component(self, value: float, threshold: float, scale: float, *, kind: str) -> float:
        s = max(_safe_float(scale, 1.0), 1e-9)
        if kind == "above":
            return max(0.0, (value - threshold) / s)
        return max(0.0, (threshold - value) / s)

    def _factor_from_states(self, fc: FactorConfig) -> float:
        score = (
            fc.intercept
            - fc.stress_weight * self.stress_state
            + fc.food_weight * self.food_state
            + fc.habitat_weight * self.habitat_state
        )
        sig = _sigmoid(score)
        scaled = fc.min + (fc.max - fc.min) * sig
        return _clamp(scaled, fc.min, fc.max)

    def _season_weight(self, *, doy: int) -> float:
        ic = self.cfg.interactions
        mode = str(ic.season_mode or "window").strip().lower()
        strength = _clamp(_safe_float(ic.season_strength, 0.85), 0.0, 1.0)
        base = 1.0 - strength

        if mode == "trig":
            start = int(ic.breeding_start_doy)
            end = int(ic.breeding_end_doy)
            center = (start + end) / 2.0
            phase = 2.0 * np.pi * ((float(doy) - center) / 365.0)
            wave = 0.5 * (1.0 + np.cos(phase))
            return _clamp(base + strength * wave, 0.0, 1.0)

        start = int(ic.breeding_start_doy)
        end = int(ic.breeding_end_doy)
        if start <= end:
            in_window = start <= int(doy) <= end
        else:
            in_window = int(doy) >= start or int(doy) <= end
        return 1.0 if in_window else base

    def _trigger_event(
        self,
        *,
        name: str,
        cfg: WeatherEventTypeConfig,
        candidate: bool,
    ) -> int:
        if not candidate or not bool(cfg.enabled):
            return 0
        p = _clamp(_safe_float(cfg.trigger_probability, 1.0), 0.0, 1.0)
        if float(self.rng.random()) >= p:
            return 0
        dur = max(1, int(cfg.effect_duration_days))
        self._active_events[name] = max(int(self._active_events.get(name, 0)), dur)
        return 1

    def _active_event_multipliers(self) -> tuple[float, float, float, dict[str, int]]:
        ev = self.cfg.events
        event_cfg = {
            "heatwave": ev.heatwave,
            "heavy_rain": ev.heavy_rain,
            "storm": ev.storm,
            "cold_snap": ev.cold_snap,
        }
        repro_mult = 1.0
        juv_mult = 1.0
        adult_mult = 1.0
        active_flags: dict[str, int] = {}

        for name, cfg in event_cfg.items():
            active = int(self._active_events.get(name, 0)) > 0 and bool(cfg.enabled)
            active_flags[name] = int(active)
            if not active:
                continue
            repro_mult *= _clamp(_safe_float(cfg.shock_reproduction_mult, 1.0), 0.01, 2.0)
            juv_mult *= _clamp(_safe_float(cfg.shock_juvenile_survival_mult, 1.0), 0.01, 2.0)
            adult_mult *= _clamp(_safe_float(cfg.shock_adult_survival_mult, 1.0), 0.01, 2.0)
        return repro_mult, juv_mult, adult_mult, active_flags

    def _decrement_active_events(self) -> None:
        for k in list(self._active_events.keys()):
            left = int(self._active_events.get(k, 0))
            self._active_events[k] = max(0, left - 1)

    def step(self, sim_date: Any) -> dict[str, Any]:
        row = self.data.get_row(sim_date, missing_date_policy=self.cfg.missing_date_policy)
        d = pd.Timestamp(sim_date).normalize()

        temp_max = _safe_float(row.get("temp_max_C"), 0.0)
        temp_min = _safe_float(row.get("temp_min_C"), 0.0)
        precip = _safe_float(row.get("precip_mm"), 0.0)
        wind_gust = _safe_float(row.get("wind_gust_max_ms"), 0.0)
        temp_mean = _safe_float(row.get("temp_mean_C"), np.nan)
        wind_mean = _safe_float(row.get("wind_mean_ms"), np.nan)
        sunshine = _safe_float(row.get("sunshine_duration_h"), np.nan)
        dayofyear = int(_safe_float(row.get("dayofyear"), d.dayofyear))

        sc = self.cfg.stress
        heat = self._component(temp_max, sc.heat_threshold, sc.heat_scale, kind="above")
        cold = self._component(temp_min, sc.cold_threshold, sc.cold_scale, kind="below")
        wet = self._component(precip, sc.wet_threshold, sc.wet_scale, kind="above")
        wind = self._component(wind_gust, sc.wind_threshold, sc.wind_scale, kind="above")

        # Phase 4: event detectors and temporary shock counters
        ev = self.cfg.events
        prev_heatwave = int(self._heatwave_streak)
        if temp_max >= float(ev.heatwave.threshold):
            self._heatwave_streak += 1
        else:
            self._heatwave_streak = 0

        prev_cold = int(self._cold_snap_streak)
        if temp_min <= float(ev.cold_snap.threshold):
            self._cold_snap_streak += 1
        else:
            self._cold_snap_streak = 0

        heatwave_candidate = (
            self._heatwave_streak >= max(1, int(ev.heatwave.min_consecutive_days))
            and prev_heatwave < max(1, int(ev.heatwave.min_consecutive_days))
        )
        cold_snap_candidate = (
            self._cold_snap_streak >= max(1, int(ev.cold_snap.min_consecutive_days))
            and prev_cold < max(1, int(ev.cold_snap.min_consecutive_days))
        )
        heavy_rain_candidate = precip >= float(ev.heavy_rain.threshold)
        storm_candidate = wind_gust >= float(ev.storm.threshold)

        event_trigger_heatwave = self._trigger_event(
            name="heatwave",
            cfg=ev.heatwave,
            candidate=heatwave_candidate,
        )
        event_trigger_heavy_rain = self._trigger_event(
            name="heavy_rain",
            cfg=ev.heavy_rain,
            candidate=heavy_rain_candidate,
        )
        event_trigger_storm = self._trigger_event(
            name="storm",
            cfg=ev.storm,
            candidate=storm_candidate,
        )
        event_trigger_cold_snap = self._trigger_event(
            name="cold_snap",
            cfg=ev.cold_snap,
            candidate=cold_snap_candidate,
        )
        shock_repro_mult, shock_juv_mult, shock_adult_mult, event_active_flags = (
            self._active_event_multipliers()
        )

        stress_in = sc.w_heat * heat + sc.w_cold * cold + sc.w_wet * wet + sc.w_wind * wind
        self.stress_state = sc.decay * self.stress_state + stress_in
        self.stress_state = _clamp(self.stress_state, -abs(sc.state_clip), abs(sc.state_clip))

        # Phase 2: food memory state (good sunshine + moderate rain, down with extremes)
        fc = self.cfg.food
        sunshine_effect = self._component(
            sunshine,
            fc.sunshine_threshold_h,
            fc.sunshine_scale_h,
            kind="above",
        )
        rain_tol = max(_safe_float(fc.rain_tolerance_mm, 1.0), 1e-9)
        moderate_rain = max(0.0, 1.0 - abs(precip - fc.rain_optimal_mm) / rain_tol)
        # Center around typical conditions to avoid monotonic positive drift.
        sunshine_centered = sunshine_effect - 0.3
        moderate_rain_centered = moderate_rain - 0.5
        food_in = (
            fc.w_sunshine * sunshine_centered
            + fc.w_moderate_rain * moderate_rain_centered
            - fc.heat_penalty_weight * heat
            - fc.wet_penalty_weight * wet
        )
        self.food_state = fc.decay * self.food_state + food_in
        self.food_state = _clamp(self.food_state, -abs(fc.state_clip), abs(fc.state_clip))

        # Phase 2: habitat memory state (30d wetness balance + storm stability)
        self._precip_history.append(precip)
        if len(self._precip_history) > 120:
            self._precip_history = self._precip_history[-120:]

        hc = self.cfg.habitat
        precip_30d_sum = _safe_float(row.get("precip_30d_sum"), np.nan)
        if not np.isfinite(precip_30d_sum):
            precip_30d_sum = float(sum(self._precip_history[-30:]))

        ic = self.cfg.interactions
        drought = self._component(
            precip_30d_sum,
            ic.drought_threshold_30d_mm,
            ic.drought_scale_30d_mm,
            kind="below",
        )
        heat_drought = heat * drought
        cold_wet = cold * wet
        season_weight = self._season_weight(doy=dayofyear)
        interaction_mix = season_weight * (
            ic.heat_drought_weight * heat_drought + ic.cold_wet_weight * cold_wet
        )

        wet_tol = max(_safe_float(hc.wetness_tolerance_30d_mm, 1.0), 1e-9)
        wet_balance = max(0.0, 1.0 - abs(precip_30d_sum - hc.wetness_optimal_30d_mm) / wet_tol)
        storm = self._component(wind_gust, hc.storm_threshold, hc.storm_scale, kind="above")
        stability_score = max(0.0, 1.0 - storm)
        wet_balance_centered = wet_balance - 0.5
        stability_centered = stability_score - 0.5

        habitat_in = (
            hc.w_wet_balance * wet_balance_centered
            + hc.w_stability * stability_centered
            - hc.storm_penalty_weight * storm
        )
        habitat_in -= ic.add_to_habitat_weight * interaction_mix
        self.habitat_state = hc.decay * self.habitat_state + habitat_in
        self.habitat_state = _clamp(self.habitat_state, -abs(hc.state_clip), abs(hc.state_clip))

        self.food_state = _clamp(
            self.food_state - ic.add_to_food_weight * interaction_mix,
            -abs(fc.state_clip),
            abs(fc.state_clip),
        )
        self.stress_state = _clamp(
            self.stress_state + ic.add_to_stress_weight * interaction_mix,
            -abs(sc.state_clip),
            abs(sc.state_clip),
        )

        reproduction_factor = self._factor_from_states(self.cfg.reproduction)
        juvenile_survival_factor = self._factor_from_states(self.cfg.juvenile_survival)
        adult_survival_factor = self._factor_from_states(self.cfg.adult_survival)
        reproduction_factor = _clamp(
            reproduction_factor * shock_repro_mult,
            0.01,
            self.cfg.reproduction.max,
        )
        juvenile_survival_factor = _clamp(
            juvenile_survival_factor * shock_juv_mult,
            0.01,
            self.cfg.juvenile_survival.max,
        )
        adult_survival_factor = _clamp(
            adult_survival_factor * shock_adult_mult,
            0.01,
            self.cfg.adult_survival.max,
        )

        out = {
            "date": d.date().isoformat(),
            "temp_mean_C": temp_mean,
            "temp_min_C": temp_min,
            "temp_max_C": temp_max,
            "precip_mm": precip,
            "wind_mean_ms": wind_mean,
            "wind_gust_max_ms": wind_gust,
            "sunshine_duration_h": sunshine,
            "dayofyear": dayofyear,
            "heat": heat,
            "cold": cold,
            "wet": wet,
            "wind": wind,
            "stress_state": self.stress_state,
            "sunshine_effect": sunshine_effect,
            "moderate_rain": moderate_rain,
            "food_state": self.food_state,
            "precip_30d_sum_used": precip_30d_sum,
            "drought": drought,
            "heat_drought": heat_drought,
            "cold_wet": cold_wet,
            "season_weight": season_weight,
            "interaction_mix": interaction_mix,
            "wet_balance": wet_balance,
            "storm": storm,
            "habitat_state": self.habitat_state,
            "event_trigger_heatwave": event_trigger_heatwave,
            "event_trigger_heavy_rain": event_trigger_heavy_rain,
            "event_trigger_storm": event_trigger_storm,
            "event_trigger_cold_snap": event_trigger_cold_snap,
            "event_active_heatwave": int(event_active_flags.get("heatwave", 0)),
            "event_active_heavy_rain": int(event_active_flags.get("heavy_rain", 0)),
            "event_active_storm": int(event_active_flags.get("storm", 0)),
            "event_active_cold_snap": int(event_active_flags.get("cold_snap", 0)),
            "shock_reproduction_mult": shock_repro_mult,
            "shock_juvenile_survival_mult": shock_juv_mult,
            "shock_adult_survival_mult": shock_adult_mult,
            "reproduction_factor": reproduction_factor,
            "juvenile_survival_factor": juvenile_survival_factor,
            "adult_survival_factor": adult_survival_factor,
        }
        self._rows.append(out)
        self._decrement_active_events()
        return out

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(
                columns=[
                    "date",
                    "temp_mean_C",
                    "temp_min_C",
                    "temp_max_C",
                    "precip_mm",
                    "wind_mean_ms",
                    "wind_gust_max_ms",
                    "sunshine_duration_h",
                    "dayofyear",
                    "heat",
                    "cold",
                    "wet",
                    "wind",
                    "stress_state",
                    "sunshine_effect",
                    "moderate_rain",
                    "food_state",
                    "precip_30d_sum_used",
                    "drought",
                    "heat_drought",
                    "cold_wet",
                    "season_weight",
                    "interaction_mix",
                    "wet_balance",
                    "storm",
                    "habitat_state",
                    "event_trigger_heatwave",
                    "event_trigger_heavy_rain",
                    "event_trigger_storm",
                    "event_trigger_cold_snap",
                    "event_active_heatwave",
                    "event_active_heavy_rain",
                    "event_active_storm",
                    "event_active_cold_snap",
                    "shock_reproduction_mult",
                    "shock_juvenile_survival_mult",
                    "shock_adult_survival_mult",
                    "reproduction_factor",
                    "juvenile_survival_factor",
                    "adult_survival_factor",
                ]
            )
        return pd.DataFrame(self._rows)

    def write_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(p, index=False)
        return p
