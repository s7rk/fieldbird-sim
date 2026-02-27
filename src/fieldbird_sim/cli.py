from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse

import yaml


# ----------------------------
#   YAML -> RunConfig schema
# ----------------------------

@dataclass(frozen=True)
class RunSection:
    use_synthetic: bool = True
    n_fields: int = 20
    n_years: int = 20
    year_to_show: int = 1
    day_to_show: int = 160

    # optional YAML hooks
    outdir_prefix: str = "outputs"
    run_name: str | None = None
    species: tuple[str, ...] | None = None  # if None -> defaults in globals.py
    n_init_first: int = 100


@dataclass(frozen=True)
class PlotSection:
    do_maps: bool = True
    do_heatmaps: bool = True
    draw_flower_rings: bool = True
    draw_hedge_buffers: bool = True
    hedge_buffer_m: float = 5.0


@dataclass(frozen=True)
class SeedsSection:
    fields: int = 2025
    rotation: int = 2025
    management: int = 2025
    hedges: int = 2025
    sim: int = 2025


@dataclass(frozen=True)
class WinterSection:
    use_year_feedback: bool = True
    winter_survival_adult: float = 0.95
    winter_survival_juvenile: float = 0.90
    juvenile_maturation_rate: float = 0.90
    juvenile_equiv_in_capacity: float = 0.5
    window_stress_weight: float = 0.5
    winter_move_radius: float = 12.0
    winter_beta: float = 2.5
    winter_distance_cost: float = 0.08
    winter_eval_days: tuple[int, int] = (330, 20)


@dataclass(frozen=True)
class CapacitySection:
    use_capacity_soft: bool = True
    k_per_ha_adults: float = 40.0
    capacity_gamma: float = 0.4


@dataclass(frozen=True)
class RefugeSection:
    eligible_crops: tuple[str, ...] = ("Winterweizen", "Sommergerste")
    refuge_assign_mult: float = 15.0
    pulse_protect_scale: float = 4.0
    refuge_share_cap: float = 0.6


@dataclass(frozen=True)
class SyntheticCoverSection:
    add_cover: bool = True
    cover_margin_m: float = 6.0
    cover_min_patch_m2: float = 80.0
    cover_fill_residual: bool = True
    square_pad_m: float = 8.0


@dataclass(frozen=True)
class SyntheticSection:
    target_row_width_m: float | None = None
    cover: SyntheticCoverSection = SyntheticCoverSection()


@dataclass(frozen=True)
class ManagementSection:
    use_calendar: bool = True
    calendar_path: str = "configs/crop_calendar.csv"
    desync_region_sd: float = 4.0
    desync_field_sd: float = 3.0
    region_bins: int = 6


# --- NEW: CSV input wiring (only if run.use_synthetic: false) ---
@dataclass(frozen=True)
class InputsSection:
    fields_csv: str = "inputs/fields.csv"
    rotation_csv: str = "inputs/rotation.csv"
    management_csv: str = "inputs/management.csv"
    hedges_csv: str = "inputs/hedges.csv"


# --- NEW: hedges knobs (works for synthetic and csv) ---
@dataclass(frozen=True)
class HedgesSection:
    p_edge: float = 0.7
    from_csv: bool = True  # only relevant when inputs.hedges_csv exists


@dataclass(frozen=True)
class ScenarioASection:
    enabled: bool = True
    shifts: tuple[int, ...] = (7, 14, 21)


@dataclass(frozen=True)
class ScenarioBSection:
    enabled: bool = True
    shares: tuple[float, ...] = (0.10, 0.20, 0.30)
    fractions: tuple[float, ...] = (0.25, 0.50, 1.00)
    select: str = "no_flower_first"
    seed: int = 2025


@dataclass(frozen=True)
class ScenarioCSection:
    enabled: bool = True
    fallow_area_shares: tuple[float, ...] = (0.01, 0.05, 0.10)
    apply_years: str | list[int] = "all"
    selection: str = "greedy"
    seed: int = 2026


@dataclass(frozen=True)
class ScenarioDSection:
    enabled: bool = True
    add_skylark_plots: int = 2
    skylark_plot_area: float = 0.02
    target_fraction: float = 0.20
    min_eligible_years: int = 1
    only_without_flower: bool = True


@dataclass(frozen=True)
class ScenariosSection:
    A: ScenarioASection = ScenarioASection()
    B: ScenarioBSection = ScenarioBSection()
    C: ScenarioCSection = ScenarioCSection()
    D: ScenarioDSection = ScenarioDSection()


@dataclass(frozen=True)
class DebugSection:
    debug_year_xfer: int = 0


@dataclass(frozen=True)
class RunConfig:
    run: RunSection = RunSection()
    plots: PlotSection = PlotSection()
    seeds: SeedsSection = SeedsSection()
    winter: WinterSection = WinterSection()
    capacity: CapacitySection = CapacitySection()
    refuge: RefugeSection = RefugeSection()
    synthetic: SyntheticSection = SyntheticSection()
    management: ManagementSection = ManagementSection()
    inputs: InputsSection = InputsSection()      # NEW
    hedges: HedgesSection = HedgesSection()      # NEW
    scenarios: ScenariosSection = ScenariosSection()
    debug: DebugSection = DebugSection()
    raw: dict[str, Any] | None = None


# ----------------------------
#      YAML loader helpers
# ----------------------------

def _dget(d: dict, key: str, default):
    v = d.get(key, default)
    return default if v is None else v


def load_run_config(path: Path) -> RunConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    run_d = data.get("run") or {}
    plots_d = data.get("plots") or {}
    seeds_d = data.get("seeds") or {}
    winter_d = data.get("winter") or {}
    capacity_d = data.get("capacity") or {}
    refuge_d = data.get("refuge") or {}
    synthetic_d = data.get("synthetic") or {}
    syn_cover_d = (synthetic_d.get("cover") or {}) if isinstance(synthetic_d, dict) else {}
    mgmt_d = data.get("management") or {}

    inputs_d = data.get("inputs") or {}   # NEW
    hedges_d = data.get("hedges") or {}   # NEW

    scenarios_d = data.get("scenarios") or {}
    A_d = scenarios_d.get("A") or {}
    B_d = scenarios_d.get("B") or {}
    C_d = scenarios_d.get("C") or {}
    D_d = scenarios_d.get("D") or {}
    debug_d = data.get("debug") or {}

    # optional: species list
    sp = _dget(run_d, "species", None)
    if sp is None:
        species_tuple = None
    else:
        species_tuple = tuple(list(sp))

    cfg = RunConfig(
        run=RunSection(
            use_synthetic=bool(_dget(run_d, "use_synthetic", True)),
            n_fields=int(_dget(run_d, "n_fields", 20)),
            n_years=int(_dget(run_d, "n_years", 20)),
            year_to_show=int(_dget(run_d, "year_to_show", 1)),
            day_to_show=int(_dget(run_d, "day_to_show", 160)),
            outdir_prefix=str(_dget(run_d, "outdir_prefix", "outputs")),
            run_name=_dget(run_d, "run_name", None),
            species=species_tuple,
            n_init_first=int(_dget(run_d, "n_init_first", 100)),
        ),
        plots=PlotSection(
            do_maps=bool(_dget(plots_d, "do_maps", True)),
            do_heatmaps=bool(_dget(plots_d, "do_heatmaps", True)),
            draw_flower_rings=bool(_dget(plots_d, "draw_flower_rings", True)),
            draw_hedge_buffers=bool(_dget(plots_d, "draw_hedge_buffers", True)),
            hedge_buffer_m=float(_dget(plots_d, "hedge_buffer_m", 5.0)),
        ),
        seeds=SeedsSection(
            fields=int(_dget(seeds_d, "fields", 2025)),
            rotation=int(_dget(seeds_d, "rotation", 2025)),
            management=int(_dget(seeds_d, "management", 2025)),
            hedges=int(_dget(seeds_d, "hedges", 2025)),
            sim=int(_dget(seeds_d, "sim", 2025)),
        ),
        winter=WinterSection(
            use_year_feedback=bool(_dget(winter_d, "use_year_feedback", True)),
            winter_survival_adult=float(_dget(winter_d, "winter_survival_adult", 0.95)),
            winter_survival_juvenile=float(_dget(winter_d, "winter_survival_juvenile", 0.90)),
            juvenile_maturation_rate=float(_dget(winter_d, "juvenile_maturation_rate", 0.90)),
            juvenile_equiv_in_capacity=float(_dget(winter_d, "juvenile_equiv_in_capacity", 0.5)),
            window_stress_weight=float(_dget(winter_d, "window_stress_weight", 0.5)),
            winter_move_radius=float(_dget(winter_d, "winter_move_radius", 12.0)),
            winter_beta=float(_dget(winter_d, "winter_beta", 2.5)),
            winter_distance_cost=float(_dget(winter_d, "winter_distance_cost", 0.08)),
            winter_eval_days=tuple(_dget(winter_d, "winter_eval_days", [330, 20])),
        ),
        capacity=CapacitySection(
            use_capacity_soft=bool(_dget(capacity_d, "use_capacity_soft", True)),
            k_per_ha_adults=float(_dget(capacity_d, "k_per_ha_adults", 40.0)),
            capacity_gamma=float(_dget(capacity_d, "capacity_gamma", 0.4)),
        ),
        refuge=RefugeSection(
            eligible_crops=tuple(_dget(refuge_d, "eligible_crops", ["Winterweizen", "Sommergerste"])),
            refuge_assign_mult=float(_dget(refuge_d, "refuge_assign_mult", 15.0)),
            pulse_protect_scale=float(_dget(refuge_d, "pulse_protect_scale", 4.0)),
            refuge_share_cap=float(_dget(refuge_d, "refuge_share_cap", 0.6)),
        ),
        synthetic=SyntheticSection(
            target_row_width_m=_dget(synthetic_d, "target_row_width_m", None),
            cover=SyntheticCoverSection(
                add_cover=bool(_dget(syn_cover_d, "add_cover", True)),
                cover_margin_m=float(_dget(syn_cover_d, "cover_margin_m", 6.0)),
                cover_min_patch_m2=float(_dget(syn_cover_d, "cover_min_patch_m2", 80.0)),
                cover_fill_residual=bool(_dget(syn_cover_d, "cover_fill_residual", True)),
                square_pad_m=float(_dget(syn_cover_d, "square_pad_m", 8.0)),
            ),
        ),
        management=ManagementSection(
            use_calendar=bool(_dget(mgmt_d, "use_calendar", True)),
            calendar_path=str(_dget(mgmt_d, "calendar_path", "configs/crop_calendar.csv")),
            desync_region_sd=float(_dget(mgmt_d, "desync_region_sd", 4.0)),
            desync_field_sd=float(_dget(mgmt_d, "desync_field_sd", 3.0)),
            region_bins=int(_dget(mgmt_d, "region_bins", 6)),
        ),

        # NEW blocks
        inputs=InputsSection(
            fields_csv=str(_dget(inputs_d, "fields_csv", "inputs/fields.csv")),
            rotation_csv=str(_dget(inputs_d, "rotation_csv", "inputs/rotation.csv")),
            management_csv=str(_dget(inputs_d, "management_csv", "inputs/management.csv")),
            hedges_csv=str(_dget(inputs_d, "hedges_csv", "inputs/hedges.csv")),
        ),
        hedges=HedgesSection(
            p_edge=float(_dget(hedges_d, "p_edge", 0.7)),
            from_csv=bool(_dget(hedges_d, "from_csv", True)),
        ),

        scenarios=ScenariosSection(
            A=ScenarioASection(
                enabled=bool(_dget(A_d, "enabled", True)),
                shifts=tuple(_dget(A_d, "shifts", [7, 14, 21])),
            ),
            B=ScenarioBSection(
                enabled=bool(_dget(B_d, "enabled", True)),
                shares=tuple(_dget(B_d, "shares", [0.10, 0.20, 0.30])),
                fractions=tuple(_dget(B_d, "fractions", [0.25, 0.50, 1.00])),
                select=str(_dget(B_d, "select", "no_flower_first")),
                seed=int(_dget(B_d, "seed", 2025)),
            ),
            C=ScenarioCSection(
                enabled=bool(_dget(C_d, "enabled", True)),
                fallow_area_shares=tuple(_dget(C_d, "fallow_area_shares", [0.01, 0.05, 0.10])),
                apply_years=_dget(C_d, "apply_years", "all"),
                selection=str(_dget(C_d, "selection", "greedy")),
                seed=int(_dget(C_d, "seed", 2026)),
            ),
            D=ScenarioDSection(
                enabled=bool(_dget(D_d, "enabled", True)),
                add_skylark_plots=int(_dget(D_d, "add_skylark_plots", 2)),
                skylark_plot_area=float(_dget(D_d, "skylark_plot_area", 0.02)),
                target_fraction=float(_dget(D_d, "target_fraction", 0.20)),
                min_eligible_years=int(_dget(D_d, "min_eligible_years", 1)),
                only_without_flower=bool(_dget(D_d, "only_without_flower", True)),
            ),
        ),
        debug=DebugSection(
            debug_year_xfer=int(_dget(debug_d, "debug_year_xfer", 0)),
        ),
        raw=data,
    )

    return cfg


# ----------------------------
#            CLI
# ----------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="fieldbird-sim")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run simulation (YAML-only)")
    r.add_argument("--config", type=Path, required=True, help="Path to YAML config")

    a = p.parse_args(argv)

    if a.cmd == "run":
        cfg = load_run_config(a.config)

        from fieldbird_sim.legacy.kpis import run as run_legacy
        run_legacy(cfg=cfg)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())