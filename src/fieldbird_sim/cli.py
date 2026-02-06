from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import os
import argparse

import yaml


@dataclass(frozen=True)
class RunSection:
    use_synthetic: bool = True
    n_fields: int = 20
    n_years: int = 20
    year_to_show: int = 1
    day_to_show: int = 160
    do_maps: bool = True
    do_heatmaps: bool = True


@dataclass(frozen=True)
class SeedsSection:
    fields: int = 2025
    rotation: int = 2025
    management: int = 2025
    hedges: int = 2025
    sim: int = 2025


@dataclass(frozen=True)
class RunConfig:
    run: RunSection = RunSection()
    seeds: SeedsSection = SeedsSection()
    raw: dict[str, Any] | None = None  # optional: keep full YAML for later


def load_run_config(path: Path) -> RunConfig:
    if not path.exists():
        # allow running without config file for now
        return RunConfig()

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    run_d = (data.get("run") or {})
    seeds_d = (data.get("seeds") or {})

    cfg = RunConfig(
        run=RunSection(
            use_synthetic=bool(run_d.get("use_synthetic", True)),
            n_fields=int(run_d.get("n_fields", 20)),
            n_years=int(run_d.get("n_years", 20)),
            year_to_show=int(run_d.get("year_to_show", 1)),
            day_to_show=int(run_d.get("day_to_show", 160)),
            do_maps=bool(run_d.get("do_maps", True)),
            do_heatmaps=bool(run_d.get("do_heatmaps", True)),
        ),
        seeds=SeedsSection(
            fields=int(seeds_d.get("fields", 2025)),
            rotation=int(seeds_d.get("rotation", 2025)),
            management=int(seeds_d.get("management", 2025)),
            hedges=int(seeds_d.get("hedges", 2025)),
            sim=int(seeds_d.get("sim", 2025)),
        ),
        raw=data,
    )

    # optional: env overrides (batch usage)
    cfg = apply_env_overrides(cfg)
    return cfg


def apply_env_overrides(cfg: RunConfig) -> RunConfig:
    # keep it minimal; expand later
    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name, "").strip()
        if v == "":
            return default
        try:
            return int(v)
        except Exception:
            return default

    def _env_bool(name: str, default: bool) -> bool:
        v = os.environ.get(name, "").strip().lower()
        if v == "":
            return default
        return v in {"1", "true", "yes", "y", "on"}

    r = cfg.run
    s = cfg.seeds
    r2 = replace(
        r,
        n_years=_env_int("N_YEARS", r.n_years),
        n_fields=_env_int("N_FIELDS", r.n_fields),
        use_synthetic=_env_bool("USE_SYNTHETIC", r.use_synthetic),
    )
    s2 = replace(
        s,
        sim=_env_int("SIM_SEED", s.sim),
    )
    return replace(cfg, run=r2, seeds=s2)


def apply_cli_overrides(
    cfg: RunConfig,
    *,
    use_synthetic: bool | None,
    years: int | None,
    fields: int | None,
    seed: int | None,
    year_to_show: int | None,
    day_to_show: int | None,
) -> RunConfig:
    r = cfg.run
    s = cfg.seeds

    r2 = replace(
        r,
        use_synthetic=r.use_synthetic if use_synthetic is None else bool(use_synthetic),
        n_years=r.n_years if years is None else int(years),
        n_fields=r.n_fields if fields is None else int(fields),
        year_to_show=r.year_to_show if year_to_show is None else int(year_to_show),
        day_to_show=r.day_to_show if day_to_show is None else int(day_to_show),
    )
    s2 = replace(
        s,
        sim=s.sim if seed is None else int(seed),
    )
    return replace(cfg, run=r2, seeds=s2)


def _apply_env_from_config(cfg: RunConfig) -> None:
    os.environ["N_FIELDS"] = str(cfg.run.n_fields)
    os.environ["N_YEARS"] = str(cfg.run.n_years)
    os.environ["YEAR_TO_SHOW"] = str(cfg.run.year_to_show)
    os.environ["DAY_TO_SHOW"] = str(cfg.run.day_to_show)

    os.environ["SEED_FIELDS"] = str(cfg.seeds.fields)
    os.environ["SEED_ROTATION"] = str(cfg.seeds.rotation)
    os.environ["SEED_MANAGEMENT"] = str(cfg.seeds.management)
    os.environ["SEED_HEDGES"] = str(cfg.seeds.hedges)
    os.environ["SIM_SEED"] = str(cfg.seeds.sim)

    # optional toggles (nur wenn du sie per ENV nutzen willst)
    os.environ["DO_MAPS"] = "1" if cfg.run.do_maps else "0"
    os.environ["DO_HEATMAPS"] = "1" if cfg.run.do_heatmaps else "0"

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fieldbird-sim")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run simulation")
    r.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    r.add_argument("--use-synthetic", dest="use_synthetic", action="store_true")
    r.add_argument("--use-csv-inputs", dest="use_synthetic", action="store_false")
    r.add_argument("--year-to-show", type=int, default=None)
    r.add_argument("--day-to-show", type=int, default=None)

    r.set_defaults(use_synthetic=True)

    # schnelle Overrides (optional)
    r.add_argument("--fields", type=int, default=None)
    r.add_argument("--years", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)

    a = p.parse_args(argv)

    if a.cmd == "run":
        # 1) Basis-Konfig laden (optional aus YAML)
        cfg = RunConfig()
        if a.config is not None:
            cfg = load_run_config(a.config)

        # 2) CLI Overrides sauber auf cfg anwenden (CLI gewinnt)
        cfg = apply_cli_overrides(
            cfg,
            use_synthetic=a.use_synthetic,
            years=a.years,
            fields=a.fields,
            seed=a.seed,
            year_to_show=a.year_to_show,
            day_to_show=a.day_to_show,
        )

        # 3) cfg -> ENV (damit globals.py die Werte sieht)
        _apply_env_from_config(cfg)

        # 4) Simulation starten: Wahrheit ist cfg.run.use_synthetic
        from fieldbird_sim.legacy.kpis import run
        run(use_synthetic=bool(cfg.run.use_synthetic))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
