from __future__ import annotations

# ---- stdlib ----
import math
import os
import pathlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# ---- third party ----
import numpy as np
import pandas as pd

# ---- project: globals zuerst (liefert OPTS, Konstanten, helpers etc.) ----
from .globals import *  # noqa: F401,F403
from .globals import _mk_outdir

# Landscape/management generation (bei dir: globals.py)
from .globals import (
    make_synthetic_fields,
    make_rotation,
    make_management,
    make_random_hedges,
    run_years_with_feedback,   # ✅ HIERHER gehört es
)

# Geometry / layout / hedges
from .layout import (
    build_hedge_geoms_on_shared_edges,
    rect_or_centroid_layout,
    heatmap,
    plot_map,
)

# Species IO
from .species import load_species_from_csv

# Simulation core
from .simulate import BirdLandscapeSim
from .simulate import to_long_timeseries


#  (module 8/8) KPIs & Sensitivität (Sprint A)
# =========================================================

def _breed_mask(days: int, win: tuple[int, int]) -> np.ndarray:
    """Boolean-Maske für das Brutfenster [a,b] (1-indexed Tage)."""
    a, b = int(win[0]), int(win[1])
    a = max(1, min(a, days))
    b = max(1, min(b, days))
    if a > b:
        a, b = b, a
    m = np.zeros(days, dtype=bool)
    m[a - 1:b] = True
    return m


def field_occupancy(A_f: np.ndarray, breeding_win: tuple[int, int]) -> np.ndarray:
    """Anteil Brut-Tage mit >=1 Weibchen je Feld (Rückgabe: Vektor je Feld)."""
    days, _ = A_f.shape
    m = _breed_mask(days, breeding_win)
    sub = A_f[m, :] > 0
    with np.errstate(invalid="ignore"):
        return sub.mean(axis=0)


def colonization_rate(prev: dict, curr: dict, breeding_win: tuple[int, int]) -> float:
    A_prev = prev["A_f"]
    A_curr = curr["A_f"]
    days, _ = A_prev.shape
    m = _breed_mask(days, breeding_win)
    occ_prev = (A_prev[m, :] > 0).any(axis=0)
    occ_curr = (A_curr[m, :] > 0).any(axis=0)
    empty_prev = ~occ_prev
    denom = int(empty_prev.sum())
    if denom <= 0:
        return 0.0
    return float(((empty_prev) & (occ_curr)).sum()) / float(denom)


def mayfield_daily_from_log(repro_df: pd.DataFrame, breeding_win: tuple[int, int]) -> float:
    """Grober Mayfield-Proxy aus dem Tageslog: mittlere Tages-Überlebensrate im Brutfenster."""
    if repro_df is None or len(repro_df) == 0:
        return float("nan")
    a, b = int(breeding_win[0]), int(breeding_win[1])
    df = repro_df[(repro_df["day"] >= a) & (repro_df["day"] <= b)]
    if len(df) == 0:
        return float("nan")
    return float(np.nanmean(df["mean_nest_surv"]))


def compute_kpis_one_year(res: dict, cfg) -> dict:
    """Kompakte Kennzahlen pro Jahr (landschaftsweit)."""
    A_m = res["A_m"]
    A_f = res["A_f"]
    days, _ = A_f.shape

    # Adults bei Ankunft
    arr_idx = max(1, int(cfg.arrival_day)) - 1
    adults_arrival = int((A_m[arr_idx, :] + A_f[arr_idx, :]).sum())

    # Geburten (flügge Jungvögel)
    daily_fledg = res.get("daily_fledglings")
    if daily_fledg is not None:
        daily_fledg_tot = daily_fledg.sum(axis=1)
        births_total = int(daily_fledg_tot.sum())
    else:
        daily_fledg_tot = np.zeros(days, dtype=int)
        births_total = 0
    fledglings_total = births_total

    # Erwachsene: tägliche Gesamtzahlen -> Verluste (ab Ankunft)
    adults_tot = (A_m + A_f).sum(axis=1)
    adult_deaths = int(np.maximum(adults_tot[arr_idx:-1] - adults_tot[arr_idx + 1:], 0).sum())

    # Juvenile: gebundene + freie
    J_tied = res.get("J_tied_total", res.get("J_tied"))
    if J_tied is None:
        J_tied = np.zeros_like(A_f)
    J_free = res.get("J_free", np.zeros_like(A_f))
    juven_tot = (J_tied + J_free).sum(axis=1)
    juvenile_deaths = int(
        np.maximum(
            juven_tot[arr_idx:-1] + daily_fledg_tot[arr_idx:-1] - juven_tot[arr_idx + 1:],
            0,
        ).sum()
    )

    # Female-breeding-Equivalents
    A_f_b = res.get("A_f_breeding")
    if A_f_b is None:
        A_f_b = A_f.copy()
    m = _breed_mask(days, cfg.breeding_window)
    female_breed_days = int(np.asarray(A_f_b[m, :], dtype=int).sum())
    female_equiv = female_breed_days / max(int(cfg.nest_cycle_days), 1)
    fledglings_per_female_equiv = (fledglings_total / female_equiv) if female_equiv > 0 else float("nan")

    # Feld-Occupancy & Mayfield-Proxy
    occ = field_occupancy(A_f, cfg.breeding_window)
    occ_mean = float(np.nanmean(occ)) if occ.size else float("nan")
    mayfield_daily = float("nan")
    if "debug_repro" in res and isinstance(res["debug_repro"], pd.DataFrame):
        mayfield_daily = mayfield_daily_from_log(res["debug_repro"], cfg.breeding_window)

    return dict(
        adults_arrival=adults_arrival,
        fledglings_total=fledglings_total,
        female_breeding_equiv=female_equiv,
        fledglings_per_female_equiv=fledglings_per_female_equiv,
        occupancy_mean=occ_mean,
        mayfield_daily=mayfield_daily,
        adult_deaths=adult_deaths,
        juvenile_deaths=juvenile_deaths,
        births_total=births_total,
    )


def compute_kpis_from_results(results_by_year: dict, cfg) -> pd.DataFrame:
    """KPI-Tabelle je Jahr (inkl. λ zum Vorjahr und Kolonisationsrate)."""
    rows = []
    years = sorted(results_by_year.keys())

    arr_idx = max(1, int(cfg.arrival_day)) - 1
    adults = [
        int((results_by_year[y]["A_m"][arr_idx, :] + results_by_year[y]["A_f"][arr_idx, :]).sum())
        for y in years
    ]
    lambdas = [adults[i + 1] / max(adults[i], 1e-9) for i in range(len(adults) - 1)] if len(years) > 1 else []
    lam_map = {years[i + 1]: float(lambdas[i]) for i in range(len(lambdas))}

    for i, y in enumerate(years):
        k = compute_kpis_one_year(results_by_year[y], cfg)
        k["year"] = int(y)
        k["lambda_prev"] = float(lam_map.get(y, float("nan")))
        if i > 0:
            k["colonization_rate"] = float(
                colonization_rate(results_by_year[years[i - 1]], results_by_year[y], cfg.breeding_window)
            )
        else:
            k["colonization_rate"] = float("nan")
        rows.append(k)

    df = pd.DataFrame(rows)
    Path("analysis").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path("analysis") / "kpi_yearly.csv", index=False)
    return df


# ---------------- LHS-Sampler & PRCC ----------------

def lhs_sample(n: int, vary: dict, seed: int = 123) -> pd.DataFrame:
    """Latin Hypercube über gleichmäßige Schichten in jedem Parameterintervall [lo, hi]."""
    rng = np.random.default_rng(seed)
    names = list(vary.keys())
    k = len(names)
    U = (rng.random((n, k)) + np.arange(n)[:, None]) / n
    X = np.zeros_like(U)
    for j, name in enumerate(names):
        lo, hi = map(float, vary[name])
        perm = rng.permutation(n)
        X[:, j] = lo + (hi - lo) * U[perm, j]
    return pd.DataFrame(X, columns=names)


def _rank_transform(a: np.ndarray) -> np.ndarray:
    """Rang-Transform (average ranks)."""
    return pd.DataFrame(a).rank(method="average").to_numpy()


def compute_prcc(params_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """PRCC (Partial Rank Correlation Coefficients) mit Normal-Approx p-Werten (ohne SciPy)."""
    X = params_df.to_numpy(float)
    y = np.asarray(y, float).reshape(-1)

    XR = _rank_transform(X)
    yR = _rank_transform(y.reshape(-1, 1)).ravel()
    n, k = XR.shape

    out = []
    for i in range(k):
        Xi = XR[:, i]
        Xo = np.delete(XR, i, axis=1)

        Bx, *_ = np.linalg.lstsq(np.c_[np.ones(n), Xo], Xi, rcond=None)
        rx = Xi - np.c_[np.ones(n), Xo] @ Bx

        By, *_ = np.linalg.lstsq(np.c_[np.ones(n), Xo], yR, rcond=None)
        ry = yR - np.c_[np.ones(n), Xo] @ By

        r = float(np.corrcoef(rx, ry)[0, 1])
        df = max(n - k - 1, 1)
        t = r * np.sqrt(df / max(1e-12, (1 - r**2)))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        out.append((params_df.columns[i], r, p))

    return pd.DataFrame(out, columns=["param", "prcc", "p_approx"]).sort_values(
        "prcc", key=lambda s: s.abs(), ascending=False
    )


# -------------- MC mit LHS & KPI-Logging --------------

def run_mc_with_lhs(fields_gdf, rotation, management, cfg, years,
                    n_reps: int = 200, seed: int = 123, vary: dict | None = None) -> pd.DataFrame:
    """Monte Carlo mit LHS + KPI-Outputs (λ, Produktivität, Occupancy, Kolonisation)."""
    if vary is None:
        vary = {
            "fec_scale": (0.25, 0.50),
            "fledge_rate": (0.75, 0.90),
            "alpha_density": (0.001, 0.004),
            "survival_breeding": (0.990, 0.999),
            "survival_nonbreeding": (0.998, 1.000),
            "survival_juvenile": (0.980, 0.998),
            "CAPACITY_GAMMA": (0.20, 0.60),
            "WINDOW_STRESS_WEIGHT": (0.40, 0.70),
            "REFUGE_ASSIGN_MULT": (10.0, 25.0),
            "PULSE_PROTECT_SCALE": (3.0, 6.0),
            "REFUGE_SHARE_CAP": (0.50, 0.80),
        }

    samples = lhs_sample(n_reps, vary, seed=seed)

    global CAPACITY_GAMMA, WINDOW_STRESS_WEIGHT, REFUGE_ASSIGN_MULT, PULSE_PROTECT_SCALE, REFUGE_SHARE_CAP
    _globals0 = dict(
        CAPACITY_GAMMA=CAPACITY_GAMMA,
        WINDOW_STRESS_WEIGHT=WINDOW_STRESS_WEIGHT,
        REFUGE_ASSIGN_MULT=REFUGE_ASSIGN_MULT,
        PULSE_PROTECT_SCALE=PULSE_PROTECT_SCALE,
        REFUGE_SHARE_CAP=REFUGE_SHARE_CAP,
    )

    rows = []
    for rep in range(n_reps):
        cfg2 = replace(cfg)
        sampled = samples.iloc[rep].to_dict()

        for k, v in sampled.items():
            if hasattr(cfg2, k):
                setattr(cfg2, k, float(v))
            elif k in _globals0:
                globals()[k] = float(v)

        resY = run_years_with_feedback(fields_gdf, rotation, management, cfg2, years=years, N_init_first=50)

        arr_idx = max(1, int(cfg2.arrival_day)) - 1
        adults = [int((resY[y]["A_m"][arr_idx, :] + resY[y]["A_f"][arr_idx, :]).sum()) for y in years]
        lambdas = [adults[i + 1] / max(adults[i], 1e-9) for i in range(len(adults) - 1)] if len(years) > 1 else []
        lam_mean = float(np.mean(lambdas)) if lambdas else float("nan")

        kpiY = [compute_kpis_one_year(resY[y], cfg2) for y in years]
        dfY = pd.DataFrame(kpiY)

        colls = []
        for i in range(1, len(years)):
            colls.append(colonization_rate(resY[years[i - 1]], resY[years[i]], cfg2.breeding_window))
        colonization_mean = float(np.mean(colls)) if colls else float("nan")

        row = {
            "rep": rep,
            "lambda_mean": lam_mean,
            "adults_start": adults[0],
            "adults_end": adults[-1],
            "fledglings_per_female_equiv_mean": float(dfY["fledglings_per_female_equiv"].mean()),
            "occupancy_mean": float(dfY["occupancy_mean"].mean()),
            "colonization_rate_mean": colonization_mean,
            "mayfield_daily_mean": float(dfY["mayfield_daily"].mean()),
        }
        row.update({k: float(sampled[k]) for k in sampled})
        rows.append(row)

        for k, v in _globals0.items():
            globals()[k] = v

    return pd.DataFrame(rows)


# =========================================================
#  MAIN RUNNER (ported legacy __main__)
# =========================================================
def _apply_run_config_to_globals(cfg) -> None:
    """
    Transitional adapter:
    maps RunConfig -> existing globals.py knobs so the legacy runner keeps working.
    """
    # run section
    if hasattr(cfg, "run"):
        r = cfg.run
        # OPTS
        if hasattr(r, "n_fields"):
            OPTS.N_FIELDS = int(r.n_fields)
        if hasattr(r, "n_years"):
            OPTS.N_YEARS = int(r.n_years)

        # derived globals (these are computed in globals.py at import time,
        # so we must also update the dependent ones here!)
        global YEARS_TO_SIM, YEAR_TO_SHOW, DAY_TO_SHOW
        YEARS_TO_SIM = list(range(1, int(OPTS.N_YEARS) + 1))

        if hasattr(r, "year_to_show"):
            YEAR_TO_SHOW = max(1, min(int(r.year_to_show), int(OPTS.N_YEARS)))
        if hasattr(r, "day_to_show"):
            # TOT_DAYS lives in globals.py
            DAY_TO_SHOW = max(1, min(int(r.day_to_show), int(TOT_DAYS)))

        # plotting toggles
        global DO_MAPS, DO_HEATMAPS
        if hasattr(r, "do_maps"):
            DO_MAPS = bool(r.do_maps)
        if hasattr(r, "do_heatmaps"):
            DO_HEATMAPS = bool(r.do_heatmaps)

        # synthetic flag (becomes runtime USE_SYNTHETIC)
        globals()["USE_SYNTHETIC"] = bool(getattr(r, "use_synthetic", True))

    # seeds section
    if hasattr(cfg, "seeds"):
        s = cfg.seeds
        if hasattr(s, "fields"):
            OPTS.SEED_FIELDS = int(s.fields)
        if hasattr(s, "rotation"):
            OPTS.SEED_ROTATION = int(s.rotation)
        if hasattr(s, "management"):
            OPTS.SEED_MANAGEMENT = int(s.management)
        if hasattr(s, "hedges"):
            OPTS.SEED_HEDGES = int(s.hedges)
        if hasattr(s, "sim"):
            OPTS.SIM_RNG_SEED = int(s.sim)

def run(*, cfg=None, use_synthetic: bool | None = None) -> None:
    """
    Run the simulation.
    - cfg: RunConfig (preferred)
    - use_synthetic: optional override (kept for backward compatibility)
    """
    # --- apply cfg -> globals (transitional, minimal-diff) ---
    if cfg is not None:
        _apply_run_config_to_globals(cfg)

    # --- CLI override still possible ---
    if use_synthetic is not None:
        globals()["USE_SYNTHETIC"] = bool(use_synthetic)

    USE_SYNTHETIC = bool(globals().get("USE_SYNTHETIC", True))

    outdir = _mk_outdir("outputs")

    # IMPORTANT:
    # - don't reassign YEARS_TO_SIM inside this function (would shadow the global).
    # - use a local variable instead.
    years_to_sim = list(YEARS_TO_SIM)

    if USE_SYNTHETIC:
        fields_gdf = make_synthetic_fields(n_fields=int(OPTS.N_FIELDS), seed=int(OPTS.SEED_FIELDS))

        rotation = make_rotation(fields_gdf, years_to_sim, seed=int(OPTS.SEED_ROTATION))
        management = make_management(
            fields_gdf,
            rotation,
            seed=int(OPTS.SEED_MANAGEMENT),
            use_calendar=True,
            calendar_path=CROP_CALENDAR_CSV,
            desync_region_sd=4.0,
            desync_field_sd=3.0,
            region_bins=6,
        )

        hedge_pairs = make_random_hedges(fields_gdf, p_edge=0.7, seed=int(OPTS.SEED_HEDGES))
        hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)

    else:
        unique_years = sorted(pd.unique(rotation["year"]))
        if len(unique_years) > len(years_to_sim):
            years_to_sim = unique_years[: int(OPTS.N_YEARS)]

        fields_gdf = rect_or_centroid_layout(fields)

        try:
            ckpt_dir = pathlib.Path("snapshots")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fields_gdf.to_file(ckpt_dir / "fields_latest.gpkg", layer="fields", driver="GPKG")
            fields_gdf.to_parquet(ckpt_dir / f"fields_{ts}.parquet")
            print(f"[snapshot] fields saved to {ckpt_dir}/fields_latest.gpkg and fields_{ts}.parquet")
        except Exception as e:
            print("[warn] snapshot save failed:", e)

        hedge_pairs = (
            pd.read_csv(HEDGES_CSV)
            if os.path.exists(HEDGES_CSV)
            else pd.DataFrame(columns=["f1", "f2", "hedge_quality"])
        )
        hedges_gdf = None
        if USE_HEDGES_FROM_CSV and {"f1", "f2"}.issubset(hedge_pairs.columns) and len(hedge_pairs) > 0:
            hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)
        elif AUTO_HEDGES_FROM_ADJ:
            pairs = []
            fids = fields_gdf["field_id"].astype(int).values
            for i, a in enumerate(fids):
                for b in fids[i + 1:]:
                    pairs.append({"f1": int(a), "f2": int(b), "hedge_quality": 0.6})
            hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, pd.DataFrame(pairs)) if pairs else None

    print(
        f"[Landscape] fields={len(fields_gdf)}, years={len(years_to_sim)}, "
        f"rotation_rows={len(rotation)}, synthetic={USE_SYNTHETIC}"
    )
    print(
        f"[Options] N_FIELDS={OPTS.N_FIELDS} (synthetic only) | N_YEARS={OPTS.N_YEARS} | "
        f"YEAR_TO_SHOW={YEAR_TO_SHOW} | DAY_TO_SHOW={DAY_TO_SHOW}"
    )

    # ---------- Run simulation(s) per species ----------
    for sp_name in SPECIES_LIST:
        cfg = load_species_from_csv(sp_name, base_path=".")

        # sanity check optional (nicht blockieren, wenn die Funktion nicht existiert)
        if "sanity_check_species" in globals() and callable(globals()["sanity_check_species"]):
            globals()["sanity_check_species"](cfg)

        print(
            "SPECIES:",
            cfg.name,
            "| move_radius(100m)=",
            cfg.move_radius,
            "| breeding_window=",
            cfg.breeding_window,
        )
        print(
            "surv_breeding=",
            cfg.survival_breeding,
            "surv_nonbreeding=",
            cfg.survival_nonbreeding,
            "surv_juvenile=",
            cfg.survival_juvenile,
            "fledge_rate=",
            cfg.fledge_rate,
            "base_nest_survival_day=",
            cfg.base_nest_survival_day,
        )

        sim = BirdLandscapeSim(
            fields_gdf[
                [
                    "field_id",
                    "x",
                    "y",
                    "area",
                    "perimeter",
                    "flower_strip_share",
                    "skylark_plots",
                    "skylark_plot_area",
                    "width_m",
                    "height_m",
                ]
            ],
            rotation,
            management,
            cfg,
        )

        sp_outdir = outdir / cfg.name
        sp_outdir.mkdir(parents=True, exist_ok=True)

        if USE_YEAR_FEEDBACK:
            res_years = run_years_with_feedback(
                fields_gdf, rotation, management, cfg, years=years_to_sim, N_init_first=100
            )
        else:
            res_years = {}
            for y in years_to_sim:
                dbg_days = [110, 130, 160, 200] if int(y) == int(YEAR_TO_SHOW) else None
                res_years[y] = sim.simulate(days=365, year=int(y), N_init=100, debug_days=dbg_days)
                ts = to_long_timeseries(res_years[y], cfg.name, y, scenario="baseline")
                ydir = sp_outdir / f"year_{y}"
                ydir.mkdir(parents=True, exist_ok=True)
                ts.to_csv(ydir / "timeseries_long.csv", index=False)

        save_run_bundle(
            outdir=sp_outdir,
            species_name=cfg.name,
            fields_gdf=fields_gdf,
            rotation=rotation,
            management=management,
            hedges_gdf=hedges_gdf,
            results_by_year=res_years,
            day_to_show=DAY_TO_SHOW,
        )

        # Ensure timeseries_long.csv exists for all years
        try:
            for y, res in res_years.items():
                ydir = sp_outdir / f"year_{int(y)}"
                ydir.mkdir(parents=True, exist_ok=True)
                f_long = ydir / "timeseries_long.csv"
                if not f_long.exists():
                    ts = to_long_timeseries(res, cfg.name, int(y), scenario="Baseline")
                    ts.to_csv(f_long, index=False)
        except Exception as e:
            print("[warn] could not ensure timeseries_long.csv for all years:", e)

        # Quick population summary
        arr = max(1, int(cfg.arrival_day)) - 1
        N_arrivals = [(res_years[y]["A_m"][arr, :] + res_years[y]["A_f"][arr, :]).sum() for y in years_to_sim]
        lam = [N_arrivals[i + 1] / max(N_arrivals[i], 1e-9) for i in range(len(N_arrivals) - 1)]
        print("Adults at arrival by year:", [int(x) for x in N_arrivals])
        print("Lambda (year-to-year):", np.round(lam, 3), " mean:", round(float(np.mean(lam)), 3))

        # Heatmaps
        if DO_HEATMAPS:
            fids = fields_gdf["field_id"].values
            vmax = max(res["N_total"].max() for res in res_years.values())
            for y in years_to_sim:
                heatmap(
                    res_years[y]["N_total"],
                    fids,
                    f"{cfg.name} – Heatmap Gesamt – Jahr {y}",
                    vmax=vmax,
                    savepath=sp_outdir / f"plots/heatmap_year{y}.png",
                )

        # Maps
        if DO_MAPS:
            y = int(YEAR_TO_SHOW)
            plot_map(
                fields_gdf,
                rotation,
                year=y,
                hedges_gdf=hedges_gdf,
                title=f"{cfg.name} – Karte Jahr {y}, Tag {DAY_TO_SHOW}: Kultur & Vögel",
                draw_flower_rings=DRAW_FLOWER_RINGS,
                draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                hedge_buffer_m=HEDGE_BUFFER_M,
                day_vals=res_years[y]["N_total"][DAY_TO_SHOW],
                savepath=sp_outdir / f"plots/map_day{DAY_TO_SHOW}_year{y}.png",
            )
            plot_map(
                fields_gdf,
                rotation,
                year=y,
                hedges_gdf=hedges_gdf,
                title=f"{cfg.name} – Karte Jahr {y}: Kultur & Usage (Jahressumme)",
                draw_flower_rings=DRAW_FLOWER_RINGS,
                draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                hedge_buffer_m=HEDGE_BUFFER_M,
                usage_vals=res_years[y]["usage"],
                savepath=sp_outdir / f"plots/map_usage_year{y}.png",
            )

        # === Sprint A: KPIs auf Baseline ===
        kpi_df = compute_kpis_from_results(res_years, cfg)
        print("\n[KPI] Jahres-Kennzahlen (Baseline):\n", kpi_df)

        # Optional: MC/LHS/PRCC toggle
        DO_SPRINT_A_MC = False
        if DO_SPRINT_A_MC:
            mc_df = run_mc_with_lhs(fields_gdf, rotation, management, cfg, years_to_sim, n_reps=200, seed=123, vary=None)
            mc_path = sp_outdir / "mc_lhs_kpis.csv"
            mc_df.to_csv(mc_path, index=False)
            print(f"[OK] MC+LHS geschrieben: {mc_path}")

            param_pool = [
                "fec_scale",
                "fledge_rate",
                "alpha_density",
                "survival_breeding",
                "survival_nonbreeding",
                "survival_juvenile",
                "CAPACITY_GAMMA",
                "WINDOW_STRESS_WEIGHT",
                "REFUGE_ASSIGN_MULT",
                "PULSE_PROTECT_SCALE",
                "REFUGE_SHARE_CAP",
            ]
            param_cols = [c for c in mc_df.columns if c in param_pool]
            prcc_df = compute_prcc(mc_df[param_cols], mc_df["lambda_mean"].to_numpy())
            print("\n[PRCC] Treiber von lambda_mean:\n", prcc_df)

    print(f"\nAll files written under: {outdir}\n")
