from __future__ import annotations

# ---- stdlib ----
import math
from dataclasses import replace
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
    run_years_with_feedback,
    save_run_bundle,
)

# Geometry / layout / hedges
from .layout import (
    rect_or_centroid_layout,
    heatmap,
    plot_map,
)

# Species IO
from .species import load_species_from_csv

# Simulation core
from .simulate import BirdLandscapeSim, to_long_timeseries
from .plot_controls import get_plot_controls
from .hedge_setup import build_hedges_from_cfg, attach_hedge_metrics_to_fields
from .scenario_plan import build_scenarios_to_run, apply_scenario_to_inputs
from .weather_builder import build_weather_module_from_cfg
from .debug_utils import sanity_check_species
from fieldbird_sim.resources import resolve_calendar_path, resolve_species_dir


# =========================================================
#  Helpers
# =========================================================


def _resolve_species_list(species_raw) -> list[str]:
    if species_raw is None:
        return list(SPECIES_LIST)

    if isinstance(species_raw, str):
        s = species_raw.strip()
        return [s] if s else list(SPECIES_LIST)

    try:
        vals = [str(x).strip() for x in species_raw]
    except TypeError:
        s = str(species_raw).strip()
        return [s] if s else list(SPECIES_LIST)

    vals = [v for v in vals if v]
    return vals if vals else list(SPECIES_LIST)


# =========================================================
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


def compute_kpis_from_results(
    results_by_year: dict,
    cfg,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
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
    if output_path is None:
        output_path = Path("analysis") / "kpi_yearly.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
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
#  YAML-first adapter: RunConfig -> legacy globals
# =========================================================

def _apply_run_config_to_globals(cfg) -> None:
    """maps RunConfig -> globals.py knobs (legacy runner keeps working)."""
    # run basics
    OPTS.N_FIELDS = int(cfg.run.n_fields)
    OPTS.N_YEARS = int(cfg.run.n_years)
    OPTS.YEAR_TO_SHOW = int(cfg.run.year_to_show)
    OPTS.DAY_TO_SHOW = int(cfg.run.day_to_show)

    global YEARS_TO_SIM, YEAR_TO_SHOW, DAY_TO_SHOW
    YEARS_TO_SIM = list(range(1, int(OPTS.N_YEARS) + 1))
    YEAR_TO_SHOW = max(1, min(int(OPTS.YEAR_TO_SHOW), int(OPTS.N_YEARS)))
    DAY_TO_SHOW = max(1, min(int(OPTS.DAY_TO_SHOW), int(TOT_DAYS)))

    # seeds
    OPTS.SEED_FIELDS = int(cfg.seeds.fields)
    OPTS.SEED_ROTATION = int(cfg.seeds.rotation)
    OPTS.SEED_MANAGEMENT = int(cfg.seeds.management)
    OPTS.SEED_HEDGES = int(cfg.seeds.hedges)
    OPTS.SIM_RNG_SEED = int(cfg.seeds.sim)

    # plotting
    global DO_MAPS, DO_HEATMAPS, DRAW_FLOWER_RINGS, DRAW_HEDGE_BUFFERS, HEDGE_BUFFER_M
    DO_MAPS = bool(cfg.plots.do_maps)
    DO_HEATMAPS = bool(cfg.plots.do_heatmaps)
    DRAW_FLOWER_RINGS = bool(cfg.plots.draw_flower_rings)
    DRAW_HEDGE_BUFFERS = bool(cfg.plots.draw_hedge_buffers)
    HEDGE_BUFFER_M = float(cfg.plots.hedge_buffer_m)

    # winter/year feedback
    global USE_YEAR_FEEDBACK, WINTER_SURVIVAL_ADULT, WINTER_SURVIVAL_JUV, JUV_MATURATION_RATE
    global JUV_EQUIV_IN_CAPACITY, WINDOW_STRESS_WEIGHT, WINTER_MOVE_RADIUS, WINTER_BETA
    global WINTER_DISTANCE_COST, WINTER_EVAL_DAYS

    USE_YEAR_FEEDBACK = bool(cfg.winter.use_year_feedback)
    WINTER_SURVIVAL_ADULT = float(cfg.winter.winter_survival_adult)
    WINTER_SURVIVAL_JUV = float(cfg.winter.winter_survival_juvenile)
    JUV_MATURATION_RATE = float(cfg.winter.juvenile_maturation_rate)
    JUV_EQUIV_IN_CAPACITY = float(cfg.winter.juvenile_equiv_in_capacity)
    WINDOW_STRESS_WEIGHT = float(cfg.winter.window_stress_weight)
    WINTER_MOVE_RADIUS = float(cfg.winter.winter_move_radius)
    WINTER_BETA = float(cfg.winter.winter_beta)
    WINTER_DISTANCE_COST = float(cfg.winter.winter_distance_cost)
    WINTER_EVAL_DAYS = tuple(cfg.winter.winter_eval_days)

    # capacity
    global USE_CAPACITY_SOFT, K_PER_HA_ADULTS, CAPACITY_GAMMA
    USE_CAPACITY_SOFT = bool(cfg.capacity.use_capacity_soft)
    K_PER_HA_ADULTS = float(cfg.capacity.k_per_ha_adults)
    CAPACITY_GAMMA = float(cfg.capacity.capacity_gamma)

    # refuge/skylark
    global ELIGIBLE_REFUGE_CROPS, REFUGE_ASSIGN_MULT, PULSE_PROTECT_SCALE, REFUGE_SHARE_CAP
    ELIGIBLE_REFUGE_CROPS = set(cfg.refuge.eligible_crops)
    REFUGE_ASSIGN_MULT = float(cfg.refuge.refuge_assign_mult)
    PULSE_PROTECT_SCALE = float(cfg.refuge.pulse_protect_scale)
    REFUGE_SHARE_CAP = float(cfg.refuge.refuge_share_cap)

    # synthetic layout
    global TARGET_ROW_WIDTH_M
    TARGET_ROW_WIDTH_M = cfg.synthetic.target_row_width_m

    # scenarios
    global SCENARIO_A_ENABLED, SCENARIO_A_SHIFTS, SCENARIO_A
    global SCENARIO_B_ENABLED, SCENARIO_B_SHARES, SCENARIO_B_FRACTIONS, SCENARIO_B_SELECT, SCENARIO_B_SEED
    global SCENARIO_C_ENABLED, SCENARIO_C_FALLOW_AREA_SHARES, SCENARIO_C_APPLY_YEARS, SCENARIO_C_SELECTION, SCENARIO_C_SEED
    global SCENARIO_D_ENABLED, SCENARIO_D

    SCENARIO_A_ENABLED = bool(cfg.scenarios.A.enabled)
    SCENARIO_A_SHIFTS = list(cfg.scenarios.A.shifts)
    SCENARIO_A = {"shifts": SCENARIO_A_SHIFTS}

    SCENARIO_B_ENABLED = bool(cfg.scenarios.B.enabled)
    SCENARIO_B_SHARES = list(cfg.scenarios.B.shares)
    SCENARIO_B_FRACTIONS = list(cfg.scenarios.B.fractions)
    SCENARIO_B_SELECT = str(cfg.scenarios.B.select)
    SCENARIO_B_SEED = int(cfg.scenarios.B.seed)

    SCENARIO_C_ENABLED = bool(cfg.scenarios.C.enabled)
    SCENARIO_C_FALLOW_AREA_SHARES = list(cfg.scenarios.C.fallow_area_shares)
    SCENARIO_C_APPLY_YEARS = cfg.scenarios.C.apply_years
    SCENARIO_C_SELECTION = str(cfg.scenarios.C.selection)
    SCENARIO_C_SEED = int(cfg.scenarios.C.seed)

    SCENARIO_D_ENABLED = bool(cfg.scenarios.D.enabled)
    SCENARIO_D = dict(
        add_skylark_plots=int(cfg.scenarios.D.add_skylark_plots),
        skylark_plot_area=float(cfg.scenarios.D.skylark_plot_area),
        target_fraction=float(cfg.scenarios.D.target_fraction),
        min_eligible_years=int(cfg.scenarios.D.min_eligible_years),
        only_without_flower=bool(cfg.scenarios.D.only_without_flower),
    )

    # debug
    global DEBUG_YEAR_XFER
    DEBUG_YEAR_XFER = int(cfg.debug.debug_year_xfer)

# =========================================================
# Scenario A Helpers
# =========================================================
def _clamp_day(d: int, lo: int = 1, hi: int = 365) -> int:
    return int(max(lo, min(int(d), hi)))


def _shift_window_str(win: str, shift: int) -> str:
    """
    Shift window strings like "200-210" by +shift and clamp to [1..365].
    If parsing fails, returns original.
    """
    if win is None:
        return win
    s = str(win).strip()
    if not s or s.lower() in {"nan", "none"}:
        return win
    if "-" not in s:
        return win
    try:
        a_s, b_s = s.split("-", 1)
        a = _clamp_day(int(float(a_s.strip())) + shift)
        b = _clamp_day(int(float(b_s.strip())) + shift)
        if b < a:
            a, b = b, a
        return f"{a}-{b}"
    except Exception:
        return win


def apply_scenario_A_shift_management(management_df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """
    Scenario A: shift harvest timing by +shift days.
    - shifts 'harvest_date' (if present)
    - shifts 'harvest_win' (if present, format "a-b")
    Clamps to [1..365].
    """
    if management_df is None or len(management_df) == 0:
        return management_df

    df = management_df.copy()
    shift = int(shift)

    if "harvest_date" in df.columns:
        # shift only non-null
        h = pd.to_numeric(df["harvest_date"], errors="coerce")
        mask = h.notna()
        df.loc[mask, "harvest_date"] = (h.loc[mask].astype(int) + shift).clip(1, 365).astype(int)

    if "harvest_win" in df.columns:
        df["harvest_win"] = df["harvest_win"].apply(lambda w: _shift_window_str(w, shift))

    return df

# =========================================================
# Scenario B helpers
# =========================================================

def apply_scenario_B_flower_strips(
    fields_gdf_in,
    *,
    share: float,
    fraction: float,
    select: str = "no_flower_first",
    seed: int = 0,
):
    """
    Scenario B: set flower_strip_share on a subset of fields.

    Parameters
    ----------
    share : fraction of fields to modify (0..1)
    fraction : new flower_strip_share for selected fields (0..1)
    select : "no_flower_first" | "random"
    seed : rng seed for "random"

    Returns
    -------
    fields_gdf_out : copy of fields_gdf with modified flower_strip_share
    """
    if fields_gdf_in is None or len(fields_gdf_in) == 0:
        return fields_gdf_in

    df = fields_gdf_in.copy()

    if "flower_strip_share" not in df.columns:
        raise ValueError("Scenario B requires fields_gdf to have column 'flower_strip_share'.")

    share = float(share)
    fraction = float(fraction)
    if share <= 0:
        return df

    # clamp
    share = max(0.0, min(1.0, share))
    fraction = max(0.0, min(1.0, fraction))

    n = len(df)
    k = int(round(share * n))
    k = max(0, min(n, k))
    if k == 0:
        return df

    sel = str(select).strip().lower()
    rng = np.random.default_rng(int(seed))

    # ensure numeric
    fs = pd.to_numeric(df["flower_strip_share"], errors="coerce").fillna(0.0).to_numpy()

    if sel == "random":
        idx = rng.choice(np.arange(n), size=k, replace=False)
    else:
        # default: no_flower_first (smallest flower_strip_share first)
        idx = np.argsort(fs)[:k]

    # Apply: set to fraction (or you can do max(old, fraction) if you prefer)
    new_fs = fs.copy()
    new_fs[idx] = fraction
    df["flower_strip_share"] = new_fs

    print(
        f"[Scenario B] set flower_strip_share={fraction:.2f} "
        f"on {k}/{n} fields (share={share:.2f}, select={sel}, seed={seed})"
    )
    return df
# =========================================================
# Scenario C helpers
# =========================================================
def _choose_fields_best_match(field_ids: np.ndarray, field_areas: np.ndarray, target_area: float) -> list[int]:
    """
    Choose subset of fields (whole fields) whose summed area best matches target_area.
    Minimizes absolute error |sum(area)-target|.
    Tie-break: fewer fields.
    Exact search via brute force (fast for n<=25ish).
    """
    n = len(field_ids)
    if n == 0:
        return []

    # If n is small, brute force exact
    if n <= 25:
        best_mask = 0
        best_err = float("inf")
        best_k = 10**9

        # iterate all non-empty subsets
        for mask in range(1, 1 << n):
            # sum areas for mask
            s = 0.0
            k = 0
            for i in range(n):
                if mask & (1 << i):
                    s += float(field_areas[i])
                    k += 1
            err = abs(s - target_area)

            # primary: smaller error, secondary: fewer fields
            if (err < best_err) or (err == best_err and k < best_k):
                best_err = err
                best_k = k
                best_mask = mask

        chosen = [int(field_ids[i]) for i in range(n) if best_mask & (1 << i)]
        return chosen

    # Fallback for larger n: greedy by closeness + local improvement
    # (If you ever scale to many fields)
    order = np.argsort(np.abs(field_areas - target_area))
    chosen = []
    s = 0.0
    for idx in order:
        # try adding if it improves absolute error
        new_s = s + float(field_areas[idx])
        if abs(new_s - target_area) <= abs(s - target_area):
            chosen.append(int(field_ids[idx]))
            s = new_s
        # early stop if exact enough
        if abs(s - target_area) / max(target_area, 1e-9) < 1e-6:
            break
    if not chosen:
        # at least one field: pick closest single field
        best_i = int(np.argmin(np.abs(field_areas - target_area)))
        chosen = [int(field_ids[best_i])]
    return chosen

def apply_scenario_C_fallow(
    *,
    fields_gdf_in,
    rotation_in: pd.DataFrame,
    fallow_area_share: float,
    years_to_apply,
    selection: str = "best_match",   # <- default: best match
    seed: int = 0,
    fallow_crop_name: str = "Fallow",
) -> pd.DataFrame:
    if rotation_in is None or len(rotation_in) == 0:
        return rotation_in
    if fields_gdf_in is None or len(fields_gdf_in) == 0:
        return rotation_in

    rot = rotation_in.copy()

    required = {"year", "field_id", "crop"}
    if not required.issubset(rot.columns):
        raise ValueError(f"Scenario C requires rotation columns {sorted(required)}")
    for col in ("field_id", "area"):
        if col not in fields_gdf_in.columns:
            raise ValueError(f"Scenario C requires fields_gdf to have column '{col}'")

    fallow_area_share = float(max(0.0, min(1.0, fallow_area_share)))
    if fallow_area_share <= 0:
        return rot

    years_to_apply = [int(y) for y in years_to_apply]

    # --- area lookup (all fields) ---
    areas = (
        fields_gdf_in[["field_id", "area"]]
        .copy()
        .assign(field_id=lambda d: pd.to_numeric(d["field_id"], errors="coerce").astype(int))
    )
    area_map = dict(zip(areas["field_id"].tolist(), areas["area"].astype(float).tolist()))
    total_area = float(np.nansum(areas["area"].to_numpy()))
    if total_area <= 0:
        return rot
    target_area = fallow_area_share * total_area

    # --- determine ARABLE candidates from rotation in selected years ---
    rot_sel = rot[rot["year"].isin(years_to_apply)].copy()
    rot_sel["crop_norm"] = (
        rot_sel["crop"]
        .astype(str).str.strip().str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )

    NON_ARABLE = {
        fallow_crop_name.lower().replace(" ", ""),
        "permanentcover",
        "permanent_cover",
        "permanentcover",   # keep
        "grassland",
        "gruenland",
        "grünland",
        "meadow",
    }

    arable_ids = (
        rot_sel.loc[~rot_sel["crop_norm"].isin(NON_ARABLE), "field_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    # keep only ids we have area for
    arable_ids = [fid for fid in arable_ids if fid in area_map]
    if len(arable_ids) == 0:
        print("[Scenario C] no eligible arable fields -> no fallow applied.")
        return rot

    field_ids = np.array(sorted(arable_ids), dtype=int)
    field_areas = np.array([area_map[fid] for fid in field_ids], dtype=float)

    # --- choose subset with best match ---
    chosen = _choose_fields_best_match(field_ids, field_areas, target_area)  # your helper
    if not chosen:
        # safety: pick closest single field
        best_i = int(np.argmin(np.abs(field_areas - target_area)))
        chosen = [int(field_ids[best_i])]

    chosen_set = set(chosen)
    acc = float(sum(area_map[int(fid)] for fid in chosen_set))

    # --- apply to rotation ---
    mask = rot["year"].isin(years_to_apply) & rot["field_id"].astype(int).isin(chosen_set)
    n_rows = int(mask.sum())
    rot.loc[mask, "crop"] = fallow_crop_name

    print(
        f"[Scenario C] set crop='{fallow_crop_name}' for {len(chosen_set)} fields "
        f"({acc/total_area:.2%} of area; target={fallow_area_share:.2%}) "
        f"in years={years_to_apply} | rows_changed={n_rows} | selection=best_match | seed={seed}"
    )
    return rot

# =========================================================
# Scenario D helpers
# =========================================================

def apply_scenario_D_skylark(
    *,
    fields_gdf_in,
    rotation_in: pd.DataFrame,
    years_to_sim: list[int],
    add_skylark_plots: int,
    skylark_plot_area: float,
    target_fraction: float,
    min_eligible_years: int = 1,
    only_without_flower: bool = True,
    seed: int = 0,
) :
    """
    Scenario D: Assign skylark plots to a subset of fields by modifying fields_gdf columns:
      - skylark_plots (int, e.g. 2)
      - skylark_plot_area (float, e.g. 0.02)

    Selection rule (minimal, robust):
      - Eligible fields must appear in rotation for at least `min_eligible_years` within years_to_sim.
      - If only_without_flower=True: require flower_strip_share == 0 (or very close)
      - Choose fields until sum(area) reaches target_fraction * total_area (area-based target)

    Returns:
      fields_out (copy), rotation_out (unchanged), management_out (None here -> caller keeps)
    """
    if fields_gdf_in is None or len(fields_gdf_in) == 0:
        return fields_gdf_in

    fields = fields_gdf_in.copy()

    # sanity: required columns
    for col in ("field_id", "area"):
        if col not in fields.columns:
            raise ValueError(f"Scenario D requires fields_gdf to have column '{col}'")

    # make sure target cols exist
    if "skylark_plots" not in fields.columns:
        fields["skylark_plots"] = 0
    if "skylark_plot_area" not in fields.columns:
        fields["skylark_plot_area"] = 0.0
    if "flower_strip_share" not in fields.columns:
        # if missing, treat as no flower strips (so D can still run)
        fields["flower_strip_share"] = 0.0

    add_skylark_plots = int(add_skylark_plots)
    skylark_plot_area = float(skylark_plot_area)
    target_fraction = float(target_fraction)
    min_eligible_years = int(min_eligible_years)
    only_without_flower = bool(only_without_flower)

    if add_skylark_plots <= 0 or skylark_plot_area <= 0 or target_fraction <= 0:
        print("[Scenario D] nothing to apply (plots/area/target_fraction <= 0).")
        return fields

    # total area target
    total_area = float(np.nansum(fields["area"].to_numpy(dtype=float)))
    if total_area <= 0:
        return fields
    target_area = max(0.0, min(1.0, target_fraction)) * total_area

    # eligible by rotation presence
    eligible_ids = None
    if rotation_in is not None and len(rotation_in) > 0 and {"year", "field_id"}.issubset(rotation_in.columns):
        rot = rotation_in.copy()
        rot = rot[rot["year"].isin([int(y) for y in years_to_sim])]
        counts = rot.groupby("field_id")["year"].nunique()
        eligible_ids = set(counts[counts >= max(min_eligible_years, 1)].index.astype(int).tolist())
    else:
        # if no rotation -> assume all fields eligible
        eligible_ids = set(pd.to_numeric(fields["field_id"], errors="coerce").dropna().astype(int).tolist())

    # filter fields to eligible
    f = fields.copy()
    f["field_id"] = pd.to_numeric(f["field_id"], errors="coerce").astype(int)
    f = f[f["field_id"].isin(eligible_ids)].copy()

    # optional: only fields without flower strips
    if only_without_flower:
        f = f[f["flower_strip_share"].astype(float) <= 1e-12].copy()

    if len(f) == 0:
        print("[Scenario D] no eligible fields after filters -> no skylark plots assigned.")
        return fields

    # choose by area (largest first), randomized tie-break via seed
    rng = np.random.default_rng(int(seed))
    f = f.assign(_rand=rng.random(len(f)))
    f = f.sort_values(["area", "_rand"], ascending=[False, True])

    chosen = []
    acc = 0.0
    for _, row in f.iterrows():
        if acc >= target_area:
            break
        fid = int(row["field_id"])
        a = float(row["area"])
        chosen.append(fid)
        acc += a

    chosen_set = set(chosen)

    # apply to original fields
    mask = fields["field_id"].astype(int).isin(chosen_set)
    fields.loc[mask, "skylark_plots"] = add_skylark_plots
    fields.loc[mask, "skylark_plot_area"] = skylark_plot_area

    print(
        f"[Scenario D] skylark_plots={add_skylark_plots}, skylark_plot_area={skylark_plot_area} "
        f"assigned to {len(chosen_set)} fields ({acc/total_area:.2%} of area; target={target_fraction:.2%}); "
        f"min_eligible_years={min_eligible_years}, only_without_flower={only_without_flower}, seed={seed}"
    )
    return fields

# =========================================================
#  MAIN RUNNER (YAML-first)
# =========================================================

def run(*, cfg) -> None:
    """
    Run the simulation (YAML-first).
    cfg: RunConfig from fieldbird_sim.cli (or compatible object with .run/.seeds/...).
    """
    if cfg is None:
        raise ValueError("kpis.run requires cfg (RunConfig).")

    # 1) Apply cfg -> legacy globals
    _apply_run_config_to_globals(cfg)

    # 2) Synthetic flag from cfg
    use_synth = bool(getattr(cfg.run, "use_synthetic", True))

    # 3) Output dir
    out_prefix = str(getattr(cfg.run, "outdir_prefix", "outputs"))
    run_name = getattr(cfg.run, "run_name", None)
    outdir = _mk_outdir(out_prefix, name=run_name)

    # 4) Build landscape inputs + years_to_sim
    if use_synth:
        n_years = int(getattr(cfg.run, "n_years", int(OPTS.N_YEARS)))
        years_to_sim = list(range(1, n_years + 1))

        fields_gdf = make_synthetic_fields(
            n_fields=int(cfg.run.n_fields),
            seed=int(cfg.seeds.fields),
            add_cover=bool(cfg.synthetic.cover.add_cover),
            cover_margin_m=float(cfg.synthetic.cover.cover_margin_m),
            cover_min_patch_m2=float(cfg.synthetic.cover.cover_min_patch_m2),
            cover_fill_residual=bool(cfg.synthetic.cover.cover_fill_residual),
            square_pad_m=float(cfg.synthetic.cover.square_pad_m),
        )

        rotation = make_rotation(fields_gdf, years_to_sim, seed=int(cfg.seeds.rotation))

        calendar_path = resolve_calendar_path(
            str(cfg.management.calendar_path),
            base_dir=getattr(cfg, "config_dir", None),
        )
        management = make_management(
            fields_gdf,
            rotation,
            seed=int(cfg.seeds.management),
            use_calendar=bool(cfg.management.use_calendar),
            calendar_path=str(calendar_path),
            desync_region_sd=float(cfg.management.desync_region_sd),
            desync_field_sd=float(cfg.management.desync_field_sd),
            region_bins=int(cfg.management.region_bins),
        )

    else:
        # CSV inputs
        if not hasattr(cfg, "inputs"):
            raise ValueError("run.use_synthetic=false but cfg.inputs missing.")

        fields_path = Path(cfg.inputs.fields_csv)
        rotation_path = Path(cfg.inputs.rotation_csv)
        management_path = Path(cfg.inputs.management_csv)

        if not fields_path.exists():
            raise FileNotFoundError(f"fields_csv not found: {fields_path}")
        if not rotation_path.exists():
            raise FileNotFoundError(f"rotation_csv not found: {rotation_path}")
        if not management_path.exists():
            raise FileNotFoundError(f"management_csv not found: {management_path}")

        fields = pd.read_csv(fields_path)
        rotation = pd.read_csv(rotation_path)
        management = pd.read_csv(management_path)

        # sanity
        if "field_id" not in fields.columns:
            raise ValueError("fields_csv missing required column 'field_id'")
        for col in ("year", "field_id", "crop"):
            if col not in rotation.columns:
                raise ValueError(f"rotation_csv missing required column '{col}'")
        for col in ("year", "field_id"):
            if col not in management.columns:
                raise ValueError(f"management_csv missing required column '{col}'")

        # types
        fields["field_id"] = pd.to_numeric(fields["field_id"], errors="coerce").astype(int)
        rotation["field_id"] = pd.to_numeric(rotation["field_id"], errors="coerce").astype(int)
        rotation["year"] = pd.to_numeric(rotation["year"], errors="coerce").astype(int)
        management["field_id"] = pd.to_numeric(management["field_id"], errors="coerce").astype(int)
        management["year"] = pd.to_numeric(management["year"], errors="coerce").astype(int)

        fields_gdf = rect_or_centroid_layout(fields)

        unique_years = sorted(pd.unique(rotation["year"]))
        if len(unique_years) == 0:
            raise ValueError("rotation_csv contains no years")

        years_to_sim = unique_years[: int(cfg.run.n_years)]

    # IMPORTANT: plot controls AFTER years_to_sim is final (CSV overrides years_to_sim)
    plot_ctl = get_plot_controls(cfg, years_to_sim=years_to_sim, year_to_show=int(YEAR_TO_SHOW))

    # 5) Logging
    print(
        f"[Landscape] fields={len(fields_gdf)}, years={len(years_to_sim)}, "
        f"rotation_rows={len(rotation)}, synthetic={use_synth}"
    )
    print(
        f"[Options] N_FIELDS={int(cfg.run.n_fields)} | "
        f"N_YEARS={int(cfg.run.n_years)} | "
        f"YEAR_TO_SHOW={YEAR_TO_SHOW} | DAY_TO_SHOW={DAY_TO_SHOW}"
    )

    species_list = _resolve_species_list(getattr(cfg.run, "species", None))
    n_init_first = int(getattr(cfg.run, "n_init_first", 100))

    # =========================================================
    #  Scenario plan (one scenario run covers ALL species)
    # =========================================================
    scenarios_to_run = build_scenarios_to_run(cfg)

    # Baseline results store (for future deltas)
    baseline_results_by_species: dict[str, dict] = {}

    # =========================================================
    #  Run scenarios (each scenario includes ALL species)
    # =========================================================
    for scen in scenarios_to_run:
        scen_name = scen["name"]
        scen_kind = scen["kind"]

        print(f"\n=== RUN SCENARIO: {scen_name} ({scen_kind}) ===")

        scenario_outdir = outdir / scen_name
        scenario_outdir.mkdir(parents=True, exist_ok=True)

        fields_sc, rot_sc, mgmt_sc = apply_scenario_to_inputs(
            scen=scen,
            fields_gdf_in=fields_gdf,
            rotation_in=rotation,
            management_in=management,
            years_to_sim=years_to_sim,
            apply_A_shift_management=apply_scenario_A_shift_management,
            apply_B_flower_strips=apply_scenario_B_flower_strips,
            apply_C_fallow=apply_scenario_C_fallow,
            apply_D_skylark=apply_scenario_D_skylark,
        )

        # IMPORTANT: rebuild hedges per scenario (will matter once scenarios change fields)
        hedges_sc = build_hedges_from_cfg(cfg=cfg, fields_gdf=fields_sc)
        fields_sc = attach_hedge_metrics_to_fields(fields_gdf=fields_sc, hedges_gdf=hedges_sc)

        for sp_name in species_list:
            species_dir = resolve_species_dir(
                sp_name,
                configured_dir=getattr(cfg.run, "species_dir", None),
                base_dir=getattr(cfg, "config_dir", None),
            )
            sp_cfg = load_species_from_csv(sp_name, base_path=species_dir)
            weather_module, weather_start_date = build_weather_module_from_cfg(cfg)

            sanity_check_species(sp_cfg)

            print(
                "SPECIES:",
                sp_cfg.name,
                "| move_radius(100m)=",
                sp_cfg.move_radius,
                "| breeding_window=",
                sp_cfg.breeding_window,
            )

            sim = BirdLandscapeSim(
                fields_sc[
                    [
                        "field_id",
                        "x",
                        "y",
                        "area",
                        "perimeter",
                        "flower_strip_share",
                        "skylark_plots",
                        "skylark_plot_area",
                        "hedge_length_m",
                        "hedge_quality_weighted_m",
                        "hedge_quality_mean",
                        "hedge_length_m_per_ha",
                        "hedge_quality_weighted_m_per_ha",
                        "width_m",
                        "height_m",
                    ]
                ],
                rot_sc,
                mgmt_sc,
                sp_cfg,
            )

            sp_outdir = scenario_outdir / sp_cfg.name
            sp_outdir.mkdir(parents=True, exist_ok=True)

            if USE_YEAR_FEEDBACK:
                res_years = run_years_with_feedback(
                    fields_sc,
                    rot_sc,
                    mgmt_sc,
                    sp_cfg,
                    years=years_to_sim,
                    N_init_first=n_init_first,
                    weather_module=weather_module,
                    timeline_start_date=weather_start_date,
                )
            else:
                res_years = {}
                for year_idx, y in enumerate(years_to_sim):
                    dbg_days = [110, 130, 160, 200] if int(y) == int(YEAR_TO_SHOW) else None
                    res_years[y] = sim.simulate(
                        days=365,
                        year=int(y),
                        N_init=n_init_first,
                        debug_days=dbg_days,
                        weather_module=weather_module,
                        timeline_start_date=weather_start_date,
                        timeline_day_offset=int(year_idx) * 365,
                    )

                    ts = to_long_timeseries(res_years[y], sp_cfg.name, y, scenario=scen_name)
                    ydir = sp_outdir / f"year_{y}"
                    ydir.mkdir(parents=True, exist_ok=True)
                    ts.to_csv(ydir / "timeseries_long.csv", index=False)

            save_run_bundle(
                outdir=sp_outdir,
                species_name=sp_cfg.name,
                fields_gdf=fields_sc,
                rotation=rot_sc,
                management=mgmt_sc,
                hedges_gdf=hedges_sc,
                results_by_year=res_years,
                day_to_show=DAY_TO_SHOW,
            )

            if weather_module is not None:
                weather_log_path = weather_module.write_csv(sp_outdir / "weather_impact_log.csv")
                print(f"[weather] log written: {weather_log_path}")

            # Quick population summary
            arr = max(1, int(sp_cfg.arrival_day)) - 1
            N_arrivals = [(res_years[y]["A_m"][arr, :] + res_years[y]["A_f"][arr, :]).sum() for y in years_to_sim]
            lam = [N_arrivals[i + 1] / max(N_arrivals[i], 1e-9) for i in range(len(N_arrivals) - 1)]
            print("Adults at arrival by year:", [int(x) for x in N_arrivals])
            if lam:
                print("Lambda (year-to-year):", np.round(lam, 3), " mean:", round(float(np.mean(lam)), 3))
            else:
                print("Lambda (year-to-year):", np.array([]), " mean: n/a (n_years < 2)")

            # Heatmaps (per scenario)
            if DO_HEATMAPS:
                fids = fields_sc["field_id"].values

                global_vmax = None
                if plot_ctl["heatmap_vmax_mode"] == "global":
                    global_vmax = max(res["N_total"].max() for res in res_years.values())

                for y in plot_ctl["heatmaps_years"]:
                    y = int(y)
                    if y not in res_years:
                        continue

                    vmax = global_vmax
                    if plot_ctl["heatmap_vmax_mode"] == "per_year":
                        vmax = float(res_years[y]["N_total"].max())

                    heatmap(
                        res_years[y]["N_total"],
                        fids,
                        f"{sp_cfg.name} – Heatmap Gesamt – {scen_name} – Jahr {y}",
                        vmax=vmax,
                        savepath=sp_outdir / f"plots/heatmap_year{y}.png",
                    )

            # Maps (per scenario)
            if DO_MAPS:
                for yshow in plot_ctl["maps_years"]:
                    yshow = int(yshow)
                    if yshow not in res_years:
                        continue

                    plot_map(
                        fields_sc,
                        rot_sc,
                        year=yshow,
                        hedges_gdf=hedges_sc,
                        title=f"{sp_cfg.name} – {scen_name} – Jahr {yshow}, Tag {DAY_TO_SHOW}",
                        draw_flower_rings=DRAW_FLOWER_RINGS,
                        draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                        hedge_buffer_m=HEDGE_BUFFER_M,
                        day_vals=res_years[yshow]["N_total"][DAY_TO_SHOW],
                        savepath=sp_outdir / f"plots/map_day{DAY_TO_SHOW}_year{yshow}.png",
                    )

                    if plot_ctl["usage_map"]:
                        plot_map(
                            fields_sc,
                            rot_sc,
                            year=yshow,
                            hedges_gdf=hedges_sc,
                            title=f"{sp_cfg.name} – {scen_name} – Jahr {yshow}: Usage (Jahressumme)",
                            draw_flower_rings=DRAW_FLOWER_RINGS,
                            draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                            hedge_buffer_m=HEDGE_BUFFER_M,
                            usage_vals=res_years[yshow]["usage"],
                            savepath=sp_outdir / f"plots/map_usage_year{yshow}.png",
                        )

            # KPI
            kpi_df = compute_kpis_from_results(
                res_years,
                sp_cfg,
                output_path=sp_outdir / "analysis" / "kpi_yearly.csv",
            )
            print("\n[KPI] Jahres-Kennzahlen:\n", kpi_df)

            # store baseline for later delta step
            if scen_kind == "baseline":
                baseline_results_by_species[sp_cfg.name] = res_years

    print(f"\nAll files written under: {outdir}\n")
