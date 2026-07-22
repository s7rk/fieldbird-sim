from __future__ import annotations

# =========================================================
#                         BirdSIM
# =========================================================
# YAML-first/legacy globals module:
# - provides defaults + functions used by legacy.kpis runner
# - values are overwritten at runtime by kpis._apply_run_config_to_globals(cfg)
# =========================================================

# ------------------- Import area -------------------

# Standard library
import json
import math
import os
import pathlib
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

# Third-party
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from fieldbird_sim.provenance import runtime_metadata

# ------------------- Global controls -------------------

TOT_DAYS = 365  # zentrale Referenz für Jahr-Länge (Index = DOY)
DEBUG_YEAR_XFER = 1

# === Defaults (werden später in legacy.kpis._apply_run_config_to_globals überschrieben) ===
OPTS = SimpleNamespace(
    N_FIELDS=20,
    N_YEARS=20,
    YEAR_TO_SHOW=1,
    DAY_TO_SHOW=160,
    SEED_FIELDS=2025,
    SEED_ROTATION=2025,
    SEED_MANAGEMENT=2025,
    SEED_HEDGES=2025,
    SIM_RNG_SEED=2025,
)

# Plot defaults (YAML überschreibt)
DO_MAPS = True
DO_HEATMAPS = True

# Derived defaults (kpis.py setzt die nach YAML-apply erneut!)
YEARS_TO_SIM = list(range(1, int(OPTS.N_YEARS) + 1))
YEAR_TO_SHOW = max(1, min(int(OPTS.YEAR_TO_SHOW), int(OPTS.N_YEARS)))
DAY_TO_SHOW = max(1, min(int(OPTS.DAY_TO_SHOW), TOT_DAYS))

SPECIES_LIST = ["Feldlerche", "Rebhuhn"]

# Layout
USE_RECT_PACKING = True
TARGET_ROW_WIDTH_M = None

# Hedges / visuals
USE_HEDGES_FROM_CSV = True
AUTO_HEDGES_FROM_ADJ = False
DRAW_HEDGE_BUFFERS = True
HEDGE_BUFFER_M = 5.0
DRAW_FLOWER_RINGS = True

# Refuge settings
ELIGIBLE_REFUGE_CROPS = {"Winterweizen", "Sommergerste"}
REFUGE_ASSIGN_MULT = 15.0
PULSE_PROTECT_SCALE = 4.0
REFUGE_SHARE_CAP = 0.6

# Input paths
FIELDS_CSV = "inputs/fields_template.csv"
ROTATION_CSV_MAIN = "inputs/rotation.csv"
ROTATION_CSV_TPL = "inputs/rotation_template.csv"
MANAGEMENT_CSV_MAIN = "inputs/management.csv"
MANAGEMENT_CSV_TPL = "inputs/management_template.csv"
HEDGES_CSV = "inputs/hedges_template.csv"
SKYLARK_WINDOWS_CSV = Path("configs/skylark_windows.csv")
CROP_CALENDAR_CSV = "configs/crop_calendar.csv"

# Scenarios defaults (YAML überschreibt)
SCENARIO_A_ENABLED = True
SCENARIO_A_SHIFTS = [7, 14, 21]
SCENARIO_A = {"shifts": SCENARIO_A_SHIFTS}

SCENARIO_B_ENABLED = True
SCENARIO_B_SHARES = [0.10, 0.20, 0.30]
SCENARIO_B_FRACTIONS = [0.25, 0.50, 1.00]
SCENARIO_B_SELECT = "no_flower_first"
SCENARIO_B_SEED = 2025

SCENARIO_C_ENABLED = True
SCENARIO_C_FALLOW_AREA_SHARES = [0.01, 0.05, 0.10]
SCENARIO_C_APPLY_YEARS = "all"
SCENARIO_C_SELECTION = "greedy"
SCENARIO_C_SEED = 2026

SCENARIO_D_ENABLED = True
SCENARIO_D = dict(
    add_skylark_plots=2,
    skylark_plot_area=0.02,
    target_fraction=0.20,
    min_eligible_years=1,
    only_without_flower=True,
)

# Winter / feedback defaults
USE_YEAR_FEEDBACK = True
WINTER_SURVIVAL_ADULT = 0.95
WINTER_SURVIVAL_JUV = 0.90
JUV_MATURATION_RATE = 0.90
JUV_EQUIV_IN_CAPACITY = 0.5
WINDOW_STRESS_WEIGHT = 0.5

WINTER_MOVE_RADIUS = 12.0
WINTER_BETA = 2.5
WINTER_DISTANCE_COST = 0.08
WINTER_EVAL_DAYS = (330, 20)

# Capacity defaults
USE_CAPACITY_SOFT = True
K_PER_HA_ADULTS = 40.0
CAPACITY_GAMMA = 0.4

USE_DENSITY_FACTOR = True


# =========================================================
# Output directory & plot saving
# =========================================================

def _mk_outdir(prefix: str = "outputs", *, name: Optional[str] = None) -> pathlib.Path:
    """YAML-first: keine ENV-Tags; name optional, sonst timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(prefix) / (name or ts)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_fig(path, tight: bool = True, dpi: int = 180, transparent: bool = False) -> None:
    """Save current matplotlib figure robustly and close it."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf()
    bbox = None
    pad_inches = None
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
        bbox = "tight"
        pad_inches = 0.02

    try:
        fig.savefig(p, dpi=dpi, bbox_inches=bbox, pad_inches=pad_inches, transparent=transparent)
    finally:
        plt.close(fig)


# =========================================================
# Daily full export helper (used by save_run_bundle)
# =========================================================

def _build_daily_full_df(
    fields_gdf: pd.DataFrame,
    rotation_df: pd.DataFrame,
    management_df: pd.DataFrame,
    results_by_year: dict,
    species_name: str,
    run_id: str,
) -> Optional[pd.DataFrame]:
    """
    Build 'birdsim_daily_full' long table.
    Returns None if results_by_year is empty.
    """
    if not results_by_year:
        return None

    fields_df = fields_gdf.copy()
    if "field_id" not in fields_df.columns:
        raise ValueError("fields_gdf must contain 'field_id'")

    fields_df["field_id"] = pd.to_numeric(fields_df["field_id"], errors="coerce").astype("Int64")

    # x/y from geometry if needed
    if "x" not in fields_df.columns or "y" not in fields_df.columns:
        if "geometry" in fields_df.columns:
            try:
                cent = fields_df.geometry.centroid
                if "x" not in fields_df.columns:
                    fields_df["x"] = cent.x
                if "y" not in fields_df.columns:
                    fields_df["y"] = cent.y
            except Exception:
                fields_df["x"] = fields_df.get("x", np.nan)
                fields_df["y"] = fields_df.get("y", np.nan)
        else:
            fields_df["x"] = fields_df.get("x", np.nan)
            fields_df["y"] = fields_df.get("y", np.nan)

    # Area ha
    if "area_ha" in fields_df.columns:
        fields_df["area_ha"] = pd.to_numeric(fields_df["area_ha"], errors="coerce")
    elif "area" in fields_df.columns:
        fields_df["area_ha"] = pd.to_numeric(fields_df["area"], errors="coerce")
    else:
        fields_df["area_ha"] = np.nan
    fields_df["area_ha"] = fields_df["area_ha"].astype(float)

    fields_df["flower_strip_share"] = (
        pd.to_numeric(fields_df.get("flower_strip_share", 0.0), errors="coerce").fillna(0.0)
    )
    fields_df["skylark_plots"] = (
        pd.to_numeric(fields_df.get("skylark_plots", 0), errors="coerce").fillna(0).astype(int)
    )
    if "skylark_plot_area_ha" in fields_df.columns:
        fields_df["skylark_plot_area_ha"] = pd.to_numeric(
            fields_df["skylark_plot_area_ha"], errors="coerce"
        ).fillna(0.0)
    elif "skylark_plot_area" in fields_df.columns:
        fields_df["skylark_plot_area_ha"] = pd.to_numeric(
            fields_df["skylark_plot_area"], errors="coerce"
        ).fillna(0.0)
    else:
        fields_df["skylark_plot_area_ha"] = 0.0

    fields_df["width_m"] = pd.to_numeric(fields_df.get("width_m", np.nan), errors="coerce")
    fields_df["height_m"] = pd.to_numeric(fields_df.get("height_m", np.nan), errors="coerce")
    fields_df["perimeter"] = pd.to_numeric(fields_df.get("perimeter", np.nan), errors="coerce")

    # refuge_share
    if "refuge_share" in fields_df.columns:
        fields_df["refuge_share"] = pd.to_numeric(fields_df["refuge_share"], errors="coerce").clip(0.0, 1.0)
    else:
        area_vals = fields_df["area_ha"].to_numpy(float)
        plots = fields_df["skylark_plots"].to_numpy(float)
        plot_area = fields_df["skylark_plot_area_ha"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            refuge = np.where(area_vals > 0, (plots * plot_area) / area_vals, 0.0)
        fields_df["refuge_share"] = np.clip(refuge, 0.0, 1.0)

    fields_lookup = fields_df.set_index("field_id")

    # rotation typing
    rotation_df = rotation_df.copy()
    rotation_df["year"] = pd.to_numeric(rotation_df["year"], errors="coerce").astype("Int64")
    rotation_df["field_id"] = pd.to_numeric(rotation_df["field_id"], errors="coerce").astype("Int64")

    # management typing
    management_df = management_df.copy()
    if len(management_df):
        if {"year", "field_id"} - set(management_df.columns):
            raise ValueError("management_df must contain 'year' and 'field_id'")
        management_df["year"] = pd.to_numeric(management_df["year"], errors="coerce").astype("Int64")
        management_df["field_id"] = pd.to_numeric(management_df["field_id"], errors="coerce").astype("Int64")

    def _parse_days(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return np.array([], dtype=int)
        if isinstance(s, (list, tuple, np.ndarray)):
            return np.array([int(x) for x in s if 1 <= int(x) <= 365], dtype=int)
        txt = str(s).strip()
        if not txt:
            return np.array([], dtype=int)
        parts = [p.strip() for p in re.split(r"[;, ]+", txt) if p.strip()]
        out = []
        for p in parts:
            try:
                v = int(float(p))
                if 1 <= v <= 365:
                    out.append(v)
            except Exception:
                pass
        return np.array(out, dtype=int)

    def _build_management_map(year: int) -> dict[int, dict]:
        if len(management_df) == 0:
            return {}
        sub = management_df.loc[management_df["year"] == int(year)]
        if sub.empty:
            return {}
        out = {}
        for _, row in sub.iterrows():
            fid = int(row["field_id"])

            def _clean_day(val):
                try:
                    v = int(float(val))
                except Exception:
                    return None
                return v if 1 <= v <= 365 else None

            harvest_day = _clean_day(row.get("harvest_date", None))
            insect_day = _clean_day(row.get("insecticide_date", None))
            mow_arr = _parse_days(row.get("mow_dates", None))
            mow_set = set()
            for d in mow_arr.tolist():
                for dd in (d - 1, d, d + 1):
                    if 1 <= dd <= 365:
                        mow_set.add(int(dd))

            out[fid] = {"harvest": harvest_day, "insecticide": insect_day, "mowing_days": mow_set}
        return out

    def _neighbor_counts(xy: np.ndarray, fids: np.ndarray, thr_m: float) -> dict[int, int]:
        n = len(fids)
        if n == 0:
            return {}
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        D = np.sqrt(dx * dx + dy * dy)
        D[np.arange(n), np.arange(n)] = np.inf
        counts = (D <= float(thr_m)).sum(axis=1).astype(int)
        return {int(fids[i]): int(counts[i]) for i in range(n)}

    out_rows = []
    for year in sorted(results_by_year.keys()):
        res = results_by_year[year]

        A_m = res.get("A_m")
        A_f = res.get("A_f")
        J_free = res.get("J_free")
        A_f_breeding = res.get("A_f_breeding")
        if A_f_breeding is None and A_f is not None:
            A_f_breeding = np.zeros_like(A_f, dtype=int)

        J_tied = res.get("J_tied_total", res.get("J_tied"))
        if J_tied is None and A_f is not None:
            J_tied = np.zeros_like(A_f, dtype=int)

        new_nests = res.get("daily_new_nests")
        fledglings = res.get("daily_fledglings")
        N_total = res.get("N_total")

        new_nests_c1 = res.get("daily_new_nests_c1")
        new_nests_c2 = res.get("daily_new_nests_c2")
        new_nests_c3 = res.get("daily_new_nests_c3")

        if N_total is None and all(v is not None for v in (A_m, A_f, J_free, J_tied)):
            N_total = (A_m.astype(int) + A_f.astype(int) + J_free.astype(int) + J_tied.astype(int))

        for nm in ("A_m", "A_f", "J_free", "N_total"):
            if locals()[nm] is None:
                raise ValueError(f"results_by_year[{year!r}] missing '{nm}' – export impossible.")

        days, nfields = A_m.shape

        if "fields" in res and isinstance(res["fields"], pd.DataFrame) and "field_id" in res["fields"].columns:
            res_fids = res["fields"]["field_id"].astype(int).to_numpy()
        else:
            res_fids = fields_lookup.index.dropna().astype(int).sort_values().to_numpy()
            if len(res_fids) != nfields:
                raise ValueError(f"Cannot determine field_id order safely (year={year}).")

        rot_y = rotation_df.loc[rotation_df["year"] == int(year)]
        crop_map = {int(r.field_id): str(r.crop) for r in rot_y.itertuples(index=False)}
        crops_vec = np.array([crop_map.get(int(fid), "Unbekannt") for fid in res_fids], dtype=object)

        mg_map = _build_management_map(int(year))

        XY = np.c_[fields_lookup.loc[res_fids, "x"].to_numpy(float), fields_lookup.loc[res_fids, "y"].to_numpy(float)]
        neigh300 = _neighbor_counts(XY, res_fids, thr_m=300.0)
        neigh600 = _neighbor_counts(XY, res_fids, thr_m=600.0)

        width = fields_lookup.loc[res_fids, "width_m"].to_numpy(float)
        height = fields_lookup.loc[res_fids, "height_m"].to_numpy(float)
        perim = fields_lookup.loc[res_fids, "perimeter"].to_numpy(float)
        area_ha_vec = fields_lookup.loc[res_fids, "area_ha"].to_numpy(float)
        area_m2_vec = np.where(np.isfinite(area_ha_vec), area_ha_vec * 10_000.0, np.nan)

        aspect_ratio_vec = np.where(
            np.isfinite(width) & np.isfinite(height) & (width > 0) & (height > 0),
            np.maximum(width, height) / np.maximum(1e-9, np.minimum(width, height)),
            np.nan,
        )
        edge_density_vec = perim / np.maximum(area_ha_vec, 1e-9)
        compactness_vec = np.where(
            np.isfinite(perim) & (perim > 0) & np.isfinite(area_m2_vec),
            4 * math.pi * area_m2_vec / (perim**2),
            np.nan,
        )

        x_vec = fields_lookup.loc[res_fids, "x"].to_numpy(float)
        y_vec = fields_lookup.loc[res_fids, "y"].to_numpy(float)
        flower_share_vec = fields_lookup.loc[res_fids, "flower_strip_share"].to_numpy(float)
        skl_plots_vec = fields_lookup.loc[res_fids, "skylark_plots"].to_numpy(int)
        skl_area_vec = fields_lookup.loc[res_fids, "skylark_plot_area_ha"].to_numpy(float)
        refuge_share_vec = fields_lookup.loc[res_fids, "refuge_share"].to_numpy(float)

        def flat(arr: np.ndarray) -> np.ndarray:
            return np.asarray(arr, dtype=float).reshape(-1, order="C")

        day_col = np.repeat(np.arange(1, days + 1, dtype=int), nfields)
        fid_col = np.tile(res_fids, days)
        crop_col = np.tile(crops_vec, days)

        doy = day_col.astype(int)
        doy_sin = np.sin(2 * math.pi * (doy / 365.0))
        doy_cos = np.cos(2 * math.pi * (doy / 365.0))

        is_harvest = np.zeros_like(day_col, dtype=bool)
        is_insect = np.zeros_like(day_col, dtype=bool)
        is_mowing = np.zeros_like(day_col, dtype=bool)

        hmap = {
            int(fid): mg_map.get(int(fid), {"harvest": None, "insecticide": None, "mowing_days": set()})
            for fid in res_fids
        }
        for i in range(day_col.shape[0]):
            m = hmap.get(int(fid_col[i]))
            if not m:
                continue
            d = int(day_col[i])
            if m["harvest"] == d:
                is_harvest[i] = True
            if m["insecticide"] == d:
                is_insect[i] = True
            if d in m["mowing_days"]:
                is_mowing[i] = True

        A_m_f = flat(A_m)
        A_f_f = flat(A_f)
        A_f_breed_f = flat(A_f_breeding) if A_f_breeding is not None else np.zeros_like(A_f_f)
        J_free_f = flat(J_free)
        J_tied_f = flat(J_tied) if J_tied is not None else np.zeros_like(A_f_f)
        new_nests_f = flat(new_nests) if new_nests is not None else np.zeros_like(A_f_f)
        new_nests_c1_f = flat(new_nests_c1) if new_nests_c1 is not None else np.zeros_like(A_f_f)
        new_nests_c2_f = flat(new_nests_c2) if new_nests_c2 is not None else np.zeros_like(A_f_f)
        new_nests_c3_f = flat(new_nests_c3) if new_nests_c3 is not None else np.zeros_like(A_f_f)
        fledglings_f = flat(fledglings) if fledglings is not None else np.zeros_like(A_f_f)
        N_total_f = flat(N_total)

        in_breeding = (A_f_breed_f > 0) | (new_nests_f > 0)

        def rep(v):
            return np.tile(v, days)

        x_f = rep(x_vec)
        y_f = rep(y_vec)
        area_f = rep(area_ha_vec)
        flow_f = rep(flower_share_vec)
        sklp_f = rep(skl_plots_vec)
        skla_f = rep(skl_area_vec)
        ref_f = rep(refuge_share_vec)
        w_f = rep(width)
        h_f = rep(height)
        p_f = rep(perim)
        ar_f = rep(aspect_ratio_vec)
        ed_f = rep(edge_density_vec)
        comp_f = rep(compactness_vec)
        n300_f = rep(np.array([neigh300.get(int(fid), 0) for fid in res_fids], dtype=int))
        n600_f = rep(np.array([neigh600.get(int(fid), 0) for fid in res_fids], dtype=int))

        df = pd.DataFrame(
            {
                "year": int(year),
                "day": day_col.astype(int),
                "field_id": fid_col.astype(int),
                "A_m": A_m_f.astype(int),
                "A_f": A_f_f.astype(int),
                "A_f_breeding": A_f_breed_f.astype(int),
                "J_free": J_free_f.astype(int),
                "J_tied": J_tied_f.astype(int),
                "new_nests": new_nests_f.astype(int),
                "new_nests_c1": new_nests_c1_f.astype(int),
                "new_nests_c2": new_nests_c2_f.astype(int),
                "new_nests_c3": new_nests_c3_f.astype(int),
                "fledglings": fledglings_f.astype(int),
                "N_total": N_total_f.astype(int),
                "crop": crop_col.astype(str),
                "is_harvest_day": is_harvest.astype(bool),
                "is_insecticide_day": is_insect.astype(bool),
                "is_mowing_day": is_mowing.astype(bool),
                "x": x_f.astype(float),
                "y": y_f.astype(float),
                "area_ha": area_f.astype(float),
                "flower_strip_share": flow_f.astype(float),
                "skylark_plots": sklp_f.astype(int),
                "skylark_plot_area_ha": skla_f.astype(float),
                "refuge_share": ref_f.astype(float),
                "width_m": w_f.astype(float),
                "height_m": h_f.astype(float),
                "perimeter": p_f.astype(float),
                "doy_sin": doy_sin.astype(float),
                "doy_cos": doy_cos.astype(float),
                "in_breeding": in_breeding.astype(bool),
                "aspect_ratio": ar_f.astype(float),
                "edge_density": ed_f.astype(float),
                "compactness": comp_f.astype(float),
                "neighbors_300m": n300_f.astype(int),
                "neighbors_600m": n600_f.astype(int),
                "species": str(species_name),
                "run_id": str(run_id),
            }
        )

        out_rows.append(df)

    out = pd.concat(out_rows, ignore_index=True)

    desired_cols = [
        "year",
        "day",
        "field_id",
        "A_m",
        "A_f",
        "A_f_breeding",
        "J_free",
        "J_tied",
        "new_nests",
        "new_nests_c1",
        "new_nests_c2",
        "new_nests_c3",
        "fledglings",
        "N_total",
        "crop",
        "is_harvest_day",
        "is_insecticide_day",
        "is_mowing_day",
        "x",
        "y",
        "area_ha",
        "flower_strip_share",
        "skylark_plots",
        "skylark_plot_area_ha",
        "refuge_share",
        "width_m",
        "height_m",
        "perimeter",
        "doy_sin",
        "doy_cos",
        "in_breeding",
        "aspect_ratio",
        "edge_density",
        "compactness",
        "neighbors_300m",
        "neighbors_600m",
        "species",
        "run_id",
    ]
    for c in desired_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[desired_cols]


# =========================================================
# save_run_bundle (needed by legacy.kpis)
# =========================================================

def save_run_bundle(
    outdir,
    species_name,
    fields_gdf,
    rotation,
    management,
    hedges_gdf,
    results_by_year,
    day_to_show,
):
    """
    Writes per-run snapshots + per-species outputs.
    Keeps API compatible with legacy.kpis.
    """
    outdir = pathlib.Path(outdir)
    try:
        resolved_outdir = outdir.resolve()
    except Exception:
        resolved_outdir = outdir

    parent_name = resolved_outdir.parent.name if resolved_outdir.parent != resolved_outdir else ""
    run_id = parent_name or resolved_outdir.name

    # Root snapshot: outputs/<run_id>/data
    run_root = outdir.parent
    root_data_dir = run_root / "data"
    root_data_dir.mkdir(parents=True, exist_ok=True)

    root_fields_csv = root_data_dir / "fields_flat.csv"
    root_rotation_csv = root_data_dir / "rotation.csv"
    root_management_csv = root_data_dir / "management.csv"

    if not root_fields_csv.exists():
        try:
            root_fields_flat = fields_gdf.drop(columns=[c for c in ["geometry"] if c in fields_gdf.columns]).copy()
            root_fields_flat.to_csv(root_fields_csv, index=False)
            rotation.to_csv(root_rotation_csv, index=False)
            management.to_csv(root_management_csv, index=False)

            try:
                fields_gdf.to_file(root_data_dir / "fields.geojson", driver="GeoJSON")
            except Exception as e:
                print("[warn] writing root fields.geojson failed:", e)

            if hedges_gdf is not None and len(hedges_gdf) > 0:
                try:
                    hedges_gdf.drop(columns=[c for c in ["geometry"] if c in hedges_gdf.columns]).to_csv(
                        root_data_dir / "hedges.csv", index=False
                    )
                except Exception:
                    pass

            print(f"[data] root snapshot geschrieben: {root_data_dir}")
        except Exception as exc:
            print(f"[warn] root data snapshot failed: {exc}")

    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    fields_flat = fields_gdf.drop(columns=[c for c in ["geometry"] if c in fields_gdf.columns]).copy()

    max_day = int(globals().get("TOT_DAYS", 365))
    meta = {
        "species": species_name,
        "years": list(results_by_year.keys()),
        "day_to_show": int(max(1, min(int(day_to_show), max_day))),
        "runtime": runtime_metadata(),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Per year arrays
    for y, res in results_by_year.items():
        ydir = outdir / f"year_{y}"
        (ydir / "arrays").mkdir(parents=True, exist_ok=True)
        (ydir / "plots").mkdir(parents=True, exist_ok=True)

        np.savez_compressed(ydir / "arrays" / "N_total.npz", N_total=res["N_total"])
        np.savez_compressed(ydir / "arrays" / "A_m.npz", A_m=res["A_m"])
        np.savez_compressed(ydir / "arrays" / "A_f.npz", A_f=res["A_f"])
        np.savez_compressed(ydir / "arrays" / "J_free.npz", J_free=res["J_free"])

        J_tied_arr = res.get("J_tied_total", res.get("J_tied"))
        if J_tied_arr is not None:
            np.savez_compressed(ydir / "arrays" / "J_tied_total.npz", J_tied_total=J_tied_arr)
        if "J_age_series" in res:
            np.savez_compressed(ydir / "arrays" / "J_age_series.npz", J_age_series=res["J_age_series"])
        if "A_f_breeding" in res:
            np.savez_compressed(ydir / "arrays" / "A_f_breeding.npz", A_f_breeding=res["A_f_breeding"])

        usage_df = pd.DataFrame({"field_id": res["fields"]["field_id"].values, "usage": res["usage"]})
        usage_df.to_csv(ydir / "usage.csv", index=False)

        daily_totals = res["N_total"].sum(axis=1)
        pd.DataFrame({"day": np.arange(len(daily_totals)), "total_individuals": daily_totals}).to_csv(
            ydir / "daily_totals.csv", index=False
        )

        if "debug_repro" in res and isinstance(res["debug_repro"], pd.DataFrame):
            res["debug_repro"].to_csv(ydir / "repro_log.csv", index=False)
        if isinstance(res.get("debug"), dict):
            daily_debug = res["debug"].get("daily")
            if isinstance(daily_debug, pd.DataFrame):
                daily_debug.to_csv(ydir / "debug_daily.csv", index=False)

    # birdsim_daily_full.csv
    try:
        daily_full_df = _build_daily_full_df(
            fields_gdf,
            rotation,
            management,
            results_by_year,
            species_name=species_name,
            run_id=run_id,
        )
        if daily_full_df is not None and len(daily_full_df) > 0:
            daily_path = outdir / "birdsim_daily_full.csv"
            daily_full_df.to_csv(daily_path, index=False)
            print(f"[daily] gespeichert: {daily_path}")
    except Exception as exc:
        print(f"[warn] daily full export failed: {exc}")

    # layout export (optional)
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        from .layout import save_fields_layout_csv
        save_fields_layout_csv(fields_flat, analysis_dir, filename="fields_layout_realized.csv")
    except Exception as e:
        print(f"[warn] Layout-Export fehlgeschlagen: {e}")


# =========================================================
# Synthetic landscape helpers
# =========================================================

def add_permanent_cover(
    fields_rects,
    square_poly,
    margin_width_m=6,
    min_patch_area_m2=80,
    fill_residual=True,
):
    union_fields = unary_union(fields_rects)
    out = []

    # rim
    if margin_width_m and margin_width_m > 0:
        rim = union_fields.buffer(margin_width_m).intersection(square_poly).difference(union_fields)
        geoms = [rim] if (not rim.is_empty and rim.geom_type == "Polygon") else (list(rim.geoms) if not rim.is_empty else [])
        out += [(g, "rim") for g in geoms if g.area >= min_patch_area_m2]

    # residual
    if fill_residual:
        residual = square_poly.difference(union_fields)
        geoms = [residual] if (not residual.is_empty and residual.geom_type == "Polygon") else (list(residual.geoms) if not residual.is_empty else [])
        out += [(g, "residual") for g in geoms if g.area >= min_patch_area_m2]

    return out


def make_synthetic_fields(
    n_fields=150,
    seed=42,
    add_cover=True,
    cover_margin_m=6,
    cover_min_patch_m2=80,
    cover_fill_residual=True,
    square_pad_m=8,
):
    """
    Synthetische Felder + optional PermanentCover Patch (gemerged).
    Returns GeoDataFrame.
    """
    rng = np.random.default_rng(seed)

    area = np.clip(np.exp(rng.normal(np.log(3.0), 0.6, size=n_fields)), 0.5, 12.0)
    aspect = np.exp(rng.normal(0.0, 0.5, size=n_fields))
    width_m = np.sqrt(area * 10_000 * aspect)
    height_m = (area * 10_000) / width_m

    df = pd.DataFrame(
        {
            "field_id": np.arange(1, n_fields + 1, dtype=int),
            "area": area.astype(float),
            "width_m": width_m.astype(float),
            "height_m": height_m.astype(float),
        }
    )

    fs_mask = rng.random(n_fields) < 0.4
    df["flower_strip_share"] = np.where(fs_mask, rng.beta(2, 10, size=n_fields) * 0.25, 0.0)
    df["skylark_plots"] = 0
    df["skylark_plot_area"] = 0.0

    from .layout import rect_shelf_pack
    gdf = rect_shelf_pack(df, target_row_width=TARGET_ROW_WIDTH_M)

    if "geometry" not in gdf.columns or gdf.geometry.isna().any():
        geoms = [box(r["x"], r["y"], r["x"] + r["width_m"], r["y"] + r["height_m"]) for _, r in gdf.iterrows()]
        gdf = gpd.GeoDataFrame(gdf, geometry=geoms, crs="EPSG:3857")

    gdf["is_permanent_cover"] = False
    gdf["cover_source"] = ""

    if not add_cover:
        return gdf

    fields_rects = list(gdf.geometry)
    union_fields = unary_union(fields_rects)
    minx, miny, maxx, maxy = union_fields.bounds
    square_poly = box(minx - square_pad_m, miny - square_pad_m, maxx + square_pad_m, maxy + square_pad_m)

    cover_parts = add_permanent_cover(
        fields_rects=fields_rects,
        square_poly=square_poly,
        margin_width_m=cover_margin_m,
        min_patch_area_m2=cover_min_patch_m2,
        fill_residual=cover_fill_residual,
    )

    if cover_parts:
        start_id = int(gdf["field_id"].max()) + 1
        polys = [poly for (poly, _src) in cover_parts]
        merged = unary_union(polys)

        minx, miny, maxx, maxy = merged.bounds
        w = maxx - minx
        h = maxy - miny

        cover_row = {
            "field_id": start_id,
            "area": merged.area / 10_000.0,
            "width_m": w,
            "height_m": h,
            "flower_strip_share": 0.0,
            "skylark_plots": 0,
            "skylark_plot_area": 0.0,
            "x": minx,
            "y": miny,
            "perimeter": merged.length,
            "is_permanent_cover": True,
            "cover_source": "merged",
        }
        cover_gdf = gpd.GeoDataFrame(pd.DataFrame([cover_row]), geometry=[merged], crs=gdf.crs)
        gdf = pd.concat([gdf, cover_gdf], ignore_index=True)

    share_cover = gdf.loc[gdf["is_permanent_cover"], "area"].sum() / gdf["area"].sum()
    print(
        f"[make_synthetic_fields] PermanentCover: {share_cover:.2%} der Fläche | "
        f"Patches={gdf['is_permanent_cover'].sum()} (rim+residual)"
    )
    return gdf


def make_rotation(fields_gdf, years, seed=43, crop_mix=None):
    """Arable rotates; PermanentCover stays constant."""
    rng = np.random.default_rng(seed)
    if crop_mix is None:
        crop_mix = {
            "Winterweizen": 0.36,
            "Sommergerste": 0.22,
            "Mais": 0.17,
            "Raps": 0.11,
            "Zuckerrübe": 0.10,
            "Ackerbohne": 0.04,
        }

    crops = list(crop_mix.keys())
    probs = np.array(list(crop_mix.values()), float)
    probs /= probs.sum()

    mask_cover = fields_gdf.get("is_permanent_cover", pd.Series(False, index=fields_gdf.index)).astype(bool)
    fids_cover = fields_gdf.loc[mask_cover, "field_id"].astype(int).to_numpy()
    fids_arable = fields_gdf.loc[~mask_cover, "field_id"].astype(int).to_numpy()

    rows = []
    for y in years:
        rows.extend({"year": int(y), "field_id": int(fid), "crop": "PermanentCover"} for fid in fids_cover)

    if len(fids_arable) > 0:
        prev = rng.choice(crops, size=len(fids_arable), p=probs)
        for y in years:
            cur = prev.copy()
            mask = rng.random(len(fids_arable)) < 0.6
            cur[mask] = rng.choice(crops, size=int(mask.sum()), p=probs)
            same = np.where(cur == prev)[0]
            for i in same:
                choices = [c for c in crops if c != prev[i]]
                cur[i] = rng.choice(choices)
            rows.extend({"year": int(y), "field_id": int(fid), "crop": str(c)} for fid, c in zip(fids_arable, cur))
            prev = cur

    return pd.DataFrame(rows)


def _read_crop_calendar(path: str | Path) -> dict:
    """Read crop calendar csv into dict[crop] -> row-dict."""
    path = str(path)
    if path.strip() == "":
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Crop calendar CSV not found: {path}")

    df = pd.read_csv(path)
    crop_col = None
    for c in df.columns:
        if c.lower().strip() == "crop":
            crop_col = c
            break
    if crop_col is None:
        raise ValueError(f"Crop calendar CSV must contain a 'crop' column. Got: {list(df.columns)}")

    out = {}
    for _, row in df.iterrows():
        crop = str(row[crop_col]).strip()
        if crop == "" or crop.lower() in {"nan", "none"}:
            continue
        d = {}
        for c in df.columns:
            if c == crop_col:
                continue
            v = row[c]
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            d[c] = v
        out[crop] = d
    return out


def make_management(
    fields_gdf,
    rotation_df,
    seed=44,
    use_calendar: bool = True,
    calendar_path: str | Path = CROP_CALENDAR_CSV,
    desync_region_sd: float = 4.0,
    desync_field_sd: float = 3.0,
    region_bins: int = 6,
):
    """Create per field/year management schedule (harvest/insect/windows)."""
    rng = np.random.default_rng(seed)

    XY = fields_gdf[["x", "y"]].to_numpy(float)
    if region_bins < 1:
        region_bins = 1

    x_min, y_min = XY.min(axis=0)
    x_max, y_max = XY.max(axis=0)
    bx = max((x_max - x_min) / region_bins, 1.0)
    by = max((y_max - y_min) / region_bins, 1.0)

    def _region_id(x, y):
        ix = int((x - x_min) // bx)
        iy = int((y - y_min) // by)
        ix = min(max(ix, 0), region_bins - 1)
        iy = min(max(iy, 0), region_bins - 1)
        return iy * region_bins + ix

    region_by_fid = {
        int(fid): _region_id(float(x), float(y))
        for fid, x, y in fields_gdf[["field_id", "x", "y"]].itertuples(index=False)
    }
    unique_regions = np.unique(list(region_by_fid.values()))
    region_offset = {rid: rng.normal(0.0, desync_region_sd) for rid in unique_regions}

    cal = _read_crop_calendar(calendar_path) if use_calendar else {}

    rows = []
    for _, r in rotation_df.iterrows():
        crop = str(r["crop"])
        fid = int(r["field_id"])
        y = int(r["year"])

        if crop == "PermanentCover":
            rows.append(
                dict(
                    year=y,
                    field_id=fid,
                    mow_dates="",
                    harvest_date=None,
                    insecticide_date=None,
                    sow_win=None,
                    harvest_win=None,
                    other_ops_win=None,
                )
            )
            continue

        rid = int(region_by_fid.get(fid, int(rng.integers(0, max(1, len(unique_regions))))))
        off_reg = float(region_offset.get(rid, 0.0))
        off_field = float(rng.normal(0.0, desync_field_sd))

        harv = None
        insect = None
        sow_win = None
        harv_win = None
        other_win = None

        if crop in cal:
            c = cal[crop]
            hc = c.get("harvest_center", None)
            if hc is not None:
                base = rng.normal(float(hc), float(c.get("harvest_sd", 0.0) or 0.0))
                hw = c.get("harvest_width", None)
                if hw is not None:
                    a = base - float(hw)
                    b = base + float(hw)
                    harv = int(np.clip(rng.uniform(a, b) + off_reg + off_field, 1, 365))
                else:
                    harv = int(np.clip(base + off_reg + off_field, 1, 365))

                span = int(c.get("harvest_win_span", 10) or 10)
                harv_win = f"{max(1, harv - span//2)}-{min(365, harv + span//2)}"

            sc = c.get("sow_center", None)
            sw = c.get("sow_width", None)
            if sc is not None and sw is not None:
                a = float(sc) - float(sw) + off_reg + off_field
                b = float(sc) + float(sw) + off_reg + off_field
                aa = int(np.clip(min(a, b), 1, 365))
                bb = int(np.clip(max(a, b), 1, 365))
                sow_win = f"{aa}-{bb}"

            if float(c.get("other_prob", 0.0) or 0.0) > 0 and rng.random() < float(c.get("other_prob", 0.0)):
                oc = c.get("other_center", None)
                ow = c.get("other_width", None)
                if oc is not None and ow is not None:
                    a = float(oc) - float(ow) + off_reg + off_field
                    b = float(oc) + float(ow) + off_reg + off_field
                    aa = int(np.clip(min(a, b), 1, 365))
                    bb = int(np.clip(max(a, b), 1, 365))
                    other_win = f"{aa}-{bb}"

            if float(c.get("insect_prob", 0.0) or 0.0) > 0 and rng.random() < float(c.get("insect_prob", 0.0)):
                ic = c.get("insect_center", None)
                iw = c.get("insect_width", None)
                if ic is not None:
                    if iw is not None:
                        insect = int(np.clip(rng.uniform(float(ic) - float(iw), float(ic) + float(iw)) + off_reg + off_field, 1, 365))
                    else:
                        insect = int(np.clip(float(ic) + off_reg + off_field, 1, 365))
        else:
            base = None
            if crop == "Winterweizen":
                base = rng.normal(210, 7)
            elif crop == "Sommergerste":
                base = rng.normal(190, 6)
            elif crop == "Mais":
                base = rng.normal(250, 10)
            elif crop == "Raps":
                base = rng.normal(195, 7)
            elif crop in ("Zuckerrübe", "Zuckerruebe"):
                base = rng.normal(260, 10)
            elif crop == "Ackerbohne":
                base = rng.normal(200, 7)

            if base is not None:
                harv = int(np.clip(base + off_reg + off_field, 1, 365))
                harv_win = f"{max(1, harv - 5)}-{min(365, harv + 5)}"

            if crop in ("Winterweizen", "Sommergerste"):
                a = np.clip(rng.normal(100, 8) + off_reg + off_field, 70, 130)
                b = np.clip(rng.normal(125, 8) + off_reg + off_field, 95, 160)
                sow_win = f"{int(min(a, b))}-{int(max(a, b))}"

            if rng.random() < 0.4:
                a = np.clip(rng.normal(140, 10) + off_reg + off_field, 110, 190)
                b = np.clip(rng.normal(155, 10) + off_reg + off_field, 120, 210)
                other_win = f"{int(min(a, b))}-{int(max(a, b))}"

            if crop in ("Winterweizen", "Sommergerste", "Raps") and rng.random() < 0.3:
                insect = int(np.clip(rng.normal(150, 7) + off_reg + off_field, 100, 220))

        rows.append(
            dict(
                year=y,
                field_id=fid,
                mow_dates="",
                harvest_date=harv,
                insecticide_date=insect,
                sow_win=sow_win,
                harvest_win=harv_win,
                other_ops_win=other_win,
            )
        )

    return pd.DataFrame(rows)


def make_random_hedges(fields_gdf, p_edge=0.7, seed=45):
    """Propose hedge pairs between near neighbors."""
    rng = np.random.default_rng(seed)
    fids = fields_gdf["field_id"].astype(int).to_numpy()
    XY = fields_gdf[["x", "y"]].to_numpy(float)

    pairs = set()
    for i in range(len(fids)):
        d = np.sqrt(((XY - XY[i]) ** 2).sum(axis=1))
        order = np.argsort(d)
        for j in order[1:6]:
            a, b = sorted((int(fids[i]), int(fids[j])))
            if rng.random() < p_edge:
                pairs.add((a, b))

    return pd.DataFrame(
        [{"f1": a, "f2": b, "hedge_quality": float(rng.uniform(0.4, 0.9))} for (a, b) in pairs]
    )


# =========================================================
# Cross-year feedback helper used by legacy.kpis
# =========================================================

def _balanced_half_split(
    counts: np.ndarray, *, year: int, ensure_both_if_total_ge2: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(counts if counts is not None else [], dtype=int)
    if c.ndim == 0:
        c = np.array([int(c)], dtype=int)
    c = np.clip(c, 0, None)

    tot = int(c.sum())
    if tot == 0:
        return np.zeros_like(c), np.zeros_like(c)

    target_m = tot // 2
    if ensure_both_if_total_ge2 and tot >= 2:
        target_m = max(1, target_m)

    m = c // 2
    rem = (c & 1).astype(bool)
    need = int(target_m - m.sum())
    if need > 0:
        idx = np.flatnonzero(rem)
        if idx.size > 0:
            start_idx = int(year) % idx.size
            give = np.concatenate([idx[start_idx:], idx[:start_idx]])[:need]
            m[give] += 1

    f = c - m

    if tot == 1:
        mi = int(m.sum())
        fi = int(f.sum())
        if mi == 0 and fi == 1 and (year % 2) == 0:
            j = int(np.argmax(f))
            f[j] -= 1
            m[j] += 1
        elif fi == 0 and mi == 1 and (year % 2) == 1:
            j = int(np.argmax(m))
            m[j] -= 1
            f[j] += 1

    if int(m.sum()) == 0 and int(f.sum()) > 0:
        j = int(np.argmax(f))
        f[j] -= 1
        m[j] += 1
    elif int(f.sum()) == 0 and int(m.sum()) > 0:
        j = int(np.argmax(m))
        m[j] -= 1
        f[j] += 1

    return m.astype(int), f.astype(int)


def run_years_with_feedback(
    fields_gdf,
    rotation,
    management,
    cfg,
    years: list[int],
    N_init_first: int = 100,
    outdir: Optional[Path] = None,
    extra_filename: Optional[str] = None,
    weather_module=None,
    timeline_start_date=None,
):
    """
    Multi-year simulation with winter redistribution feedback.
    Keeps API compatible with legacy.kpis usage.
    """
    from .simulate import BirdLandscapeSim

    fields_gdf = fields_gdf.copy()
    for col, default in [
        ("hedge_length_m", 0.0),
        ("hedge_quality_weighted_m", 0.0),
        ("hedge_quality_mean", 0.0),
        ("hedge_length_m_per_ha", 0.0),
        ("hedge_quality_weighted_m_per_ha", 0.0),
    ]:
        if col not in fields_gdf.columns:
            fields_gdf[col] = default

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
                "hedge_length_m",
                "hedge_quality_weighted_m",
                "hedge_quality_mean",
                "hedge_length_m_per_ha",
                "hedge_quality_weighted_m_per_ha",
                "width_m",
                "height_m",
            ]
        ],
        rotation,
        management,
        cfg,
    )
    rng_local = sim.rng

    results_by_year: Dict[int, dict] = {}
    start_state = None

    juv_qc_rows: List[dict] = []

    for year_idx, y in enumerate(years):
        res = sim.simulate(
            days=365,
            year=int(y),
            N_init=(0 if start_state is not None else N_init_first),
            start_state=start_state,
            debug_days=None,
            log_repro=False,
            keep_age_series=False,
            weather_module=weather_module,
            timeline_start_date=timeline_start_date,
            timeline_day_offset=int(year_idx) * 365,
        )
        results_by_year[int(y)] = res

        if not USE_YEAR_FEEDBACK:
            start_state = None
            continue

        T = res["A_m"].shape[0]
        dep_idx = int(np.clip(int(getattr(cfg, "departure_day", 365)), 1, T)) - 1

        A_m_dep = res["A_m"][dep_idx, :].astype(int)
        A_f_dep = res["A_f"][dep_idx, :].astype(int)

        J_tied_mat = res.get("J_tied_total", res.get("J_tied"))
        if J_tied_mat is None:
            J_tied_mat = np.zeros_like(res["A_m"])

        J_dep = (J_tied_mat[dep_idx, :].astype(int) + res["J_free"][dep_idx, :].astype(int))

        juv_mat_rate = float(getattr(cfg, "juvenile_maturation_rate", JUV_MATURATION_RATE))
        winter_surv_ad = float(getattr(cfg, "winter_survival_adult", WINTER_SURVIVAL_ADULT))
        winter_surv_juv = float(getattr(cfg, "winter_survival_juvenile", WINTER_SURVIVAL_JUV))

        J_mature = rng_local.binomial(J_dep, juv_mat_rate).astype(int)
        add_m_pre, add_f_pre = _balanced_half_split(J_mature, year=int(y))

        J_nonmature = (J_dep - J_mature).astype(int)
        J_nonmature_surv = rng_local.binomial(J_nonmature, winter_surv_juv).astype(int)
        add_m_spring, add_f_spring = _balanced_half_split(J_nonmature_surv, year=int(y))

        pool_tot = (A_m_dep + A_f_dep + add_m_pre + add_f_pre).astype(int)
        surv_tot = rng_local.binomial(pool_tot, winter_surv_ad).astype(int)
        surv_m_ad, surv_f_ad = _balanced_half_split(surv_tot, year=int(y))

        surv_m_total = (surv_m_ad + add_m_spring).astype(int)
        surv_f_total = (surv_f_ad + add_f_spring).astype(int)

        m_next = sim._winter_redistribute(surv_m_total, year=int(y)).astype(int)
        f_next = sim._winter_redistribute(surv_f_total, year=int(y)).astype(int)

        start_state = {"A_m0": m_next, "A_f0": f_next}

        # minimal QC logging (optional)
        juv_qc_rows.append(
            dict(
                year=int(y),
                tot_J_dep=int(J_dep.sum()),
                tot_start=int(m_next.sum() + f_next.sum()),
            )
        )

    # optional export
    if outdir is not None:
        species_name = str(getattr(cfg, "name", "species"))
        base_dir = Path(outdir)
        target_dir = (base_dir if base_dir.name.lower() == species_name.lower() else base_dir / species_name) / "analysis"
        target_dir.mkdir(parents=True, exist_ok=True)
        suf = f"_{extra_filename}" if (extra_filename and str(extra_filename).strip()) else ""
        pd.DataFrame(juv_qc_rows).to_csv(target_dir / f"juvenile_transition_qc{suf}.csv", index=False)

    return results_by_year
