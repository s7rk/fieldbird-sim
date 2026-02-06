from __future__ import annotations

# =========================================================
#                         BirdSIM                         #
# =========================================================
#  We simulate birds across many farm fields over multiple years.
#  Each day, every field gets a habitat score from the crop’s food & cover
#  and any disturbance (harvest/mowing/insecticide pulses and operation windows).
#  Birds move toward better fields (breeding females stay on nests).
#  During the breeding window, pairs start nests; some nests go into
#  skylark windows which shield them from pulse disturbance and improve
#  survival. Nests can fail or fledge; juveniles age, then roam.
#  Crowding reduces breeding success and we apply a soft carrying capacity.
#  At the end of the season, birds survive winter and redistribute,
#  seeding the next year. We can test management scenarios
#  (later harvest, more flower strips or skylark plots)
#  and export tidy summaries and maps.
# =========================================================

# ------------------- Import area -------------------

# Standard library
import csv
import inspect
import json
import math
import os
import pathlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# Third-party
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

# ------------------- Global controls -------------------

TOT_DAYS = 365  # zentrale Referenz für Jahr-Länge (Index = DOY)
DEBUG_YEAR_XFER=1
# === (1) Quick-Options für die Simulation – HIER meiste Einstellungen vornehmen ===
OPTS = SimpleNamespace(
    N_FIELDS=20,        # Anzahl synthetischer Felder (wirksam nur bei USE_SYNTHETIC=True)
    N_YEARS=20,        # Anzahl zu simulierender Jahre
    YEAR_TO_SHOW=1,     # Jahr für Beispiel-Karten/Plots
    DAY_TO_SHOW=160,    # Tag (DOY) für Tages-Karte
    # Reproduzierbarkeit: Seeds für Synthese/Rotation/Management/Hecken + Simulation
    SEED_FIELDS=2025,
    SEED_ROTATION=2025,
    SEED_MANAGEMENT=2025,
    SEED_HEDGES=2025,
    SIM_RNG_SEED=2025,
)
# multiple runs
def _env_int(name, default):
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v != "" else int(default)
    except Exception:
        return int(default)

# Seeds aus ENV erlauben (fallback = bisherige OPTS)
OPTS.SEED_FIELDS   = _env_int("SEED_FIELDS",   OPTS.SEED_FIELDS)
OPTS.SEED_ROTATION = _env_int("SEED_ROTATION", OPTS.SEED_ROTATION)
OPTS.SEED_MANAGEMENT = _env_int("SEED_MANAGEMENT", OPTS.SEED_MANAGEMENT)
OPTS.SEED_HEDGES   = _env_int("SEED_HEDGES",   OPTS.SEED_HEDGES)
OPTS.SIM_RNG_SEED  = _env_int("SIM_SEED",      OPTS.SIM_RNG_SEED)

OPTS.N_FIELDS     = _env_int("N_FIELDS",     OPTS.N_FIELDS)
OPTS.N_YEARS      = _env_int("N_YEARS",      OPTS.N_YEARS)
OPTS.YEAR_TO_SHOW = _env_int("YEAR_TO_SHOW", OPTS.YEAR_TO_SHOW)
OPTS.DAY_TO_SHOW  = _env_int("DAY_TO_SHOW",  OPTS.DAY_TO_SHOW)

# optional:
DO_MAPS     = bool(int(os.environ.get("DO_MAPS", "1")))
DO_HEATMAPS = bool(int(os.environ.get("DO_HEATMAPS", "1")))

# (optional) Szenario-spezifische Seeds erlauben, falls genutzt
try:
    SCENARIO_B_SEED = _env_int("SCENARIO_B_SEED", SCENARIO_B_SEED)
except NameError:
    pass
try:
    SCENARIO_C_SEED = _env_int("SCENARIO_C_SEED", SCENARIO_C_SEED)
except NameError:
    pass

# Sanity (früh scheitern statt leise falsche Ergebnisse)
#assert int(OPTS.N_YEARS) >= 1, "OPTS.N_YEARS muss >= 1 sein"

# === Werte, die aus OPTS abgeleitet werden – NICHT ändern ===
YEARS_TO_SIM = list(range(1, int(OPTS.N_YEARS) + 1))
YEAR_TO_SHOW = max(1, min(int(OPTS.YEAR_TO_SHOW), int(OPTS.N_YEARS)))
DAY_TO_SHOW  = max(1, min(int(OPTS.DAY_TO_SHOW), TOT_DAYS))  # <— clamp auf [1..365]
SPECIES_LIST = [
    "Feldlerche",
    "Rebhuhn"]  # weitere Arten hier ergänzen (Strings)

# === Feldlayout (synthetische Geometrien) ===
USE_RECT_PACKING   = True   # True: Rechteck-Packlayout; False: einfache Zentroid-Quadrate
TARGET_ROW_WIDTH_M = None   # None => automatisch aus Feldgrößen ableiten

# === Hecken-Darstellung/Quelle ===
USE_HEDGES_FROM_CSV = True  # True: Hecken aus CSV (HEDGES_CSV) lesen
AUTO_HEDGES_FROM_ADJ = False# True: sehr einfache Auto-Nachbarschaft als Hecken
DRAW_HEDGE_BUFFERS   = True # True: Pufferflächen um Hecken visualisieren
HEDGE_BUFFER_M       = 5.0  # Pufferbreite in Metern

# === Lerchenfenster/Refugien (tuning, biologisch plausible Defaults) ===
ELIGIBLE_REFUGE_CROPS = {"Winterweizen", "Sommergerste"}  # Kulturen, in denen Fenster wirken
REFUGE_ASSIGN_MULT    = 15.0  # Stärke, mit der neue Nester Fenster bevorzugen (vs. reine Flächenquote)
PULSE_PROTECT_SCALE   = 4.0   # wie stark refuge_share Puls-Störungen abschirmt
REFUGE_SHARE_CAP      = 0.6   # max. Anteil Nester in Fenstern (Kappung)

# === Rein visuelle Overlays ===
DRAW_FLOWER_RINGS = True  # True: Blühstreifen als Randringe anzeigen

# === Dateipfade (falls reale Inputs; sonst Fallback auf Templates) ===
FIELDS_CSV           = "inputs/fields_template.csv"
ROTATION_CSV_MAIN    = "inputs/rotation.csv"
ROTATION_CSV_TPL     = "inputs/rotation_template.csv"
MANAGEMENT_CSV_MAIN  = "inputs/management.csv"
MANAGEMENT_CSV_TPL   = "inputs/management_template.csv"
HEDGES_CSV           = "inputs/hedges_template.csv"
SKYLARK_WINDOWS_CSV  = Path("configs/skylark_windows.csv")
CROP_CALENDAR_CSV    = "configs/crop_calendar.csv"

# === Szenarien (Beispiele; werden im Hauptteil angewendet) ===
# A: spätere Ernte (einzelner Wert ODER Liste von Werten in Tagen)
SCENARIO_A_ENABLED = True
SCENARIO_A_SHIFTS = [7, 14, 21]   # auch erlaubt: 21  oder  [21]
SCENARIO_A = {"shifts": SCENARIO_A_SHIFTS}


# B: Blühstreifen – mehrere Intensitäten/Feldanteile + gezielte Auswahl
SCENARIO_B_ENABLED = True
SCENARIO_B_SHARES = [0.10, 0.20, 0.30]   # additive Erhöhung je ausgewähltem Feld (0..0.5); auch scalar möglich
SCENARIO_B_FRACTIONS = [0.25, 0.50, 1.00]  # Anteil der Felder, die Blühstreifen erhalten; auch scalar möglich
SCENARIO_B_SELECT = "no_flower_first"      # "random" | "no_flower_first"
SCENARIO_B_SEED = 2025

# --- NEU: C – Brache-Anteil an Gesamtfläche (statt in Baseline) ---
SCENARIO_C_ENABLED = True  # Master-Toggle für „Brache“-Szenarien
SCENARIO_C_FALLOW_AREA_SHARES = [0.01, 0.05, 0.10]  # 1%, 5%, 10% der Gesamtfläche
SCENARIO_C_APPLY_YEARS = "all"  # "all" oder Liste z. B. [1,2,3]
SCENARIO_C_SELECTION = "largest"  # "largest" (größte Felder zuerst) oder "weighted_random"
SCENARIO_C_SEED = 2026  # für weighted_random
SCENARIO_C_SELECTION = "greedy"  # statt "largest" oder "weighted_random"


# D – gezielte Lerchenfenster (empfohlen)
# Info (Kurzfassung):
#  - Mindestgröße je Fenster: ≥ 20 m²
#  - empfohlene Dichte: 2–10 Fenster/ha
#  - sinnvoll in Getreide; bei (Winter)gerste i. d. R. weniger geeignet
#  - Mindestabstände zu Ortschaften/Strukturen beachten
#  - ökologischer Nutzen steigt mit Extensivierung/Blühstreifen-Nachbarschaft
SCENARIO_D_ENABLED = True  # <— Szenario D global an-/abschalten
SCENARIO_D = dict(
    add_skylark_plots=2,      # Fensterzahl je Ziel-Feld
    skylark_plot_area=0.02,   # ha pro Fenster (0.02 ha = 200 m²; konsistent zu ≥20 m²)
    target_fraction=0.20,     # Top-Anteil der Felder nach Eignung (Rotation/ohne Blühstreifen)
    min_eligible_years=1,     # mind. so viele Jahre mit „eligible crops“
    only_without_flower=True  # Felder ohne Blühstreifen bevorzugen
)
# === (10) Cross-year Feedback & Winter (Überwinterung, Umverteilung) ===
USE_YEAR_FEEDBACK     = True   # True: Jahresende -> Winter -> Startbestand Folgejahr
WINTER_SURVIVAL_ADULT = 0.95   # adulte tägliche/umgerechnete Überlebensrate im Winterpfad
WINTER_SURVIVAL_JUV   = 0.90   # juvenile Überlebensrate über Winter
JUV_MATURATION_RATE   = 0.90   # Anteil spätjähriger Juveniler, die vor Winter adult werden
JUV_EQUIV_IN_CAPACITY = 0.5    # 0..1 Anteil, wie stark Juvenile zur Kapazitätslast zählen
WINDOW_STRESS_WEIGHT  = 0.5    # Gewicht chronischer Fenster-Störung relativ zu Pulsen
# Winterbewegung (Umverteilung Überwinterer)
WINTER_MOVE_RADIUS    = 12.0   # Distanzradius (in 100 m Einheiten)
WINTER_BETA           = 2.5    # Schärfe der Wahl (softmax)
WINTER_DISTANCE_COST  = 0.08   # Distanzkosten in der Winterpräferenz
WINTER_EVAL_DAYS      = (330, 20)  # Tage zur Habitatbewertung (Spätherbst, Frühling)

# === (11) Tragfähigkeit (weiche Kapazität) ===
USE_CAPACITY_SOFT = True
K_PER_HA_ADULTS   = 40.0  # max. *adulte* pro ha, ab der Strafe greift
CAPACITY_GAMMA    = 0.4   # Stärke der Nutzenstrafe, wenn über K

# === (12) Weitere globale Schalter ===
# Dichtefaktor bei der Reproduktion berücksichtigen (wir verdrahten das unten in simulate)
USE_DENSITY_FACTOR = True
# =========================================================
#           Output directory & plot saving
# =========================================================

def _mk_outdir(prefix="outputs", *, name: Optional[str] = None) -> pathlib.Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # NEW: optionaler Tag aus ENV (z.B. rep_001)
    tag = os.environ.get("RUN_TAG", "").strip()
    if tag:
        ts = f"{ts}_{tag}"
    outdir = pathlib.Path(prefix) / (name or ts)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _save_fig(path, tight: bool = True, dpi: int = 180, transparent: bool = False) -> None:
    """
    Speichert die aktuelle Figure robust:
      - legt Zielordner an
      - nutzt tight bbox (falls gewünscht) mit kleiner padding
      - schließt explizit die Figure (kein globales plt.close())
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf()
    bbox = None
    pad_inches = None

    if tight:
        try:
            fig.tight_layout()
        except Exception:
            # Tight-Layout kann selten bei komplexen Artists fehlschlagen -> ignorieren
            pass
        bbox = "tight"
        pad_inches = 0.02

    try:
        fig.savefig(p, dpi=dpi, bbox_inches=bbox, pad_inches=pad_inches, transparent=transparent)
    finally:
        plt.close(fig)

def _build_daily_full_df(fields_gdf: pd.DataFrame,
                         rotation_df: pd.DataFrame,
                         management_df: pd.DataFrame,
                         results_by_year: dict,
                         species_name: str,
                         run_id: str) -> Optional[pd.DataFrame]:
    """
    Baut die Tabelle 'birdsim_daily_full' im Long-Format (year, day, field_id, ...).

    Spalten (genau in dieser Reihenfolge):
      year,day,field_id,A_m,A_f,A_f_breeding,J_free,J_tied,new_nests,fledglings,N_total,
      crop,is_harvest_day,is_insecticide_day,is_mowing_day,
      x,y,area_ha,flower_strip_share,skylark_plots,skylark_plot_area_ha,refuge_share,
      width_m,height_m,perimeter,doy_sin,doy_cos,in_breeding,
      aspect_ratio,edge_density,compactness,neighbors_300m,neighbors_600m,
      species,run_id
    """
    import numpy as np
    import pandas as pd
    import math

    if not results_by_year:
        return None

    # ---------------- Static field attributes (Lookup je field_id) ----------------
    fields_df = fields_gdf.copy()
    if "field_id" not in fields_df.columns:
        raise ValueError("fields_gdf must contain 'field_id'")

    fields_df["field_id"] = pd.to_numeric(fields_df["field_id"], errors="coerce").astype("Int64")

    # x/y auffüllen (falls Geometrie vorhanden)
    if "x" not in fields_df.columns or "y" not in fields_df.columns:
        if "geometry" in fields_df.columns:
            try:
                cent = fields_df.geometry.centroid
                if "x" not in fields_df.columns: fields_df["x"] = cent.x
                if "y" not in fields_df.columns: fields_df["y"] = cent.y
            except Exception:
                fields_df["x"] = fields_df.get("x", np.nan)
                fields_df["y"] = fields_df.get("y", np.nan)
        else:
            fields_df["x"] = fields_df.get("x", np.nan)
            fields_df["y"] = fields_df.get("y", np.nan)

    # Area (ha)
    if "area_ha" in fields_df.columns:
        fields_df["area_ha"] = pd.to_numeric(fields_df["area_ha"], errors="coerce")
    elif "area" in fields_df.columns:
        fields_df["area_ha"] = pd.to_numeric(fields_df["area"], errors="coerce")
    else:
        fields_df["area_ha"] = np.nan
    fields_df["area_ha"] = fields_df["area_ha"].astype(float)

    # Flower / Skylark
    fields_df["flower_strip_share"]   = pd.to_numeric(fields_df.get("flower_strip_share", 0.0), errors="coerce").fillna(0.0)
    fields_df["skylark_plots"]        = pd.to_numeric(fields_df.get("skylark_plots", 0), errors="coerce").fillna(0).astype(int)
    if "skylark_plot_area_ha" in fields_df.columns:
        fields_df["skylark_plot_area_ha"] = pd.to_numeric(fields_df["skylark_plot_area_ha"], errors="coerce").fillna(0.0)
    elif "skylark_plot_area" in fields_df.columns:
        fields_df["skylark_plot_area_ha"] = pd.to_numeric(fields_df["skylark_plot_area"], errors="coerce").fillna(0.0)
    else:
        fields_df["skylark_plot_area_ha"] = 0.0

    # Perimeter / width / height
    fields_df["width_m"]  = pd.to_numeric(fields_df.get("width_m", np.nan),  errors="coerce")
    fields_df["height_m"] = pd.to_numeric(fields_df.get("height_m", np.nan), errors="coerce")
    fields_df["perimeter"] = pd.to_numeric(fields_df.get("perimeter", np.nan), errors="coerce")

    # Refuge share (falls nicht vorhanden: aus Skylark-Fenstern approximieren)
    if "refuge_share" in fields_df.columns:
        fields_df["refuge_share"] = pd.to_numeric(fields_df["refuge_share"], errors="coerce").clip(0.0, 1.0)
    else:
        area_vals = fields_df["area_ha"].to_numpy(float)
        plots     = fields_df["skylark_plots"].to_numpy(float)
        plot_area = fields_df["skylark_plot_area_ha"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            refuge = np.where(area_vals > 0, (plots * plot_area) / area_vals, 0.0)
        fields_df["refuge_share"] = np.clip(refuge, 0.0, 1.0)

    fields_lookup = fields_df.set_index("field_id")

    # ---------------- Rotation & Management Maps ----------------
    rotation_df = rotation_df.copy()
    if "year" not in rotation_df.columns:
        raise ValueError("rotation_df must contain 'year'")
    rotation_df["year"]     = pd.to_numeric(rotation_df["year"], errors="coerce").astype("Int64")
    rotation_df["field_id"] = pd.to_numeric(rotation_df["field_id"], errors="coerce").astype("Int64")

    management_df = management_df.copy()
    if len(management_df):
        if {"year","field_id"} - set(management_df.columns):
            raise ValueError("management_df must contain 'year' and 'field_id'")
        management_df["year"]     = pd.to_numeric(management_df["year"], errors="coerce").astype("Int64")
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
            insect_day  = _clean_day(row.get("insecticide_date", None))
            mow_arr     = _parse_days(row.get("mow_dates", None))
            mow_set     = set()
            for d in mow_arr.tolist():
                for dd in (d-1, d, d+1):   # ±1 Puffer
                    if 1 <= dd <= 365:
                        mow_set.add(int(dd))

            out[fid] = {"harvest": harvest_day, "insecticide": insect_day, "mowing_days": mow_set}
        return out

    # ---------------- Helper: Nachbarn pro (Teil-)Landschaft ----------------
    def _neighbor_counts(xy: np.ndarray, fids: np.ndarray, thr_m: float) -> dict[int, int]:
        n = len(fids)
        if n == 0:
            return {}
        # Distanzmatrix (O(n^2) – hier ok)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        D = np.sqrt(dx*dx + dy*dy)
        # exclude self
        D[np.arange(n), np.arange(n)] = np.inf
        counts = (D <= float(thr_m)).sum(axis=1).astype(int)
        return {int(fids[i]): int(counts[i]) for i in range(n)}

    # ---------------- Sammeln pro Jahr ----------------
    out_rows = []
    years_sorted = sorted(results_by_year.keys())
    for year in years_sorted:
        res = results_by_year[year]
        # Arrays (days x fields)
        A_m = res.get("A_m")
        A_f = res.get("A_f")
        J_free = res.get("J_free")
        # A_f_breeding (optional)
        A_f_breeding = res.get("A_f_breeding")
        if A_f_breeding is None and A_f is not None:
            A_f_breeding = np.zeros_like(A_f, dtype=int)
        # J_tied (optional; im Modell oft 0, wird über Kohorten abgebildet)
        J_tied = res.get("J_tied_total", res.get("J_tied"))
        if J_tied is None and A_f is not None:
            J_tied = np.zeros_like(A_f, dtype=int)
        new_nests   = res.get("daily_new_nests")
        fledglings  = res.get("daily_fledglings")
        N_total     = res.get("N_total")

        new_nests_c1 = res.get("daily_new_nests_c1")
        new_nests_c2 = res.get("daily_new_nests_c2")
        new_nests_c3 = res.get("daily_new_nests_c3")

        # Absicherung: N_total berechnen, falls fehlt
        if N_total is None and all(v is not None for v in (A_m, A_f, J_free, J_tied)):
            N_total = (A_m.astype(int) + A_f.astype(int) + J_free.astype(int) + J_tied.astype(int))

        # Pflicht: A_m/A_f/J_free und N_total sollten da sein
        for nm in ("A_m", "A_f", "J_free", "N_total"):
            if locals()[nm] is None:
                raise ValueError(f"results_by_year[{year!r}] fehlt '{nm}' – Export nicht möglich.")

        days, nfields = A_m.shape
        # field_id-Reihenfolge aus res (maßgeblich für Arrays)
        if "fields" in res and "field_id" in res["fields"].columns:
            res_fids = res["fields"]["field_id"].astype(int).to_numpy()
        else:
            # Fallback: aus fields_lookup aufsteigend
            res_fids = fields_lookup.index.dropna().astype(int).sort_values().to_numpy()
            if len(res_fids) != nfields:
                raise ValueError(f"Kann field_id-Reihenfolge nicht sicher bestimmen (year={year}).")

        # Crop je Feld (dieses Jahr)
        rot_y = rotation_df.loc[rotation_df["year"] == int(year)]
        crop_map = {int(r.field_id): str(r.crop) for r in rot_y.itertuples(index=False)}
        crops_vec = np.array([crop_map.get(int(fid), "Unbekannt") for fid in res_fids], dtype=object)

        # Management-Flags Map
        mg_map = _build_management_map(int(year))

        # Nachbarn (300/600m) und statische Feldmerkmale je Feld hier einmal berechnen
        XY = np.c_[fields_lookup.loc[res_fids, "x"].to_numpy(float),
                   fields_lookup.loc[res_fids, "y"].to_numpy(float)]
        neigh300 = _neighbor_counts(XY, res_fids, thr_m=300.0)
        neigh600 = _neighbor_counts(XY, res_fids, thr_m=600.0)

        width  = fields_lookup.loc[res_fids, "width_m"].to_numpy(float)
        height = fields_lookup.loc[res_fids, "height_m"].to_numpy(float)
        perim  = fields_lookup.loc[res_fids, "perimeter"].to_numpy(float)
        area_ha_vec = fields_lookup.loc[res_fids, "area_ha"].to_numpy(float)
        area_m2_vec = np.where(np.isfinite(area_ha_vec), area_ha_vec * 10_000.0, np.nan)

        aspect_ratio_vec = np.where(
            np.isfinite(width) & np.isfinite(height) & (width > 0) & (height > 0),
            np.maximum(width, height) / np.maximum(1e-9, np.minimum(width, height)),
            np.nan
        )
        edge_density_vec = perim / np.maximum(area_ha_vec, 1e-9)
        compactness_vec  = np.where(
            np.isfinite(perim) & (perim > 0) & np.isfinite(area_m2_vec),
            4 * math.pi * area_m2_vec / (perim ** 2),
            np.nan
        )

        x_vec = fields_lookup.loc[res_fids, "x"].to_numpy(float)
        y_vec = fields_lookup.loc[res_fids, "y"].to_numpy(float)
        flower_share_vec = fields_lookup.loc[res_fids, "flower_strip_share"].to_numpy(float)
        skl_plots_vec    = fields_lookup.loc[res_fids, "skylark_plots"].to_numpy(int)
        skl_area_vec     = fields_lookup.loc[res_fids, "skylark_plot_area_ha"].to_numpy(float)
        refuge_share_vec = fields_lookup.loc[res_fids, "refuge_share"].to_numpy(float)

        # --------- Flatten-Helfer (day-major, fields variieren schneller) ---------
        def flat(arr: np.ndarray) -> np.ndarray:
            return np.asarray(arr, dtype=float).reshape(-1, order="C")

        # Grundgerüst (year, day, field_id)
        day_col   = np.repeat(np.arange(1, days + 1, dtype=int), nfields)
        fid_col   = np.tile(res_fids, days)
        crop_col  = np.tile(crops_vec, days)

        # Tages-Features
        doy = day_col.astype(int)
        doy_sin = np.sin(2 * math.pi * (doy / 365.0))
        doy_cos = np.cos(2 * math.pi * (doy / 365.0))

        # Management-Flags vektorisieren
        is_harvest = np.zeros_like(day_col, dtype=bool)
        is_insect  = np.zeros_like(day_col, dtype=bool)
        is_mowing  = np.zeros_like(day_col, dtype=bool)
        # Map von field_id -> (h, i, mow_set)
        hmap = {int(fid): mg_map.get(int(fid), {"harvest": None, "insecticide": None, "mowing_days": set()})
                for fid in res_fids}
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

        # Population & Repro flatten
        A_m_f          = flat(A_m)
        A_f_f          = flat(A_f)
        A_f_breed_f    = flat(A_f_breeding) if A_f_breeding is not None else np.zeros_like(A_f_f)
        J_free_f       = flat(J_free)
        J_tied_f       = flat(J_tied) if J_tied is not None else np.zeros_like(A_f_f)
        new_nests_f    = flat(new_nests) if new_nests is not None else np.zeros_like(A_f_f)
        new_nests_c1_f = flat(new_nests_c1) if new_nests_c1 is not None else np.zeros_like(A_f_f)
        new_nests_c2_f = flat(new_nests_c2) if new_nests_c2 is not None else np.zeros_like(A_f_f)
        new_nests_c3_f = flat(new_nests_c3) if new_nests_c3 is not None else np.zeros_like(A_f_f)
        fledglings_f   = flat(fledglings) if fledglings is not None else np.zeros_like(A_f_f)
        N_total_f      = flat(N_total)

        # in_breeding (True, wenn dieses Feld an diesem Tag Brutaktivität zeigt)
        in_breeding = (A_f_breed_f > 0) | (new_nests_f > 0)

        # Statische Feldmerkmale flatten (per day einfach wiederholen)
        def rep(v): return np.tile(v, days)

        x_f     = rep(x_vec)
        y_f     = rep(y_vec)
        area_f  = rep(area_ha_vec)
        flow_f  = rep(flower_share_vec)
        sklp_f  = rep(skl_plots_vec)
        skla_f  = rep(skl_area_vec)
        ref_f   = rep(refuge_share_vec)
        w_f     = rep(width)
        h_f     = rep(height)
        p_f     = rep(perim)
        ar_f    = rep(aspect_ratio_vec)
        ed_f    = rep(edge_density_vec)
        comp_f  = rep(compactness_vec)
        n300_f  = rep(np.array([neigh300.get(int(fid), 0) for fid in res_fids], dtype=int))
        n600_f  = rep(np.array([neigh600.get(int(fid), 0) for fid in res_fids], dtype=int))

        # DataFrame bauen (Spalten in exakt gewünschter Reihenfolge)
        df = pd.DataFrame({
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
        })

        out_rows.append(df)

    out = pd.concat(out_rows, ignore_index=True)

    # Ordnung der Spalten exakt fixieren (falls Pandas gemischt hat)
    desired_cols = [
        "year","day","field_id",
        "A_m","A_f","A_f_breeding","J_free","J_tied","new_nests","new_nests_c1","new_nests_c2","new_nests_c3","fledglings","N_total",
        "crop","is_harvest_day","is_insecticide_day","is_mowing_day",
        "x","y","area_ha","flower_strip_share","skylark_plots","skylark_plot_area_ha","refuge_share",
        "width_m","height_m","perimeter","doy_sin","doy_cos","in_breeding",
        "aspect_ratio","edge_density","compactness","neighbors_300m","neighbors_600m",
        "species","run_id"
    ]
    # Fehlende (sollte es nicht geben) zur Sicherheit ergänzen
    for c in desired_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[desired_cols]

    return out



    # --- Assemble rows -----------------------------------------------------------
    rows: List[dict] = []
    col_order = [
        "species", "run_id", "sim_year", "doy", "field_id", "area_ha", "x", "y",
        "flower_strip_share", "refuge_share", "skylark_plots", "skylark_plot_area_ha",
        "crop", "harvest", "mowing", "insecticide",
        "adults_f", "adults_m", "juveniles", "subadults",
    ]

    for year in sorted(results_by_year.keys()):
        res = results_by_year[year]
        if res is None:
            continue

        fields_order = res.get("fields")
        if not isinstance(fields_order, pd.DataFrame) or "field_id" not in fields_order.columns:
            continue
        field_ids = fields_order["field_id"].astype(int).to_numpy()

        A_f = np.asarray(res.get("A_f"))
        A_m = np.asarray(res.get("A_m"))
        J_free = np.asarray(res.get("J_free"))
        J_tied = res.get("J_tied_total", res.get("J_tied"))
        if J_tied is None:
            J_tied = np.zeros_like(J_free)
        else:
            J_tied = np.asarray(J_tied)
            if J_tied.shape != J_free.shape:
                J_tied = np.zeros_like(J_free)

        if A_f.ndim != 2 or A_m.ndim != 2:
            continue

        days = A_f.shape[0]

        crop_map = {
            int(fid): str(crop)
            for fid, crop in rotation_df.loc[rotation_df["year"] == int(year), ["field_id", "crop"]].itertuples(index=False)
        }
        mg_map = _build_management_map(int(year))

        for day_idx in range(days):
            doy = day_idx + 1
            adults_f_day = A_f[day_idx, :]
            adults_m_day = A_m[day_idx, :]
            juv_free_day = J_free[day_idx, :] if J_free.ndim == 2 else np.zeros_like(adults_f_day)
            juv_tied_day = J_tied[day_idx, :] if J_tied.ndim == 2 else np.zeros_like(adults_f_day)
            juveniles_day = juv_free_day.astype(int) + juv_tied_day.astype(int)

            for idx_field, fid in enumerate(field_ids):
                if fid not in fields_lookup.index:
                    continue
                attrs = fields_lookup.loc[fid]
                mg_info = mg_map.get(int(fid), None)
                harvest_flag = int(bool(mg_info and mg_info.get("harvest") == doy))
                mowing_flag = int(bool(mg_info and doy in mg_info.get("mowing_days", set())))
                insect_flag = int(bool(mg_info and mg_info.get("insecticide") == doy))

                rows.append({
                    "species": str(species_name),
                    "run_id": str(run_id),
                    "sim_year": int(year),
                    "doy": int(doy),
                    "field_id": int(fid),
                    "area_ha": _to_float(attrs.get("area_ha", np.nan)),
                    "x": _to_float(attrs.get("x", np.nan)),
                    "y": _to_float(attrs.get("y", np.nan)),
                    "flower_strip_share": _to_float(attrs.get("flower_strip_share", 0.0), 0.0),
                    "refuge_share": _to_float(attrs.get("refuge_share", 0.0), 0.0),
                    "skylark_plots": _to_int(attrs.get("skylark_plots", 0), 0),
                    "skylark_plot_area_ha": _to_float(attrs.get("skylark_plot_area_ha", 0.0), 0.0),
                    "crop": str(crop_map.get(int(fid), "Unbekannt")),
                    "harvest": harvest_flag,
                    "mowing": mowing_flag,
                    "insecticide": insect_flag,
                    "adults_f": int(adults_f_day[idx_field]),
                    "adults_m": int(adults_m_day[idx_field]),
                    "juveniles": int(juveniles_day[idx_field]),
                    "subadults": 0,
                })

    if not rows:
        return None

    df_out = pd.DataFrame(rows)
    # Ensure column order and dtypes (where sensible).
    df_out = df_out[col_order]
    return df_out

def save_run_bundle(outdir, species_name, fields_gdf, rotation, management, hedges_gdf, results_by_year, day_to_show):
    """
    Saves (per run + per species):
      - outputs/<run_id>/data/:
          fields_flat.csv, rotation.csv, management.csv, fields.geojson (optional), hedges.csv (optional)
        (written only once per run, i.e. on first species call)
      - outputs/<run_id>/<species>/:
          run_meta.json, birdsim_daily_full.csv,
          year_X/arrays/*.npz, year_X/usage.csv, year_X/daily_totals.csv,
          year_X/repro_log.csv (if res['debug_repro'] present),
          year_X/debug_daily.csv (if res['debug']['daily'] present),
          analysis/fields_layout_realized.csv
      - outputs/<run_id>/<species>/plots/ (created)
    """
    outdir = pathlib.Path(outdir)
    try:
        resolved_outdir = outdir.resolve()
    except Exception:
        resolved_outdir = outdir
    parent_name = resolved_outdir.parent.name if resolved_outdir.parent != resolved_outdir else ""
    run_id = parent_name or resolved_outdir.name

    # ---------------- Root-level Inputs snapshot (einmal pro Run) ----------------
    # outdir = outputs/<run_id>/<species>  -> run_root = outputs/<run_id>
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

            # optional: Geometrie einmal auf Run-Root speichern (breit kompatibel)
            try:
                fields_gdf.to_file(root_data_dir / "fields.geojson", driver="GeoJSON")
            except Exception as e:
                print("[warn] writing root fields.geojson failed:", e)

            # optional: hedges einmal auf Run-Root speichern (ohne Geometrie)
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

    # ---------------- Species-level dirs ----------------
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    # für Layout-Export (ohne Geometrie)
    fields_flat = fields_gdf.drop(columns=[c for c in ["geometry"] if c in fields_gdf.columns]).copy()

    # ---------------- Metadata ----------------
    # DOY clamp (verhindert Out-of-range)
    try:
        max_day = int(globals().get("TOT_DAYS", 365))
    except Exception:
        max_day = 365
    meta = {
        "species": species_name,
        "years": list(results_by_year.keys()),
        "day_to_show": int(max(1, min(int(day_to_show), max_day)))
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------- Per year arrays/CSVs ----------------
    for y, res in results_by_year.items():
        ydir = outdir / f"year_{y}"
        (ydir / "arrays").mkdir(parents=True, exist_ok=True)
        (ydir / "plots").mkdir(parents=True, exist_ok=True)

        # Arrays (komprimiert)
        np.savez_compressed(ydir / "arrays" / "N_total.npz", N_total=res["N_total"])
        np.savez_compressed(ydir / "arrays" / "A_m.npz", A_m=res["A_m"])
        np.savez_compressed(ydir / "arrays" / "A_f.npz", A_f=res["A_f"])
        np.savez_compressed(ydir / "arrays" / "J_free.npz", J_free=res["J_free"])

        # J_tied_total evtl. legacy-Name absichern:
        J_tied_arr = res.get("J_tied_total", res.get("J_tied"))
        if J_tied_arr is not None:
            np.savez_compressed(ydir / "arrays" / "J_tied_total.npz", J_tied_total=J_tied_arr)
        if "J_age_series" in res:
            np.savez_compressed(ydir / "arrays" / "J_age_series.npz", J_age_series=res["J_age_series"])
        if "A_f_breeding" in res:
            np.savez_compressed(ydir / "arrays" / "A_f_breeding.npz", A_f_breeding=res["A_f_breeding"])

        # Usage & Jahressummen
        usage_df = pd.DataFrame({
            "field_id": res["fields"]["field_id"].values,
            "usage": res["usage"]
        })
        usage_df.to_csv(ydir / "usage.csv", index=False)

        daily_totals = res["N_total"].sum(axis=1)
        pd.DataFrame({"day": np.arange(len(daily_totals)), "total_individuals": daily_totals}).to_csv(
            ydir / "daily_totals.csv", index=False
        )

        # Optionale Debug-Logs
        if "debug_repro" in res and isinstance(res["debug_repro"], pd.DataFrame):
            res["debug_repro"].to_csv(ydir / "repro_log.csv", index=False)
        if isinstance(res.get("debug"), dict):
            daily_debug = res["debug"].get("daily")
            if isinstance(daily_debug, pd.DataFrame):
                daily_debug.to_csv(ydir / "debug_daily.csv", index=False)

    # ---------------- BirdSIM daily full export (species-level) ----------------
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

    # ---------------- Layout-Export (species-level) ----------------
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        from .layout import save_fields_layout_csv
        save_fields_layout_csv(fields_flat, analysis_dir, filename="fields_layout_realized.csv")
    except Exception as e:
        print(f"[warn] Layout-Export fehlgeschlagen: {e}")

# ---------- Synthetic landscape & inputs ----------
def make_clip_poly_from_fields(fields_gdf,
                               pad_left=0.0, pad_right=0.0,
                               pad_bottom=0.0, pad_top=0.0):
    # benutzt die echten Feldausdehnungen
    minx = float(fields_gdf["x"].min())
    miny = float(fields_gdf["y"].min())
    maxx = float((fields_gdf["x"] + fields_gdf["width_m"]).max())
    maxy = float((fields_gdf["y"] + fields_gdf["height_m"]).max())
    return box(minx - pad_left, miny - pad_bottom,
               maxx + pad_right, maxy + pad_top)

def add_permanent_cover(fields_rects, square_poly,
                        margin_width_m=6,
                        min_patch_area_m2=80,
                        fill_residual=True):
    union_fields = unary_union(fields_rects)
    out = []

    # (A) Randsaum
    if margin_width_m and margin_width_m > 0:
        rim = union_fields.buffer(margin_width_m).intersection(square_poly).difference(union_fields)
        geoms = [rim] if (not rim.is_empty and rim.geom_type == "Polygon") else (list(rim.geoms) if not rim.is_empty else [])
        out += [(g, "rim") for g in geoms if g.area >= min_patch_area_m2]

    # (B) Restflächen
    if fill_residual:
        residual = square_poly.difference(union_fields)
        geoms = [residual] if (not residual.is_empty and residual.geom_type == "Polygon") else (list(residual.geoms) if not residual.is_empty else [])
        out += [(g, "residual") for g in geoms if g.area >= min_patch_area_m2]

    return out


# ---------- Synthetic landscape & inputs ----------
def make_synthetic_fields(n_fields=150, seed=42,
                          add_cover=True,
                          cover_margin_m=6,
                          cover_min_patch_m2=80,
                          cover_fill_residual=True,
                          square_pad_m=8):
    """
    Wie bisher – erweitert um PermanentCover (Randsaum + Restflächen).
    Gibt ein GeoDataFrame zurück; Dauergrün ist in den Zeilen mit 'is_permanent_cover'==True.
    """
    rng = np.random.default_rng(seed)

    # areas (ha): log-normalish 0.5..12ha, skewed to small
    area = np.clip(np.exp(rng.normal(np.log(3.0), 0.6, size=n_fields)), 0.5, 12.0)

    # aspect ratios ~ lognormal around 1
    aspect = np.exp(rng.normal(0.0, 0.5, size=n_fields))
    width_m  = np.sqrt(area * 10_000 * aspect)
    height_m = (area * 10_000) / width_m

    df = pd.DataFrame({
        "field_id": np.arange(1, n_fields + 1, dtype=int),
        "area": area.astype(float),
        "width_m": width_m.astype(float),
        "height_m": height_m.astype(float),
    })

    # features
    fs_mask = rng.random(n_fields) < 0.4
    flower_strip_share = np.where(fs_mask, rng.beta(2, 10, size=n_fields) * 0.25, 0.0)
    skylark_plots = np.zeros(n_fields, dtype=int)
    skylark_plot_area = np.zeros(n_fields, dtype=float)

    df["flower_strip_share"] = flower_strip_share
    df["skylark_plots"] = skylark_plots
    df["skylark_plot_area"] = skylark_plot_area

    # pack into a grid-like layout and compute geometry/perimeter/x/y
    from .layout import rect_shelf_pack
    gdf = rect_shelf_pack(df, target_row_width=TARGET_ROW_WIDTH_M)

    # Falls rect_shelf_pack keine Geometrie setzt: erzeugen
    if "geometry" not in gdf.columns or gdf.geometry.isna().any():
        geoms = []
        for _, r in gdf.iterrows():
            geoms.append(box(r["x"], r["y"], r["x"] + r["width_m"], r["y"] + r["height_m"]))
        gdf = gpd.GeoDataFrame(gdf, geometry=geoms, crs="EPSG:3857")

    gdf["is_permanent_cover"] = False
    gdf["cover_source"] = ""

    if not add_cover:
        return gdf

    # Äußeres Rechteck exakt an die Felder andocken (Hüllbox + kleiner Puffer)
    fields_rects = list(gdf.geometry)
    union_fields = unary_union(fields_rects)
    minx, miny, maxx, maxy = union_fields.bounds
    square_poly = box(minx - square_pad_m, miny - square_pad_m,
                      maxx + square_pad_m, maxy + square_pad_m)

    # Dauergrün-Patches generieren
    cover_parts = add_permanent_cover(
        fields_rects=fields_rects,
        square_poly=square_poly,
        margin_width_m=cover_margin_m,
        min_patch_area_m2=cover_min_patch_m2,
        fill_residual=cover_fill_residual
    )

    if cover_parts:
        start_id = int(gdf["field_id"].max()) + 1

        polys = [poly for (poly, _src) in cover_parts]
        merged = unary_union(polys)

        minx, miny, maxx, maxy = merged.bounds
        w = maxx - minx
        h = maxy - miny

        cover_row = {
            "field_id": start_id,  # <- nur eine ID
            "area": merged.area / 10_000.0,  # ha
            "width_m": w,
            "height_m": h,
            "flower_strip_share": 0.0,
            "skylark_plots": 0,
            "skylark_plot_area": 0.0,
            "x": minx, "y": miny,
            "perimeter": merged.length,
            "is_permanent_cover": True,
            "cover_source": "merged"  # oder "rim+residual"
        }

        cover_gdf = gpd.GeoDataFrame(pd.DataFrame([cover_row]), geometry=[merged], crs=gdf.crs)
        gdf = pd.concat([gdf, cover_gdf], ignore_index=True)

    # Debug/Monitoring
    share_cover = gdf.loc[gdf["is_permanent_cover"], "area"].sum() / gdf["area"].sum()
    print(f"[make_synthetic_fields] PermanentCover: {share_cover:.2%} der Fläche | "
          f"Patches={gdf['is_permanent_cover'].sum()} (rim+residual)")

    return gdf


def make_rotation(fields_gdf, years, seed=43, crop_mix=None):
    """Arable-Felder rotieren wie gehabt; PermanentCover bleibt konstant."""
    rng = np.random.default_rng(seed)
    if crop_mix is None:
        crop_mix = {
            "Winterweizen": 0.36, "Sommergerste": 0.22, "Mais": 0.17,
            "Raps": 0.11, "Zuckerrübe": 0.10, "Ackerbohne": 0.04
        }

    crops = list(crop_mix.keys())
    probs = np.array(list(crop_mix.values()), float); probs /= probs.sum()

    mask_cover = fields_gdf.get("is_permanent_cover", pd.Series(False, index=fields_gdf.index)).astype(bool)
    fids_cover  = fields_gdf.loc[mask_cover, "field_id"].astype(int).to_numpy()
    fids_arable = fields_gdf.loc[~mask_cover, "field_id"].astype(int).to_numpy()

    rows = []

    # (A) PermanentCover – fest auf allen Jahren
    for y in years:
        rows.extend({"year": int(y), "field_id": int(fid), "crop": "PermanentCover"} for fid in fids_cover)

    # (B) Arable – wie bisher
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
            rows.extend({"year": int(y), "field_id": int(fid), "crop": str(c)}
                        for fid, c in zip(fids_arable, cur))
            prev = cur

    rot = pd.DataFrame(rows)
    return rot


from pathlib import Path
import numpy as np
import pandas as pd

def make_management(fields_gdf, rotation_df, seed=44,
                    use_calendar: bool = True,
                    calendar_path: str | Path = CROP_CALENDAR_CSV,
                    desync_region_sd: float = 4.0,   # gemeinsame Verschiebung (Tage) je Nachbarschaft
                    desync_field_sd: float = 3.0,    # individueller Jitter je Feld
                    region_bins: int = 6):
    """
    Erzeugt Management-Events je Feld+Jahr, desynchronisiert nach Kultur,
    mit optionalem Kultur-Kalender (CSV) und zweistufiger Streuung (Region + Feld).

    - 'PermanentCover' erhält grundsätzlich KEINE Arbeiten/Termine.
    - Wenn 'use_calendar=True', werden pro Kultur (falls vorhanden) folgende Keys gelesen:
        harvest_center, harvest_sd, harvest_width, harvest_win_span,
        sow_center, sow_width,
        other_prob, other_center, other_width,
        insect_prob, insect_center, insect_width
    - Fallback-Heuristiken greifen, wenn Kultur nicht im Kalender steht.
    """
    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1) Regionale Cluster (grob)
    # -----------------------------
    # einfache Gitter-Nachbarschaften auf Basis von x/y
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

    # Map: field_id -> region_id
    region_by_fid = {
        int(fid): _region_id(float(x), float(y))
        for fid, x, y in fields_gdf[["field_id", "x", "y"]].itertuples(index=False)
    }
    unique_regions = np.unique(list(region_by_fid.values()))
    region_offset = {rid: rng.normal(0.0, desync_region_sd) for rid in unique_regions}

    # -----------------------------
    # 2) Kultur-Kalender laden
    # -----------------------------
    cal = _read_crop_calendar(calendar_path) if use_calendar else {}

    # -----------------------------
    # 3) Events je Feld+Jahr bauen
    # -----------------------------
    rows = []
    for _, r in rotation_df.iterrows():
        crop = str(r["crop"])
        fid  = int(r["field_id"])
        y    = int(r["year"])

        # ---- Dauergrün: keine Maßnahmen ----
        if crop == "PermanentCover":
            rows.append({
                "year": y, "field_id": fid,
                "mow_dates": "",               # optional: wenn Mahd separat modelliert wird
                "harvest_date": None,
                "insecticide_date": None,
                "sow_win": None,
                "harvest_win": None,
                "other_ops_win": None
            })
            continue

        # Offsets ermitteln (Region + Feld)
        rid = int(region_by_fid.get(fid, int(rng.integers(0, max(1, len(unique_regions))))))
        off_reg = float(region_offset.get(rid, 0.0))
        off_field = float(rng.normal(0.0, desync_field_sd))

        harv = None; insect = None; sow_win = None; harv_win = None; other_win = None

        if crop in cal:
            c = cal[crop]

            # --- Ernte ---
            hc = c.get("harvest_center", None)
            if hc is not None:
                base = rng.normal(hc, float(c.get("harvest_sd", 0.0) or 0.0))
                if c.get("harvest_width", None) is not None:
                    a = base - float(c["harvest_width"])
                    b = base + float(c["harvest_width"])
                    harv = int(np.clip(rng.uniform(a, b) + off_reg + off_field, 1, 365))
                else:
                    harv = int(np.clip(base + off_reg + off_field, 1, 365))

                span = int(c.get("harvest_win_span", 10) or 10)
                harv_win = f"{max(1, harv - span//2)}-{min(365, harv + span//2)}"

            # --- Saat-Fenster ---
            sc = c.get("sow_center", None)
            sw = c.get("sow_width", None)
            if sc is not None and sw is not None:
                a = sc - sw + off_reg + off_field
                b = sc + sw + off_reg + off_field
                aa = int(np.clip(min(a, b), 1, 365))
                bb = int(np.clip(max(a, b), 1, 365))
                sow_win = f"{aa}-{bb}"

            # --- Sonstige Arbeiten ---
            if float(c.get("other_prob", 0.0) or 0.0) > 0 and rng.random() < float(c.get("other_prob", 0.0)):
                oc = c.get("other_center", None); ow = c.get("other_width", None)
                if oc is not None and ow is not None:
                    a = oc - ow + off_reg + off_field
                    b = oc + ow + off_reg + off_field
                    aa = int(np.clip(min(a, b), 1, 365))
                    bb = int(np.clip(max(a, b), 1, 365))
                    other_win = f"{aa}-{bb}"

            # --- Insektizid ---
            if float(c.get("insect_prob", 0.0) or 0.0) > 0 and rng.random() < float(c.get("insect_prob", 0.0)):
                ic = c.get("insect_center", None); iw = c.get("insect_width", None)
                if ic is not None:
                    if iw is not None:
                        insect = int(np.clip(rng.uniform(ic - iw, ic + iw) + off_reg + off_field, 1, 365))
                    else:
                        insect = int(np.clip(ic + off_reg + off_field, 1, 365))

        else:
            # -------- Fallback-Heuristiken (wie besprochen) --------
            # Ernte-Basis je Kultur
            if crop == "Winterweizen": base = rng.normal(210, 7)
            elif crop == "Sommergerste": base = rng.normal(190, 6)
            elif crop == "Mais": base = rng.normal(250, 10)
            elif crop == "Raps": base = rng.normal(195, 7)
            elif crop in ("Zuckerrübe", "Zuckerruebe"): base = rng.normal(260, 10)
            elif crop == "Ackerbohne": base = rng.normal(200, 7)
            else: base = None

            if base is not None:
                harv = int(np.clip(base + off_reg + off_field, 1, 365))
                harv_win = f"{max(1, harv - 5)}-{min(365, harv + 5)}"

            # Saatfenster für WW/SG
            if crop in ("Winterweizen", "Sommergerste"):
                a = np.clip(rng.normal(100, 8) + off_reg + off_field, 70, 130)
                b = np.clip(rng.normal(125, 8) + off_reg + off_field, 95, 160)
                sow_win = f"{int(min(a, b))}-{int(max(a, b))}"

            # Sonstige Arbeiten (probabilistisch)
            if rng.random() < 0.4:
                a = np.clip(rng.normal(140, 10) + off_reg + off_field, 110, 190)
                b = np.clip(rng.normal(155, 10) + off_reg + off_field, 120, 210)
                other_win = f"{int(min(a, b))}-{int(max(a, b))}"

            # Insektizid für WW/SG/Raps (prob.)
            if crop in ("Winterweizen", "Sommergerste", "Raps") and rng.random() < 0.3:
                insect = int(np.clip(rng.normal(150, 7) + off_reg + off_field, 100, 220))

        rows.append({
            "year": y, "field_id": fid,
            "mow_dates": "",               # optional: separat modellieren
            "harvest_date": harv,
            "insecticide_date": insect,
            "sow_win": sow_win,
            "harvest_win": harv_win,
            "other_ops_win": other_win
        })

    return pd.DataFrame(rows)

def _read_crop_calendar(path: str) -> dict:
    """
    Lies eine Crop-Calendar CSV ein und liefere ein dict:
        crop_name -> dict mit Management-Infos (z.B. harvest days etc.)

    Erwartet mindestens eine Spalte 'crop' (oder 'Crop').
    Alle weiteren Spalten werden mit übernommen.
    Felder mit Tageslisten können z.B. Strings wie "120,121,122" oder "120-125" enthalten.
    """
    import os
    import pandas as pd

    if path is None or str(path).strip() == "":
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"Crop calendar CSV not found: {path}")

    df = pd.read_csv(path)

    # crop-Spalte robust finden
    crop_col = None
    for c in df.columns:
        if c.lower().strip() == "crop":
            crop_col = c
            break
    if crop_col is None:
        raise ValueError(f"Crop calendar CSV must contain a 'crop' column. Got columns: {list(df.columns)}")

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
            # NaNs raus
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass

            # Optional: wenn deine Pipeline parse_days erwartet, wandle typische day-list columns um.
            # Wir machen das nur, wenn parse_days existiert (kommt z.B. aus layout.py).
            if "day" in c.lower() or "doy" in c.lower():
                if "parse_days" in globals():
                    try:
                        d[c] = parse_days(v)  # type: ignore[name-defined]
                        continue
                    except Exception:
                        pass

            d[c] = v

        out[crop] = d

    return out


def make_random_hedges(fields_gdf, p_edge=0.7, seed=45):
    """Propose hedge pairs mostly between nearest neighbors; only where edges share."""
    rng = np.random.default_rng(seed)
    fids = fields_gdf["field_id"].astype(int).to_numpy()
    XY = fields_gdf[["x", "y"]].to_numpy(float)

    pairs = set()
    for i in range(len(fids)):
        d = np.sqrt(((XY - XY[i])**2).sum(axis=1))
        order = np.argsort(d)
        for j in order[1:6]:
            a, b = sorted((int(fids[i]), int(fids[j])))
            if rng.random() < p_edge:
                pairs.add((a, b))
    return pd.DataFrame([{"f1": a, "f2": b, "hedge_quality": float(rng.uniform(0.4, 0.9))}
                         for (a, b) in pairs])


def _crop_for_year(rotation_df: pd.DataFrame, year: int) -> pd.Series:
    """
    Liefert eine Series: index=field_id (int), value=crop (str) für das angegebene Jahr.
    - robust gegen Jahr als float/str
    - säubert field_id/crop-Typen
    - füllt fehlende/leer getrimmte Crops als 'Unbekannt'
    - löst Duplikate deterministisch und warnt
    """
    # Filter & Kernspalten
    y_int = int(year)
    rot_y = rotation_df.loc[rotation_df["year"].astype(int) == y_int, ["field_id", "crop"]].copy()
    if rot_y.empty:
        return pd.Series(dtype=object, name="crop")

    # Typen & Säuberung
    rot_y["field_id"] = rot_y["field_id"].astype(int)
    rot_y["crop"] = rot_y["crop"].astype(str).str.strip()

    # Leere als 'Unbekannt'
    rot_y["crop"] = rot_y["crop"].replace({"nan": "", "NaN": ""})
    rot_y.loc[rot_y["crop"] == "", "crop"] = "Unbekannt"

    # Duplikate pro field_id im selben Jahr behandeln
    dup_mask = rot_y.duplicated(subset=["field_id"], keep=False)
    if dup_mask.any():
        # Behalte die erste nicht-leere Angabe je field_id (deterministisch via Reihenfolge)
        rot_y = (rot_y
                 .groupby("field_id", as_index=False)["crop"]
                 .agg(lambda s: next((v for v in s if str(v).strip() != ""), "Unbekannt")))
        print(f"[warn] rotation: multiple rows for some field_ids in year {y_int}; "
              f"kept first non-empty per field.")
    else:
        rot_y = rot_y.drop_duplicates(subset=["field_id"])

    s = rot_y.set_index("field_id")["crop"]
    s.name = "crop"
    return s


def summarize_one_year(res: dict, year: int, fields_df: pd.DataFrame,
                       rotation_df: pd.DataFrame, species_name: str, scenario_name: str,
                       breeding_window: tuple[int, int]) -> pd.DataFrame:
    """
    Tidy per-field summary für ein Jahr. Robuster bzgl.:
      - DOY vs. 0-basierte Indizes (kein Off-by-one)
      - Field-Reihenfolge (immer an res['fields'] ausrichten)
      - Fehlende Arrays (J_tied, Clutch-Arrays)
    """
    # --- Kerndaten & Ausrichtung auf Sim-Feldreihenfolge ---
    N   = res["N_total"]          # (days x fields)
    A_m = res["A_m"]
    A_f = res["A_f"]
    J_free = res["J_free"]
    J_tied = res.get("J_tied_total", res.get("J_tied"))
    if J_tied is None:
        J_tied = np.zeros_like(J_free, dtype=np.int32)

    newN  = res.get("daily_new_nests")
    fledg = res.get("daily_fledglings")

    # Clutch-Starts (optional)
    newN_c1 = res.get("daily_new_nests_c1")
    newN_c2 = res.get("daily_new_nests_c2")
    newN_c3 = res.get("daily_new_nests_c3")

    days, n_fields = N.shape
    day_ix0 = np.arange(days)  # 0..days-1
    # DOY = day_ix0 + 1  --> Fenster korrekt clampen
    b0_raw, b1_raw = breeding_window
    b0 = int(max(1, min(int(b0_raw), days)))
    b1 = int(max(1, min(int(b1_raw), days)))
    if b1 < b0:
        # leeres Fenster
        mask_breed = np.zeros(days, dtype=bool)
    else:
        mask_breed = ((day_ix0 + 1) >= b0) & ((day_ix0 + 1) <= b1)


    # --- Metriken (alle in Sim-Reihenfolge) ---
    usage_total    = N.sum(axis=0)
    usage_breeding = N[mask_breed, :].sum(axis=0) if mask_breed.any() else np.zeros(n_fields, dtype=int)
    adults_mean    = (A_m + A_f).mean(axis=0)
    adults_max     = (A_m + A_f).max(axis=0)
    juveniles_mean = (J_tied + J_free).mean(axis=0)

    fledgl_total = fledg.sum(axis=0) if isinstance(fledg, np.ndarray) else (
        np.zeros(n_fields, dtype=int) if fledg is None else np.asarray(fledg).sum(axis=0)
    )
    newnests_total = newN.sum(axis=0) if isinstance(newN, np.ndarray) else (
        np.zeros(n_fields, dtype=int) if newN is None else np.asarray(newN).sum(axis=0)
    )

    def _sum_or_zeros(arr):
        if arr is None:
            return np.zeros(n_fields, dtype=int)
        arr = np.asarray(arr)
        return arr.sum(axis=0) if arr.ndim == 2 else np.zeros(n_fields, dtype=int)

    newnests_c1_total = _sum_or_zeros(newN_c1)
    newnests_c2_total = _sum_or_zeros(newN_c2)
    newnests_c3_total = _sum_or_zeros(newN_c3)

    # --- Statische Felddaten & Kultur mergen (über field_id) ---
    sim_fields = res["fields"][["field_id"]].copy()
    sim_fields["field_id"] = sim_fields["field_id"].astype(int)

    F = fields_df.copy()
    if "field_id" not in F.columns:
        raise ValueError("fields_df must contain 'field_id'")
    F["field_id"] = F["field_id"].astype(int)

    # Pflichtspalten weich ergänzen
    for col, default in [("area", 0.0), ("flower_strip_share", 0.0),
                         ("skylark_plots", 0), ("skylark_plot_area", 0.0)]:
        if col not in F.columns:
            F[col] = default

    # Kultur-Karte für dieses Jahr
    crop_map = _crop_for_year(rotation_df, int(year))

    # Merge: Sim-Reihenfolge beibehalten
    base = sim_fields.merge(
        F[["field_id", "area", "flower_strip_share", "skylark_plots", "skylark_plot_area"]],
        on="field_id", how="left"
    )

    # --- Refuge-Share sauber berechnen ---
    area_ha = base["area"].astype(float).to_numpy()
    skl_cnt = base["skylark_plots"].astype(float).to_numpy()
    skl_ha  = base["skylark_plot_area"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        refuge_share = np.clip(np.where(area_ha > 0, (skl_cnt * skl_ha) / area_ha, 0.0), 0.0, 1.0)

    # --- DataFrame mit Metriken in Sim-Reihenfolge ---
    out = pd.DataFrame({
        "species": species_name,
        "scenario": scenario_name,
        "year": int(year),
        "field_id": sim_fields["field_id"].astype(int).values,
        "crop": sim_fields["field_id"].map(crop_map).fillna("Unbekannt").astype(str).values,
        "area_ha": area_ha,
        "flower_strip_share": base["flower_strip_share"].astype(float).values,
        "skylark_plots": base["skylark_plots"].astype(int).values,
        "skylark_plot_area_ha": skl_ha,
        "refuge_share": refuge_share.astype(float),
        "usage_total": usage_total.astype(int),
        "usage_breeding_win": usage_breeding.astype(int),
        "adults_mean": adults_mean.astype(float),
        "adults_max": adults_max.astype(int),
        "juveniles_mean": juveniles_mean.astype(float),
        "fledglings_total": fledgl_total.astype(int),
        "new_nests_total": newnests_total.astype(int),
        "new_nests_c1_total": newnests_c1_total.astype(int),
        "new_nests_c2_total": newnests_c2_total.astype(int),
        "new_nests_c3_total": newnests_c3_total.astype(int),
    })

    return out

def summarize_overall(per_field_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-field summaries to landscape-level (per species/year/scenario).
    Robust gegen NaNs/Typen und erweitert um Clutch-Summen.
    """
    if per_field_df is None or len(per_field_df) == 0:
        return pd.DataFrame(columns=[
            "species","scenario","year","fields","area_ha","usage_total","usage_breeding_win",
            "adults_mean","adults_max","juveniles_mean",
            "fledglings_total","new_nests_total",
            "new_nests_c1_total","new_nests_c2_total","new_nests_c3_total"
        ])

    df = per_field_df.copy()

    # Sicherstellen, dass Gruppierschlüssel korrekt typisiert sind
    for k in ("species", "scenario"):
        if k in df.columns:
            df[k] = df[k].astype(str)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)

    # Numerische Spalten robust casten (NaN -> 0)
    num_cols = [
        "area_ha","usage_total","usage_breeding_win",
        "adults_mean","adults_max","juveniles_mean",
        "fledglings_total","new_nests_total",
        "new_nests_c1_total","new_nests_c2_total","new_nests_c3_total"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Gruppierung
    agg_dict = dict(
        fields=("field_id", "nunique"),
        area_ha=("area_ha", "sum"),
        usage_total=("usage_total", "sum"),
        usage_breeding_win=("usage_breeding_win", "sum"),
        adults_mean=("adults_mean", "sum"),        # Summe über Felder = mittlere Landschaftsbelegung (tageweise gemittelt)
        adults_max=("adults_max", "sum"),          # Hinweis: Summe der Feldmaxima (nicht Landschafts-Maximum)
        juveniles_mean=("juveniles_mean", "sum"),
        fledglings_total=("fledglings_total", "sum"),
        new_nests_total=("new_nests_total", "sum"),
    )

    # Clutch-Totals nur aggregieren, wenn vorhanden
    if "new_nests_c1_total" in df.columns:
        agg_dict["new_nests_c1_total"] = ("new_nests_c1_total", "sum")
    if "new_nests_c2_total" in df.columns:
        agg_dict["new_nests_c2_total"] = ("new_nests_c2_total", "sum")
    if "new_nests_c3_total" in df.columns:
        agg_dict["new_nests_c3_total"] = ("new_nests_c3_total", "sum")

    grp = (df
           .groupby(["species", "scenario", "year"], as_index=False)
           .agg(**agg_dict))

    # Stabil sortieren
    grp = grp.sort_values(["species", "scenario", "year"], kind="stable").reset_index(drop=True)
    return grp

def add_deltas_vs_baseline(per_field_df: pd.DataFrame, baseline_name: str = "Baseline") -> pd.DataFrame:
    """
    Compute deltas to Baseline per species/year/field, with safe % changes (no div-by-zero warnings).
    Produces rows only for scenarios != baseline.
    - robust gegen NaNs/Typen/Duplikate
    - erweitert die Metrikliste automatisch um vorhandene Clutch-Summen
    """
    if per_field_df is None or len(per_field_df) == 0:
        return pd.DataFrame(columns=[
            "species","scenario","year","field_id","crop","area_ha",
            "usage_total","usage_total_delta","usage_total_rel_pct",
            "usage_breeding_win","usage_breeding_win_delta","usage_breeding_win_rel_pct",
            "adults_mean","adults_mean_delta","adults_mean_rel_pct",
            "adults_max","adults_max_delta","adults_max_rel_pct",
            "juveniles_mean","juveniles_mean_delta","juveniles_mean_rel_pct",
            "fledglings_total","fledglings_total_delta","fledglings_total_rel_pct",
            "new_nests_total","new_nests_total_delta","new_nests_total_rel_pct",
            "new_nests_c1_total","new_nests_c1_total_delta","new_nests_c1_total_rel_pct",
            "new_nests_c2_total","new_nests_c2_total_delta","new_nests_c2_total_rel_pct",
            "new_nests_c3_total","new_nests_c3_total_delta","new_nests_c3_total_rel_pct",
        ])

    df = per_field_df.copy()

    # === 1) Metriken bestimmen (Basis + optionale Clutch-Spalten) ===
    base_metrics = [
        "usage_total", "usage_breeding_win",
        "adults_mean", "adults_max", "juveniles_mean",
        "fledglings_total", "new_nests_total",
    ]
    clutch_opts = [c for c in ("new_nests_c1_total", "new_nests_c2_total", "new_nests_c3_total") if c in df.columns]
    metrics = base_metrics + clutch_opts

    # === 2) Numerische Typen erzwingen & NaNs -> 0 ===
    for c in metrics:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Schlüssel-Typen stabilisieren
    df["species"] = df["species"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    df["year"] = df["year"].astype(int)
    df["field_id"] = df["field_id"].astype(int)
    if "area_ha" in df.columns:
        df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce").fillna(0)
    if "crop" in df.columns:
        df["crop"] = df["crop"].astype(str)

    # === 3) Duplikate pro (species,year,field,scenario) zusammenfassen (Summe) ===
    group_keys = ["species", "year", "field_id", "scenario"]
    agg_dict = {m: "sum" for m in metrics}
    # Info-Spalten (crop, area_ha) deterministisch "erste" nehmen
    for info_col in ("crop", "area_ha"):
        if info_col in df.columns:
            agg_dict[info_col] = "first"
    df_g = df.groupby(group_keys, as_index=False).agg(agg_dict)

    # === 4) Baseline-Subset & Index ===
    base = (df_g[df_g["scenario"] == baseline_name]
            .set_index(["species", "year", "field_id"])
            .sort_index())

    if base.empty:
        return pd.DataFrame(columns=[
            "species","scenario","year","field_id","crop","area_ha",
            *sum([[m, f"{m}_delta", f"{m}_rel_pct"] for m in metrics], [])
        ])

    rows = []
    for scen, dfS in df_g.groupby("scenario", sort=False):
        if scen == baseline_name:
            continue

        S = dfS.set_index(["species", "year", "field_id"]).sort_index()
        common = S.index.intersection(base.index)
        if len(common) == 0:
            continue

        M = S.loc[common].copy()
        B = base.loc[common]

        # Deltas/Rel.-% pro Metrik
        for col in metrics:
            delta = M[col].to_numpy(dtype=float) - B[col].to_numpy(dtype=float)
            M[f"{col}_delta"] = delta
            M[f"{col}_rel_pct"] = _safe_rel_pct(delta, B[col].to_numpy(dtype=float))

        # Scenario-Label und zurück zu Spalten
        M["scenario"] = scen
        M = M.reset_index()

        keep_cols = ["species", "scenario", "year", "field_id"]
        # Info-Spalten, falls vorhanden
        for info_col in ("crop", "area_ha"):
            if info_col in M.columns:
                keep_cols.append(info_col)
        # Metriken + Deltas + Rel.-%
        for col in metrics:
            keep_cols += [col, f"{col}_delta", f"{col}_rel_pct"]

        rows.append(M[keep_cols])

    if not rows:
        return pd.DataFrame(columns=[
            "species","scenario","year","field_id","crop","area_ha",
            *sum([[m, f"{m}_delta", f"{m}_rel_pct"] for m in metrics], [])
        ])

    out = pd.concat(rows, ignore_index=True)
    # Stabil sortieren
    out = out.sort_values(["species", "year", "scenario", "field_id"], kind="stable").reset_index(drop=True)
    return out

def _safe_rel_pct(delta_arr: np.ndarray, base_arr: np.ndarray) -> np.ndarray:
    """Return 100*delta/base without warnings.
    Rules:
      - 0.0 if base≈0 and delta≈0
      - NaN if base≈0 and delta≠0
      - else 100 * delta / base
    NaNs in inputs propagate to NaN in output.
    """
    delta = np.asarray(delta_arr, dtype=np.float64)
    base  = np.asarray(base_arr,  dtype=np.float64)
    out   = np.full(delta.shape, np.nan, dtype=np.float64)

    # Toleranz für "≈0" (vermeidet numerische Artefakte)
    eps = 1e-12

    # Gültige (finite) Werte betrachten
    finite = np.isfinite(delta) & np.isfinite(base)

    # Fälle: base≈0
    base_zero = finite & (np.abs(base) <= eps)
    both_zero = base_zero & (np.abs(delta) <= eps)
    out[both_zero] = 0.0  # 0/0 → 0 %

    # Fälle: base≠0
    nonzero = finite & (np.abs(base) > eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[nonzero] = 100.0 * (delta[nonzero] / base[nonzero])

    # Alle nicht-finiten Ergebnisse als NaN
    out[~np.isfinite(out)] = np.nan
    return out

def _balanced_half_split(counts: np.ndarray, *, year: int,
                         ensure_both_if_total_ge2: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Teilt counts[i] deterministisch in M/F je Feld i, so dass SUM(M)≈SUM(F).
    - Keine neuen Individuen; nur Aufteilung vorhandener Counts.
    - ensure_both_if_total_ge2: wenn Gesamtzahl ≥2, gibt es landschaftsweit min. 1 M und 1 F.
    - Gibt IMMER (m, f) zurück (np.int-Arrays), nie None.
    """
    # Defensive: counts -> integer, keine Negativen
    c = np.asarray(counts if counts is not None else [], dtype=int)
    if c.ndim == 0:
        c = np.array([int(c)], dtype=int)
    c = np.clip(c, 0, None)

    tot = int(c.sum())

    # Fall 0: nichts zu verteilen
    if tot == 0:
        m = np.zeros_like(c)
        f = np.zeros_like(c)
        return m, f

    # Zielzahl Männchen
    target_m = tot // 2
    if ensure_both_if_total_ge2 and tot >= 2:
        target_m = max(1, target_m)

    # Grundverteilung (hälftig je Feld)
    m = c // 2
    rem = (c & 1).astype(bool)  # Felder mit „Rest 1“

    # Aufrunden bis target_m (runde Reste zyklisch, Startindex hängt vom Jahr ab)
    need = int(target_m - m.sum())
    if need > 0:
        idx = np.flatnonzero(rem)
        if idx.size > 0:
            start_idx = int(year) % idx.size
            give = np.concatenate([idx[start_idx:], idx[:start_idx]])[:need]
            m[give] += 1
        # wenn es keine Rest-Indizes gibt, bleibt m wie es ist (immer noch wohl-definiert)

    f = c - m

    # Spezialfall: tot == 1 → alternierendes Geschlecht nach Jahresparität
    if tot == 1:
        mi = int(m.sum()); fi = int(f.sum())
        if mi == 0 and fi == 1 and (year % 2) == 0:
            j = int(np.argmax(f))
            f[j] -= 1; m[j] += 1
        elif fi == 0 and mi == 1 and (year % 2) == 1:
            j = int(np.argmax(m))
            m[j] -= 1; f[j] += 1

    # Garantiekorrektur (nur Umverteilung, keine neuen Individuen)
    sum_m = int(m.sum()); sum_f = int(f.sum())
    if sum_m == 0 and sum_f > 0:
        j = int(np.argmax(f))
        f[j] -= 1; m[j] += 1
    elif sum_f == 0 and sum_m > 0:
        j = int(np.argmax(m))
        m[j] -= 1; f[j] += 1

    return m.astype(int), f.astype(int)


# --- Hilfsfunktion: landschaftsweit balancierter Split ohne Zufall ---
def run_years_with_feedback(fields_gdf, rotation, management, cfg,
                            years: list[int], N_init_first: int = 100,
                            outdir: Optional[Path] = None,
                            extra_filename: Optional[str] = None):
    """
    Mehrjahreslauf mit Cross-Year-Feedback (ohne Bestands-Booster).
    Speichert juvenile_transition(_qc).csv wenn outdir gesetzt ist.
    """
    from .simulate import BirdLandscapeSim

    sim = BirdLandscapeSim(
        fields_gdf[[
            "field_id","x","y","area","perimeter",
            "flower_strip_share","skylark_plots","skylark_plot_area",
            "width_m","height_m"
        ]],
        rotation, management, cfg
    )
    rng_local = sim.rng

    results_by_year: Dict[int, dict] = {}
    start_state = None

    juv_log: List[dict] = []
    juv_qc_rows: List[dict] = []

    for yi, y in enumerate(years):
        res = sim.simulate(
            days=365, year=int(y),
            N_init=(0 if start_state is not None else N_init_first),
            start_state=start_state,
            debug_days=None,
            log_repro=False,
            keep_age_series=False
        )

        results_by_year[y] = res

        if not USE_YEAR_FEEDBACK:
            start_state = None
            continue

        # === Abflug-Index und Bestände am Abflugtag
        T = res["A_m"].shape[0]
        dep_idx = int(np.clip(int(getattr(cfg, "departure_day", 365)), 1, T)) - 1

        A_m_dep = res["A_m"][dep_idx, :].astype(int)
        A_f_dep = res["A_f"][dep_idx, :].astype(int)
        J_tied_mat = res.get("J_tied_total", res.get("J_tied"))
        J_dep = (J_tied_mat[dep_idx, :].astype(int) + res["J_free"][dep_idx, :].astype(int))
        # >>> Debug: Warum ist J_tied_total am Abflugtag ggf. 0?
        last_fledge_idx = int(np.max(np.where(res["daily_fledglings"].sum(axis=1) > 0)[0])) if (
                    res["daily_fledglings"].sum(axis=1) > 0).any() else -1
        last_fledge_day = last_fledge_idx + 1 if last_fledge_idx >= 0 else -1
        juvenile_free_age = int(getattr(cfg, "juvenile_free_age", 30))
        latest_tied_day = (last_fledge_day + juvenile_free_age - 1) if last_fledge_day >= 0 else -1

        if int(os.environ.get("DEBUG_YEAR_XFER", "0")) == 1:
            # Optional nur loggen, wenn am Abflugtag auffällig (z.B. keine gebundenen Juvenilen):
            if int(J_tied_mat[dep_idx, :].sum()) == 0:
                print(f"[YEAR-XFER y={y}] dep_idx={dep_idx}  dep_day={dep_idx + 1}  "
                      f"J_free={int(res['J_free'][dep_idx, :].sum())}  "
                      f"J_tied_total={int(J_tied_mat[dep_idx, :].sum())}  "
                      f"last_fledge_day={last_fledge_day}  latest_tied_day={latest_tied_day}")

        if J_tied_mat is None:
            J_tied_mat = np.zeros_like(res["A_m"])


        # >>> OPTIONALER CHECK: zeigt sofort, ob J_tied_total greift
        if int(os.environ.get("DEBUG_YEAR_XFER", "0")) == 1:
            print(f"[YEAR-XFER y={y}] dep_idx={dep_idx}  "
                  f"J_free={int(res['J_free'][dep_idx, :].sum())}  "
                  f"J_tied_total={int(J_tied_mat[dep_idx, :].sum())}  "
                  f"J_dep_total={int(J_dep.sum())}  "
                  f"A_dep_total={int(A_m_dep.sum() + A_f_dep.sum())}")
            # Sanity: keine Negativen
            assert J_dep.min() >= 0

        # === Parameter
        juv_mat_rate    = float(getattr(cfg, "juvenile_maturation_rate", JUV_MATURATION_RATE))
        winter_surv_ad  = float(getattr(cfg, "winter_survival_adult", WINTER_SURVIVAL_ADULT))
        winter_surv_juv = float(getattr(cfg, "winter_survival_juvenile", WINTER_SURVIVAL_JUV))

        # =====================================================
        # 1) Vorwinter-Reifung (Juv -> adult)  + 50:50-Split
        # =====================================================
        J_mature = rng_local.binomial(J_dep, juv_mat_rate).astype(int)
        add_m_pre, add_f_pre = _balanced_half_split(J_mature, year=int(y))                # <<< 50:50

        # =====================================================
        # 2) Nicht gereifte Juv. überleben den Winter, reifen
        #    im Frühling + 50:50-Split
        # =====================================================
        J_nonmature = (J_dep - J_mature).astype(int)
        J_nonmature_surv = rng_local.binomial(J_nonmature, winter_surv_juv).astype(int)
        add_m_spring, add_f_spring = _balanced_half_split(J_nonmature_surv, year=int(y))  # <<< 50:50

        # =====================================================
        # 3) Erwachsene über Winter: erst gesamt überleben,
        #    dann 50:50-Split (kein M/F-Bias)
        # =====================================================
        pool_tot = (A_m_dep + A_f_dep + add_m_pre + add_f_pre).astype(int)
        surv_tot = rng_local.binomial(pool_tot, winter_surv_ad).astype(int)
        surv_m_ad, surv_f_ad = _balanced_half_split(surv_tot, year=int(y))                # <<< 50:50

        # Frühjahrs-Bestand vor Umverteilung
        surv_m_total = (surv_m_ad + add_m_spring).astype(int)
        surv_f_total = (surv_f_ad + add_f_spring).astype(int)

        # (Optionaler Mini-Guard, ändert nur Zuordnung, erzeugt keine Individuen)
        tot_all = int(surv_m_total.sum() + surv_f_total.sum())
        if tot_all >= 1:
            if int(surv_m_total.sum()) == 0 and int(surv_f_total.sum()) > 0:
                j = int(np.argmax(surv_f_total))
                surv_m_total[j] += 1;
                surv_f_total[j] -= 1
            elif int(surv_f_total.sum()) == 0 and int(surv_m_total.sum()) > 0:
                j = int(np.argmax(surv_m_total))
                surv_f_total[j] += 1;
                surv_m_total[j] -= 1

        # === Winter-Umverteilung
        m_next = sim._winter_redistribute(surv_m_total, year=int(y)).astype(int)
        f_next = sim._winter_redistribute(surv_f_total, year=int(y)).astype(int)

        tot_next = int(m_next.sum() + f_next.sum())
        if tot_next == 0:
            print(f"[ZERO-ARRIVAL DEBUG y={y}] "
                  f"A_dep_total={int(A_m_dep.sum() + A_f_dep.sum())}  "
                  f"J_dep_total={int(J_dep.sum())}  "
                  f"winter_surv_adult={cfg.winter_survival_adult}  "
                  f"juv_mat_rate={cfg.juvenile_maturation_rate}  "
                  f"winter_surv_juv={cfg.winter_survival_juvenile}")

        start_state = {"A_m0": m_next, "A_f0": f_next}

        # === Logging (wie zuvor)
        tot_J_dep            = int(J_dep.sum())
        tot_J_mature         = int(J_mature.sum())
        tot_J_nonmature      = int(J_nonmature.sum())
        tot_J_nonmature_surv = int(J_nonmature_surv.sum())
        obs_mat_rate         = (tot_J_mature / tot_J_dep) if tot_J_dep > 0 else float("nan")
        obs_juv_winter_surv  = (tot_J_nonmature_surv / tot_J_nonmature) if tot_J_nonmature > 0 else float("nan")

        tot_surv_m_total = int(surv_m_total.sum())
        tot_surv_f_total = int(surv_f_total.sum())
        tot_m_next       = int(m_next.sum())
        tot_f_next       = int(f_next.sum())
        denom_next       = max(tot_m_next + tot_f_next, 1)
        sex_share_next_F = (tot_f_next / denom_next) if (tot_m_next + tot_f_next) > 0 else float("nan")
        sex_ratio_next_F_over_M = (tot_f_next / max(tot_m_next, 1)) if (tot_m_next + tot_f_next) > 0 else float("nan")

        v_mature_gt_dep               = int((J_mature > J_dep).sum())
        v_pre_split_mismatch          = 0  # deterministisch
        v_spring_split_mismatch       = 0
        v_nonmature_surv_gt_nonmature = int((J_nonmature_surv > J_nonmature).sum())

        juv_log.append({
            "year": int(y),
            "J_dep": J_dep.tolist(),
            "J_mature": J_mature.tolist(),
            "add_m_pre": add_m_pre.tolist(),
            "add_f_pre": add_f_pre.tolist(),
            "J_nonmature_surv": J_nonmature_surv.tolist(),
            "add_m_spring": add_m_spring.tolist(),
            "add_f_spring": add_f_spring.tolist(),
            "surv_m_total": surv_m_total.tolist(),
            "surv_f_total": surv_f_total.tolist(),
            "start_m_next": m_next.tolist(),
            "start_f_next": f_next.tolist(),
        })

        juv_qc_rows.append({
            "year": int(y),
            "n_fields": int(len(J_dep)),
            "tot_J_dep": tot_J_dep,
            "tot_J_mature": tot_J_mature,
            "tot_J_nonmature": tot_J_nonmature,
            "tot_J_nonmature_surv": tot_J_nonmature_surv,
            "obs_mat_rate": float(obs_mat_rate),
            "obs_juv_winter_surv": float(obs_juv_winter_surv),
            "tot_surv_m_total": tot_surv_m_total,
            "tot_surv_f_total": tot_surv_f_total,
            "tot_start_m_next": tot_m_next,
            "tot_start_f_next": tot_f_next,
            "sex_share_next_F": float(sex_share_next_F),
            "sex_ratio_next_F_over_M": float(sex_ratio_next_F_over_M),
            "violation_J_mature_gt_J_dep": v_mature_gt_dep,
            "violation_pre_split_mismatch": v_pre_split_mismatch,
            "violation_spring_split_mismatch": v_spring_split_mismatch,
            "violation_J_nonmature_surv_gt_J_nonmature": v_nonmature_surv_gt_nonmature,
            "param_juv_mat_rate": juv_mat_rate,
            "param_winter_surv_ad": winter_surv_ad,
            "param_winter_surv_juv": winter_surv_juv,
        })

        tot_m_next = int(m_next.sum())
        tot_f_next = int(f_next.sum())
        tot_next = tot_m_next + tot_f_next

        if tot_next >= 2:
            f_share = tot_f_next / tot_next
            if f_share < 0.30 or f_share > 0.70:
                print(f"[warn][year {y}] Ungewöhnliche Geschlechterverteilung am Start: F-Anteil={f_share:.2f}")

    # === Speichern
    species_name = str(getattr(cfg, "name", "species"))
    if outdir is not None:
        base_dir = Path(outdir)
        target_dir = (base_dir if base_dir.name.lower() == species_name.lower() else base_dir / species_name) / "analysis"
        target_dir.mkdir(parents=True, exist_ok=True)
        suf = f"_{extra_filename}" if (extra_filename and str(extra_filename).strip()) else ""
        pd.DataFrame(juv_log).to_csv(target_dir / f"juvenile_transition{suf}.csv", index=False)
        pd.DataFrame(juv_qc_rows).to_csv(target_dir / f"juvenile_transition_qc{suf}.csv", index=False)
        print(f"[analysis] gespeichert: {target_dir / f'juvenile_transition{suf}.csv'}")
        print(f"[analysis] gespeichert: {target_dir / f'juvenile_transition_qc{suf}.csv'}")

    return results_by_year


# =========================================================
