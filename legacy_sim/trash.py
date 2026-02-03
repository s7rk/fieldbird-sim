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

from __future__ import annotations

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

# === (2) Werte, die aus OPTS abgeleitet werden – NICHT ändern ===
YEARS_TO_SIM = list(range(1, int(OPTS.N_YEARS) + 1))
YEAR_TO_SHOW = max(1, min(int(OPTS.YEAR_TO_SHOW), int(OPTS.N_YEARS)))
DAY_TO_SHOW  = max(1, min(int(OPTS.DAY_TO_SHOW), TOT_DAYS))  # <— clamp auf [1..365]
SPECIES_LIST = [
    "Feldlerche",
    "Rebhuhn"]  # weitere Arten hier ergänzen (Strings)

# === (3) Plot-Ausgabe (ein/aus) ===
DO_HEATMAPS = True  # True: pro Jahr Heatmaps schreiben
DO_MAPS     = True  # True: Karten für YEAR_TO_SHOW erzeugen

# === (4) Feldlayout (synthetische Geometrien) ===
USE_RECT_PACKING   = True   # True: Rechteck-Packlayout; False: einfache Zentroid-Quadrate
TARGET_ROW_WIDTH_M = None   # None => automatisch aus Feldgrößen ableiten

# === (5) Hecken-Darstellung/Quelle ===
USE_HEDGES_FROM_CSV = True  # True: Hecken aus CSV (HEDGES_CSV) lesen
AUTO_HEDGES_FROM_ADJ = False# True: sehr einfache Auto-Nachbarschaft als Hecken
DRAW_HEDGE_BUFFERS   = True # True: Pufferflächen um Hecken visualisieren
HEDGE_BUFFER_M       = 5.0  # Pufferbreite in Metern

# === (6) Lerchenfenster/Refugien (tuning, biologisch plausible Defaults) ===
ELIGIBLE_REFUGE_CROPS = {"Winterweizen", "Sommergerste"}  # Kulturen, in denen Fenster wirken
REFUGE_ASSIGN_MULT    = 15.0  # Stärke, mit der neue Nester Fenster bevorzugen (vs. reine Flächenquote)
PULSE_PROTECT_SCALE   = 4.0   # wie stark refuge_share Puls-Störungen abschirmt
REFUGE_SHARE_CAP      = 0.6   # max. Anteil Nester in Fenstern (Kappung)

# === (7) Rein visuelle Overlays ===
DRAW_FLOWER_RINGS = True  # True: Blühstreifen als Randringe anzeigen

# === (8) Dateipfade (falls reale Inputs; sonst Fallback auf Templates) ===
FIELDS_CSV           = "inputs/fields_template.csv"
ROTATION_CSV_MAIN    = "inputs/rotation.csv"
ROTATION_CSV_TPL     = "inputs/rotation_template.csv"
MANAGEMENT_CSV_MAIN  = "inputs/management.csv"
MANAGEMENT_CSV_TPL   = "inputs/management_template.csv"
HEDGES_CSV           = "inputs/hedges_template.csv"
SKYLARK_WINDOWS_CSV  = Path("configs/skylark_windows.csv")
CROP_CALENDAR_CSV    = "configs/crop_calendar.csv"

# === (9) Szenarien (Beispiele; werden im Hauptteil angewendet) ===
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
        "A_m","A_f","A_f_breeding","J_free","J_tied","new_nests","fledglings","N_total",
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
    Saves:
      - data/: fields_flat.csv, rotation.csv, management.csv, hedges.(geojson|csv)
      - year_X/arrays/*.npz, year_X/usage.csv, year_X/daily_totals.csv,
        year_X/repro_log.csv (if res['debug_repro'] present),
        year_X/debug_daily.csv (if res['debug']['daily'] present)
      - analysis/fields_layout_realized.csv  (einmalig)
      - run_meta.json
    """
    outdir = pathlib.Path(outdir)
    try:
        resolved_outdir = outdir.resolve()
    except Exception:
        resolved_outdir = outdir
    parent_name = resolved_outdir.parent.name if resolved_outdir.parent != resolved_outdir else ""
    run_id = parent_name or resolved_outdir.name

    (outdir / "data").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    # ---------------- Inputs snapshot ----------------
    fields_flat = fields_gdf.drop(columns=[c for c in ["geometry"] if c in fields_gdf.columns]).copy()
    (outdir / "data" / "fields_flat.csv").parent.mkdir(parents=True, exist_ok=True)
    fields_flat.to_csv(outdir / "data" / "fields_flat.csv", index=False)
    rotation.to_csv(outdir / "data" / "rotation.csv", index=False)
    management.to_csv(outdir / "data" / "management.csv", index=False)

    # Geometrie: beste Formate versuchen, robust fallen lassen
    try:
        fields_gdf.to_file(outdir / "data" / "fields.gpkg", layer="fields", driver="GPKG")
    except Exception as e:
        print("[warn] writing fields.gpkg failed:", e)

    try:
        # Requires geopandas >= 0.14 and pyarrow
        fields_gdf.to_parquet(outdir / "data" / "fields.parquet")
    except Exception as e:
        print("[warn] writing fields.parquet failed:", e)

    # GeoJSON fallback (breit lesbar)
    try:
        fields_gdf.to_file(outdir / "data" / "fields.geojson", driver="GeoJSON")
    except Exception as e:
        print("[warn] writing fields.geojson failed:", e)

    if hedges_gdf is not None and len(hedges_gdf) > 0:
        try:
            hedges_gdf.to_file(outdir / "data" / "hedges.geojson", driver="GeoJSON")
        except Exception:
            # Fallback: CSV ohne Geometrie
            hedges_gdf.drop(columns=[c for c in ["geometry"] if c in hedges_gdf.columns]).to_csv(
                outdir / "data" / "hedges.csv", index=False
            )

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

        # ---------------- BirdSIM daily full export ----------------
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

    # ---------------- einmaliger Layout-Export (species-level) ----------------
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        save_fields_layout_csv(fields_flat, analysis_dir, filename="fields_layout_realized.csv")
        print(f"[layout] gespeichert: {analysis_dir / 'fields_layout_realized.csv'}")
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
        rows = []
        geoms = []
        for k, (poly, src) in enumerate(cover_parts, start=0):
            minx, miny, maxx, maxy = poly.bounds
            w = maxx - minx
            h = maxy - miny
            rows.append({
                "field_id": start_id + k,
                "area": poly.area / 10_000.0,      # ha
                "width_m": w,
                "height_m": h,
                "flower_strip_share": 0.0,
                "skylark_plots": 0,
                "skylark_plot_area": 0.0,
                "x": minx, "y": miny,
                "perimeter": poly.length,
                "is_permanent_cover": True,
                "cover_source": src
            })
            geoms.append(poly)

        cover_gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=gdf.crs)
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
#  (module 2/7) Parsers, layout, hedges, plots
# =========================================================
def parse_days(x) -> np.ndarray:
    """
    Parsen von Tageslisten (DOY):
    - akzeptiert: None/NaN, Zahl, String, Liste/Tuple/np.ndarray/pd.Series
    - Trennzeichen: ',', ';', '|', '/', Leerraum, Zeilenumbrüche
    - Ranges werden expandiert: '120-125', '120–125', '120—125' (inkl. Unicode-Dashes)
    - Duplikate entfernt, sortiert
    - gültiger Bereich: 1..365 (alles andere wird verworfen)
    """
    import re
    import numpy as np
    import math

    # --- Früh-Exit bei None/NaN ---
    if x is None:
        return np.array([], dtype=int)
    if isinstance(x, float) and math.isnan(x):
        return np.array([], dtype=int)

    # --- In String vereinheitlichen ---
    #  - Sequenzen (Liste/Tuple/ndarray/Series) -> kommaseparierter String
    try:
        import pandas as pd  # optional, falls vorhanden
        is_series = isinstance(x, pd.Series)
    except Exception:
        is_series = False

    if isinstance(x, (list, tuple, np.ndarray)) or is_series:
        xs = ",".join(str(v) for v in (x.tolist() if hasattr(x, "tolist") else x))
    else:
        xs = str(x)

    xs = xs.strip()
    if xs == "" or xs.lower() in {"na", "nan", "none", "null"}:
        return np.array([], dtype=int)

    # --- Alle gängigen Trenner vereinheitlichen (außer '-','–','—' für Ranges) ---
    # Ersetze ; | / \n \t und mehrere Leerzeichen durch Komma
    xs = re.sub(r"[;\|/]", ",", xs)
    xs = re.sub(r"[\r\n\t ]+", ",", xs)
    # Kommas bereinigen (doppelte zu einfachen)
    xs = re.sub(r",+", ",", xs).strip(",")
    if xs == "":
        return np.array([], dtype=int)

    # --- Tokens zerlegen & parsen ---
    vals = []
    for tok in xs.split(","):
        t = tok.strip()
        if not t:
            continue

        # Range? Unterstützt '-', '–' (EN dash), '—' (EM dash)
        m = re.match(r"^\s*(\d+)\s*[-–—]\s*(\d+)\s*$", t)
        if m:
            a = int(m.group(1)); b = int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            # Expand inklusiv
            vals.extend(range(lo, hi + 1))
            continue

        # Einzelwert (Zahl)
        try:
            v = int(float(t))
            vals.append(v)
        except Exception:
            # ignorieren, falls unlesbar
            continue

    if not vals:
        return np.array([], dtype=int)

    # --- DOY-Bounds anwenden & aufräumen ---
    arr = np.array(vals, dtype=int)
    # gültiger DOY-Bereich
    arr = arr[(arr >= 1) & (arr <= 365)]
    if arr.size == 0:
        return np.array([], dtype=int)

    # Duplikate entfernen & sortieren
    arr = np.unique(arr)
    return arr


def parse_window(x) -> Optional[Tuple[int, int]]:
    """
    Parse a DOY window into (start, end) with 1..365 bounds.
    Accepts:
      - None/NaN -> None
      - "120-150", "120–150", "120—150", "120 to 150"
      - "120,150" or "120;150"
      - tuple/list/array like (120, 150)
      - dict with keys {"start","end"} or {"a","b"}
    Returns (min(a,b), max(a,b)) or None on failure.
    """
    import math, re
    import numpy as np

    # helpers
    def _clamp(v: float) -> int:
        return int(max(1, min(365, int(round(v)))))

    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None

    # tuple/list/array
    if isinstance(x, (tuple, list, np.ndarray)):
        vals = []
        for v in x:
            try:
                vals.append(float(v))
            except Exception:
                pass
        if len(vals) >= 2:
            a, b = _clamp(vals[0]), _clamp(vals[1])
            return (a, b) if a <= b else (b, a)
        return None

    # dict mit start/end
    if isinstance(x, dict):
        for ka, kb in (("start","end"), ("a","b"), ("von","bis")):
            if ka in x and kb in x:
                try:
                    a = _clamp(float(x[ka]))
                    b = _clamp(float(x[kb]))
                    return (a, b) if a <= b else (b, a)
                except Exception:
                    return None
        return None

    # string-normalisierung
    xs = str(x).strip()
    if xs == "" or xs.lower() in {"na", "nan", "none", "null"}:
        return None

    # unify separators: en/em dash → '-', ' to ' → '-', commas/semicolons/whitespace → single comma
    xs = xs.replace("–", "-").replace("—", "-")
    xs = re.sub(r"\bto\b", "-", xs, flags=re.IGNORECASE)
    xs = re.sub(r"[;,]", ",", xs)
    xs = re.sub(r"[\r\n\t ]+", "", xs)

    # Fälle: "a-b" oder "a,b"
    if "-" in xs:
        parts = xs.split("-")
    elif "," in xs:
        parts = xs.split(",")
    else:
        # als Fallback: zwei Zahlen aus dem String extrahieren
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", xs)
        parts = nums[:2]

    if len(parts) != 2:
        return None

    try:
        a = _clamp(float(parts[0]))
        b = _clamp(float(parts[1]))
    except Exception:
        return None

    return (a, b) if a <= b else (b, a)


def in_window(doy: int, win: Optional[Tuple[int, int]]) -> bool:
    if win is None:
        return False
    try:
        a, b = int(win[0]), int(win[1])
    except Exception:
        return False
    if a > b:
        a, b = b, a
    try:
        td = int(globals().get("TOT_DAYS", 365))
    except Exception:
        td = 365
    try:
        d = int(doy)
    except Exception:
        return False
    if not (1 <= d <= td):
        return False
    a = max(1, min(td, a))
    b = max(1, min(td, b))
    return a <= d <= b



def ensure_rect_sides(fields: pd.DataFrame) -> pd.DataFrame:
    df = fields.copy().reset_index(drop=True)

    # Spalten sicherstellen & numerisch casten
    if "width_m" not in df.columns:  df["width_m"]  = np.nan
    if "height_m" not in df.columns: df["height_m"] = np.nan
    df["width_m"]  = pd.to_numeric(df["width_m"],  errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")

    area_m2 = pd.to_numeric(df.get("area", np.nan), errors="coerce").fillna(0.0) * 10_000.0

    # Falls area fehlt/≤0 aber width*height vorhanden → daraus ableiten
    wh_area = (df["width_m"] * df["height_m"]).where(df["width_m"].notna() & df["height_m"].notna())
    area_m2 = np.where((area_m2 <= 0.0) & wh_area.notna(), wh_area, area_m2)

    # Beide Seiten fehlend → Quadrat aus Fläche
    both_na = df["width_m"].isna() & df["height_m"].isna()
    if both_na.any():
        side = np.sqrt(np.maximum(area_m2, 1e-6))
        df.loc[both_na, "width_m"]  = side[both_na]
        df.loc[both_na, "height_m"] = side[both_na]

    # Nur width fehlt
    w_na = df["width_m"].isna() & df["height_m"].notna()
    if w_na.any():
        df.loc[w_na, "width_m"] = np.maximum(area_m2[w_na] / df.loc[w_na, "height_m"].to_numpy(), 1e-6)

    # Nur height fehlt
    h_na = df["height_m"].isna() & df["width_m"].notna()
    if h_na.any():
        df.loc[h_na, "height_m"] = np.maximum(area_m2[h_na] / df.loc[h_na, "width_m"].to_numpy(), 1e-6)

    return df

def rect_shelf_pack(fields: pd.DataFrame, target_row_width: Optional[float] = None):
    from shapely.geometry import Polygon
    df = ensure_rect_sides(fields).copy().reset_index(drop=True)

    W = df["width_m"].astype(float).to_numpy()
    H = df["height_m"].astype(float).to_numpy()
    n = len(df)

    if target_row_width is None:
        # heuristik: Gesamtbreite / sqrt(n) → brauchbare Regalbreite
        target_row_width = float(np.sum(W) / max(math.sqrt(max(n, 1)), 1.0))

    order = np.argsort(-H)  # hohe zuerst
    rows, cur, cur_w = [], [], 0.0
    for idx in order:
        w = float(W[idx])
        if cur_w + w <= target_row_width or len(cur) == 0:
            cur.append(idx); cur_w += w
        else:
            rows.append(cur); cur = [idx]; cur_w = w
    if cur:
        rows.append(cur)

    geoms = [None] * n
    x0, y0 = 0.0, 0.0
    for row in rows:
        row_h = float(max(H[i] for i in row))
        x = x0
        for i in row:
            w, h = float(W[i]), float(H[i])
            # robust gegen winzige/negative Werte (sollten schon in ensure_rect_sides abgefangen sein)
            w = max(w, 1e-6); h = max(h, 1e-6)
            poly = Polygon([(x, y0), (x + w, y0), (x + w, y0 + h), (x, y0 + h)])
            # selten: invalid (Numerik) → sauber puffern
            if not poly.is_valid:
                poly = poly.buffer(0)
            geoms[i] = poly
            x += w
        y0 += row_h

    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:3857")

    # Geometrie-basierte Kennzahlen
    gdf["area"] = gdf.geometry.area / 10_000.0  # ha
    gdf["perimeter"] = gdf.geometry.length

    c = gdf.geometry.centroid
    gdf["x"] = c.x
    gdf["y"] = c.y
    return gdf

def build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs_df):
    """
    Erzeuge Hecken-Geometrien entlang gemeinsam genutzter Feldgrenzen.
    - robust gegen invalide Geometrien (buffer(0))
    - extrahiert nur linienhafte Überschneidungen (keine Punkte/Flächen)
    - mergen mehrerer Teilsegmente (linemerge / unary_union)
    - filtert sehr kurze Kanten (MIN_LEN)
    - dedupliziert (f1,f2) und (f2,f1)
    Returns: GeoDataFrame oder None, wenn nichts gefunden.
    """
    import numpy as np
    from shapely.ops import unary_union, linemerge
    from shapely.geometry import LineString, MultiLineString, GeometryCollection

    MIN_LEN = 0.10  # Meter; alles darunter wird ignoriert

    if hedge_pairs_df is None or len(hedge_pairs_df) == 0:
        return None
    if "geometry" not in fields_gdf.columns:
        return None

    # Feld-Polygone mappen & ggf. "cleanen"
    id2poly = {}
    for fid, geom in zip(fields_gdf["field_id"].astype(int), fields_gdf.geometry):
        if geom is None:
            continue
        try:
            g = geom if geom.is_valid else geom.buffer(0)
        except Exception:
            # falls buffer(0) scheitert, Geometrie überspringen
            continue
        if g is not None and (not g.is_empty):
            id2poly[int(fid)] = g

    rows = []
    for _, r in hedge_pairs_df.iterrows():
        try:
            f1 = int(r["f1"]); f2 = int(r["f2"])
        except Exception:
            continue
        if f1 == f2 or f1 not in id2poly or f2 not in id2poly:
            continue

        q = float(r.get("hedge_quality", 1.0))
        # optional clamp
        q = float(np.clip(q, 0.0, 1.0))

        p1 = id2poly[f1]
        p2 = id2poly[f2]

        # Grenze schneiden
        try:
            inter = p1.boundary.intersection(p2.boundary)
        except Exception:
            # notfalls Polygone "cleanen" und erneut
            try:
                p1c = p1 if p1.is_valid else p1.buffer(0)
                p2c = p2 if p2.is_valid else p2.buffer(0)
                inter = p1c.boundary.intersection(p2c.boundary)
            except Exception:
                continue

        if inter.is_empty:
            continue

        # Linien extrahieren & mergen
        line_geoms = []
        try:
            if isinstance(inter, LineString):
                if inter.length > MIN_LEN:
                    line_geoms.append(inter)
            elif isinstance(inter, MultiLineString):
                for g in inter.geoms:
                    if g.length > MIN_LEN:
                        line_geoms.append(g)
            elif isinstance(inter, GeometryCollection):
                for g in inter.geoms:
                    if isinstance(g, (LineString, MultiLineString)):
                        if isinstance(g, LineString) and g.length > MIN_LEN:
                            line_geoms.append(g)
                        elif isinstance(g, MultiLineString):
                            for gg in g.geoms:
                                if gg.length > MIN_LEN:
                                    line_geoms.append(gg)
            # sonst (Points/Polygone) ignorieren
        except Exception:
            continue

        if not line_geoms:
            continue

        # zusammenführen (ein Segment, wenn möglich)
        try:
            merged = linemerge(unary_union(line_geoms))
        except Exception:
            # fallback: nimm die einzelnen Teile
            merged = unary_union(line_geoms)

        # Multi vs Single
        geoms_out = []
        if isinstance(merged, (LineString, MultiLineString)):
            geoms_out = [merged] if isinstance(merged, LineString) else list(merged.geoms)
        else:
            # seltene Fälle (z. B. GeometryCollection) — extrahiere Lines
            if hasattr(merged, "geoms"):
                geoms_out = [g for g in merged.geoms if isinstance(g, LineString)]
        geoms_out = [g for g in geoms_out if g.length > MIN_LEN]

        if not geoms_out:
            continue

        # (f1,f2) deduplizieren: immer sortierte Key-Reihenfolge verwenden
        a, b = (f1, f2) if f1 < f2 else (f2, f1)
        for g in geoms_out:
            rows.append({"f1": a, "f2": b, "hedge_quality": q, "geometry": g})

    if not rows:
        return None

    # Duplikate mit gleicher (f1,f2) verschmelzen
    df = pd.DataFrame(rows)
    grouped = []
    for (a, b), sub in df.groupby(["f1", "f2"], sort=False):
        try:
            geom = unary_union(list(sub["geometry"]))
        except Exception:
            # fallback: längstes Segment behalten
            geom = max(sub["geometry"], key=lambda gg: gg.length)
        # hedge_quality: Mittelwert (oder max); hier Mittelwert
        q = float(np.mean(sub["hedge_quality"]))
        grouped.append({"f1": int(a), "f2": int(b), "hedge_quality": q, "geometry": geom})

    if not grouped:
        return None

    return gpd.GeoDataFrame(grouped, geometry="geometry", crs=fields_gdf.crs)

def save_fields_layout_csv(fields_df, out_dir: Path, filename: str = "fields_layout_realized.csv"):
    """
    Speichert das Feld-Layout (x,y,width_m,height_m) in Meter-Koordinaten je field_id.
    Akzeptierte Eingaben (mind. eine Variante):
      - x0,y0,x1,y1   (beliebige Reihenfolge; wird normalisiert auf min/max)
      - x,y,w,h       (wird zu width_m/height_m umbenannt)
      - x,y,width_m,height_m
      - geometry      (Bounding-Box wird benutzt; CRS sollte metrisch sein, z.B. EPSG:3857)
    """
    import numpy as np
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    df = fields_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # field_id sicherstellen (fallback: index)
    if "field_id" not in df.columns:
        df["field_id"] = np.arange(1, len(df) + 1)
    df["field_id"] = pd.to_numeric(df["field_id"], errors="coerce").astype("Int64")

    # Normalisieren auf x,y,width_m,height_m
    has_xywh   = {"x", "y", "width_m", "height_m"}.issubset(df.columns)
    has_xywh2  = {"x", "y", "w", "h"}.issubset(df.columns)
    has_bbox   = {"x0", "y0", "x1", "y1"}.issubset(df.columns)
    has_geom   = "geometry" in df.columns

    if has_bbox:
        # toleriert vertauschte Ecken
        x0 = pd.to_numeric(df["x0"], errors="coerce")
        y0 = pd.to_numeric(df["y0"], errors="coerce")
        x1 = pd.to_numeric(df["x1"], errors="coerce")
        y1 = pd.to_numeric(df["y1"], errors="coerce")
        df["x"] = np.minimum(x0, x1)
        df["y"] = np.minimum(y0, y1)
        df["width_m"]  = (np.maximum(x0, x1) - np.minimum(x0, x1)).astype(float)
        df["height_m"] = (np.maximum(y0, y1) - np.minimum(y0, y1)).astype(float)
    elif has_xywh2:
        df = df.rename(columns={"w": "width_m", "h": "height_m"})
        # cast
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df["width_m"]  = pd.to_numeric(df["width_m"], errors="coerce")
        df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    elif has_xywh:
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df["width_m"]  = pd.to_numeric(df["width_m"], errors="coerce")
        df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    elif has_geom:
        # Fallback: Bounding Box aus Geometrie (Achtung: CRS muss metrisch sein!)
        xs, ys, ws, hs = [], [], [], []
        for g in df["geometry"]:
            if g is None or getattr(g, "is_empty", True):
                xs.append(np.nan); ys.append(np.nan); ws.append(np.nan); hs.append(np.nan)
                continue
            try:
                x0, y0, x1, y1 = g.bounds
                x = min(x0, x1); y = min(y0, y1)
                w = abs(x1 - x0); h = abs(y1 - y0)
            except Exception:
                x = y = w = h = np.nan
            xs.append(x); ys.append(y); ws.append(w); hs.append(h)
        df["x"], df["y"], df["width_m"], df["height_m"] = xs, ys, ws, hs
    else:
        got = list(df.columns)
        raise ValueError(
            "save_fields_layout_csv: unbekanntes Feldschema. "
            "Erwartet eine der Varianten: "
            "{x0,y0,x1,y1} ODER {x,y,w,h} ODER {x,y,width_m,height_m} ODER {geometry}. "
            f"Vorhandene Spalten: {got}"
        )

    # Validierungen
    for col in ["x", "y", "width_m", "height_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # unplausible Werte erkennen
    bad = (
        df["x"].isna() | df["y"].isna() |
        ~np.isfinite(df["x"]) | ~np.isfinite(df["y"]) |
        df["width_m"].isna() | df["height_m"].isna() |
        ~np.isfinite(df["width_m"]) | ~np.isfinite(df["height_m"]) |
        (df["width_m"] <= 0) | (df["height_m"] <= 0)
    )
    if bad.any():
        # Zeige ein paar problematische Zeilen zur Diagnose
        sample = df.loc[bad, ["field_id", "x", "y", "width_m", "height_m"]].head(5)
        raise ValueError(
            "save_fields_layout_csv: ungültige oder fehlende Werte in x/y/width_m/height_m "
            f"({bad.sum()} Zeile(n)). Beispiel:\n{sample}"
        )

    # Doppelte field_id vermeiden (letzten behalten)
    df = df.drop_duplicates(subset=["field_id"], keep="last")

    # Ausgabe
    cols = ["field_id", "x", "y", "width_m", "height_m"]
    df_out = df[cols].copy().sort_values("field_id").astype({
        "field_id": int, "x": float, "y": float, "width_m": float, "height_m": float
    })

    target = out_dir / filename
    df_out.to_csv(target, index=False)
    return target

def rect_or_centroid_layout(fields: pd.DataFrame):
    """
    Liefert ein GeoDataFrame mit Rechtecken:
      - Wenn USE_RECT_PACKING=True: nutzt rect_shelf_pack(fields, TARGET_ROW_WIDTH_M)
      - Sonst: baut Quadrate mit Fläche 'area' (ha) um (x,y)-Zentren.
        Fehlen x/y, wird ein einfacher Gitter-Fallback erzeugt.
    Ergänzt/überschreibt: x, y, area (aus Geometrie), perimeter.
    CRS: EPSG:3857 (Meter)
    """
    from shapely.geometry import Polygon

    df = fields.copy().reset_index(drop=True)
    if USE_RECT_PACKING:
        return rect_shelf_pack(df, target_row_width=TARGET_ROW_WIDTH_M)

    # --- numerische Spalten sicherstellen ---
    df["area"] = pd.to_numeric(df.get("area", np.nan), errors="coerce").fillna(0.0)

    # x/y vorhanden?
    has_xy = {"x", "y"}.issubset(df.columns)
    if not has_xy or df["x"].isna().any() or df["y"].isna().any():
        # Fallback: einfache Gitter-Positionen
        n = len(df)
        cols = int(math.ceil(math.sqrt(max(n, 1))))
        # Abstand abhängig von max Feldfläche (robust)
        max_area_ha = float(np.nanmax(df["area"].values)) if n > 0 else 1.0
        spacing = float(math.sqrt(max(max_area_ha, 1e-6) * 10_000.0) * 1.5 + 10.0)
        xs, ys = [], []
        for i in range(n):
            r, c = divmod(i, cols)
            xs.append(c * spacing)
            ys.append(r * spacing)
        df["x"] = xs
        df["y"] = ys
    else:
        df["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0.0)
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)

    # --- Geometrien bauen (Quadrate um (x,y)) ---
    geoms = []
    for _, r in df.iterrows():
        A_ha = float(r["area"])
        side = math.sqrt(max(A_ha, 1e-9) * 10_000.0)  # Meter
        h = side / 2.0
        cx, cy = float(r["x"]), float(r["y"])
        poly = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])
        if not poly.is_valid:
            poly = poly.buffer(0)
        geoms.append(poly)

    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:3857")

    # Kennzahlen aus Geometrie ableiten (konsistent halten)
    c = gdf.geometry.centroid
    gdf["x"] = c.x
    gdf["y"] = c.y
    gdf["area"] = gdf.geometry.area / 10_000.0  # ha
    gdf["perimeter"] = gdf.geometry.length
    return gdf

def heatmap(N_mat, field_ids, title, vmax=None, savepath=None):
    """
    Heatmap über Tage (Zeilen) × Felder (Spalten).
    - prüft Shape & field_ids-Länge (bei Mismatch werden IDs 1..F verwendet)
    - reduziert xticks bei vielen Feldern
    - wählt standardmäßig vmax als 99. Perzentil (robust gegen Ausreißer)
    """
    from matplotlib.ticker import MaxNLocator

    N = np.asarray(N_mat)
    if N.ndim != 2:
        raise ValueError("heatmap: N_mat muss 2D (days × fields) sein.")
    days, F = N.shape

    fids = np.asarray(field_ids)
    if fids.size != F:
        fids = np.arange(1, F + 1)

    vmin = 0.0
    if vmax is None:
        finite = np.isfinite(N)
        vmax = float(np.nanpercentile(N[finite], 99.0)) if finite.any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(max(6, F * 0.25), 6))
    im = ax.imshow(N, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")

    ax.set_title(title)
    ax.set_ylabel("Tag (DOY)")
    ax.set_xlabel("Felder (field_id)")

    # X-Ticks: bei vielen Feldern ausdünnen
    if F <= 40:
        ax.set_xticks(np.arange(F))
        ax.set_xticklabels(fids.astype(int), rotation=90)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=40, integer=True))
        ax.set_xticklabels([])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Individuen")

    if savepath:
        p = Path(savepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_fig(p)
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)

def plot_map(fields_gdf, rotation_df, year: int,
             hedges_gdf=None, title="Karte",
             draw_flower_rings=True, draw_hedge_buffers=True, hedge_buffer_m=5.0,
             day_vals=None, usage_vals=None, savepath=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(9, 7))

    # --- Datenkopie & defensive Geometrie-Reparatur ---
    fg = fields_gdf.copy()
    if "geometry" in fg.columns:
        try:
            fg["geometry"] = fg["geometry"].apply(
                lambda g: (g if (g is None or g.is_valid) else g.buffer(0))
            )
        except Exception:
            pass

    # --- Kultur pro Feld (Jahr) ---
    rot_y = rotation_df[rotation_df["year"] == year]
    crops_map = dict(zip(rot_y["field_id"].astype(int), rot_y["crop"].astype(str)))
    fg["crop"] = fg["field_id"].astype(int).map(crops_map).fillna("Unbekannt")

    # --- Farbpalette (bekannte + Fallback für Unbekannte) ---
    base_palette = {
        "Winterweizen": "#9ecae1", "Sommergerste": "#c7e9c0", "Mais": "#fff7bc",
        "Raps": "#fdd0a2", "Zuckerrübe": "#c994c7", "Ackerbohne": "#a1d99b",
        "Brache": "#d9d9d9", "Unbekannt": "#f0f0f0"
    }
    unique_crops = list(pd.unique(fg["crop"]))
    fallback_colors = plt.cm.tab20.colors
    pal = {}
    k = 0
    for c in unique_crops:
        if c in base_palette:
            pal[c] = base_palette[c]
        else:
            pal[c] = fallback_colors[k % len(fallback_colors)]
            k += 1

    fg.plot(
        ax=ax,
        facecolor=fg["crop"].map(pal),
        edgecolor="black",
        linewidth=0.8,
        alpha=0.95
    )

    # --------- Blühstreifen-Ringe (robust & defensiv) ---------
    rings = []
    if draw_flower_rings and "flower_strip_share" in fg.columns and fg["flower_strip_share"].fillna(0).gt(0).any():
        for _, fr in fg.iterrows():
            try:
                fs = float(fr.get("flower_strip_share", 0.0) or 0.0)
                if fs <= 0.0:
                    continue
                geom = fr.geometry
                if geom is None or geom.is_empty:
                    continue
                A = float(fr.get("area", 0.0)) * 10_000.0
                P = float(fr.get("perimeter", 0.0))
                if A <= 0.0 or P <= 0.0:
                    continue
                w = (fs * A) / max(P, 1e-9)
                if w <= 0.0:
                    continue
                try:
                    inner = geom.buffer(-w)
                except Exception:
                    geom = geom.buffer(0)
                    inner = geom.buffer(-w)
                raw_ring = geom if inner.is_empty else geom.difference(inner)
                if raw_ring.is_empty:
                    continue
                if isinstance(raw_ring, Polygon):
                    rings.append(raw_ring)
                elif isinstance(raw_ring, MultiPolygon):
                    rings.extend([g for g in raw_ring.geoms if not g.is_empty])
                elif isinstance(raw_ring, GeometryCollection):
                    for g in raw_ring.geoms:
                        if isinstance(g, Polygon):
                            rings.append(g)
                        elif isinstance(g, MultiPolygon):
                            rings.extend([gg for gg in g.geoms if not gg.is_empty])
            except Exception:
                pass

    if rings:
        ring_gdf = gpd.GeoDataFrame(geometry=rings, crs=fg.crs)
        ring_gdf.plot(
            ax=ax,
            facecolor="#fff2b2",
            edgecolor="#fdae6b",
            linewidth=1.0,
            alpha=0.7,
            hatch="///",
            zorder=4
        )

    # --------- Hecken + optionale Buffer ---------
    if hedges_gdf is not None and len(hedges_gdf) > 0:
        q = hedges_gdf.get("hedge_quality", pd.Series([0.6] * len(hedges_gdf)))
        lw = 1.0 + 3.0 * (q - q.min()) / (q.max() - q.min() + 1e-9)
        try:
            hedges_gdf.plot(ax=ax, color="#237a00", linewidth=lw, alpha=0.95, zorder=5)
        except Exception:
            # Fallback ohne variable linewidth
            hedges_gdf.plot(ax=ax, color="#237a00", linewidth=1.5, alpha=0.95, zorder=5)

        if draw_hedge_buffers:
            buffers = hedges_gdf.copy()
            try:
                buffers["geometry"] = buffers.buffer(hedge_buffer_m, cap_style=2, join_style=2)
                for g in buffers.geometry:
                    if g is None or g.is_empty:
                        continue
                    if isinstance(g, Polygon):
                        geoms_to_draw = [g]
                    elif isinstance(g, MultiPolygon):
                        geoms_to_draw = [gg for gg in g.geoms if not gg.is_empty]
                    else:
                        geoms_to_draw = []
                    for gg in geoms_to_draw:
                        x, y = gg.exterior.xy
                        ax.fill(x, y, facecolor="#a1d99b", edgecolor="none", alpha=0.35, hatch="xx", zorder=4)
            except Exception as e:
                print("Hedge buffer zeichnen fehlgeschlagen:", e)

    # --------- Dichteoverlay (Tag / Usage) ---------
    vals, label = None, None
    raw = day_vals if day_vals is not None else usage_vals
    if raw is not None:
        # dict/Series (per field_id) → in Feld-Reihenfolge bringen
        if isinstance(raw, dict):
            m = {int(k): float(v) for k, v in raw.items()}
            vals = fg["field_id"].astype(int).map(m).astype(float).to_numpy()
            label = "Vögel (Tag)" if day_vals is not None else "Vögel (Usage)"
        elif isinstance(raw, pd.Series):
            s = raw.copy()
            try:
                s.index = s.index.astype(int)
                vals = fg["field_id"].astype(int).map(s).astype(float).to_numpy()
            except Exception:
                vals = np.asarray(s.values, dtype=float)
            label = "Vögel (Tag)" if day_vals is not None else "Vögel (Usage)"
        else:
            vals = np.asarray(raw, dtype=float)
            label = "Vögel (Tag)" if day_vals is not None else "Vögel (Usage)"

    if vals is not None:
        if len(vals) != len(fg):
            print(f"[warn] plot_map: length mismatch in overlay values (got {len(vals)}, fields {len(fg)}); overlay skipped.")
        else:
            finite = np.isfinite(vals)
            if not finite.any():
                print("[warn] plot_map: overlay values are all NaN/inf; overlay skipped.")
            else:
                vmin = float(np.nanmin(vals[finite]))
                vmax = float(np.nanpercentile(vals[finite], 99.0))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                denom = (vmax - vmin)
                cmap = plt.cm.Reds

                for i, (_, fr) in enumerate(fg.iterrows()):
                    try:
                        level = 0.2 + 0.8 * ((vals[i] - vmin) / denom) if denom > 0 else 1.0
                        color = cmap(np.clip(level, 0.0, 1.0))
                        geom = fr.geometry
                        if geom is None or geom.is_empty:
                            continue
                        if isinstance(geom, Polygon):
                            xs, ys = geom.exterior.xy
                            ax.fill(xs, ys, facecolor=color, edgecolor=None, alpha=0.30, zorder=3)
                        elif isinstance(geom, MultiPolygon):
                            for gg in geom.geoms:
                                if gg.is_empty:
                                    continue
                                xs, ys = gg.exterior.xy
                                ax.fill(xs, ys, facecolor=color, edgecolor=None, alpha=0.30, zorder=3)
                    except Exception:
                        pass

                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                mappable.set_array(vals)
                cb = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
                cb.set_label(label)

    # Feld-IDs (optional: nur bei überschaubarer Anzahl, sonst unleserlich)
    try:
        if len(fg) <= 120:
            for _, r in fg.iterrows():
                ax.text(float(r["x"]), float(r["y"]), f"{int(r['field_id'])}",
                        ha="center", va="center", fontsize=9, color="white", weight="bold", zorder=7)
    except Exception:
        pass

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.15)

    # Kleine Crop-Legende (max 10 Einträge, rest zusammengefasst)
    try:
        crops_present = pd.Series(fg["crop"]).value_counts()
        top_items = list(crops_present.index[:10])
        handles = [Patch(facecolor=pal[c], edgecolor="black", label=f"{c} ({crops_present[c]})") for c in top_items]
        if len(crops_present) > 10:
            handles.append(Patch(facecolor="#dddddd", edgecolor="black", label=f"+{len(crops_present)-10} weitere"))
        ax.legend(handles=handles, title="Kulturen", loc="upper right", frameon=True, fontsize=8)
    except Exception:
        pass

    if savepath:
        p = Path(savepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_fig(p)
    else:
        plt.tight_layout()
        plt.show()

# =========================================================
#  (module 3/7) Habitat/utility functions  — robust version
# =========================================================

def _clamp_doy(d: int) -> int:
    try:
        td = int(globals().get("TOT_DAYS", 365))
    except Exception:
        td = 365
    try:
        x = int(d)
    except Exception:
        x = 1
    return int(min(td, max(1, x)))


# kanonisierte Kultur-Kürzel
_CROP_ALIASES = {
    "WW": "WINTERWEIZEN",
    "WINTERWEIZEN": "WINTERWEIZEN",
    "SOMMERGERSTE": "SOMMERGERSTE",
    "SG": "SOMMERGERSTE",
    "MAIS": "MAIS",
    "RAPS": "RAPS",
    "ZUCKERRÜBE": "ZUCKERRÜBE",
    "ZUCKERRUEBE": "ZUCKERRÜBE",
    "ACKERBOHNE": "ACKERBOHNE",
    "BRACHE": "BRACHE",
    "FALLOW": "BRACHE",          # toleranter Alias (falls mal englische Inputs auftauchen)
    "SET-ASIDE": "BRACHE",
}

def _crop_key(crop: str) -> str:
    """Upper-case & alias mapping to canonical crop key; unknown -> original upper."""
    if crop is None:
        return "UNBEKANNT"
    k = str(crop).strip().upper()
    return _CROP_ALIASES.get(k, k)

def _gauss(d: int, mu: float, sd: float, scale: float = 1.0, base: float = 0.0) -> float:
    """kleine Hilfsfunktion für glockenförmige Indizes."""
    x = float(_clamp_doy(d))
    try:
        return float(math.exp(-((x - mu) ** 2) / (2.0 * sd ** 2)) * scale + base)
    except Exception:
        return float(base)

def food_index(crop: str, doy: int) -> float:
    """
    Nahrungsindex 0..~2 (modellintern), abhängig von Kultur & DOY.
    Form bleibt wie bisher; nur robustere Guards.
    """
    c = _crop_key(crop)
    d = _clamp_doy(doy)

    if c == "WINTERWEIZEN":   return _gauss(d, 160, 25, 1.0, 0.2)
    if c == "SOMMERGERSTE":   return _gauss(d, 150, 22, 1.0, 0.25)
    if c == "MAIS":           return _gauss(d, 190, 30, 1.2, 0.1)
    if c == "RAPS":           return _gauss(d, 130, 18, 1.1, 0.2)
    if c == "ZUCKERRÜBE":     return _gauss(d, 185, 28, 0.9, 0.15)
    if c == "ACKERBOHNE":     return _gauss(d, 165, 20, 1.0, 0.2)
    if c == "BRACHE":         return _gauss(d, 170, 35, 0.5, 0.4)  # identisch zur bisherigen Formel
    # Fallback (unbekannte Kultur): breite Glocke um 170
    return _gauss(d, 170, 40, 0.6, 0.3)

def cover_index(crop: str, doy: int) -> float:
    """
    Deckungsindex 0..1 (ggf. gedämpft in später Phase für gewisse Kulturen).
    """
    c = _crop_key(crop)
    d = _clamp_doy(doy)

    if c == "WINTERWEIZEN":
        base = 1.0 / (1.0 + math.exp(-(d - 120) / 15.0))
        return float(max(0.0, base - (0.6 if d > 200 else 0.0)))
    if c == "SOMMERGERSTE":
        base = 1.0 / (1.0 + math.exp(-(d - 140) / 15.0))
        return float(max(0.0, base - (0.6 if d > 190 else 0.0)))
    if c == "MAIS":
        return float(1.0 / (1.0 + math.exp(-(d - 170) / 18.0)))
    if c == "RAPS":
        base = 1.0 / (1.0 + math.exp(-(d - 115) / 13.0))
        return float(max(0.0, base - (0.5 if d > 190 else 0.0)))
    if c == "ZUCKERRÜBE":
        return float(1.0 / (1.0 + math.exp(-(d - 175) / 17.0)))
    if c == "ACKERBOHNE":
        base = 1.0 / (1.0 + math.exp(-(d - 150) / 14.0))
        return float(max(0.0, base - (0.4 if d > 190 else 0.0)))
    if c == "BRACHE":
        return float(0.6)
    # Fallback
    return float(1.0 / (1.0 + math.exp(-(d - 150) / 20.0)))

def disturbance_components(
    doy: int,
    mow_vec: np.ndarray,
    harvest_day: Optional[int],
    insecticide_day: Optional[int],
    sow_win: Optional[Tuple[int, int]],
    harv_win: Optional[Tuple[int, int]],
    other_win: Optional[Tuple[int, int]],
    field_area_ha: float,
    *,
    base_penalties: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Pulse (Mähen/Ernte/Insektizid) + Window (Saat/Erntefenster/sonstige).
    Multi-Pass-Ernte & Insektizid-Persistenz integriert.
    """
    d = _clamp_doy(doy)

    # default nicht mutieren
    if base_penalties is None:
        base_pen = {"sowing": -0.5, "harvest": -1.2, "other": -0.6}
    else:
        base_pen = dict(base_penalties)

    HARVEST_PASS_DAYS = 5
    INSECT_PERSIST_DAYS = 10
    INSECT_HALF_LIFE = 3.0

    pulse = 0.0

    # Mähen: ±1 Tag Puls (mehrfache Termine sind okay)
    try:
        mv = np.asarray(mow_vec, dtype=float)
        if mv.size > 0 and np.any(np.isfinite(mv)):
            if np.any(np.abs(d - mv) <= 1):
                pulse += -1.0
    except Exception:
        pass

    # Multi-pass Ernte: Dreiecksprofil
    if harvest_day is not None and not pd.isna(harvest_day):
        try:
            dd = int(d) - int(harvest_day)
            hspan = max(0, (HARVEST_PASS_DAYS - 1) // 2)
            if abs(dd) <= hspan:
                frac = (hspan + 1 - abs(dd)) / (hspan + 1)
                pulse += base_pen.get("harvest", -1.2) * float(frac)
        except Exception:
            pass

    # Insektizid: exponentieller Abbau
    if insecticide_day is not None and not pd.isna(insecticide_day):
        try:
            dt = int(d) - int(insecticide_day)
            if 0 <= dt <= INSECT_PERSIST_DAYS:
                persist = 0.5 ** (dt / INSECT_HALF_LIFE)
                pulse += -0.8 * float(persist)
        except Exception:
            pass

    # Windows (Skalierung mit Feldgröße)
    fa = float(field_area_ha) if pd.notna(field_area_ha) else 1.0
    fa = max(1e-6, fa)
    scale = 1.0 + 0.1 * math.log1p(fa / 10.0)

    window = 0.0
    if in_window(d, sow_win):
        window += base_pen.get("sowing", -0.5) * scale
    if in_window(d, harv_win):
        window += base_pen.get("harvest", -1.2) * scale
    if in_window(d, other_win):
        window += base_pen.get("other", -0.6) * scale

    return float(pulse), float(window)

def measure_modifiers(
    flower_strip_share: float = 0.0,
    skylark_plots: int = 0,
    skylark_plot_area: float = 0.0,
    perimeter_m: float = 0.0,
    median_perimeter_m: float = 1.0,
    field_area_ha: float = 1.0
):
    """
    Returns:
      delta_food, delta_cover, disturbance_mult, refuge_area, refuge_share
    refuge_share ist *flächen-normalisiert*.
    """
    try:
        fs = np.clip(float(flower_strip_share), 0.0, 0.5)
    except Exception:
        fs = 0.0

    perim = float(perimeter_m) if pd.notna(perimeter_m) else 0.0
    medp  = float(median_perimeter_m) if pd.notna(median_perimeter_m) else 1.0
    medp  = max(medp, 1e-6)

    edge_scale  = 0.5 * perim / medp
    delta_food  = 0.6 * fs * edge_scale
    delta_cover = 0.5 * fs * edge_scale

    try:
        refuge_area = max(0.0, float(skylark_plots) * float(skylark_plot_area))  # ha
    except Exception:
        refuge_area = 0.0

    fa = float(field_area_ha) if pd.notna(field_area_ha) else 0.0
    refuge_share = 0.0 if fa <= 0 else np.clip(refuge_area / fa, 0.0, 1.0)

    # kleine Dämpfung der Störung mit wachsendem Refugienanteil
    disturbance_mult = 1.0 - np.clip(0.3 * refuge_share, 0.0, 0.3)

    return float(delta_food), float(delta_cover), float(disturbance_mult), float(refuge_area), float(refuge_share)

def softmax(v, beta: float = 3.0):
    """
    Numerisch stabile Softmax mit beta-Schärfe.
    Fällt auf Gleichverteilung zurück, wenn Summe 0/NaN/Inf.
    """
    arr = np.asarray(v, dtype=float)
    if arr.size == 0:
        return arr
    # maskiere nicht-finite Werte (ersetzen durch sehr kleine Zahl)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.ones_like(arr) / arr.size
    safe = arr.copy()
    safe[~finite] = np.nanmin(arr[finite])
    vmax = np.nanmax(safe)
    ex = np.exp(np.clip(beta * (safe - vmax), -60.0, 60.0))  # clip gegen Overflow
    s = np.nansum(ex)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(arr) / arr.size
    return ex / s

def load_skylark_windows_config(path: str | Path) -> SimpleNamespace:
    """Liest configs/skylark_windows.csv (key,value)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Skylark windows config not found: {p}")
    kv = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row.get("key") or "").strip()
            v = (row.get("value") or "").strip()
            if not k:
                continue
            kv[k] = v

    def _getf(k, d):
        try:
            return float(str(kv.get(k, d)).replace(",", "."))
        except Exception:
            return float(d)
    def _geti(k, d):
        try:
            return int(float(str(kv.get(k, d)).replace(",", ".")))
        except Exception:
            return int(d)
    def _getb(k, d):
        v = str(kv.get(k, d)).strip().lower()
        return v in ("1", "true", "yes", "y")
    def _gets(k, d):
        return str(kv.get(k, d)) if k in kv else d

    return SimpleNamespace(
        size_m2=_getf("size_m2", 20),
        target_per_ha=_getf("target_per_ha", 2),
        design_min_per_ha=_getf("design_min_per_ha", 3),
        design_max_per_ha=_getf("design_max_per_ha", 10),
        effect_crops=[c.strip() for c in _gets("effect_crops", "").split(",") if c.strip()],
        min_dist_edge_m=_getf("min_dist_edge_m", 25),
        min_dist_tall_struct_m=_getf("min_dist_tall_struct_m", 50),
        effect_only_second_half=_getb("effect_only_second_half", True),
        fledglings_mult_at_target=_getf("fledglings_mult_at_target", 1.5),
        pref_bonus=_getf("pref_bonus", 0.15),
        nest_survival_mult=_getf("nest_survival_mult", 0.05),
    )

# =========================================================
#  (module 4/7) Species-CSV loader  — robust version
# =========================================================

@dataclass
class SpeciesConfig:
    name: str
    breeding_window: Tuple[int,int]
    arrival_day: int
    departure_day: int
    move_radius: float        # in 100m units
    beta: float
    distance_cost: float
    breeding_move_multiplier: float
    survival_breeding: float
    survival_nonbreeding: float
    survival_juvenile: float
    fec_scale: float
    nest_cycle_days: int
    base_nest_survival_day: float
    nest_disturb_sensitivity: float
    clutch_size: float
    fledge_rate: float
    max_broods_per_female: int
    juvenile_free_age: int
    alpha_density: float
    density_affects_start: bool
    nest_refuge_gamma: float
    territory_size_ha: float = 0.0                # mittlere Reviergröße (ha), 0 => deaktiviert
    max_pairs_per_ha: float = 0.0                 # Kapazität (Paare/ha), 0 => deaktiviert
    renest_cooldown_days: int = 0                 # Tage Sperrfrist nach Nestverlust, 0 => aus
    winter_survival_adult: float = 0.95
    winter_survival_juvenile: float = 0.90
    juvenile_maturation_rate: float = 0.90
    juvenile_equiv_in_capacity: float = 0.5
    winter_move_radius: float = 12.0  # in 100m-Einheiten wie dist_mat
    winter_beta: float = 2.5
    winter_distance_cost: float = 0.08
    winter_eval_day_autumn: int = 330  # für _winter_utility
    winter_eval_day_spring: int = 20
    winter_eval_days: Tuple[int, int] = (330, 20)

    # prefs bekommt einen Default-Factory, um Default-Felder vorher zuzulassen
    prefs: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Art-spezifische Flags / Zusatzpräferenzen
    benefits_from_skylark_windows: bool = False
    # DOY/Clutch-Präferenzen je Kultur (uppercased crop -> {1:{start,end,p},2:{...},3:{...}})
    clutch_prefs: Dict[str, Dict[int, Dict[str, Optional[float]]]] = field(default_factory=dict)


def load_species_from_csv(name: str, base_path: str | Path = ".") -> SpeciesConfig:
    base_path = Path(base_path)

    def _read_any(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path)
        # Unnamed-Spalten weg und Spaltennamen trimmen
        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _norm_prob(x, default):
        try:
            if x is None: return float(default)
            s = str(x).strip()
            if s == "" or s.lower() in {"na", "nan", "none", "null"}: return float(default)
            s = s.replace(",", ".")
            v = float(s)
            if 1.0 < v <= 100.0:
                v /= 100.0
            v = max(0.0, min(1.0, v))
            return float(v)
        except Exception:
            return float(default)

    def _getf(params, k, default):
        v = params.get(k, default)
        try:
            return float(str(v).replace(",", "."))
        except Exception:
            return float(default)

    def _geti(params, k, default):
        v = params.get(k, default)
        try:
            return int(float(str(v).replace(",", ".")))
        except Exception:
            return int(default)

    def _getb(params, k, default):
        s = str(params.get(k, default)).strip().lower()
        return s in {"1", "true", "t", "yes", "y", "ja"}

    def _gets(params, k, default):
        v = params.get(k, default)
        return str(v) if v is not None else str(default)

    # --- Dateien finden ---
    p_params = base_path / f"species_{name}.csv"
    pref_candidates = [
        f"species_{name}_prefs.csv",
        f"species_{name}_pref.csv",
        f"species_{name.lower()}_prefs.csv",
        f"species_{name.lower()}_pref.csv",
        f"species_{name}.csv",              # falls zusammengeführt
        f"species_{name.lower()}.csv",
    ]
    p_prefs: Optional[Path] = None
    for cand in pref_candidates:
        test = base_path / cand
        if test.exists():
            p_prefs = test
            break
    if p_prefs is None:
        raise FileNotFoundError(f"Missing species prefs CSV for {name}. Tried: {pref_candidates}")
    if not p_params.exists():
        raise FileNotFoundError(f"Missing species parameter CSV: {p_params}")

    # --- Parameter-CSV lesen (key,value) ---
    raw_params = _read_any(p_params)
    key_alias   = {"param", "parameter", "key", "schluessel", "name"}
    value_alias = {"value", "wert", "val", "v"}
    key_col = next((c for c in raw_params.columns if c.lower() in key_alias), None)
    val_col = next((c for c in raw_params.columns if c.lower() in value_alias), None)
    if key_col is None or val_col is None:
        if raw_params.shape[1] >= 2:
            key_col, val_col = raw_params.columns[:2]
        elif raw_params.shape[1] == 1:
            kv = []
            for s in raw_params.iloc[:, 0].astype(str):
                s = s.strip()
                if "=" in s: k, v = s.split("=", 1)
                elif ";" in s: k, v = s.split(";", 1)
                elif "," in s: k, v = s.split(",", 1)
                else: continue
                kv.append((k.strip(), v.strip()))
            if not kv:
                raise ValueError(f"Cannot interpret {p_params}.")
            raw_params = pd.DataFrame(kv, columns=["param", "value"])
            key_col, val_col = "param", "value"

    params = {}
    for _, r in raw_params.iterrows():
        k = str(r[key_col]).strip()
        v = r[val_col]
        if k != "":
            params[k] = v

    # --- Prefs-CSV lesen (Food/Cover/Disturb & ggf. Clutch) ---
    prefs = _read_any(p_prefs)
    rename = {}
    for c in prefs.columns:
        cl = c.lower()
        if cl in {"kultur", "frucht", "crop"}:                 rename[c] = "crop"
        elif cl in {"w_food", "food", "nahrung"}:              rename[c] = "w_food"
        elif cl in {"w_cover", "cover", "deckung"}:            rename[c] = "w_cover"
        elif cl in {"w_disturb_base", "disturb", "stoerung", "störung"}:
            rename[c] = "w_disturb_base"
    if rename:
        prefs = prefs.rename(columns=rename)

    # falls w_disturb_base fehlt → Spalte anlegen
    if "w_disturb_base" not in prefs.columns:
        prefs["w_disturb_base"] = pd.NA

    # Spezialfall "crop" wie "Winterweizen; 0.4" → Disturb-Base extrahieren
    def _maybe_float(x):
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return None

    new_rows = []
    for _, r in prefs.iterrows():
        crop_raw = str(r.get("crop", ""))
        r = r.copy()
        if ";" in crop_raw:
            left, right = crop_raw.split(";", 1)
            crop_clean = left.strip()
            extra = _maybe_float(right)
            if pd.isna(r.get("w_disturb_base", pd.NA)) and extra is not None:
                r["w_disturb_base"] = -abs(extra)
            r["crop"] = crop_clean
        new_rows.append(r)
    prefs = pd.DataFrame(new_rows)

    # Pflichtspalten prüfen + 0..1-Klipps robust (auch bei NaN)
    req = {"crop", "w_food", "w_cover", "w_disturb_base"}
    if not req.issubset(set(prefs.columns)):
        raise ValueError(f"{p_prefs}: Missing columns. Need {sorted(req)}, got {list(prefs.columns)}")

    prefs["crop"] = prefs["crop"].astype(str)

    def _clip01(x, default=0.3):
        try:
            v = float(str(x).replace(",", "."))
            if math.isnan(v):
                return float(default)
            return float(min(1.0, max(0.0, v)))
        except Exception:
            return float(default)

    prefs["w_food"]  = prefs["w_food"].apply(_clip01)
    prefs["w_cover"] = prefs["w_cover"].apply(_clip01)
    prefs["w_disturb_base"] = prefs["w_disturb_base"].apply(
        lambda x: float(str(x).replace(",", ".")) if pd.notna(x) and str(x).strip() != "" else 0.0
    )

    # ---------- DOY/Clutch-Präferenzen aus prefs-CSV ziehen (optional) ----------
    def _sint(x):
        try:
            return int(float(str(x).replace(",", ".").strip()))
        except Exception:
            return None

    def _sfloat(x, default=0.0):
        try:
            return float(str(x).replace(",", ".").strip())
        except Exception:
            return default

    have_clutch_cols = any(
        c.lower().startswith(("clutch", "p_clutch")) for c in prefs.columns
    )
    clutch_prefs: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}

    if have_clutch_cols:
        cols = {c.lower(): c for c in prefs.columns}

        def getcol(name: str) -> str:
            # toleriert Groß/Kleinschreibung
            return cols.get(name, name)

        for _, r in prefs.iterrows():
            crop_u = str(r["crop"]).strip().upper()
            if not crop_u:
                continue
            rec: Dict[int, Dict[str, Optional[float]]] = {}
            for k in (1, 2, 3):
                pcol = getcol(f"p_clutch{k}")
                scol = getcol(f"clutch{k}_start")
                ecol = getcol(f"clutch{k}_end")
                p = _sfloat(r.get(pcol, 0.0), 0.0)
                s = _sint(r.get(scol, None))
                e = _sint(r.get(ecol, None))
                # wenn p==0 → Zeitfenster ignorieren
                if p <= 0.0:
                    s, e = None, None
                rec[k] = {"start": s, "end": e, "p": float(min(1.0, max(0.0, p)))}
            clutch_prefs[crop_u] = rec

    # --- interpret survivals as daily/annual and convert ---
    survival_mode = str(params.get("survival_mode", "daily")).strip().lower()

    b0 = _geti(params, "breeding_window_start", 110)
    b1 = _geti(params, "breeding_window_end", 200)
    breed_days = max(1, b1 - b0 + 1)
    nonbreed_days = max(1, 365 - breed_days)

    surv_b = _norm_prob(params.get("survival_breeding"), 0.996)
    surv_nb = _norm_prob(params.get("survival_nonbreeding"), 0.999)
    surv_j = _norm_prob(params.get("survival_juvenile"), 0.997)

    if survival_mode in ("annual", "year", "yearly"):
        surv_b  = surv_b  ** (1.0 / float(breed_days))
        surv_nb = surv_nb ** (1.0 / float(nonbreed_days))
        surv_j  = surv_j  ** (1.0 / 365.0)

    cfg = SpeciesConfig(
        name=_gets(params, "name", name),

        # Brutfenster & Phänologie
        breeding_window=(b0, b1),
        arrival_day=_geti(params, "arrival_day", 1),
        departure_day=_geti(params, "departure_day", 365),

        # Bewegung / Habitatwahl
        move_radius=_getf(params, "move_radius", 8.0),
        beta=_getf(params, "beta", 3.0),
        distance_cost=_getf(params, "distance_cost", 0.12),
        breeding_move_multiplier=_getf(params, "breeding_move_multiplier", 0.4),

        # Überleben (täglich, ggf. aus "annual" umgerechnet)
        survival_breeding=surv_b,
        survival_nonbreeding=surv_nb,
        survival_juvenile=surv_j,

        # Reproduktion
        fec_scale=_getf(params, "fec_scale", 0.3),
        nest_cycle_days=_geti(params, "nest_cycle_days", 28),
        base_nest_survival_day=_norm_prob(params.get("base_nest_survival_day"), 0.98),
        nest_disturb_sensitivity=_getf(params, "nest_disturb_sensitivity", 0.25),
        clutch_size=_getf(params, "clutch_size", 3.0),
        fledge_rate=_norm_prob(params.get("fledge_rate"), 0.85),
        max_broods_per_female=_geti(params, "max_broods_per_female", 2),

        # Juvenile
        juvenile_free_age=_geti(params, "juvenile_free_age", 21),

        # Dichte-Effekte
        alpha_density=_getf(params, "alpha_density", 0.002),
        density_affects_start=_getb(params, "density_affects_start", True),

        # Refugien / Lerchenfenster
        nest_refuge_gamma=_norm_prob(params.get("nest_refuge_gamma"), 0.15),
        benefits_from_skylark_windows=_getb(params, "benefits_from_skylark_windows", True),

        # Revier-/Kapazitäts-Defaults
        territory_size_ha=_getf(params, "territory_size_ha", 0.0),
        max_pairs_per_ha=_getf(params, "max_pairs_per_ha", 0.0),

        # Winter & Feedback
        winter_survival_adult=_norm_prob(params.get("winter_survival_adult"), 0.95),
        winter_survival_juvenile=_norm_prob(params.get("winter_survival_juvenile"), 0.90),
        juvenile_maturation_rate=_norm_prob(params.get("juvenile_maturation_rate"), 0.90),
        juvenile_equiv_in_capacity=_getf(params, "juvenile_equiv_in_capacity", 0.5),
        winter_move_radius=_getf(params, "winter_move_radius", 12.0),
        winter_beta=_getf(params, "winter_beta", 2.5),
        winter_distance_cost=_getf(params, "winter_distance_cost", 0.08),
        winter_eval_day_autumn=_geti(params, "winter_eval_day_autumn", 330),
        winter_eval_day_spring=_geti(params, "winter_eval_day_spring", 20),
        winter_eval_days=(
            _geti(params, "winter_eval_day_autumn", 330),
            _geti(params, "winter_eval_day_spring", 20)
        ),

        # Präferenzen-Tabellen
        prefs=prefs,                   # Food/Cover/Disturb (DataFrame aus derselben CSV)
        clutch_prefs=clutch_prefs      # DOY/Gelege-Präferenzen (Dict aus derselben CSV)
    )

    # ---- optionale Revier-Parameter konsistent ableiten ----
    # 1) territory_radius_m direkt oder aus territory_size_ha ableiten (Kreisannahme)
    t_rad = params.get("territory_radius_m", None)
    if t_rad is not None and str(t_rad).strip() != "":
        try:
            setattr(cfg, "territory_radius_m", float(str(t_rad).replace(",", ".")))
        except Exception:
            pass
    elif float(getattr(cfg, "territory_size_ha", 0.0) or 0.0) > 0.0:
        try:
            area_ha = float(cfg.territory_size_ha)
            r_m = math.sqrt(max(area_ha, 0.0) * 10_000.0 / math.pi)
            setattr(cfg, "territory_radius_m", r_m)
        except Exception:
            pass

    # 2) max. Paare / ha defensiv setzen
    try:
        cfg.max_pairs_per_ha = max(0.0,
            float(str(params.get("max_pairs_per_ha", cfg.max_pairs_per_ha)).replace(",", "."))
        )
    except Exception:
        cfg.max_pairs_per_ha = float(getattr(cfg, "max_pairs_per_ha", 0.0) or 0.0)

    # 3) Winter-Eval-Tage als Tuple fixieren
    cfg.winter_eval_days = (int(cfg.winter_eval_day_autumn), int(cfg.winter_eval_day_spring))

    return cfg

def _read_crop_calendar(path: str | Path) -> dict:
    """
    Liest eine (optionale) Fruchtfolge-Kalenderdatei ein und liefert pro Kultur
    ein Dict mit z. B. Ernte-/Saat-Zentren und -Breiten sowie Wahrscheinlichkeiten.
    Rückgabe: {crop: {harvest_center: float|None, harvest_width: float|None, ...}, ...}
    Fehlende Datei -> {}.
    """
    p = Path(path)
    if not p.exists():
        return {}

    # Erst: flexibles Parsing mit Sep-Autodetektion; Fallback: Standard
    try:
        df = pd.read_csv(p, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(p)

    df.columns = [c.strip() for c in df.columns]
    out: dict[str, dict] = {}

    for _, r in df.iterrows():
        crop = str(r.get("crop", "")).strip()
        if not crop:
            continue

        def _num(name: str, default=None):
            v = r.get(name, None)
            if v is None:
                return default
            s = str(v).strip()
            if s == "" or s.lower() in {"na", "nan", "none", "null"}:
                return default
            try:
                return float(s.replace(",", "."))
            except Exception:
                return default

        def _prob(name: str, default: float = 0.0) -> float:
            v = _num(name, default)
            if v is None:
                return 0.0
            return float(min(1.0, max(0.0, v)))

        out[crop] = dict(
            harvest_center    = _num("harvest_center"),
            harvest_width     = _num("harvest_width"),
            harvest_sd        = _num("harvest_sd", 0.0),
            harvest_win_span  = int(_num("harvest_win_span", 10) or 10),
            sow_center        = _num("sow_center"),
            sow_width         = _num("sow_width"),
            other_prob        = _prob("other_prob", 0.0),
            other_center      = _num("other_center"),
            other_width       = _num("other_width"),
            insect_prob       = _prob("insect_prob", 0.0),
            insect_center     = _num("insect_center"),
            insect_width      = _num("insect_width"),
        )
    return out


def clutch_multiplier_for(
    crop: str,
    doy: int,
    clutch_number: int,
    prefs_lookup: Dict[str, Dict[int, Dict[str, Optional[float]]]]
) -> float:
    """
    Liefert einen Multiplikator (0..1) für Gelege Nr. 'clutch_number' am Tag 'doy'
    in Kultur 'crop' basierend auf DOY-Fenstern und p_clutch.

    Regeln:
      - Kein prefs_lookup oder Kultur nicht vorhanden -> 1.0 (keine Modulation).
      - p_clutch <= 0 -> 0.0 (Gelege-Typ in dieser Kultur deaktiviert).
      - Kein Start/Ende angegeben -> p wirkt ganzjährig.
      - Fenster über Jahreswechsel (z.B. start=350, end=30) werden unterstützt.
    """
    if not prefs_lookup:
        return 1.0

    crop_key = str(crop).strip().upper()
    rec = prefs_lookup.get(crop_key)
    if rec is None:
        return 1.0

    blk = rec.get(int(clutch_number), {"start": None, "end": None, "p": 0.0})
    try:
        p = float(blk.get("p", 0.0) or 0.0)
    except Exception:
        p = 0.0
    # sauber 0..1 clippen
    p = float(min(1.0, max(0.0, p)))

    if p <= 0.0:
        return 0.0

    s = blk.get("start")
    e = blk.get("end")
    if s is None or e is None:
        # kein Fenster -> p gilt ganzjährig
        return p

    try:
        s_i = int(s)
        e_i = int(e)
        d_i = int(doy)
    except Exception:
        # falls etwas schiefgeht, lieber neutral sein
        return p

    if s_i <= e_i:
        # normales Fenster innerhalb des Jahres
        return p if (s_i <= d_i <= e_i) else 0.0
    else:
        # Fenster über Jahreswechsel, z.B. 350..365 und 1..30
        in_wrap = (d_i >= s_i) or (d_i <= e_i)
        return p if in_wrap else 0.0

# =========================================================
#  (module 5/7) Debug tracer & Repro logger
# =========================================================
class DebugTracer:
    """Collects daily snapshots for selected days."""
    def __init__(self):
        self.rows: list[dict] = []

    def capture_day(
        self,
        doy,
        U,
        today_m,
        today_f,
        breeding_f,
        new_nests,
        fledglings,
        density_factor,
        nest_surv_day
    ):
        self.rows.append({
            "doy": int(doy),
            "U_min": float(np.min(U)), "U_max": float(np.max(U)), "U_mean": float(np.mean(U)),
            "M_sum": int(np.sum(today_m)), "F_sum": int(np.sum(today_f)),
            "F_breeding_sum": int(np.sum(breeding_f)),
            "new_nests_sum": int(np.sum(new_nests)),
            "fledglings_sum": int(np.sum(fledglings)),
            "density_min": float(np.min(density_factor)),
            "density_max": float(np.max(density_factor)),
            "nest_surv_min": float(np.min(nest_surv_day)),
            "nest_surv_max": float(np.max(nest_surv_day)),
        })

    def to_frames(self):
        return {"daily": pd.DataFrame(self.rows)}


def sanity_check_species(cfg) -> None:
    """
    Robust statt assert: sammelt alle Validierungsfehler und wirft 1x ValueError.
    Deckt typische Fehl-Parametrisierungen ab (Fenster, Raten, Integergrenzen).
    """
    errs: list[str] = []

    def in01(x, name):
        try:
            v = float(getattr(cfg, name))
            if not (0.0 <= v <= 1.0):
                errs.append(f"{name}={v} not in [0,1]")
        except Exception as e:
            errs.append(f"{name} invalid: {e}")

    def ge0(x, name):
        try:
            v = float(getattr(cfg, name))
            if not (v >= 0.0):
                errs.append(f"{name}={v} must be >= 0")
        except Exception as e:
            errs.append(f"{name} invalid: {e}")

    def ge1_int(name):
        try:
            v = int(getattr(cfg, name))
            if v < 1:
                errs.append(f"{name}={v} must be >= 1")
        except Exception as e:
            errs.append(f"{name} invalid: {e}")

    # Tage & Fenster
    try:
        arr = int(getattr(cfg, "arrival_day"))
        dep = int(getattr(cfg, "departure_day"))
        if not (1 <= arr <= 365):
            errs.append(f"arrival_day={arr} out of [1,365]")
        if not (1 <= dep <= 365):
            errs.append(f"departure_day={dep} out of [1,365]")
        if arr > dep:
            errs.append(f"arrival_day({arr}) > departure_day({dep})")
    except Exception as e:
        errs.append(f"arrival/departure invalid: {e}")

    try:
        b0, b1 = getattr(cfg, "breeding_window")
        b0, b1 = int(b0), int(b1)
        if not (1 <= b0 <= 365 and 1 <= b1 <= 365):
            errs.append(f"breeding_window={b0,b1} out of [1,365]")
        if b0 > b1:
            errs.append(f"breeding_window start>end: {b0}>{b1}")
    except Exception as e:
        errs.append(f"breeding_window invalid: {e}")

    # Raten in [0,1]
    for p in (
        "survival_breeding",
        "survival_nonbreeding",
        "survival_juvenile",
        "base_nest_survival_day",
        "fledge_rate",
        "juvenile_maturation_rate",
        "winter_survival_adult",
        "winter_survival_juvenile",
    ):
        in01(getattr(cfg, p, np.nan), p)

    # Nicht-negative/positive Größen
    ge0(getattr(cfg, "alpha_density", 0.0), "alpha_density")
    ge0(getattr(cfg, "distance_cost", 0.0), "distance_cost")
    ge0(getattr(cfg, "move_radius", 0.0), "move_radius")
    ge0(getattr(cfg, "beta", 0.0), "beta")
    ge0(getattr(cfg, "breeding_move_multiplier", 0.0), "breeding_move_multiplier")
    ge0(getattr(cfg, "fec_scale", 0.0), "fec_scale")
    ge0(getattr(cfg, "nest_disturb_sensitivity", 0.0), "nest_disturb_sensitivity")
    ge0(getattr(cfg, "clutch_size", 0.0), "clutch_size")
    ge0(getattr(cfg, "nest_refuge_gamma", 0.0), "nest_refuge_gamma")
    ge0(getattr(cfg, "juvenile_equiv_in_capacity", 0.0), "juvenile_equiv_in_capacity")
    ge0(getattr(cfg, "max_pairs_per_ha", 0.0), "max_pairs_per_ha")
    ge0(getattr(cfg, "territory_size_ha", 0.0), "territory_size_ha")
    ge0(getattr(cfg, "winter_distance_cost", 0.0), "winter_distance_cost")
    ge0(getattr(cfg, "winter_beta", 0.0), "winter_beta")
    ge0(getattr(cfg, "winter_move_radius", 0.0), "winter_move_radius")

    # Ganzzahlen-Grenzen
    ge1_int("nest_cycle_days")
    try:
        jfa = int(getattr(cfg, "juvenile_free_age"))
        if jfa < 0:
            errs.append(f"juvenile_free_age={jfa} must be >= 0")
    except Exception as e:
        errs.append(f"juvenile_free_age invalid: {e}")

    try:
        mbpf = int(getattr(cfg, "max_broods_per_female"))
        if mbpf < 0:
            errs.append(f"max_broods_per_female={mbpf} must be >= 0")
    except Exception as e:
        errs.append(f"max_broods_per_female invalid: {e}")

    # Prefs-DF Minimalprüfung
    prefs = getattr(cfg, "prefs", None)
    if isinstance(prefs, pd.DataFrame):
        needed = {"crop", "w_food", "w_cover", "w_disturb_base"}
        miss = needed.difference({c.lower() for c in prefs.columns})
        if miss:
            errs.append(f"prefs missing columns: {sorted(miss)}")

    # Clutch-Präferenzen (falls vorhanden) – p in [0,1]
    clutch_prefs = getattr(cfg, "clutch_prefs", None)
    if isinstance(clutch_prefs, dict):
        for cu, rec in clutch_prefs.items():
            if not isinstance(rec, dict):
                errs.append(f"clutch_prefs[{cu}] not a dict")
                continue
            for k, blk in rec.items():
                try:
                    p = float((blk or {}).get("p", 0.0))
                    if not (0.0 <= p <= 1.0):
                        errs.append(f"clutch_prefs[{cu}][{k}].p={p} not in [0,1]")
                except Exception:
                    errs.append(f"clutch_prefs[{cu}][{k}].p invalid")

    if errs:
        raise ValueError("SpeciesConfig validation failed:\n - " + "\n - ".join(errs))


class ReproLogger:
    """Stores reproduction-relevant aggregates per day for debugging."""
    def __init__(self):
        self.rows: list[dict] = []

    def log(
        self,
        day,
        in_breed,
        U,
        p_start_vec,
        density_factor,
        nest_surv_day,
        new_nests_vec,
        fledglings_vec,
        A_m_today,
        A_f_today,
        J_tied_today,
        J_free_today
    ):
        self.rows.append(dict(
            day=int(day),
            in_breed=bool(in_breed),
            mean_U=float(np.mean(U)),
            mean_p_start=float(np.mean(p_start_vec)),
            mean_density=float(np.mean(density_factor)),
            mean_nest_surv=float(np.mean(nest_surv_day)),
            total_new_nests=int(np.sum(new_nests_vec)),
            total_fledglings=int(np.sum(fledglings_vec)),
            A_m_total=int(np.sum(A_m_today)),
            A_f_total=int(np.sum(A_f_today)),
            J_tied_total=int(np.sum(J_tied_today)),
            J_free_total=int(np.sum(J_free_today)),
        ))

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

# =========================================================
#  (module 6/7) Simulator with age-cohort juveniles + tidy export
# =========================================================
class BirdLandscapeSim:
    """
    Landschafts-Simulator mit optionalen Revier-Slots (Territorien) und Re-Nest-Cooldown.
    - Revier-Slots: begrenzen die parallel aktiven Bruten pro Feld.
      Quellen (optional, pro Species):
        - cfg.territory_area_ha: mittlere Reviergröße (ha)  -> slots_by_area = floor(area_ha / territory_area_ha)
        - cfg.max_pairs_per_ha: maximale Brutpaare pro ha   -> slots_by_pph  = floor(max_pairs_per_ha * area_ha)
      Wenn beides fehlt, ist die Begrenzung de facto deaktiviert.
    - Re-Nest-Cooldown: Anzahl Tage, die ein Weibchen nach Brutverlust nicht neu beginnt
      Quelle (optional): cfg.renest_cooldown_days (int, >=0). Fehlend/0 => deaktiviert.
    """
    def __init__(self, fields_df, rotation_df, management_df, species_cfg: SpeciesConfig, rng_seed: Optional[int] = None):
        self.fields = fields_df.copy().reset_index(drop=True)
        for col, default in [("flower_strip_share", 0.0), ("skylark_plots", 0), ("skylark_plot_area", 0.0)]:
            if col not in self.fields.columns:
                self.fields[col] = default
        if "perimeter" not in self.fields.columns:
            self.fields["perimeter"] = self.fields["area"].apply(lambda a: 4.0 * math.sqrt(a * 10_000.0))

        self.rotation = rotation_df.copy()
        self.management = management_df.copy()
        self.cfg = species_cfg

        # reproducible RNG
        if rng_seed is None:
            rng_seed = getattr(self.cfg, "sim_rng_seed", None)
        if rng_seed is None:
            rng_seed = globals().get("OPTS", SimpleNamespace()).__dict__.get("SIM_RNG_SEED", 1)
        self.rng = np.random.default_rng(int(rng_seed))

        # Lerchenfenster-Konfiguration laden (falls Datei fehlt, Defaults nehmen)
        try:
            self.skylark_cfg = load_skylark_windows_config(SKYLARK_WINDOWS_CSV)
        except Exception:
            # Fallback-Defaults
            self.skylark_cfg = SimpleNamespace(
                size_m2=20.0,
                target_per_ha=2.0,
                design_min_per_ha=3.0,
                design_max_per_ha=10.0,
                effect_crops=[],
                min_dist_edge_m=25.0,
                min_dist_tall_struct_m=50.0,
                effect_only_second_half=True,
                fledglings_mult_at_target=1.5,
                pref_bonus=0.15,
                nest_survival_mult=0.05
            )

        # Art-Flag: profitiert diese Art von Lerchenfenstern?
        self.species_benefits_from_skylark_windows = bool(
            getattr(self.cfg, "benefits_from_skylark_windows", False)
        )

        # Distanzen (in 100-m-Einheiten)
        coords = self.fields[["x", "y"]].to_numpy(float)
        raw = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        self._dist_scale_m = 100.0  # 1 Einheit = 100 m
        self.dist_mat = raw / self._dist_scale_m

        self._median_perimeter = float(np.median(self.fields["perimeter"].values))
        self.K_adults = (self.fields["area"].astype(float).to_numpy() * float(K_PER_HA_ADULTS)).astype(float)

        # ---------- optionale Revier-Slots je Feld ----------
        self.territory_slots = self._compute_territory_slots()
        # ---- schnelle Indizes & Pref-Cache
        self._fid2ix = {int(fid): i for i, fid in enumerate(self.fields["field_id"].astype(int).tolist())}
        self._pref_cache = {
            str(row.crop).strip().upper(): (float(row.w_food), float(row.w_cover), float(row.w_disturb_base))
            for _, row in self.cfg.prefs.iterrows()
        }

        # Year-spezifische Caches (lazy gebaut)
        self._cache_year = None
        self._cache_crop = None  # np.array[str] je Feld (Rotation)
        self._cache_mow_sets = None  # List[set[int]] je Feld (Mäh-Tage ±1)
        self._cache_harvest = None  # np.array[int] (=-1 falls None)
        self._cache_insect = None  # np.array[int] (=-1 falls None)
        self._cache_sow_win = None  # List[tuple|None]
        self._cache_harv_win = None
        self._cache_other_win = None

    # ---------- Revier-Slots berechnen (optional) ----------
    def _compute_territory_slots(self) -> np.ndarray:
        """
        Liefert int-Array len(fields): maximale parallel aktive Paare je Feld.
        Wenn weder territory_area_ha noch max_pairs_per_ha gesetzt ist, wird sehr groß gewählt (keine harte Begrenzung).
        """
        area_ha = self.fields["area"].astype(float).to_numpy()
        # akzeptiere beide mögliche Namen: territory_area_ha ODER territory_size_ha
        terr_area = float(
            getattr(self.cfg, "territory_area_ha", 0.0) or
            getattr(self.cfg, "territory_size_ha", 0.0) or 0.0
        )
        max_pph = float(getattr(self.cfg, "max_pairs_per_ha", 0.0) or 0.0)

        BIG = 10**9  # "unbegrenzt" (effektiv)
        slots_area = np.full_like(area_ha, BIG, dtype=float)
        slots_pph = np.full_like(area_ha, BIG, dtype=float)

        if terr_area > 0:
            slots_area = np.floor(area_ha / terr_area)
        if max_pph > 0:
            slots_pph = np.floor(max_pph * area_ha)

        slots = np.minimum(slots_area, slots_pph)
        slots = np.clip(slots, 0.0, BIG).astype(int)
        return slots

    # ---------- WINTER-Hilfen ----------
    def _prepare_year_cache(self, year: int):
        if self._cache_year == int(year):
            return
        y = int(year)

        # --- Rotation: Crop je Feld ---
        rot_y = self.rotation.loc[self.rotation["year"].astype(int) == y, ["field_id", "crop"]]
        cmap = {int(fid): str(c) for fid, c in zip(rot_y["field_id"], rot_y["crop"])}
        crops = []
        for fid in self.fields["field_id"].astype(int).tolist():
            crops.append(cmap.get(int(fid), "Winterweizen"))
        self._cache_crop = np.array(crops, dtype=object)

        # --- Management: alles vorparsen ---
        mg_y = self.management.loc[self.management["year"].astype(int) == y].copy()
        mg_y["field_id"] = mg_y["field_id"].astype(int)
        mg_by_f = {int(r.field_id): r for _, r in mg_y.iterrows()}

        self._cache_mow_sets = []
        self._cache_harvest = np.full(len(self.fields), -1, dtype=np.int32)
        self._cache_insect = np.full(len(self.fields), -1, dtype=np.int32)
        self._cache_sow_win = [None] * len(self.fields)
        self._cache_harv_win = [None] * len(self.fields)
        self._cache_other_win = [None] * len(self.fields)

        for i, fid in enumerate(self.fields["field_id"].astype(int).tolist()):
            r = mg_by_f.get(int(fid))
            if r is None:
                self._cache_mow_sets.append(set())
                continue

            # Mähen → DOY-Set mit ±1 Puffer
            days = parse_days(getattr(r, "mow_dates", None))
            mowset = set()
            for d in days.tolist():
                if 1 <= d <= 365:
                    for dd in (d - 1, d, d + 1):
                        if 1 <= dd <= 365:
                            mowset.add(int(dd))
            self._cache_mow_sets.append(mowset)

            # Ernte/Insektizid als ints
            try:
                h = int(getattr(r, "harvest_date", -1))
                self._cache_harvest[i] = h if 1 <= h <= 365 else -1
            except Exception:
                pass
            try:
                ins = int(getattr(r, "insecticide_date", -1))
                self._cache_insect[i] = ins if 1 <= ins <= 365 else -1
            except Exception:
                pass

            # Fenster 1× parsen
            self._cache_sow_win[i] = parse_window(getattr(r, "sow_win", None))
            self._cache_harv_win[i] = parse_window(getattr(r, "harvest_win", None))
            self._cache_other_win[i] = parse_window(getattr(r, "other_ops_win", None))

        self._cache_year = y

    def _pref_row_safe(self, crop: str) -> dict:
        sub = self.cfg.prefs[self.cfg.prefs["crop"].str.upper() == str(crop).upper()]
        if len(sub) == 0:
            return {"w_food": 0.3, "w_cover": 0.3, "w_disturb_base": 0.0}
        r = sub.iloc[0]
        return {"w_food": float(r.w_food), "w_cover": float(r.w_cover), "w_disturb_base": float(r.w_disturb_base)}

    def _winter_utility(self, year: int) -> np.ndarray:
        d1, d2 = getattr(self.cfg, "winter_eval_days", WINTER_EVAL_DAYS)
        U = np.zeros(len(self.fields), dtype=float)
        rot_y = self.rotation[self.rotation["year"] == year]
        rot_y2 = self.rotation[self.rotation["year"] == year + 1]
        crop_map_aut = dict(zip(rot_y["field_id"].astype(int), rot_y["crop"].astype(str)))
        crop_map_spr = dict(zip(rot_y2["field_id"].astype(int), rot_y2["crop"].astype(str))) if len(rot_y2) else crop_map_aut
        for i, frow in self.fields.iterrows():
            fid = int(frow.field_id)
            crop1 = crop_map_aut.get(fid, "Winterweizen")
            crop2 = crop_map_spr.get(fid, crop1)
            pref1 = self._pref_row_safe(crop1)
            pref2 = self._pref_row_safe(crop2)
            u1 = pref1["w_food"] * food_index(crop1, d1) + pref1["w_cover"] * cover_index(crop1, d1)
            u2 = pref2["w_food"] * food_index(crop2, d2) + pref2["w_cover"] * cover_index(crop2, d2)
            U[i] = 0.5 * (u1 + u2)
        return U

    def _winter_redistribute(self, counts: np.ndarray, year: int) -> np.ndarray:
        if counts.sum() == 0:
            return counts.copy()
        U = self._winter_utility(year=year)
        out = np.zeros_like(counts, dtype=int)
        for i, cnt in enumerate(counts.astype(int)):
            if cnt <= 0:
                continue
            R = float(getattr(self.cfg, "winter_move_radius", WINTER_MOVE_RADIUS))
            BETA = float(getattr(self.cfg, "winter_beta", WINTER_BETA))
            DCOST = float(getattr(self.cfg, "winter_distance_cost", WINTER_DISTANCE_COST))

            cand = np.where(self.dist_mat[i, :] <= R)[0]
            if cand.size == 0:
                cand = np.array([i])
            scores = U[cand] - DCOST * self.dist_mat[i, cand]
            probs = softmax(scores, beta=BETA)
            moved = self.rng.multinomial(int(cnt), probs)
            out[cand] += moved
        return out

    # ---------- Präferenzen/Utilitäten ----------
    def _pref_row(self, crop: str) -> Dict[str, float]:
        k = str(crop).strip().upper()
        w = self._pref_cache.get(k)
        if w is None:
            return {"w_food": 0.3, "w_cover": 0.3, "w_disturb_base": 0.0}
        return {"w_food": w[0], "w_cover": w[1], "w_disturb_base": w[2]}

    def _utilities_for_day(self, doy: int, year: int, prev_total: Optional[np.ndarray] = None):
        self._prepare_year_cache(int(year))
        n = len(self.fields)
        U = np.zeros(n, dtype=np.float32)
        pulse_stress = np.zeros(n, dtype=np.float32)
        window_stress = np.zeros(n, dtype=np.float32)
        refuge_bonus = np.zeros(n, dtype=np.float32)
        refuge_share = np.zeros(n, dtype=np.float32)
        p_ref_nest = np.zeros(n, dtype=np.float32)

        for i, frow in self.fields.iterrows():
            field_id = int(frow.field_id)

            # Kultur (aus Cache)
            crop = self._cache_crop[i]
            pref = self._pref_row(crop)

            # Disturbance schnell aus Cache
            d = int(doy)
            # Mähen: Set-Mitgliedschaft (±1 bereits eingearbeitet)
            p_stress = -1.0 if d in self._cache_mow_sets[i] else 0.0

            # Multi-pass Ernte (Dreieck ±2 um harvest_day)
            hd = int(self._cache_harvest[i])
            if hd != -1:
                hspan = 2  # 5-Tage-Fenster
                dd = d - hd
                if abs(dd) <= hspan:
                    frac = (hspan + 1 - abs(dd)) / (hspan + 1)
                    p_stress += -1.2 * float(frac)

            # Insektizid: exponentieller Abbau über 10 Tage (T½=3)
            ins = int(self._cache_insect[i])
            if ins != -1:
                dt = d - ins
                if 0 <= dt <= 10:
                    persist = 0.5 ** (dt / 3.0)
                    p_stress += -0.8 * float(persist)

            # Fenster (mit Feldgrößen-Skalierung)
            fa = float(frow.area)
            scale = 1.0 + 0.1 * math.log1p(max(fa, 1e-6) / 10.0)
            w_stress = 0.0
            if in_window(d, self._cache_sow_win[i]):   w_stress += -0.5 * scale
            if in_window(d, self._cache_harv_win[i]):  w_stress += -1.2 * scale
            if in_window(d, self._cache_other_win[i]): w_stress += -0.6 * scale

            # Maßnahmen (Blühstreifen, Lerchenfenster)
            d_food, d_cover, d_mult, refuge_area, ref_share = measure_modifiers(
                flower_strip_share=float(getattr(frow, "flower_strip_share", 0.0)),
                skylark_plots=int(getattr(frow, "skylark_plots", 0)),
                skylark_plot_area=float(getattr(frow, "skylark_plot_area", 0.0)),
                perimeter_m=float(getattr(frow, "perimeter", 0.0)),
                median_perimeter_m=self._median_perimeter,
                field_area_ha=float(frow.area),
            )

            # Basis-Utility
            base_u = (
                pref["w_food"] * food_index(crop, doy)
                + pref["w_cover"] * cover_index(crop, doy)
                + pref["w_disturb_base"]
                + (p_stress + w_stress) * d_mult
            )
            base_u += pref["w_food"] * d_food + pref["w_cover"] * d_cover
            U[i] = base_u

            # Soft-Kapazität
            if USE_CAPACITY_SOFT and prev_total is not None:
                K = max(1.0, float(self.K_adults[i]))
                load = float(prev_total[i])
                if load > K:
                    over = (load - K) / K
                    U[i] -= float(CAPACITY_GAMMA) * over

            # Refugium (Bonus sanft gekappt)
            if refuge_area > 0:
                bonus_raw = float(self.cfg.nest_refuge_gamma * (refuge_area / (refuge_area + 2.0)))
                refuge_bonus[i] = min(bonus_raw, 0.30)
            else:
                refuge_bonus[i] = 0.0
            refuge_share[i] = ref_share

            # Anteil neuer Nester in Refugien
            if crop in ELIGIBLE_REFUGE_CROPS and ref_share > 0:
                p_ref = np.clip(REFUGE_ASSIGN_MULT * ref_share, 0.0, REFUGE_SHARE_CAP)
            else:
                p_ref = 0.0
            p_ref_nest[i] = p_ref

            pulse_stress[i] = p_stress
            window_stress[i] = w_stress

        return U, pulse_stress, window_stress, refuge_bonus, refuge_share, p_ref_nest

    def _skylark_window_multipliers_vector(self, *, day: int, year: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Liefert Vektoren (len(fields)):
          ns_mult – Multiplikator für tägliches Nest-Überleben (>=1.0)
          fl_mult – Multiplikator für Fledge-Rate (>=1.0)
        Wirkt nur, wenn:
          - self.skylark_cfg vorhanden
          - self.species_benefits_from_skylark_windows True
          - Kultur ∈ effect_crops (falls Liste nicht leer)
          - (optional) nur 2. Brut-Hälfte, falls effect_only_second_half=True
        """
        n = len(self.fields)
        ns_mult = np.ones(n, dtype=np.float32)
        fl_mult = np.ones(n, dtype=np.float32)

        if not self.species_benefits_from_skylark_windows or self.skylark_cfg is None:
            return ns_mult, fl_mult

        # nur innerhalb Brutfenster (und ggf. nur 2. Hälfte)
        a, b = int(self.cfg.breeding_window[0]), int(self.cfg.breeding_window[1])
        if not (a <= day <= b):
            return ns_mult, fl_mult
        if getattr(self.skylark_cfg, "effect_only_second_half", False) and day < (a + b) // 2:
            return ns_mult, fl_mult

        # Fenster/ha
        area = self.fields["area"].astype(float).to_numpy()
        plots = (
            self.fields["skylark_plots"].fillna(0).astype(float).to_numpy()
            if "skylark_plots" in self.fields.columns else np.zeros(n)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            per_ha = np.where(area > 0, plots / area, 0.0)

        # Kultur-Filter (aktuelles Jahr)
        rot_y = self.rotation[self.rotation["year"] == int(year)]
        crop_map = dict(zip(rot_y["field_id"].astype(int), rot_y["crop"].astype(str)))
        fids = self.fields["field_id"].astype(int).to_numpy()
        crops = np.array([crop_map.get(int(fid), "Unbekannt") for fid in fids], dtype=object)

        effect_crops = set(getattr(self.skylark_cfg, "effect_crops", []))
        if effect_crops:
            mask = np.array([c in effect_crops for c in crops], dtype=bool)
        else:
            mask = np.ones(n, dtype=bool)

        # lineare Skalierung zwischen design_min_per_ha .. design_max_per_ha
        wmin = float(getattr(self.skylark_cfg, "design_min_per_ha", 2.0))
        wmax = float(getattr(self.skylark_cfg, "design_max_per_ha", max(wmin, 10.0)))
        scale = np.clip((per_ha - wmin) / max(1e-9, (wmax - wmin)), 0.0, 1.0)

        # Mults
        ns_add = float(getattr(self.skylark_cfg, "nest_survival_mult", 0.0))
        fl_tar = float(getattr(self.skylark_cfg, "fledglings_mult_at_target", 1.0))
        ns_mult = 1.0 + ns_add * scale
        fl_mult = 1.0 + (fl_tar - 1.0) * scale

        ns_mult = np.where(mask, ns_mult, 1.0).astype(np.float32)
        fl_mult = np.where(mask, fl_mult, 1.0).astype(np.float32)
        return ns_mult, fl_mult

    # ---------- Hauptsimulation ----------
    def simulate(self, days: int = 365, year: int = 1, N_init: int = 50,
                 debug_days=None, log_repro: bool = True, keep_age_series: bool = True,
                 start_state: Optional[dict] = None):

        n = len(self.fields)
        rng = self.rng

        # Output-Arrays (days x fields)
        A_m = np.zeros((days, n), dtype=np.int32)
        A_f = np.zeros((days, n), dtype=np.int32)
        J_free = np.zeros((days, n), dtype=np.int32)
        J_tied_total = np.zeros((days, n), dtype=np.int32)
        daily_new_nests = np.zeros((days, n), dtype=np.int32)
        daily_fledglings = np.zeros((days, n), dtype=np.int32)
        A_f_breeding = np.zeros((days, n), dtype=np.int32)
        repro = ReproLogger() if log_repro else None

        # --- NEU: Neststarts je Gelege (täglich) ---
        daily_new_nests_c1 = np.zeros((days, n), dtype=np.int32)
        daily_new_nests_c2 = np.zeros((days, n), dtype=np.int32)
        daily_new_nests_c3 = np.zeros((days, n), dtype=np.int32)

        # Nester (Alterspools open/refuge)
        nest_age_len = self.cfg.nest_cycle_days
        nests_open = np.zeros((nest_age_len, n), dtype=np.float32)
        nests_refuge = np.zeros((nest_age_len, n), dtype=np.float32)

        # Juvenile Alterskohorten
        age_len = max(1, int(self.cfg.juvenile_free_age))
        J_age = np.zeros((age_len, n), dtype=np.int32)
        J_age_series = np.zeros((days, age_len, n), dtype=np.int32) if keep_age_series else None

        # Re-Nest-Cooldown (optional)
        cooldown_len = int(max(0, int(getattr(self.cfg, "renest_cooldown_days", 0) or 0)))
        cooldown_f = np.zeros((max(1, cooldown_len), n), dtype=np.int32) if cooldown_len > 0 else None

        broods_started_per_f = np.zeros(n, dtype=np.float32)

        arr = max(1, int(self.cfg.arrival_day))
        dep = min(days, int(self.cfg.departure_day))

        # Startpopulation
        if start_state is not None and 1 <= arr <= days:
            A_m[arr - 1, :] = np.asarray(start_state.get("A_m0", np.zeros(n, int)), dtype=int)
            A_f[arr - 1, :] = np.asarray(start_state.get("A_f0", np.zeros(n, int)), dtype=int)
        else:
            males0 = int(N_init) // 2
            females0 = int(N_init) - males0
            if 1 <= arr <= days:
                A_m[arr - 1, :] = rng.multinomial(males0, np.full(n, 1.0 / n))
                A_f[arr - 1, :] = rng.multinomial(females0, np.full(n, 1.0 / n))

        dbg = DebugTracer() if debug_days else None
        debug_days = set(debug_days or [])

        t0 = max(arr - 1, 0)
        tEnd = min(dep - 1, days - 2)
        for t in range(t0, tEnd + 1):
            doy = t + 1
            in_breed = (self.cfg.breeding_window[0] <= doy <= self.cfg.breeding_window[1])

            prev_idx = t
            next_idx = t + 1

            prev_m = A_m[prev_idx, :].copy()
            prev_f = A_f[prev_idx, :].copy()
            prev_free = J_free[prev_idx, :].copy()

            # Re-Nest-Cooldown Tagesfortschritt
            if cooldown_len > 0 and cooldown_f is not None:
                cooldown_f = np.roll(cooldown_f, shift=1, axis=0)
                cooldown_f[0, :] = 0

            today_m = np.zeros(n, dtype=np.int32)
            today_f = np.zeros(n, dtype=np.int32)

            # Aktive Nester = gebundene Brut-Weibchen (Belegung aus vorhandenen Nestern)
            active_nests_field = (nests_open.sum(axis=0) + nests_refuge.sum(axis=0))
            breeding_f = np.minimum(prev_f.astype(float), active_nests_field).astype(int)
            nonbreeding_f = prev_f - breeding_f
            today_f += breeding_f
            A_f_breeding[next_idx, :] = breeding_f.astype(int)

            # Soft-Kapazitätslast
            juv_equiv = float(getattr(self.cfg, "juvenile_equiv_in_capacity", JUV_EQUIV_IN_CAPACITY))
            prev_total = (
                (prev_m + prev_f).astype(float)
                + juv_equiv * (prev_free + J_age.sum(axis=0)).astype(float)
            )

            (U, pulse_stress, window_stress, refuge_bonus, refuge_share, p_ref_nest) = \
                self._utilities_for_day(doy, year, prev_total=prev_total)

            pulse_shield = np.clip(self.cfg.nest_refuge_gamma * PULSE_PROTECT_SCALE * refuge_share, 0.0, 0.9)

            move_radius = self.cfg.move_radius
            beta = self.cfg.beta
            distance_cost = self.cfg.distance_cost
            move_pen = distance_cost * (1.0 if not in_breed else (1.0 / max(self.cfg.breeding_move_multiplier, 1e-6)))

            # non-breeding Females bewegen
            for i, cnt in enumerate(nonbreeding_f):
                if cnt <= 0:
                    continue
                cand = np.where(self.dist_mat[i, :] <= move_radius)[0]
                if cand.size == 0:
                    cand = np.array([i])
                scores = U[cand] - move_pen * self.dist_mat[i, cand]
                probs = softmax(scores, beta=beta)
                moved = rng.multinomial(int(cnt), probs)
                today_f[cand] += moved

            # alle Males bewegen
            for i, cnt in enumerate(prev_m):
                if cnt <= 0:
                    continue
                cand = np.where(self.dist_mat[i, :] <= move_radius)[0]
                if cand.size == 0:
                    cand = np.array([i])
                scores = U[cand] - move_pen * self.dist_mat[i, cand]
                probs = softmax(scores, beta=beta)
                moved = rng.multinomial(int(cnt), probs)
                today_m[cand] += moved

            # Dichtefaktor (optional, nur für Repro genutzt – siehe unten)
            if USE_DENSITY_FACTOR and self.cfg.density_affects_start:
                density_factor = 1.0 / (1.0 + self.cfg.alpha_density * np.maximum(prev_total, 0.0))
                density_factor = np.clip(density_factor, 0.1, 1.0)
            else:
                density_factor = 1.0

            # ---------------- Reproduktion / Neststarts ----------------
            new_nests = np.zeros(n, dtype=int)
            p_start_vec = None
            if in_breed:
                # Weibchen im Cooldown blockieren
                blocked_by_cooldown = cooldown_f.sum(axis=0) if (cooldown_len > 0 and cooldown_f is not None) else np.zeros(n, dtype=np.int32)

                available_f = np.maximum(today_f - breeding_f - blocked_by_cooldown, 0)
                available_m = today_m
                potential_pairs = np.minimum(available_f, available_m)

                # Max. Bruten pro Weibchen einhalten
                max_broods_total = self.cfg.max_broods_per_female * today_f
                started_total = (broods_started_per_f * today_f).astype(float)
                remaining_capacity = np.maximum(0.0, max_broods_total - started_total).astype(int)

                # Revier-Slots – aktive Paare ≈ aktive Nester heute
                active_pairs_approx = (nests_open.sum(axis=0) + nests_refuge.sum(axis=0)).astype(int)
                free_slots = np.maximum(self.territory_slots - active_pairs_approx, 0)

                # Brutstart-Wahrscheinlichkeit
                x = np.asarray(U, dtype=np.float64) * float(self.cfg.fec_scale)
                x = np.clip(x, -50.0, 50.0)
                p_start_vec = 1.0 / (1.0 + np.exp(-x))
                p_start_vec = np.clip(p_start_vec, 0.0, 0.35)  # Tagesobergrenze
                if USE_DENSITY_FACTOR and self.cfg.density_affects_start:
                    p_start_vec = np.clip(p_start_vec * density_factor, 0.0, 0.9)

                # Kultur je Feld (für Clutch-Prefs)
                rot_y = self.rotation[self.rotation["year"] == year]
                crop_map = dict(zip(rot_y["field_id"].astype(int), rot_y["crop"].astype(str)))
                field_ids_vec = self.fields["field_id"].astype(int).to_numpy()
                crops_vec = np.array([crop_map.get(int(fid), "Winterweizen") for fid in field_ids_vec], dtype=object)

                # aktuelles Gelege je Feld: floor(broods_started_pro_Frau) + 1
                clutch_idx = (np.floor(broods_started_per_f).astype(int) + 1)
                clutch_idx = np.clip(clutch_idx, 1, int(self.cfg.max_broods_per_female))

                # Multiplikator je Feld (Clutch/DOY)
                mult_vec = np.ones_like(p_start_vec, dtype=float)
                if getattr(self.cfg, "clutch_prefs", None):
                    for i in range(len(mult_vec)):
                        mult_vec[i] = clutch_multiplier_for(
                            crops_vec[i], doy=int(doy), clutch_number=int(clutch_idx[i]),
                            prefs_lookup=self.cfg.clutch_prefs
                        )
                p_start_vec = np.clip(p_start_vec * mult_vec, 0.0, 0.9)

                can_start = np.minimum(potential_pairs, remaining_capacity)
                can_start = np.minimum(can_start, free_slots)
                can_start = np.maximum(can_start, 0)

                new_nests = rng.binomial(n=np.maximum(can_start, 0), p=p_start_vec)
                daily_new_nests[next_idx, :] = new_nests.astype(int)

                # --- Starts des heutigen Tages je Gelege aufsplitten ---
                c1_mask = (clutch_idx == 1)
                c2_mask = (clutch_idx == 2)
                c3_mask = (clutch_idx >= 3)

                d_c1 = np.where(c1_mask, new_nests, 0)
                d_c2 = np.where(c2_mask, new_nests, 0)
                d_c3 = np.where(c3_mask, new_nests, 0)

                daily_new_nests_c1[next_idx, :] = d_c1.astype(int)
                daily_new_nests_c2[next_idx, :] = d_c2.astype(int)
                daily_new_nests_c3[next_idx, :] = d_c3.astype(int)

                # Anteil in Refuge verteilen
                new_to_ref = rng.binomial(new_nests, np.clip(p_ref_nest, 0.0, REFUGE_SHARE_CAP))
                new_to_open = new_nests - new_to_ref

                nests_open = np.roll(nests_open, shift=1, axis=0);    nests_open[0, :] = new_to_open.astype(float)
                nests_refuge = np.roll(nests_refuge, shift=1, axis=0); nests_refuge[0, :] = new_to_ref.astype(float)

                with np.errstate(divide='ignore', invalid='ignore'):
                    broods_started_per_f += np.where(today_f > 0, new_nests / today_f, 0.0)
            else:
                p_start_vec = np.zeros_like(U)
                nests_open = np.roll(nests_open, shift=1, axis=0);     nests_open[0, :] = 0.0
                nests_refuge = np.roll(nests_refuge, shift=1, axis=0);  nests_refuge[0, :] = 0.0

            # ---------------- Nest-Überleben & Fledglings ----------------
            base = float(np.clip(self.cfg.base_nest_survival_day, 0.0001, 0.999))
            sens = float(self.cfg.nest_disturb_sensitivity)

            eff_open_stress = np.maximum(0.0, -(pulse_stress + float(WINDOW_STRESS_WEIGHT) * window_stress))
            nest_surv_open = np.clip(base * np.exp(-sens * eff_open_stress), 0.0001, 0.999)

            eff_ref_stress = np.maximum(
                0.0,
                -((1.0 - pulse_shield) * pulse_stress + float(WINDOW_STRESS_WEIGHT) * window_stress)
            )
            nest_surv_ref = base * np.exp(-sens * eff_ref_stress)
            nest_surv_ref *= (1.0 + refuge_bonus)
            nest_surv_ref = np.clip(nest_surv_ref, 0.0001, 0.999)

            # Lerchenfenster-Multiplikatoren
            ns_mult_vec, fl_mult_vec = self._skylark_window_multipliers_vector(day=doy, year=year)
            if self.skylark_cfg is not None and self.species_benefits_from_skylark_windows:
                nest_surv_open = np.clip(nest_surv_open * ns_mult_vec, 0.0001, 0.999)
                nest_surv_ref  = np.clip(nest_surv_ref  * ns_mult_vec, 0.0001, 0.999)

            # tägliche Fehlschläge zählen (für Cooldown)
            failed_open_today = np.zeros(n, dtype=np.int32)
            failed_ref_today = np.zeros(n, dtype=np.int32)

            for a in range(nest_age_len):
                if nests_open[a, :].sum() > 0:
                    fails_o = rng.binomial(n=np.maximum(nests_open[a, :].astype(int), 0), p=(1.0 - nest_surv_open))
                    failed_open_today += fails_o.astype(int)
                    nests_open[a, :] = np.maximum(0.0, nests_open[a, :] - fails_o)
                if nests_refuge[a, :].sum() > 0:
                    fails_r = rng.binomial(n=np.maximum(nests_refuge[a, :].astype(int), 0), p=(1.0 - nest_surv_ref))
                    failed_ref_today += fails_r.astype(int)
                    nests_refuge[a, :] = np.maximum(0.0, nests_refuge[a, :] - fails_r)

            failed_today = failed_open_today + failed_ref_today
            if cooldown_len > 0 and cooldown_f is not None:
                to_cool = np.minimum(A_f_breeding[next_idx, :].astype(int), failed_today.astype(int))
                cooldown_f[0, :] += to_cool

            # Fledglings am Ende des Zyklus
            fledging_n = nests_open[-1, :].copy() + nests_refuge[-1, :].copy()
            nests_open[-1, :] = 0.0
            nests_refuge[-1, :] = 0.0
            fledglings_raw = fledging_n * self.cfg.clutch_size * self.cfg.fledge_rate
            fledglings_raw = fledglings_raw * fl_mult_vec
            # WICHTIG: optional Dichtefaktor auch auf Jungvögel (wie früher) – nur wenn aktiviert
            if USE_DENSITY_FACTOR and self.cfg.density_affects_start:
                fledglings_raw = fledglings_raw * density_factor
            fledglings = fledglings_raw.astype(int)
            daily_fledglings[next_idx, :] = fledglings.astype(int)

            # Adult Survival
            surv_b = float(np.clip(self.cfg.survival_breeding, 0.0, 1.0))
            surv_nb = float(np.clip(self.cfg.survival_nonbreeding, 0.0, 1.0))
            # ---------------- Adult Survival (tägliche Überlebensrate) ----------------
            p_adult = surv_b if in_breed else surv_nb
            A_m[next_idx, :] = rng.binomial(np.maximum(today_m, 0), p_adult)
            A_f[next_idx, :] = rng.binomial(np.maximum(today_f, 0), p_adult)

            # ---------------- Juvenile: altern, überleben, frei werden ----------------
            # (1) Nestlinge, die heute flügge wurden, kommen in Alterskohorte 0
            new_fledge = fledglings.astype(int)

            # (2) Alle vorhandenen Alterskohorten überleben heute als Juvenile
            if J_age.sum() > 0:
                for a in range(age_len):
                    vals = np.maximum(J_age[a, :].astype(int), 0)
                    if vals.sum() > 0:
                        J_age[a, :] = rng.binomial(vals, float(self.cfg.survival_juvenile))

            # (3) Kohorte, die heute "aus dem Raster" fällt, wird zu J_free
            released_today = J_age[-1, :].astype(int).copy()

            # (4) Altern: nach oben schieben; Index 0 wird gleich mit new_fledge gefüllt
            if age_len > 1:
                J_age[1:, :] = J_age[:-1, :]
            J_age[0, :] = 0  # wird gleich gesetzt

            # (5) Frisch flügge hinzu
            if new_fledge.sum() > 0:
                J_age[0, :] += new_fledge

            # (6) Freie Juvenile (bereits "out of nest" + aus Alterskohorten entlassen)
            prev_free_surv = rng.binomial(np.maximum(prev_free, 0), float(self.cfg.survival_juvenile))
            J_free[next_idx, :] = prev_free_surv + released_today
            # >>> NEU: gebundene Juvenile (Summe aller Alterskohorten) mitschreiben
            J_tied_total[next_idx, :] = J_age.sum(axis=0).astype(int)

            # (7) Debug/Logging
            if repro is not None:
                # mittlere "Nest-Überleben heute" für Log (open/refuge gemittelt)
                nest_surv_day = 0.5 * (nest_surv_open + nest_surv_ref)
                repro.log(
                    day=doy,
                    in_breed=in_breed,
                    U=U,
                    p_start_vec=(p_start_vec if p_start_vec is not None else np.zeros(n)),
                    density_factor=(density_factor if isinstance(density_factor, np.ndarray) else np.full(n, float(
                        density_factor))),
                    nest_surv_day=nest_surv_day,
                    new_nests_vec=new_nests,
                    fledglings_vec=new_fledge,
                    A_m_today=today_m,
                    A_f_today=today_f,
                    J_tied_today=np.zeros(n, dtype=int),  # wir führen Nestlinge explizit nicht separat
                    J_free_today=J_free[next_idx, :],
                )

            if keep_age_series and J_age_series is not None:
                J_age_series[next_idx, :, :] = J_age.astype(int)

            # (8) N_total (für diesen Tag+1)
            N_total_today = (
                    A_m[next_idx, :].astype(int) +
                    A_f[next_idx, :].astype(int) +
                    J_free[next_idx, :].astype(int) +
                    J_age.sum(axis=0).astype(int)
            )
            if 'N_total' not in locals():
                N_total = np.zeros((days, n), dtype=np.int32)
            N_total[next_idx, :] = N_total_today.astype(int)

            # (9) Optionaler Debug-Snapshot für ausgewählte Tage
            if dbg is not None and (doy in debug_days):
                # grobe „mittlere Nestüberlebensrate“ für den Tag
                nest_surv_day_scalar = float(np.mean(0.5 * (nest_surv_open + nest_surv_ref)))
                dbg.capture_day(
                    doy=doy,
                    U=U,
                    today_m=today_m,
                    today_f=today_f,
                    breeding_f=breeding_f,
                    new_nests=new_nests,
                    fledglings=new_fledge,
                    density_factor=(density_factor if isinstance(density_factor, np.ndarray) else np.full(n, float(
                        density_factor))),
                    nest_surv_day=np.full(n, nest_surv_day_scalar, dtype=float)
                )

        # ---- Anfangszustand (arr-1) für N_total füllen, falls leer ----
        if 'N_total' not in locals():
            N_total = np.zeros((days, n), dtype=np.int32)
        if arr - 1 >= 0:
            N_total[arr - 1, :] = (A_m[arr - 1, :] + A_f[arr - 1, :]).astype(int)

        # ---- Rückgabeobjekt bauen ----
        usage = N_total.sum(axis=0).astype(int)
        res_out = {
            "fields": self.fields.copy(),
            "A_m": A_m.astype(int),
            "A_f": A_f.astype(int),
            "J_free": J_free.astype(int),
            "J_tied_total": J_tied_total.astype(int),  # hier 0; Nestlinge werden via J_age abgebildet
            "A_f_breeding": A_f_breeding.astype(int),
            "daily_new_nests": daily_new_nests.astype(int),
            "daily_new_nests_c1": daily_new_nests_c1.astype(int),
            "daily_new_nests_c2": daily_new_nests_c2.astype(int),
            "daily_new_nests_c3": daily_new_nests_c3.astype(int),
            "daily_fledglings": daily_fledglings.astype(int),
            "N_total": N_total.astype(int),
            "usage": usage,
            "J_tied_total": J_tied_total.astype(int),
        }
        if keep_age_series and J_age_series is not None:
            res_out["J_age_series"] = J_age_series.astype(int)
        if repro is not None:
            try:
                res_out["debug_repro"] = repro.to_df()
            except Exception:
                pass
        res_out["debug"] = (dbg.to_frames() if dbg is not None else {})

        return res_out


# ---------------- Tidy time-series export ----------------
def to_long_timeseries(res, species, year, scenario="Baseline"):
    """
    Long-Format-Zeitreihe:
      Spalten: day (1..TOT_DAYS), year, field_id, stage, n, species, scenario
      Stages: A_m, A_f, J_tied, J_free, N_total
    """
    import numpy as np
    import pandas as pd

    N   = res["N_total"]
    A_m = res["A_m"]
    A_f = res["A_f"]
    Jt  = res.get("J_tied_total", res.get("J_tied"))
    if Jt is None:
        Jt = np.zeros_like(N, dtype=np.int32)
    Jf  = res["J_free"]

    field_ids = res["fields"]["field_id"].astype(int).to_numpy()

    def _pack(arr: np.ndarray, stage_name: str) -> pd.DataFrame:
        df = pd.DataFrame(arr, columns=field_ids)
        df = df.reset_index().melt(id_vars="index", var_name="field_id", value_name="n")
        df = df.rename(columns={"index": "day"})
        df["stage"] = stage_name
        return df

    parts = [
        _pack(A_m, "A_m"),
        _pack(A_f, "A_f"),
        _pack(Jt,  "J_tied"),
        _pack(Jf,  "J_free"),
        _pack(N,   "N_total"),
    ]
    out = pd.concat(parts, ignore_index=True)

    # 0-basierter Index -> DOY 1..TOT_DAYS
    out["day"] = out["day"].astype(int) + 1
    out["year"] = int(year)
    out["field_id"] = out["field_id"].astype(int)
    out["n"] = pd.to_numeric(out["n"], errors="coerce").fillna(0).astype(int)
    out["species"] = str(species)
    out["scenario"] = str(scenario)

    try:
        td = int(globals().get("TOT_DAYS", 365))
    except Exception:
        td = 365
    out = out[(out["day"] >= 1) & (out["day"] <= td)]
    return out

# =========================================================
#  (module 7/7) Scenarios, compare, main
# =========================================================

# ---------- Scenario helpers ----------
def choose_flower_target_fields(
    fields_df: pd.DataFrame,
    *,
    fraction: float,
    mode: str = "random",
    seed: int = 2025
) -> list[int]:
    """Wählt Feld-IDs für Blühstreifen aus."""
    F = fields_df[["field_id", "flower_strip_share"]].copy()
    if "flower_strip_share" not in F.columns:
        F["flower_strip_share"] = 0.0
    F["flower_strip_share"] = F["flower_strip_share"].fillna(0.0)

    k = int(round(max(0.0, min(1.0, float(fraction))) * len(F)))
    if k <= 0:
        return []

    if mode == "no_flower_first":
        # zuerst Felder mit 0, dann aufwärts; Ties deterministisch per Seed mischen
        rng = np.random.default_rng(int(seed))
        # stabile Reihenfolge: erst sortieren, dann innerhalb gleicher Werte permutieren
        F = F.sort_values("flower_strip_share", ascending=True).reset_index(drop=True)
        groups = []
        for v, grp in F.groupby("flower_strip_share", sort=True):
            idx = grp.index.to_numpy()
            if len(idx) > 1:
                idx = idx[rng.permutation(len(idx))]
            groups.append(idx)
        order = np.concatenate(groups)
        chosen_idx = order[:k]
        return F.loc[chosen_idx, "field_id"].astype(int).tolist()

    # default: random
    rng = np.random.default_rng(int(seed))
    return rng.choice(F["field_id"].to_numpy(dtype=int), size=k, replace=False).tolist()

def shift_window_str(win_str: str, delta: int, *, day_min: int = 1, day_max: int = 365) -> str:
    """Fenster 'a-b' um delta verschieben, sicher auf [1..365] clampen und sortieren."""
    if not isinstance(win_str, str) or "-" not in win_str:
        return win_str
    a_str, b_str = win_str.split("-", 1)
    try:
        a = int(float(a_str.strip())); b = int(float(b_str.strip()))
    except Exception:
        return win_str
    a += int(delta); b += int(delta)
    a = max(day_min, min(day_max, a))
    b = max(day_min, min(day_max, b))
    if a > b:
        a, b = b, a
    return f"{a}-{b}"

def harvest_overlap_share(mgmt_df: pd.DataFrame, breeding_window: tuple[int,int]) -> float:
    """
    Anteil der (gültigen) harvest_date-Einträge, die im Brutfenster liegen.
    """
    if "harvest_date" not in mgmt_df.columns:
        return float("nan")
    a, b = int(breeding_window[0]), int(breeding_window[1])
    s = mgmt_df["harvest_date"].dropna()
    if s.empty:
        return float("nan")
    try:
        v = s.astype(float).astype(int)
    except Exception:
        return float("nan")
    denom = (v.between(1, 365)).sum()
    if denom == 0:
        return float("nan")
    return float(v.between(a, b).sum()) / float(denom)

def choose_skylark_target_fields(
    fields_df: pd.DataFrame,
    rotation_df: pd.DataFrame,
    *,
    eligible_crops: set[str] | None = None,
    min_years: int = 1,
    top_fraction: float = 0.20,
    only_without_flower: bool = True
) -> list[int]:
    """
    Liefert eine Liste von field_id, die sich besonders für Lerchenfenster eignen:
      - viele Jahre mit 'eligible_crops' (Standard: ELIGIBLE_REFUGE_CROPS)
      - optional: ohne Blühstreifen (only_without_flower)
      - top_fraction der Felder nach Score
    Score = Anzahl 'eligible'-Jahre (kleiner Bonus, wenn keine Blühstreifen)
    """
    if eligible_crops is None:
        eligible_crops = set(ELIGIBLE_REFUGE_CROPS)

    rot = rotation_df.copy()
    rot["crop"] = rot["crop"].astype(str)
    rot["eligible"] = rot["crop"].isin(eligible_crops)

    counts = (rot.groupby("field_id", as_index=False)["eligible"]
                 .sum()
                 .rename(columns={"eligible": "eligible_years"}))

    F = fields_df[["field_id", "flower_strip_share"]].copy()
    if "flower_strip_share" not in F.columns:
        F["flower_strip_share"] = 0.0
    F["flower_strip_share"] = F["flower_strip_share"].fillna(0.0)

    df = F.merge(counts, on="field_id", how="left").fillna({"eligible_years": 0})
    df["eligible_years"] = df["eligible_years"].astype(int)

    # Mindestanzahl eligible Jahre
    df = df[df["eligible_years"] >= int(min_years)].copy()
    if df.empty:
        return []

    # Score: bevorzugt viele eligible Jahre; leichter Bonus, wenn keine Blühstreifen
    bonus = np.where((df["flower_strip_share"] <= 1e-9) & only_without_flower, 0.25, 0.0)
    df["score"] = df["eligible_years"].astype(float) + bonus

    k = max(1, int(round(top_fraction * len(fields_df))))
    top = (df.sort_values(["score", "eligible_years"], ascending=False)
             .head(k))
    return top["field_id"].astype(int).tolist()

from typing import Tuple
import numpy as np
import pandas as pd

def apply_management_scenario(
    fields_df=None,
    management_df=None,
    # --- A: Ernte ---
    harvest_shift_days: int = 0,
    # --- B: Blühstreifen ---
    add_flower_strip: float = 0.0,          # z.B. 0.10 (= +10 %-Punkte)
    target_fraction: float | None = None,   # Anteil Felder, die Blühstreifen bekommen (0..1)
    selection: str = "random",              # "random" | "no_flower_first"
    seed: int | None = None,
    # --- D: Lerchenfenster ---
    skylark_plots: int = 0,
    skylark_plot_area: float = 0.0,         # ha je Fenster
    skylark_target_field_ids: list[int] | None = None,
    skylark_target_fraction: float = 0.0,
    skylark_min_eligible_years: int = 1,
    skylark_only_without_flower: bool = True,
    # ---- rückwärtskompatible Aliase (werden unten gemappt) ----
    fields_gdf=None,
    management=None,
    shifts=None,                            # alt für harvest_shift_days
    flower_fraction: float | None = None,   # ALT: Anteil Felder (NICHT die Additionshöhe!)
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wendet Management-Änderungen an und gibt (fields_df_neu, management_df_neu) zurück.
    Robust ggü. alten Argumentnamen aus deinem Skript.
    """

    # ---------- Alias-Zuordnung ----------
    if fields_df is None and fields_gdf is not None:
        fields_df = fields_gdf
    if management_df is None and management is not None:
        management_df = management

    # alt: shifts -> harvest_shift_days (Zahl oder Liste)
    if (harvest_shift_days in (None, 0)) and (shifts is not None):
        try:
            if isinstance(shifts, (list, tuple, np.ndarray)):
                harvest_shift_days = int(shifts[0]) if len(shifts) else 0
                print(f"[compat] apply_management_scenario: benutze shift={harvest_shift_days} aus shifts={shifts}")
            else:
                harvest_shift_days = int(shifts)
        except Exception:
            pass

    # alt: flower_fraction bedeutete *Anteil Felder* -> target_fraction (nicht add_flower_strip!)
    if (target_fraction in (None, 0.0)) and (flower_fraction not in (None, 0.0)):
        try:
            target_fraction = float(flower_fraction)
        except Exception:
            pass

    # alt: add_skylark_plots (wie in deinen Aufrufen) -> skylark_plots
    if "add_skylark_plots" in kwargs and (not skylark_plots or skylark_plots == 0):
        try:
            skylark_plots = int(kwargs["add_skylark_plots"])
        except Exception:
            pass
    # alt: skylark_fraction -> skylark_target_fraction
    if "skylark_fraction" in kwargs and (not skylark_target_fraction or skylark_target_fraction == 0.0):
        try:
            skylark_target_fraction = float(kwargs["skylark_fraction"])
        except Exception:
            pass
    # alt: flower_seed -> seed
    if "flower_seed" in kwargs and seed is None:
        try:
            seed = int(kwargs["flower_seed"])
        except Exception:
            seed = None

    # ---------- Kopien (niemals in-place ändern) ----------
    F = fields_df.copy()
    M = management_df.copy()

    # Pflichtspalten weich sicherstellen
    if "field_id" not in F.columns:
        raise ValueError("fields_df fehlt 'field_id'.")
    F["field_id"] = F["field_id"].astype(int)
    if "flower_strip_share" not in F.columns:
        F["flower_strip_share"] = 0.0
    if "skylark_plots" not in F.columns:
        F["skylark_plots"] = 0
    if "skylark_plot_area" not in F.columns:
        F["skylark_plot_area"] = 0.0

    # ========= A) Ernte verschieben =========
    def _clamp_day(x):
        try:
            d = int(round(float(x)))
            return max(1, min(365, d))
        except Exception:
            return x

    d_shift = int(harvest_shift_days or 0)
    if d_shift != 0 and len(M):
        if "harvest_day" in M.columns:
            M["harvest_day"] = pd.to_numeric(M["harvest_day"], errors="coerce").fillna(0).astype(int)
            M["harvest_day"] = M["harvest_day"].apply(lambda v: _clamp_day(v + d_shift) if v > 0 else v)
        elif "harvest_date" in M.columns:
            M["harvest_date"] = pd.to_numeric(M["harvest_date"], errors="coerce")
            M["harvest_date"] = M["harvest_date"].apply(lambda v: _clamp_day(v + d_shift) if pd.notna(v) else v)
        else:
            op_col = next((c for c in M.columns if str(c).lower() in {"op","operation"}), None)
            day_col = next((c for c in M.columns if str(c).lower() in {"day","doy","tag"}), None)
            if op_col and day_col:
                ops = M[op_col].astype(str).str.lower().str.strip()
                is_harv = ops.isin({"harvest","ernte"})
                if is_harv.any():
                    M.loc[is_harv, day_col] = (
                        pd.to_numeric(M.loc[is_harv, day_col], errors="coerce")
                          .fillna(0).astype(int)
                          .apply(lambda v: _clamp_day(v + d_shift) if v > 0 else v)
                    )

    # ========= B) Blühstreifen hinzufügen =========
    add_fs = float(add_flower_strip or 0.0)           # z. B. 0.10
    tgt_frac = float(target_fraction) if target_fraction not in (None, np.nan) else 0.0
    if add_fs > 0.0 and len(F):
        n = len(F)
        k = int(np.ceil(max(0.0, min(1.0, tgt_frac)) * n)) if tgt_frac > 0 else n
        if selection == "no_flower_first":
            order = F.sort_values("flower_strip_share", ascending=True).index
            pick_idx = order[:k]
        else:
            rng = np.random.default_rng(seed if seed is not None else 2025)
            pick_idx = rng.choice(F.index.to_numpy(), size=k, replace=False)
        F.loc[pick_idx, "flower_strip_share"] = (
            F.loc[pick_idx, "flower_strip_share"].astype(float) + add_fs
        ).clip(0.0, 0.6)  # ggf. Kappung anpassen

    # ========= D) Lerchenfenster hinzufügen =========
    skl_n = int(skylark_plots or 0)
    skl_ha = float(skylark_plot_area or 0.0)
    if skl_n > 0 and skl_ha > 0.0 and len(F):
        if skylark_target_field_ids is not None and len(skylark_target_field_ids):
            mask = F["field_id"].isin(pd.Index(skylark_target_field_ids).astype(int))
        elif skylark_target_fraction and skylark_target_fraction > 0:
            k = int(np.ceil(skylark_target_fraction * len(F)))
            order = F.sort_values("flower_strip_share", ascending=True).index
            mask = F.index.isin(order[:k])
        else:
            mask = np.ones(len(F), dtype=bool)

        F.loc[mask, "skylark_plots"] = (
            pd.to_numeric(F.loc[mask, "skylark_plots"], errors="coerce").fillna(0).astype(int) + skl_n
        )
        F.loc[mask, "skylark_plot_area"] = (
            pd.to_numeric(F.loc[mask, "skylark_plot_area"], errors="coerce").fillna(0.0).astype(float) + skl_ha
        )

    # Immer Tuple zurückgeben (wichtig für fA, mA = ...)
    return F, M

# ---------- Scenario C helper: global ----------
def apply_fallow_scenario(fields_df: pd.DataFrame,
                          rotation_df: pd.DataFrame,
                          management_df: pd.DataFrame,
                          *,
                          share: float,
                          years: list[int] | str = "all",
                          mode: str = "largest",
                          seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame, list[int], float]:
    """
    Setzt `share` der Gesamtfläche als Brache, indem ganze Felder gewählt werden.
    Gibt (rotation_new, management_new, chosen_field_ids, chosen_area_share_actual) zurück.
    - years="all": in allen Simulationsjahren; sonst Liste von Jahren (ints)
    - mode="largest": größte Felder zuerst bis Ziel erreicht
      mode="weighted_random": zufällige Auswahl, Gewicht ∝ Fläche
    Management der Brache-Felder wird beruhigt (keine Ernte/Insektizid/Fenster).
    """
    r2 = rotation_df.copy()
    m2 = management_df.copy()

    F = fields_df[["field_id", "area"]].copy()
    F["field_id"] = F["field_id"].astype(int)
    F["area"] = F["area"].astype(float)

    total_area = float(F["area"].sum())
    target_area = float(max(0.0, min(1.0, share)) * total_area)
    if target_area <= 0.0 or total_area <= 0.0:
        return r2, m2, [], 0.0

    rng = np.random.default_rng(int(seed))

    if mode == "weighted_random":
        # ziehe ohne Zurücklegen, bis Summe >= Ziel
        pool = F.copy()
        pool["p"] = pool["area"] / max(total_area, 1e-12)
        chosen = []
        acc = 0.0
        idxs = pool.index.to_numpy()
        while acc < target_area and len(idxs) > 0:
            probs = pool.loc[idxs, "p"].to_numpy()
            probs = probs / probs.sum()
            pick = rng.choice(idxs, size=1, replace=False, p=probs)[0]
            chosen.append(int(pool.at[pick, "field_id"]))
            acc += float(pool.at[pick, "area"])
            idxs = idxs[idxs != pick]
        chosen_ids = chosen
        chosen_area = acc

    elif mode == "greedy":
        # --- Hilfsfunktion: einfacher Greedy in gegebener Reihenfolge ---
        def greedy_once(df_sorted, target):
            chosen = []
            chosen_area = 0.0
            for _, row in df_sorted.iterrows():
                a = float(row["area"]);
                fid = int(row["field_id"])
                if abs((chosen_area + a) - target) <= abs(chosen_area - target):
                    chosen.append(fid)
                    chosen_area += a
            return chosen, chosen_area
        # 1) Zwei Läufe: absteigend und aufsteigend nach Fläche
        F_desc = F.sort_values("area", ascending=False).reset_index(drop=True)
        F_asc = F.sort_values("area", ascending=True).reset_index(drop=True)
        chosen_d, area_d = greedy_once(F_desc, target_area)
        chosen_a, area_a = greedy_once(F_asc, target_area)
        # Bessere der beiden Lösungen wählen
        if abs(area_a - target_area) < abs(area_d - target_area):
            chosen_ids = chosen_a[:]
            chosen_area = area_a
        else:
            chosen_ids = chosen_d[:]
            chosen_area = area_d
        # 2) Additions-Feinsuche: füge kleine Nicht-gewählte hinzu, wenn es verbessert
        not_chosen = set(F["field_id"].astype(int)) - set(chosen_ids)
        improved = True
        while improved and len(not_chosen) > 0:
            improved = False
            best_gain = 0.0
            best_id = None
            for fid in list(not_chosen):
                a = float(F.loc[F["field_id"] == fid, "area"].iloc[0])
                before = abs(chosen_area - target_area)
                after = abs((chosen_area + a) - target_area)
                gain = before - after
                if gain > best_gain + 1e-12:  # bestes Plus merken
                    best_gain = gain
                    best_id = fid
            if best_id is not None:
                chosen_ids.append(best_id)
                chosen_area += float(F.loc[F["field_id"] == best_id, "area"].iloc[0])
                not_chosen.remove(best_id)
                improved = True
        # 3) Pruning: entferne einzelne ausgewählte Felder, wenn Entfernung weiter verbessert
        improved = True
        while improved and len(chosen_ids) > 1:
            improved = False
            best_gain = 0.0
            best_id = None
            for fid in list(chosen_ids):
                a = float(F.loc[F["field_id"] == fid, "area"].iloc[0])
                before = abs(chosen_area - target_area)
                after = abs((chosen_area - a) - target_area)
                gain = before - after
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_id = fid
            if best_id is not None:
                chosen_ids.remove(best_id)
                chosen_area -= float(F.loc[F["field_id"] == best_id, "area"].iloc[0])
                improved = True


    else:
        # größte Felder zuerst bis Ziel erreicht
        F2 = F.sort_values("area", ascending=False).reset_index(drop=True)
        cum = F2["area"].cumsum()
        k = int((cum >= target_area).idxmax()) if (cum >= target_area).any() else len(F2) - 1
        chosen_ids = F2.loc[:k, "field_id"].astype(int).tolist()
        chosen_area = float(F2.loc[:k, "area"].sum())

    actual_share = chosen_area / max(total_area, 1e-12)

    # Jahre festlegen
    if years == "all":
        years_set = set(r2["year"].astype(int).unique().tolist())
    else:
        years_set = set(int(y) for y in years)

    # Rotation: auf Brache setzen
    mask_fields = r2["field_id"].astype(int).isin(chosen_ids)
    mask_years = r2["year"].astype(int).isin(years_set)
    r2.loc[mask_fields & mask_years, "crop"] = "Brache"

    # Management für Brache-Felder "leiser" (keine Ernte/Insektizid/Fenster)
    if len(m2):
        mm = m2.copy()
        mm["field_id"] = mm["field_id"].astype(int)
        mm["year"] = mm["year"].astype(int)
        m_mask = mm["field_id"].isin(chosen_ids) & mm["year"].isin(years_set)
        for col in ["harvest_date", "insecticide_date", "sow_win", "harvest_win", "other_ops_win"]:
            if col in mm.columns:
                mm.loc[m_mask, col] = None
        if "mow_dates" in mm.columns:
            mm.loc[m_mask, "mow_dates"] = ""
        m2 = mm

    print(f"[Scenario C] Ziel={share * 100:.1f}% Fläche | tatsächlich={actual_share * 100:.2f}% | Felder={len(chosen_ids)}")
    return r2, m2, chosen_ids, actual_share


    # ---------- Blühstreifen (gezielt oder flächig) ----------
    if add_flower_strip != 0.0:
        if "flower_strip_share" not in f2.columns:
            f2["flower_strip_share"] = 0.0

        # Zielmenge bestimmen: explizite IDs > Fraktion > alle
        if flower_target_field_ids is not None and len(flower_target_field_ids) > 0:
            target_idx = f2.index[f2["field_id"].isin(list(flower_target_field_ids))].to_numpy()
        elif flower_fraction is None:
            target_idx = f2.index.to_numpy()
        else:
            frac = max(0.0, min(1.0, float(flower_fraction)))
            k = int(round(frac * len(f2)))
            rng = np.random.default_rng(int(flower_seed))
            target_idx = rng.choice(f2.index.to_numpy(), size=k, replace=False) if k > 0 else np.array([], dtype=int)

        if len(target_idx) > 0:
            f2.loc[target_idx, "flower_strip_share"] = (
                    f2.loc[target_idx, "flower_strip_share"].fillna(0.0) + float(add_flower_strip)
            ).clip(0.0, 0.5)

    # ---------- Lerchenfenster (wie gehabt, optional gezielt per skylark_fraction/IDs) ----------
    if add_skylark_plots != 0:
        if "skylark_plots" not in f2.columns:
            f2["skylark_plots"] = 0
        if "skylark_plot_area" not in f2.columns:
            f2["skylark_plot_area"] = 0.0

        f2["skylark_plots"] = f2["skylark_plots"].fillna(0).astype(int)
        f2["skylark_plot_area"] = f2["skylark_plot_area"].fillna(0.0).astype(float)

        idx_all = f2.index.to_numpy()

        # --- NEU: explizite Zielfelder per field_id (hat Priorität) ---
        if skylark_target_field_ids:
            chosen_idx = f2.index[f2["field_id"].isin(skylark_target_field_ids)].to_numpy()
        else:
            # Fallback: Zufallsauswahl per Anteil
            if skylark_fraction is None:
                chosen_idx = idx_all
            else:
                frac = max(0.0, min(1.0, float(skylark_fraction)))
                k = int(round(frac * len(idx_all)))
                rng = np.random.default_rng(int(skylark_seed))
                chosen_idx = rng.choice(idx_all, size=k, replace=False) if k > 0 else np.array([], dtype=int)

        if len(chosen_idx) > 0:
            f2.loc[chosen_idx, "skylark_plots"] = f2.loc[chosen_idx, "skylark_plots"] + int(add_skylark_plots)
            if skylark_plot_area is not None:
                f2.loc[chosen_idx, "skylark_plot_area"] = np.maximum(
                    f2.loc[chosen_idx, "skylark_plot_area"], float(skylark_plot_area)
                )

    # ---------- Ernte verschieben (robust, clamped) ----------
    if int(harvest_shift_days) != 0:
        dh = int(harvest_shift_days)

        if "harvest_date" in m2.columns:
            def _shift_day(x):
                if pd.isna(x) or str(x).strip() == "":
                    return x
                try:
                    v = int(float(x)) + dh
                    return int(np.clip(v, 1, 365))
                except Exception:
                    return x

            m2["harvest_date"] = m2["harvest_date"].apply(_shift_day)

        if "harvest_win" in m2.columns:
            m2["harvest_win"] = m2["harvest_win"].apply(lambda s: shift_window_str(s, dh))

    return f2, m2


def usage_compare(field_ids, usage_base, usage_scen, label, saveprefix=None):
    x = np.arange(len(field_ids)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, usage_base, width=w, label="Baseline")
    ax.bar(x + w/2, usage_scen, width=w, label=label)
    ax.set_xticks(x); ax.set_xticklabels(field_ids)
    ax.set_xlabel("field_id"); ax.set_ylabel("Summe Individuen-Tage")
    ax.set_title(f"Gesamtnutzung je Feld – {label} vs. Baseline"); ax.legend()
    if saveprefix:
        _save_fig(f"{saveprefix}_bars.png")
    else:
        plt.tight_layout(); plt.show()

    delta = usage_scen - usage_base
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(field_ids, delta)
    ax.set_xlabel("field_id"); ax.set_ylabel("Δ Nutzung")
    ax.set_title(f"Delta Nutzung je Feld – {label}")
    if saveprefix:
        _save_fig(f"{saveprefix}_delta.png")
    else:
        plt.tight_layout(); plt.show()

def run_monte_carlo(fields_gdf, rotation, management, cfg, years,
                    n_reps=200, seed=123, vary=None, outdir="outputs/mc"):
    """
    Randomly samples model params, runs a full multi-year sim for each replicate,
    and records lambda (year-to-year adults at arrival) and end-pop size.

    'vary' maps param_name -> (low, high). Names can be SpeciesConfig fields or
    module-level globals used in the sim (e.g., CAPACITY_GAMMA).
    """
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)

    # module-level globals that affect simulate(); remember to restore
    global CAPACITY_GAMMA, WINDOW_STRESS_WEIGHT, REFUGE_ASSIGN_MULT, PULSE_PROTECT_SCALE, REFUGE_SHARE_CAP
    _globals0 = dict(
        CAPACITY_GAMMA=CAPACITY_GAMMA,
        WINDOW_STRESS_WEIGHT=WINDOW_STRESS_WEIGHT,
        REFUGE_ASSIGN_MULT=REFUGE_ASSIGN_MULT,
        PULSE_PROTECT_SCALE=PULSE_PROTECT_SCALE,
        REFUGE_SHARE_CAP=REFUGE_SHARE_CAP,
    )

    if vary is None:
        vary = {
            "fec_scale": (0.25, 0.50),
            "fledge_rate": (0.75, 0.90),
            "alpha_density": (0.001, 0.004),
            "survival_breeding": (0.990, 0.999),
            "survival_nonbreeding": (0.998, 1.000),
            "survival_juvenile": (0.980, 0.998),
            # globals:
            "CAPACITY_GAMMA": (0.20, 0.60),
            "WINDOW_STRESS_WEIGHT": (0.40, 0.70),
            "REFUGE_ASSIGN_MULT": (10.0, 25.0),
            "PULSE_PROTECT_SCALE": (3.0, 6.0),
            "REFUGE_SHARE_CAP": (0.50, 0.80),
        }

    rows = []
    for rep in range(n_reps):
        cfg2 = replace(cfg)  # shallow clone of SpeciesConfig

        # sample params
        sampled = {}
        for k, (lo, hi) in vary.items():
            v = float(rng.uniform(lo, hi))
            sampled[k] = v
            if hasattr(cfg2, k):
                setattr(cfg2, k, v)
            elif k in _globals0:
                globals()[k] = v  # set module-global used inside simulate()

        # run years with feedback
        resY = run_years_with_feedback(fields_gdf, rotation, management, cfg2,
                                       years=years, N_init_first=50)

        # adults at arrival each year + lambdas
        arr_idx = max(1, int(cfg2.arrival_day)) - 1
        adults = [int((resY[y]["A_m"][arr_idx, :] + resY[y]["A_f"][arr_idx, :]).sum()) for y in years]
        lambdas = [adults[i+1] / max(adults[i], 1e-9) for i in range(len(adults) - 1)]
        lam_mean = float(np.mean(lambdas)) if lambdas else float("nan")

        rows.append({"rep": rep, "lambda_mean": lam_mean,
                     "adults_start": adults[0], "adults_end": adults[-1], **sampled})

    # write CSV
    mc_df = pd.DataFrame(rows)
    p = pathlib.Path(outdir) / f"mc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    mc_df.to_csv(p, index=False)

    # restore globals
    for k, v in _globals0.items():
        globals()[k] = v

    print(f"[OK] Monte Carlo written: {p}")
    return mc_df, str(p)

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
    days, n = A_f.shape
    m = _breed_mask(days, breeding_win)
    sub = A_f[m, :] > 0
    with np.errstate(invalid='ignore'):
        return sub.mean(axis=0)

def colonization_rate(prev: dict, curr: dict, breeding_win: tuple[int, int]) -> float:
    A_prev = prev["A_f"]; A_curr = curr["A_f"]
    days, _ = A_prev.shape
    m = _breed_mask(days, breeding_win)
    occ_prev = (A_prev[m, :] > 0).any(axis=0)
    occ_curr = (A_curr[m, :] > 0).any(axis=0)
    empty_prev = ~occ_prev
    denom = int(empty_prev.sum())
    if denom <= 0:
        return 0.0  # statt NaN
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

    # Geburten (flügge Jungvögel) und Todesfälle rekonstruieren
    daily_fledg = res.get("daily_fledglings")
    if daily_fledg is not None:
        daily_fledg_tot = daily_fledg.sum(axis=1)
        births_total = int(daily_fledg_tot.sum())
    else:
        daily_fledg_tot = np.zeros(days, dtype=int)
        births_total = 0
    fledglings_total = births_total  # rückwärtskompatibel

    # Erwachsene: tägliche Gesamtzahlen -> Verluste
    adults_tot = (A_m + A_f).sum(axis=1)
    adult_deaths = int(
        np.maximum(adults_tot[arr_idx:-1] - adults_tot[arr_idx + 1:], 0).sum()
    )

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

    # Female-breeding-Equivalents: Summe Brut-Weibchen-Tage / Nestzyklus
    A_f_b = res.get("A_f_breeding")
    if A_f_b is None:
        # Fallback: approximativ A_f im Brutfenster
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

    # Adults at arrival -> λ
    arr_idx = max(1, int(cfg.arrival_day)) - 1
    adults = [int((results_by_year[y]["A_m"][arr_idx, :] + results_by_year[y]["A_f"][arr_idx, :]).sum())
              for y in years]
    lambdas = [adults[i + 1] / max(adults[i], 1e-9) for i in range(len(adults) - 1)] if len(years) > 1 else []
    lam_map = {years[i + 1]: float(lambdas[i]) for i in range(len(lambdas))}

    for i, y in enumerate(years):
        k = compute_kpis_one_year(results_by_year[y], cfg)
        k["year"] = int(y)
        k["lambda_prev"] = float(lam_map.get(y, float("nan")))
        # Kolonisation (gegenüber Vorjahr)
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
    # n Schichten je Dimension
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

        # Regression Xi ~ Xo
        Bx, *_ = np.linalg.lstsq(np.c_[np.ones(n), Xo], Xi, rcond=None)
        rx = Xi - np.c_[np.ones(n), Xo] @ Bx

        # Regression yR ~ Xo
        By, *_ = np.linalg.lstsq(np.c_[np.ones(n), Xo], yR, rcond=None)
        ry = yR - np.c_[np.ones(n), Xo] @ By

        r = float(np.corrcoef(rx, ry)[0, 1])
        df = max(n - k - 1, 1)
        t = r * np.sqrt(df / max(1e-12, (1 - r**2)))
        # zweiseitiger p-Wert via Normal-Approx
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        out.append((params_df.columns[i], r, p))

    return pd.DataFrame(out, columns=["param", "prcc", "p_approx"]).sort_values(
        "prcc", key=lambda s: s.abs(), ascending=False
    )

# -------------- MC mit LHS & KPI-Logging --------------

def run_mc_with_lhs(fields_gdf, rotation, management, cfg, years,
                    n_reps: int = 200, seed: int = 123, vary: dict | None = None) -> pd.DataFrame:
    """Monte Carlo mit LHS + KPI-Outputs (λ, Produktivität, Occupancy, Kolonisation)."""
    rng = np.random.default_rng(seed)

    # Standardbereiche (angelehnt an run_monte_carlo)
    if vary is None:
        vary = {
            "fec_scale": (0.25, 0.50),
            "fledge_rate": (0.75, 0.90),
            "alpha_density": (0.001, 0.004),
            "survival_breeding": (0.990, 0.999),
            "survival_nonbreeding": (0.998, 1.000),
            "survival_juvenile": (0.980, 0.998),
            # globals:
            "CAPACITY_GAMMA": (0.20, 0.60),
            "WINDOW_STRESS_WEIGHT": (0.40, 0.70),
            "REFUGE_ASSIGN_MULT": (10.0, 25.0),
            "PULSE_PROTECT_SCALE": (3.0, 6.0),
            "REFUGE_SHARE_CAP": (0.50, 0.80),
        }

    samples = lhs_sample(n_reps, vary, seed=seed)

    # Globale Backups (wie in run_monte_carlo)
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
        cfg2 = replace(cfg)  # flache Kopie

        # sampled Parameter anwenden
        sampled = samples.iloc[rep].to_dict()
        for k, v in sampled.items():
            if hasattr(cfg2, k):
                setattr(cfg2, k, float(v))
            elif k in _globals0:
                globals()[k] = float(v)

        # Lauf
        resY = run_years_with_feedback(fields_gdf, rotation, management, cfg2,
                                       years=years, N_init_first=50)

        # Adults + λ
        arr_idx = max(1, int(cfg2.arrival_day)) - 1
        adults = [int((resY[y]["A_m"][arr_idx, :] + resY[y]["A_f"][arr_idx, :]).sum()) for y in years]
        lambdas = [adults[i + 1] / max(adults[i], 1e-9) for i in range(len(adults) - 1)] if len(years) > 1 else []
        lam_mean = float(np.mean(lambdas)) if lambdas else float("nan")

        # KPIs pro Jahr
        kpiY = [compute_kpis_one_year(resY[y], cfg2) for y in years]
        dfY = pd.DataFrame(kpiY)

        # Kolonisation (jährlich, ab Jahr 2)
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

        # Globals zurücksetzen
        for k, v in _globals0.items():
            globals()[k] = v

    return pd.DataFrame(rows)
if __name__ == "__main__":
    USE_SYNTHETIC = True  # flip to False to use your CSV inputs

    outdir = _mk_outdir("outputs")

    if USE_SYNTHETIC:
        fields_gdf = make_synthetic_fields(n_fields=int(OPTS.N_FIELDS), seed=int(OPTS.SEED_FIELDS))
        rotation = make_rotation(fields_gdf, YEARS_TO_SIM, seed=int(OPTS.SEED_ROTATION))
        management = make_management(
            fields_gdf, rotation, seed=int(OPTS.SEED_MANAGEMENT),
            use_calendar=True, calendar_path=CROP_CALENDAR_CSV,
            desync_region_sd=4.0, desync_field_sd=3.0, region_bins=6
        )
        hedge_pairs = make_random_hedges(fields_gdf, p_edge=0.7, seed=int(OPTS.SEED_HEDGES))
        hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)
    else:
        # CSV-Zweig wie gehabt einlesen ...
        # (YEARS_TO_SIM aus OPTS verwenden oder – falls CSV mehr/andere Jahre enthält –
        #  optional auf die ersten OPTS.N_YEARS beschränken:)
        unique_years = sorted(pd.unique(rotation["year"]))
        if len(unique_years) > len(YEARS_TO_SIM):
            YEARS_TO_SIM = unique_years[:int(OPTS.N_YEARS)]

        # Build geometry from CSV fields
        fields_gdf = rect_or_centroid_layout(fields)

        # Quick checkpoint save of the current layout (handy for debugging)
        try:
            ckpt_dir = pathlib.Path("snapshots")
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Overwrite a "latest" file + keep a timestamped copy
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fields_gdf.to_file(ckpt_dir / "fields_latest.gpkg", layer="fields", driver="GPKG")
            fields_gdf.to_parquet(ckpt_dir / f"fields_{ts}.parquet")
            print(f"[snapshot] fields saved to {ckpt_dir}/fields_latest.gpkg and fields_{ts}.parquet")
        except Exception as e:
            print("[warn] snapshot save failed:", e)

        # Hedges (from CSV or auto-adjacency)
        hedge_pairs = (
            pd.read_csv(HEDGES_CSV)
            if os.path.exists(HEDGES_CSV)
            else pd.DataFrame(columns=["f1", "f2", "hedge_quality"])
        )
        hedges_gdf = None
        if USE_HEDGES_FROM_CSV and {"f1", "f2"}.issubset(hedge_pairs.columns) and len(hedge_pairs) > 0:
            hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)
        elif AUTO_HEDGES_FROM_ADJ:
            # super-simple adjacency candidate list (optional)
            pairs = []
            fids = fields_gdf["field_id"].astype(int).values
            for i, a in enumerate(fids):
                for b in fids[i + 1:]:
                    pairs.append({"f1": int(a), "f2": int(b), "hedge_quality": 0.6})
            hedges_gdf = build_hedge_geoms_on_shared_edges(fields_gdf, pd.DataFrame(pairs)) if pairs else None

    # Sanity line so you instantly notice if the landscape got overwritten
    print(f"[Landscape] fields={len(fields_gdf)}, years={len(YEARS_TO_SIM)}, "
          f"rotation_rows={len(rotation)}, synthetic={USE_SYNTHETIC}")
    print(f"[Options] N_FIELDS={OPTS.N_FIELDS} (synthetic only) | N_YEARS={OPTS.N_YEARS} | "
          f"YEAR_TO_SHOW={YEAR_TO_SHOW} | DAY_TO_SHOW={DAY_TO_SHOW}")


    # ---------- Run simulation(s) per species ----------
    for sp_name in SPECIES_LIST:
        cfg = load_species_from_csv(sp_name, base_path=".")
        sanity_check_species(cfg)
        print("SPECIES:", cfg.name,
              "| move_radius(100m)=", cfg.move_radius,
              "| breeding_window=", cfg.breeding_window)
        print("surv_breeding=", cfg.survival_breeding,
              "surv_nonbreeding=", cfg.survival_nonbreeding,
              "surv_juvenile=", cfg.survival_juvenile,
              "fledge_rate=", cfg.fledge_rate,
              "base_nest_survival_day=", cfg.base_nest_survival_day)

        sim = BirdLandscapeSim(
            fields_gdf[[
                "field_id", "x", "y", "area", "perimeter",
                "flower_strip_share", "skylark_plots", "skylark_plot_area",
                "width_m", "height_m"
            ]],
            rotation, management, cfg
        )

        sp_outdir = outdir / cfg.name
        sp_outdir.mkdir(parents=True, exist_ok=True)

        # Base run (with cross-year feedback)
        if USE_YEAR_FEEDBACK:
            res_years = run_years_with_feedback(fields_gdf, rotation, management, cfg,
                                                years=YEARS_TO_SIM, N_init_first=100)
        else:
            res_years = {}
            for y in YEARS_TO_SIM:
                dbg_days = [110, 130, 160, 200] if y == YEAR_TO_SHOW else None
                res_years[y] = sim.simulate(days=365, year=int(y), N_init=100, debug_days=dbg_days)
                # optional: export tidy time series
                ts = to_long_timeseries(res_years[y], cfg.name, y, scenario="baseline")
                ydir = sp_outdir / f"year_{y}"; ydir.mkdir(parents=True, exist_ok=True)
                ts.to_csv(ydir / "timeseries_long.csv", index=False)

        # Bundle save (arrays + inputs + per-year CSVs)
        save_run_bundle(
            outdir=sp_outdir,
            species_name=cfg.name,
            fields_gdf=fields_gdf,
            rotation=rotation,
            management=management,
            hedges_gdf=hedges_gdf,
            results_by_year=res_years,
            day_to_show=DAY_TO_SHOW
        )
        # Ensure timeseries_long.csv exists for all years (both feedback and no-feedback modes)
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
        N_arrivals = [(res_years[y]["A_m"][arr, :] + res_years[y]["A_f"][arr, :]).sum() for y in YEARS_TO_SIM]
        lam = [N_arrivals[i + 1] / max(N_arrivals[i], 1e-9) for i in range(len(N_arrivals) - 1)]
        print("Adults at arrival by year:", [int(x) for x in N_arrivals])
        print("Lambda (year-to-year):", np.round(lam, 3), " mean:", round(np.mean(lam), 3))

        # Heatmaps
        if DO_HEATMAPS:
            fids = fields_gdf["field_id"].values
            vmax = max(res["N_total"].max() for res in res_years.values())
            for y in YEARS_TO_SIM:
                heatmap(
                    res_years[y]["N_total"], fids,
                    f"{cfg.name} – Heatmap Gesamt – Jahr {y}",
                    vmax=vmax,
                    savepath=sp_outdir / f"plots/heatmap_year{y}.png"
                )

        # Maps
        if DO_MAPS:
            y = YEAR_TO_SHOW
            plot_map(
                fields_gdf, rotation, year=y, hedges_gdf=hedges_gdf,
                title=f"{cfg.name} – Karte Jahr {y}, Tag {DAY_TO_SHOW}: Kultur & Vögel",
                draw_flower_rings=DRAW_FLOWER_RINGS,
                draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                hedge_buffer_m=HEDGE_BUFFER_M,
                day_vals=res_years[y]["N_total"][DAY_TO_SHOW],
                savepath=sp_outdir / f"plots/map_day{DAY_TO_SHOW}_year{y}.png"
            )
            plot_map(
                fields_gdf, rotation, year=y, hedges_gdf=hedges_gdf,
                title=f"{cfg.name} – Karte Jahr {y}: Kultur & Usage (Jahressumme)",
                draw_flower_rings=DRAW_FLOWER_RINGS,
                draw_hedge_buffers=DRAW_HEDGE_BUFFERS,
                hedge_buffer_m=HEDGE_BUFFER_M,
                usage_vals=res_years[y]["usage"],
                savepath=sp_outdir / f"plots/map_usage_year{y}.png"
            )

        # ---- Scenarios across all YEARS_TO_SIM ----
        # Verwende denselben Startbestand wie Baseline, sonst sind Vergleiche verzerrt.
        baseline_init = int(N_arrivals[0])  # adults at arrival in Jahr 1 (Baseline)


        def run_scenario_across_years(fields_df, mg_df, *, N0=baseline_init):
            return run_years_with_feedback(
                fields_df, rotation, mg_df, cfg,
                years=YEARS_TO_SIM, N_init_first=int(N0),
                outdir=None  # Szenarien schreiben keine juvenile_transition-Dateien
            )


        # ----------------- Scenarios across all YEARS_TO_SIM -----------------
        scenario_runs = {}

        # Baseline (immer 4-Tuple!)
        scenario_runs["Baseline"] = (fields_gdf.copy(), management.copy(), rotation.copy(), res_years)

        # A – spätere Ernte (mehrere Shifts als einzelne Szenarien)
        if SCENARIO_A_ENABLED:
            for dh in SCENARIO_A_SHIFTS:
                fA, mA = apply_management_scenario(
                    fields_gdf.copy(),
                    management.copy(),
                    int(dh)  # harvest_shift_days positionsbasiert
                )
                scenario_runs[f"A+{int(dh)}d"] = (fA, mA, rotation.copy(), run_scenario_across_years(fA, mA))


        # B – zusätzliche Blühstreifen (über alle Shares × Fractions)
        if SCENARIO_B_ENABLED:
            for share in SCENARIO_B_SHARES:
                for frac in SCENARIO_B_FRACTIONS:
                    fB, mB = apply_management_scenario(
                        fields_df=fields_gdf.copy(),
                        management_df=management.copy(),
                        add_flower_strip=float(share),
                        flower_fraction=float(frac),
                        flower_seed=SCENARIO_B_SEED
                    )
                    code = f"B_s{int(round(100 * share))}_f{int(round(100 * frac))}"
                    scenario_runs[code] = (fB, mB, rotation.copy(), run_scenario_across_years(fB, mB))

        # C – Brache-Anteil (Rotation ändert sich!)
        if SCENARIO_C_ENABLED:
            for share in SCENARIO_C_FALLOW_AREA_SHARES:
                rC, mC, idsC, share_act = apply_fallow_scenario(
                    fields_df=fields_gdf,
                    rotation_df=rotation,
                    management_df=management,
                    share=float(share),
                    years=YEARS_TO_SIM if SCENARIO_C_APPLY_YEARS == "all" else SCENARIO_C_APPLY_YEARS,
                    mode=SCENARIO_C_SELECTION,
                    seed=SCENARIO_C_SEED
                )
                resC = run_years_with_feedback(
                    fields_gdf, rC, mC, cfg,
                    years=YEARS_TO_SIM, N_init_first=int(baseline_init),
                    outdir=None
                )

                code = f"C_{int(round(share * 100))}p"
                scenario_runs[code] = (fields_gdf.copy(), mC, rC, resC)

        # D – gezielte Lerchenfenster
        if SCENARIO_D_ENABLED:
            target_ids_D = choose_skylark_target_fields(
                fields_gdf, rotation,
                eligible_crops=ELIGIBLE_REFUGE_CROPS,
                min_years=SCENARIO_D.get("min_eligible_years", 1),
                top_fraction=SCENARIO_D.get("target_fraction", 0.20),
                only_without_flower=SCENARIO_D.get("only_without_flower", True)
            )
            fD, mD = apply_management_scenario(
                fields_df=fields_gdf.copy(),
                management_df=management.copy(),
                add_skylark_plots=SCENARIO_D.get("add_skylark_plots", 2),
                skylark_plot_area=SCENARIO_D.get("skylark_plot_area", 0.02),
                skylark_target_field_ids=target_ids_D
            )
            scenario_runs["D"] = (fD, mD, rotation.copy(), run_scenario_across_years(fD, mD))

        # Labels für Auswertung/CSV
        scenario_labels = {
            "Baseline": "Baseline",
            "D": "D gezielte Lerchenfenster"
        }
        # A-Varianten
        if SCENARIO_A_ENABLED:
            for dh in SCENARIO_A_SHIFTS:
                scenario_labels[f"A_{int(dh)}"] = f"A Ernte +{int(dh)} Tage"
        # B-Varianten
        if SCENARIO_B_ENABLED:
            for share in SCENARIO_B_SHARES:
                for frac in SCENARIO_B_FRACTIONS:
                    code = f"B_s{int(round(100 * share))}_f{int(round(100 * frac))}"
                    scenario_labels[code] = (
                        f"B +{int(round(100 * share))}% Blühstreifen, auf {int(round(100 * frac))}% der Felder"
                    )
        # C-Varianten
        if SCENARIO_C_ENABLED:
            for share in SCENARIO_C_FALLOW_AREA_SHARES:
                scenario_labels[f"C_{int(round(share * 100))}p"] = f"C Brache {int(round(share * 100))}% Fläche"

        # ---------- Per-field/overall + Deltas schreiben ----------
        per_field_rows = []
        for scen_code, payload in scenario_runs.items():
            # payload = (fields_df, management_df, rotation_df_for_this_scenario, results_by_year)
            fDF, mDF, rotDF, resY = payload
            for yy in YEARS_TO_SIM:
                per_field_rows.append(
                    summarize_one_year(
                        res=resY[yy],
                        year=int(yy),
                        fields_df=fDF,
                        rotation_df=rotDF,  # <- korrekte Rotation je Szenario
                        species_name=cfg.name,
                        scenario_name=scenario_labels.get(scen_code, scen_code),
                        breeding_window=cfg.breeding_window
                    )
                )

        per_field_all = pd.concat(per_field_rows, ignore_index=True)
        per_field_path = sp_outdir / "summary_per_field.csv"
        per_field_all.to_csv(per_field_path, index=False)

        overall = summarize_overall(per_field_all)
        overall_path = sp_outdir / "summary_overall.csv"
        overall.to_csv(overall_path, index=False)

        deltas = add_deltas_vs_baseline(per_field_all, baseline_name="Baseline")
        deltas_path = sp_outdir / "summary_delta_vs_baseline.csv"
        deltas.to_csv(deltas_path, index=False)
        print(f"[OK] Summaries written:\n  {per_field_path}\n  {overall_path}\n  {deltas_path}")

        # ---------- KPI-Summary je Szenario (Mittel über Jahre) ----------
        kpi_rows = []
        for scen_code, (fDF, mDF, rotDF, resY) in scenario_runs.items():
            # KPIs je Jahr berechnen (ohne Datei-Schreiben)
            yrs = sorted(resY.keys())
            arr_idx = max(1, int(cfg.arrival_day)) - 1
            adults_arr = [int((resY[y]["A_m"][arr_idx, :] + resY[y]["A_f"][arr_idx, :]).sum()) for y in yrs]
            lambdas = [adults_arr[i + 1] / max(adults_arr[i], 1e-9) for i in range(len(adults_arr) - 1)] if len(
                yrs) > 1 else []

            kpiY = [compute_kpis_one_year(resY[y], cfg) for y in yrs]
            kdf = pd.DataFrame(kpiY)

            # Zusatzmetrik: Ernte im Brutfenster (Anteil)
            h_overlap = harvest_overlap_share(mDF, cfg.breeding_window)

            kpi_rows.append({
                "scenario": scenario_labels.get(scen_code, scen_code),
                "lambda_mean": float(np.mean(lambdas)) if lambdas else float("nan"),
                "adults_start": adults_arr[0] if adults_arr else float("nan"),
                "adults_end": adults_arr[-1] if adults_arr else float("nan"),
                "fledglings_per_female_equiv_mean": float(kdf["fledglings_per_female_equiv"].mean()),
                "occupancy_mean": float(kdf["occupancy_mean"].mean()),
                "colonization_rate_mean": float(
                    kdf.get("colonization_rate", pd.Series([float("nan")] * len(kdf))).mean()),
                "mayfield_daily_mean": float(kdf["mayfield_daily"].mean()),
                "harvest_in_breeding_share": float(h_overlap),
                "mean_flower_strip_share": float(fDF["flower_strip_share"].fillna(0.0).mean()),
                "share_of_fields_with_flower": float((fDF["flower_strip_share"].fillna(0.0) > 0).mean()),

            })
        kpi_summary = pd.DataFrame(kpi_rows)
        kpi_summary_path = sp_outdir / "scenario_kpis_summary.csv"
        kpi_summary.to_csv(kpi_summary_path, index=False)
        print(f"[OK] Scenario KPI summary written: {kpi_summary_path}")

        # === Sprint A: KPIs auf Baseline ===
        kpi_df = compute_kpis_from_results(res_years, cfg)
        print("\n[KPI] Jahres-Kennzahlen (Baseline):\n", kpi_df)

        # === (Optional) MC + LHS + PRCC ===
        DO_SPRINT_A_MC = False  # auf True setzen, wenn du laufen lassen willst
        if DO_SPRINT_A_MC:
            mc_df = run_mc_with_lhs(fields_gdf, rotation, management, cfg, YEARS_TO_SIM,
                                    n_reps=200, seed=123, vary=None)
            mc_path = sp_outdir / "mc_lhs_kpis.csv"
            mc_df.to_csv(mc_path, index=False)
            print(f"[OK] MC+LHS geschrieben: {mc_path}")

            # PRCC auf lambda_mean
            param_pool = [
                "fec_scale", "fledge_rate", "alpha_density",
                "survival_breeding", "survival_nonbreeding", "survival_juvenile",
                "CAPACITY_GAMMA", "WINDOW_STRESS_WEIGHT", "REFUGE_ASSIGN_MULT",
                "PULSE_PROTECT_SCALE", "REFUGE_SHARE_CAP"
            ]
            param_cols = [c for c in mc_df.columns if c in param_pool]
            prcc_df = compute_prcc(mc_df[param_cols], mc_df["lambda_mean"].to_numpy())
            print("\n[PRCC] Treiber von lambda_mean:\n", prcc_df)

        # --- NEU: Szenarien auch als birdsim_daily_full.csv speichern ---
        for scen_code, (fDF, mDF, rotDF, resY) in scenario_runs.items():
            scen_dir = sp_outdir / "scenarios" / scen_code
            scen_dir.mkdir(parents=True, exist_ok=True)

            # Daily Full Table bauen
            daily_df = _build_daily_full_df(
                fields_gdf=fDF,
                rotation_df=rotDF,
                management_df=mDF,
                results_by_year=resY,
                species_name=cfg.name,
                run_id=str(outdir.name)
            )
            if daily_df is not None:
                daily_df["scenario"] = scen_code  # Szenario-Label anhängen
                daily_df.to_csv(scen_dir / "birdsim_daily_full.csv", index=False)
                print(f"[OK] Szenario-Daily gespeichert: {scen_dir / 'birdsim_daily_full.csv'}")

        # ===== Optional: Monte Carlo (toggle on if you want to run it here) =====
        DO_MONTE_CARLO = False
        if DO_MONTE_CARLO:
            mc_df, mc_path = run_monte_carlo(
                fields_gdf, rotation, management, cfg, YEARS_TO_SIM,
                n_reps=200, seed=123, vary=None, outdir=str(sp_outdir / "mc")
            )
            print(f"[OK] Monte Carlo written: {mc_path}")

    print(f"\nAll files written under: {outdir}\n")
