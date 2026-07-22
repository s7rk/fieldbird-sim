from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np
import pandas as pd

from .globals import *  # noqa: F401,F403

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
    columns = ["field_id"]
    if "flower_strip_share" in fields_df.columns:
        columns.append("flower_strip_share")
    F = fields_df[columns].copy()
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

    columns = ["field_id"]
    if "flower_strip_share" in fields_df.columns:
        columns.append("flower_strip_share")
    F = fields_df[columns].copy()
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
                a = float(row["area"])
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
