from __future__ import annotations

from .globals import *  # noqa: F401,F403

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
