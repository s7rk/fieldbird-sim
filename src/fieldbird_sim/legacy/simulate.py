from __future__ import annotations


import math
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np

from .globals import *  # noqa: F401,F403
from .debug_utils import DebugTracer, ReproLogger
from .layout import parse_days, parse_window, in_window
from .habitat import measure_modifiers, food_index, cover_index, softmax
from .species import clutch_multiplier_for


def _coerce_date(value) -> date | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    if hasattr(value, "date"):
        try:
            d = value.date()
            if isinstance(d, date):
                return d
        except Exception:
            pass

    txt = str(value).strip()
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).date()
    except Exception:
        pass
    try:
        return date.fromisoformat(txt.split("T")[0])
    except Exception:
        return None




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
        for col, default in [
            ("flower_strip_share", 0.0),
            ("skylark_plots", 0),
            ("skylark_plot_area", 0.0),
            ("hedge_length_m", 0.0),
            ("hedge_quality_weighted_m", 0.0),
            ("hedge_quality_mean", 0.0),
            ("hedge_length_m_per_ha", 0.0),
            ("hedge_quality_weighted_m_per_ha", 0.0),
        ]:
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
        # K pro ha: aus globals, sonst Fallback (wie in trash.py / globals.py: 40.0)
        K_PER_HA_ADULTS_local = float(globals().get("K_PER_HA_ADULTS", 40.0))
        self.K_adults = (self.fields["area"].astype(float).to_numpy() * K_PER_HA_ADULTS_local).astype(float)

        self._hedge_effect_enabled = bool(getattr(self.cfg, "hedge_effect_enabled", False))
        self._hedge_food_bonus = float(getattr(self.cfg, "hedge_food_bonus_per_100m_per_ha", 0.0) or 0.0)
        self._hedge_cover_bonus = float(getattr(self.cfg, "hedge_cover_bonus_per_100m_per_ha", 0.0) or 0.0)
        self._hedge_attractiveness_bias = float(
            getattr(self.cfg, "hedge_attractiveness_bias_per_100m_per_ha", 0.0) or 0.0
        )
        self._hedge_territory_area_bonus = float(
            getattr(self.cfg, "hedge_territory_area_bonus_ha_per_100m_per_ha", 0.0) or 0.0
        )
        self._hedge_effect_clip = float(getattr(self.cfg, "hedge_effect_clip_100m_per_ha", 4.0) or 4.0)
        self._hedge_effect_clip = max(0.0, self._hedge_effect_clip)

        # ---------- optionale Revier-Slots je Feld ----------
        self.territory_slots = self._compute_territory_slots()
        ts = self.territory_slots
        print(f"[territory_slots] {self.cfg.name}: min={ts.min()}, max={ts.max()}, >0={(ts > 0).sum()}/{len(ts)}")

        clutch_base = float(getattr(self.cfg, "clutch_size", 0.0) or 0.0)
        clutch_min = float(getattr(self.cfg, "clutch_size_min", clutch_base) or clutch_base)
        clutch_mode = float(getattr(self.cfg, "clutch_size_mode", clutch_base) or clutch_base)
        clutch_max = float(getattr(self.cfg, "clutch_size_max", clutch_base) or clutch_base)
        self._clutch_min = min(clutch_min, clutch_max)
        self._clutch_max = max(clutch_min, clutch_max)
        self._clutch_mode = min(max(clutch_mode, self._clutch_min), self._clutch_max)
        self._clutch_variable = abs(self._clutch_max - self._clutch_min) > 1e-9

        self._replacement_only_early_loss = bool(
            getattr(self.cfg, "replacement_brood_only_after_early_loss", False)
        )
        cutoff_default = int(self.cfg.breeding_window[0]) + 45
        cutoff_raw = int(getattr(self.cfg, "replacement_brood_cutoff_doy", cutoff_default))
        self._replacement_cutoff_doy = int(np.clip(cutoff_raw, 1, 365))

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
        if self._hedge_effect_enabled and self._hedge_territory_area_bonus != 0.0:
            if "hedge_length_m_per_ha" in self.fields.columns:
                hedge_density = pd.to_numeric(
                    self.fields["hedge_length_m_per_ha"],
                    errors="coerce",
                ).fillna(0.0).to_numpy(float)
            elif "hedge_quality_weighted_m_per_ha" in self.fields.columns:
                hedge_density = pd.to_numeric(
                    self.fields["hedge_quality_weighted_m_per_ha"],
                    errors="coerce",
                ).fillna(0.0).to_numpy(float)
            else:
                hedge_density = np.zeros_like(area_ha, dtype=float)
            hedge_scaled = np.clip(np.maximum(hedge_density, 0.0) / 100.0, 0.0, self._hedge_effect_clip)
            area_ha = np.maximum(area_ha + self._hedge_territory_area_bonus * hedge_scaled, 0.0)
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
        # FAST FIX: wenn terr_area alles auf 0 drückt (weil Felder kleiner als terr_area),
        # dann erlaube pro Feld maximal 1 Territorium (statt "unendlich").
        if terr_area > 0 and np.nanmax(slots_area) <= 0:
            slots_area = (area_ha > 0).astype(float)  # 1 für jedes Feld mit Fläche, sonst 0
        if max_pph > 0:
            slots_pph = np.floor(max_pph * area_ha)

        slots = np.minimum(slots_area, slots_pph)
        slots = np.clip(slots, 0.0, BIG).astype(int)
        return slots

    def _sample_nest_yield(self, nest_counts: np.ndarray) -> np.ndarray:
        """
        Samples per-nest clutch sizes (optional triangular distribution) and returns
        the total fledgling potential per field for newly started nests.
        """
        out = np.zeros(len(nest_counts), dtype=np.float32)
        fledge_rate = float(self.cfg.fledge_rate)

        if not self._clutch_variable:
            out[:] = nest_counts.astype(np.float32) * float(self._clutch_mode) * fledge_rate
            return out

        for i, c in enumerate(nest_counts.astype(int)):
            if c <= 0:
                continue
            clutch = self.rng.triangular(
                left=self._clutch_min,
                mode=self._clutch_mode,
                right=self._clutch_max,
                size=int(c),
            )
            out[i] = float(np.sum(clutch) * fledge_rate)
        return out

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

            hedge_effect_scaled = 0.0
            if self._hedge_effect_enabled:
                hedge_metric = float(
                    getattr(
                        frow,
                        "hedge_quality_weighted_m_per_ha",
                        getattr(frow, "hedge_length_m_per_ha", 0.0),
                    )
                    or 0.0
                )
                hedge_effect_scaled = float(
                    np.clip(max(hedge_metric, 0.0) / 100.0, 0.0, self._hedge_effect_clip)
                )
                d_food += self._hedge_food_bonus * hedge_effect_scaled
                d_cover += self._hedge_cover_bonus * hedge_effect_scaled

            # Basis-Utility
            base_u = (
                pref["w_food"] * food_index(crop, doy)
                + pref["w_cover"] * cover_index(crop, doy)
                + pref["w_disturb_base"]
                + (p_stress + w_stress) * d_mult
            )
            base_u += pref["w_food"] * d_food + pref["w_cover"] * d_cover
            base_u += self._hedge_attractiveness_bias * hedge_effect_scaled
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
        crops = np.array([crop_map.get(int(fid), "Unknown") for fid in fids], dtype=object)

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
                 start_state: Optional[dict] = None,
                 weather_module=None,
                 timeline_start_date=None,
                 timeline_day_offset: int = 0):

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
        nest_yield_open = np.zeros((nest_age_len, n), dtype=np.float32)
        nest_yield_refuge = np.zeros((nest_age_len, n), dtype=np.float32)

        # Juvenile Alterskohorten
        age_len = max(1, int(self.cfg.juvenile_free_age))
        J_age = np.zeros((age_len, n), dtype=np.int32)
        J_age_series = np.zeros((days, age_len, n), dtype=np.int32) if keep_age_series else None

        # Re-Nest-Cooldown (optional)
        cooldown_len = int(max(0, int(getattr(self.cfg, "renest_cooldown_days", 0) or 0)))
        cooldown_f = np.zeros((max(1, cooldown_len), n), dtype=np.int32) if cooldown_len > 0 else None

        broods_started_per_f = np.zeros(n, dtype=np.float32)
        renest_credits = np.zeros(n, dtype=np.int32)

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

        weather_start_date = None
        day_offset = int(timeline_day_offset)
        weather_diag_by_t = {}
        if weather_module is not None:
            start_candidate = timeline_start_date
            if start_candidate is None:
                start_candidate = getattr(weather_module, "default_start_date", None)
            weather_start_date = _coerce_date(start_candidate)
            if weather_start_date is None:
                raise ValueError(
                    "weather module enabled, but no valid timeline_start_date could be resolved."
                )

            # advance over the full year so weather memory remains continuous even outside bird activity
            for day_idx in range(days):
                sim_date = weather_start_date + timedelta(days=day_offset + day_idx)
                weather_diag_by_t[day_idx] = weather_module.step(sim_date)

        for t in range(t0, tEnd + 1):
            doy = t + 1
            in_breed = (self.cfg.breeding_window[0] <= doy <= self.cfg.breeding_window[1])

            prev_idx = t
            next_idx = t + 1

            reproduction_factor = 1.0
            juvenile_survival_factor = 1.0
            adult_survival_factor = 1.0
            if weather_module is not None:
                weather_diag = weather_diag_by_t.get(t, {})
                reproduction_factor = float(weather_diag.get("reproduction_factor", 1.0))
                juvenile_survival_factor = float(weather_diag.get("juvenile_survival_factor", 1.0))
                adult_survival_factor = float(weather_diag.get("adult_survival_factor", 1.0))

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
            # juvenile contribution to capacity: cfg overrides, else globals constant, else fallback
            JUV_EQUIV_IN_CAPACITY_local = float(globals().get("JUV_EQUIV_IN_CAPACITY", 0.5))
            juv_equiv = float(getattr(self.cfg, "juvenile_equiv_in_capacity", JUV_EQUIV_IN_CAPACITY_local))

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
                max_broods_total = int(self.cfg.max_broods_per_female) * today_f
                started_total = np.floor(np.maximum(broods_started_per_f, 0.0) * today_f).astype(int)
                started_total = np.clip(started_total, 0, max_broods_total)
                remaining_capacity = np.maximum(max_broods_total - started_total, 0)
                first_brood_capacity = np.maximum(today_f - started_total, 0)
                replacement_capacity = np.maximum(remaining_capacity - first_brood_capacity, 0)

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
                        m = clutch_multiplier_for(
                            crops_vec[i], doy=int(doy), clutch_number=int(clutch_idx[i]),
                            prefs_lookup=self.cfg.clutch_prefs
                        )
                        # --- QUICK FIX ---
                        if m is None or (isinstance(m, float) and np.isnan(m)) or m <= 0.0:
                            m = 1.0
                        mult_vec[i] = float(m)

                p_start_vec = np.clip(p_start_vec * mult_vec, 0.0, 0.9)
                p_start_vec = np.clip(p_start_vec * reproduction_factor, 0.0, 0.9)

                if self._replacement_only_early_loss:
                    credit_capacity = np.minimum(np.maximum(renest_credits, 0), replacement_capacity)
                    start_capacity = first_brood_capacity + credit_capacity
                else:
                    start_capacity = remaining_capacity

                can_start = np.minimum(potential_pairs, start_capacity)
                can_start = np.minimum(can_start, free_slots)
                can_start = np.maximum(can_start, 0)

                new_nests = rng.binomial(n=np.maximum(can_start.astype(int), 0), p=p_start_vec)
                if self._replacement_only_early_loss:
                    first_starts = np.minimum(new_nests, first_brood_capacity)
                    replacement_starts = np.minimum(
                        new_nests - first_starts,
                        np.maximum(renest_credits, 0),
                    )
                    renest_credits = np.maximum(renest_credits - replacement_starts, 0)
                else:
                    first_starts = new_nests
                    replacement_starts = np.zeros_like(new_nests)

                daily_new_nests[next_idx, :] = new_nests.astype(int)

                # --- Starts des heutigen Tages je Gelege aufsplitten ---
                if self._replacement_only_early_loss:
                    d_c1 = first_starts
                    d_c2 = replacement_starts
                    d_c3 = np.zeros_like(new_nests)
                else:
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
                new_yield_total = self._sample_nest_yield(new_nests)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ref_share = np.where(new_nests > 0, new_to_ref / np.maximum(new_nests, 1), 0.0)
                new_yield_ref = new_yield_total * ref_share
                new_yield_open = new_yield_total - new_yield_ref

                nests_open = np.roll(nests_open, shift=1, axis=0);    nests_open[0, :] = new_to_open.astype(float)
                nests_refuge = np.roll(nests_refuge, shift=1, axis=0); nests_refuge[0, :] = new_to_ref.astype(float)
                nest_yield_open = np.roll(nest_yield_open, shift=1, axis=0); nest_yield_open[0, :] = new_yield_open.astype(np.float32)
                nest_yield_refuge = np.roll(nest_yield_refuge, shift=1, axis=0); nest_yield_refuge[0, :] = new_yield_ref.astype(np.float32)

                with np.errstate(divide='ignore', invalid='ignore'):
                    broods_started_per_f += np.where(today_f > 0, new_nests / today_f, 0.0)
            else:
                p_start_vec = np.zeros_like(U)
                nests_open = np.roll(nests_open, shift=1, axis=0);     nests_open[0, :] = 0.0
                nests_refuge = np.roll(nests_refuge, shift=1, axis=0);  nests_refuge[0, :] = 0.0
                nest_yield_open = np.roll(nest_yield_open, shift=1, axis=0); nest_yield_open[0, :] = 0.0
                nest_yield_refuge = np.roll(nest_yield_refuge, shift=1, axis=0); nest_yield_refuge[0, :] = 0.0

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
                    open_n = np.maximum(nests_open[a, :].astype(int), 0)
                    fails_o = rng.binomial(n=open_n, p=(1.0 - nest_surv_open))
                    survived_o = np.maximum(open_n - fails_o, 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        keep_frac_o = np.where(open_n > 0, survived_o / open_n, 0.0)
                    failed_open_today += fails_o.astype(int)
                    nests_open[a, :] = survived_o.astype(np.float32)
                    nest_yield_open[a, :] = np.maximum(nest_yield_open[a, :] * keep_frac_o, 0.0)
                if nests_refuge[a, :].sum() > 0:
                    refuge_n = np.maximum(nests_refuge[a, :].astype(int), 0)
                    fails_r = rng.binomial(n=refuge_n, p=(1.0 - nest_surv_ref))
                    survived_r = np.maximum(refuge_n - fails_r, 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        keep_frac_r = np.where(refuge_n > 0, survived_r / refuge_n, 0.0)
                    failed_ref_today += fails_r.astype(int)
                    nests_refuge[a, :] = survived_r.astype(np.float32)
                    nest_yield_refuge[a, :] = np.maximum(nest_yield_refuge[a, :] * keep_frac_r, 0.0)

            failed_today = failed_open_today + failed_ref_today
            if self._replacement_only_early_loss and doy <= self._replacement_cutoff_doy:
                renest_credits = np.minimum(
                    renest_credits + failed_today.astype(np.int32),
                    np.iinfo(np.int32).max,
                )
            if cooldown_len > 0 and cooldown_f is not None:
                to_cool = np.minimum(A_f_breeding[next_idx, :].astype(int), failed_today.astype(int))
                cooldown_f[0, :] += to_cool

            # Fledglings am Ende des Zyklus
            fledging_n = nests_open[-1, :].copy() + nests_refuge[-1, :].copy()
            fledging_yield = nest_yield_open[-1, :].copy() + nest_yield_refuge[-1, :].copy()
            nests_open[-1, :] = 0.0
            nests_refuge[-1, :] = 0.0
            nest_yield_open[-1, :] = 0.0
            nest_yield_refuge[-1, :] = 0.0
            fledglings_raw = fledging_yield
            fledglings_raw = fledglings_raw * fl_mult_vec
            # WICHTIG: optional Dichtefaktor auch auf Jungvögel (wie früher) – nur wenn aktiviert
            if USE_DENSITY_FACTOR and self.cfg.density_affects_start:
                fledglings_raw = fledglings_raw * density_factor
            fledglings = np.maximum(fledglings_raw, 0.0).astype(int)
            daily_fledglings[next_idx, :] = fledglings.astype(int)

            # Adult Survival
            surv_b = float(np.clip(self.cfg.survival_breeding, 0.0, 1.0))
            surv_nb = float(np.clip(self.cfg.survival_nonbreeding, 0.0, 1.0))
            # ---------------- Adult Survival (tägliche Überlebensrate) ----------------
            p_adult = surv_b if in_breed else surv_nb
            p_adult = float(np.clip(p_adult * adult_survival_factor, 0.0, 1.0))
            A_m[next_idx, :] = rng.binomial(np.maximum(today_m, 0), p_adult)
            A_f[next_idx, :] = rng.binomial(np.maximum(today_f, 0), p_adult)

            # ---------------- Juvenile: altern, überleben, frei werden ----------------
            # (1) Nestlinge, die heute flügge wurden, kommen in Alterskohorte 0
            new_fledge = fledglings.astype(int)
            p_juvenile = float(np.clip(self.cfg.survival_juvenile * juvenile_survival_factor, 0.0, 1.0))

            # (2) Alle vorhandenen Alterskohorten überleben heute als Juvenile
            if J_age.sum() > 0:
                for a in range(age_len):
                    vals = np.maximum(J_age[a, :].astype(int), 0)
                    if vals.sum() > 0:
                        J_age[a, :] = rng.binomial(vals, p_juvenile)

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
            prev_free_surv = rng.binomial(np.maximum(prev_free, 0), p_juvenile)
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
