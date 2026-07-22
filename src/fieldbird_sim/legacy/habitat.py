from __future__ import annotations

import csv
from typing import Tuple

from .globals import *  # noqa: F401,F403
from .layout import in_window

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
