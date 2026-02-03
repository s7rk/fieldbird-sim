from __future__ import annotations

from .globals import *  # noqa: F401,F403

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
