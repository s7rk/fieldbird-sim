from __future__ import annotations

from .globals import *  # noqa: F401,F403

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
