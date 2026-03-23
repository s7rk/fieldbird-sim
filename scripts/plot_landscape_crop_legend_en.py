#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import Polygon

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CROP_TRANSLATIONS = {
    "Winterweizen": "Winter wheat",
    "Sommergerste": "Spring barley",
    "Mais": "Maize",
    "Raps": "Oilseed rape",
    "Zuckerrübe": "Sugar beet",
    "Zuckerruebe": "Sugar beet",
    "Ackerbohne": "Faba bean",
    "Brache": "Fallow",
    "PermanentCover": "Permanent cover",
    "Unbekannt": "Unknown",
}

BASE_PALETTE = {
    # High-contrast, colorblind-friendly base palette (Okabe-Ito inspired).
    "Winterweizen": "#56B4E9",
    "Sommergerste": "#009E73",
    "Mais": "#F0E442",
    "Raps": "#E69F00",
    "Zuckerrübe": "#CC79A7",
    "Zuckerruebe": "#CC79A7",
    "Ackerbohne": "#0072B2",
    "Brache": "#D55E00",
    "PermanentCover": "#999999",
    "Unbekannt": "#7F7F7F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a landscape map with a full English crop legend."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/20260225_145603"),
        help="Run directory that contains data/fields.geojson and data/rotation.csv",
    )
    parser.add_argument("--year", type=int, default=1, help="Simulation year to plot")
    parser.add_argument(
        "--species",
        type=str,
        default="Rebhuhn",
        help="Species label for the title only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. If omitted, a default file name is used.",
    )
    parser.add_argument(
        "--show-field-ids",
        action="store_true",
        help="Draw field IDs at centroids",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def _build_rectangles_from_flat(fields_flat: pd.DataFrame) -> gpd.GeoDataFrame:
    df = fields_flat.copy()
    df["field_id"] = pd.to_numeric(df["field_id"], errors="coerce").astype("Int64")
    df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
    df["y"] = pd.to_numeric(df.get("y"), errors="coerce")
    df["width_m"] = pd.to_numeric(df.get("width_m"), errors="coerce")
    df["height_m"] = pd.to_numeric(df.get("height_m"), errors="coerce")
    df["area"] = pd.to_numeric(df.get("area"), errors="coerce")

    geoms = []
    for row in df.itertuples(index=False):
        x = float(row.x) if pd.notna(row.x) else 0.0
        y = float(row.y) if pd.notna(row.y) else 0.0
        w = float(row.width_m) if pd.notna(row.width_m) else math.nan
        h = float(row.height_m) if pd.notna(row.height_m) else math.nan
        if not np.isfinite(w) or w <= 0 or not np.isfinite(h) or h <= 0:
            area_ha = float(row.area) if pd.notna(row.area) else 0.01
            side = math.sqrt(max(area_ha, 1e-6) * 10_000.0)
            w = side
            h = side
        x0, x1 = x - (w / 2.0), x + (w / 2.0)
        y0, y1 = y - (h / 2.0), y + (h / 2.0)
        geoms.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:3857")
    return gdf


def load_fields(data_dir: Path) -> gpd.GeoDataFrame:
    geojson = data_dir / "fields.geojson"
    fields_flat = data_dir / "fields_flat.csv"
    if geojson.exists():
        gdf = gpd.read_file(geojson)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:3857")
        return gdf
    if fields_flat.exists():
        return _build_rectangles_from_flat(pd.read_csv(fields_flat))
    raise FileNotFoundError(f"No fields file found in {data_dir}")


def make_plot(
    fields_gdf: gpd.GeoDataFrame,
    rotation_df: pd.DataFrame,
    year: int,
    species: str,
    all_crops_run: list[str],
    output_path: Path,
    show_field_ids: bool,
    dpi: int,
) -> None:
    rot_year = rotation_df.loc[rotation_df["year"] == year, ["field_id", "crop"]].copy()
    if rot_year.empty:
        available = sorted(rotation_df["year"].unique())
        raise ValueError(f"Year {year} not found. Available years: {available}")

    fg = fields_gdf.copy()
    fg["field_id"] = pd.to_numeric(fg["field_id"], errors="coerce").astype(int)
    crop_map = dict(zip(rot_year["field_id"].astype(int), rot_year["crop"].astype(str)))
    fg["crop_de"] = fg["field_id"].map(crop_map).fillna("Unbekannt")
    fg["crop_en"] = fg["crop_de"].map(CROP_TRANSLATIONS).fillna(fg["crop_de"])

    unique_crops = pd.unique(pd.concat([fg["crop_de"], pd.Series(all_crops_run, dtype="object")]))
    tab20 = plt.cm.tab20.colors
    palette: dict[str, str] = {}
    i = 0
    for crop in unique_crops:
        if crop in BASE_PALETTE:
            palette[crop] = BASE_PALETTE[crop]
        else:
            palette[crop] = tab20[i % len(tab20)]
            i += 1

    fig, ax = plt.subplots(figsize=(14, 8))
    fg.plot(
        ax=ax,
        facecolor=fg["crop_de"].map(palette),
        edgecolor="#1f1f1f",
        linewidth=0.9,
        alpha=0.96,
    )

    if show_field_ids:
        # representative_point() is guaranteed to lie within polygon geometry.
        # This avoids label confusion for ring/donut-shaped fields.
        rp = fg.geometry.representative_point()
        for fid, x, y in zip(fg["field_id"], rp.x, rp.y):
            ax.text(
                float(x),
                float(y),
                str(int(fid)),
                ha="center",
                va="center",
                fontsize=8.5,
                color="#111111",
                weight="bold",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "#ffffff",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
                zorder=6,
            )

    crop_counts = fg["crop_de"].value_counts()
    legend_crops = sorted(
        all_crops_run,
        key=lambda c: CROP_TRANSLATIONS.get(c, c).lower(),
    )
    handles = []
    for crop_de in legend_crops:
        count = int(crop_counts.get(crop_de, 0))
        crop_en = CROP_TRANSLATIONS.get(crop_de, crop_de)
        handles.append(
            Patch(
                facecolor=palette[crop_de],
                edgecolor="black",
                label=f"{crop_en} ({int(count)})",
            )
        )

    ax.legend(
        handles=handles,
        title="Crop Types",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        borderaxespad=0.0,
    )

    ax.set_title(
        f"Landscape by Crop Type (Year {year})\nSpecies: {species}",
        fontsize=13,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    data_dir = run_dir / "data"
    rotation_path = data_dir / "rotation.csv"
    if not rotation_path.exists():
        raise FileNotFoundError(f"Missing rotation file: {rotation_path}")

    output = args.output
    if output is None:
        output = (
            Path("outputs/final_ref/landscape_plots")
            / f"landscape_crop_legend_en_{run_dir.name}_year{args.year}.png"
        )

    fields_gdf = load_fields(data_dir)
    rotation_df = pd.read_csv(rotation_path)
    rotation_df["year"] = pd.to_numeric(rotation_df["year"], errors="coerce").astype(int)
    rotation_df["field_id"] = pd.to_numeric(rotation_df["field_id"], errors="coerce").astype(int)
    rotation_df["crop"] = rotation_df["crop"].astype(str)

    make_plot(
        fields_gdf=fields_gdf,
        rotation_df=rotation_df,
        year=args.year,
        species=args.species,
        all_crops_run=sorted(rotation_df["crop"].dropna().astype(str).unique()),
        output_path=output,
        show_field_ids=args.show_field_ids,
        dpi=args.dpi,
    )
    print(f"[ok] Saved plot: {output.resolve()}")


if __name__ == "__main__":
    main()
