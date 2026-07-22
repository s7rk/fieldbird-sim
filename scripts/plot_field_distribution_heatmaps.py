#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPECIES_LABELS = {
    "Rebhuhn": "Grey Partridge",
    "Feldlerche": "Skylark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot field-level bird distribution heatmaps."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Path to a birdsim_daily_full.csv file. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--fig-width-cm", type=float, default=26.2)
    parser.add_argument("--fig-height-cm", type=float, default=13.03)
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def load_field_year_means(path: Path) -> tuple[str, pd.DataFrame]:
    usecols = ["year", "day", "field_id", "N_total", "species"]
    df = pd.read_csv(path, usecols=usecols)
    species = str(df["species"].dropna().iloc[0])
    field_year = (
        df.groupby(["field_id", "year"], as_index=False)["N_total"]
        .mean()
        .rename(columns={"N_total": "mean_daily_abundance"})
    )
    return species, field_year


def make_matrix(field_year: pd.DataFrame) -> pd.DataFrame:
    matrix = field_year.pivot(
        index="field_id",
        columns="year",
        values="mean_daily_abundance",
    )
    matrix = matrix.sort_index().sort_index(axis=1)
    return matrix


def plot_heatmap(
    matrix: pd.DataFrame,
    species_label: str,
    output_path: Path,
    dpi: int,
    fig_size: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)
    values = matrix.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(values, 98)) if np.isfinite(values).any() else 1.0
    if vmax <= 0:
        vmax = 1.0

    im = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap="YlGnBu",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(f"Field-Level Bird Distribution: {species_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Field ID")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([str(int(c)) for c in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(int(i)) for i in matrix.index])
    ax.tick_params(axis="both", labelsize=9)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.018)
    cbar.set_label("Mean daily abundance per field", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.16, top=0.86)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_size = (args.fig_width_cm / 2.54, args.fig_height_cm / 2.54)

    source_rows = []
    for input_path in args.input:
        species, field_year = load_field_year_means(input_path)
        label = SPECIES_LABELS.get(species, species)
        matrix = make_matrix(field_year)
        stem = f"field_distribution_heatmap_{safe_name(label)}"

        field_year.to_csv(args.output_dir / f"{stem}.csv", index=False)
        plot_heatmap(
            matrix=matrix,
            species_label=label,
            output_path=args.output_dir / f"{stem}.png",
            dpi=args.dpi,
            fig_size=fig_size,
        )
        source_rows.append(
            {
                "species": label,
                "source": str(input_path),
                "n_fields": int(matrix.shape[0]),
                "n_years": int(matrix.shape[1]),
                "output_png": str(args.output_dir / f"{stem}.png"),
                "output_csv": str(args.output_dir / f"{stem}.csv"),
            }
        )

    pd.DataFrame(source_rows).to_csv(args.output_dir / "field_distribution_heatmap_sources.csv", index=False)
    print(f"[ok] Wrote heatmaps to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
