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
    "Rebhuhn": "Grey Partridge (Rebhuhn)",
    "Feldlerche": "Skylark (Feldlerche)",
}

SPECIES_COLORS = {
    "Rebhuhn": "#b20d3a",
    "Feldlerche": "#0b3b63",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot total bird abundance by species across simulation years."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Scenario run directory containing one subfolder per species.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--rolling-days", type=int, default=14)
    parser.add_argument("--fig-width-cm", type=float, default=26.2)
    parser.add_argument("--fig-height-cm", type=float, default=13.03)
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def load_daily_totals(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*/birdsim_daily_full.csv")):
        usecols = ["year", "day", "field_id", "N_total", "species"]
        df = pd.read_csv(path, usecols=usecols)
        grouped = (
            df.groupby(["species", "year", "day"], as_index=False)["N_total"]
            .sum()
            .rename(columns={"N_total": "total_birds"})
        )
        frames.append(grouped)

    if not frames:
        raise FileNotFoundError(f"No birdsim_daily_full.csv files found below {run_dir}")

    out = pd.concat(frames, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["day"] = pd.to_numeric(out["day"], errors="coerce").astype(int)
    out["simulation_time_years"] = out["year"] + (out["day"] - 1) / 365.0
    out = out.sort_values(["species", "year", "day"]).reset_index(drop=True)
    return out


def add_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out["total_birds_rolling"] = (
        out.groupby("species")["total_birds"]
        .transform(lambda s: s.rolling(window=window, min_periods=1, center=True).mean())
    )
    return out


def style_axis(ax: plt.Axes, years: list[int]) -> None:
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Total bird abundance (individuals)")
    ax.set_xticks(years)
    ax.set_xlim(min(years), max(years) + 1)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.grid(axis="x", alpha=0.12, linewidth=0.6)
    for year in years:
        ax.axvline(year, color="#9aa3aa", lw=0.45, alpha=0.35, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_species_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    species: str,
    years: list[int],
    rolling_days: int,
) -> None:
    sub = data[data["species"] == species].copy()
    label = SPECIES_LABELS.get(species, species)
    color = SPECIES_COLORS.get(species, "#263238")

    ax.plot(
        sub["simulation_time_years"],
        sub["total_birds"],
        color=color,
        alpha=0.22,
        lw=0.8,
        label="Daily total",
    )
    ax.plot(
        sub["simulation_time_years"],
        sub["total_birds_rolling"],
        color=color,
        alpha=0.98,
        lw=2.1,
        label=f"{rolling_days}-day rolling mean",
    )

    ax.set_title(label, fontsize=12, fontweight="bold")
    style_axis(ax, years)
    ax.legend(frameon=False, loc="upper left", fontsize=8)


def plot_combined(data: pd.DataFrame, output_path: Path, dpi: int, rolling_days: int, fig_size: tuple[float, float]) -> None:
    species_list = list(data["species"].drop_duplicates())
    years = sorted(data["year"].unique().tolist())

    fig, axes = plt.subplots(1, len(species_list), figsize=fig_size, sharex=True, sharey=False)
    axes = np.atleast_1d(axes)
    for ax, species in zip(axes, species_list):
        plot_species_panel(ax, data, species, years, rolling_days)

    fig.suptitle("Total Bird Abundance Across Simulation Years", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_single_species(data: pd.DataFrame, output_dir: Path, dpi: int, rolling_days: int, fig_size: tuple[float, float]) -> None:
    years = sorted(data["year"].unique().tolist())
    for species in data["species"].drop_duplicates():
        label = SPECIES_LABELS.get(species, species)
        fig, ax = plt.subplots(figsize=fig_size)
        plot_species_panel(ax, data, species, years, rolling_days)
        ax.set_title(f"Total Bird Abundance: {label}", fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.86)
        output_path = output_dir / f"total_bird_abundance_{safe_name(label)}.png"
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals = add_rolling(load_daily_totals(args.run_dir), args.rolling_days)
    totals.to_csv(args.output_dir / "total_bird_abundance_daily_by_species.csv", index=False)

    fig_size = (args.fig_width_cm / 2.54, args.fig_height_cm / 2.54)
    plot_combined(
        totals,
        args.output_dir / "total_bird_abundance_by_species_panels.png",
        args.dpi,
        args.rolling_days,
        fig_size,
    )
    plot_single_species(totals, args.output_dir, args.dpi, args.rolling_days, fig_size)

    print(f"[ok] Wrote plots and CSV to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
