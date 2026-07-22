from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["Baseline", "MLP", "CNN", "GNN"]
MODEL_COLORS = {
    "Baseline": "#8c8c8c",
    "MLP": "#1f77b4",
    "CNN": "#d62728",
    "GNN": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-field daily trajectory plots from run10 unified predictions."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("~/Desktop/präsi/csv/run10_unified_predictions_4models.csv").expanduser(),
        help="Unified CSV with columns: model, field_id, day, y_true_level, y_pred_level.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/presentation_plots/run10_new/fields"),
        help="Directory for per-field plots.",
    )
    return parser.parse_args()


def compute_metrics(g: pd.DataFrame) -> tuple[float, float, float]:
    y_true = g["y_true_level"].to_numpy(dtype=float)
    y_pred = g["y_pred_level"].to_numpy(dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return mae, rmse, r2


def create_overlay_plot(field_df: pd.DataFrame, field_id: int, output_path: Path) -> None:
    daily = (
        field_df.groupby(["model", "day"], as_index=False)[["y_true_level", "y_pred_level"]]
        .mean()
        .sort_values(["model", "day"])
    )
    gt = daily.groupby("day", as_index=False)["y_true_level"].mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gt["day"], gt["y_true_level"], color="black", lw=2.5, label="Ground Truth")

    for model in MODEL_ORDER:
        if model not in daily["model"].values:
            continue
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["y_pred_level"], lw=2.0, color=MODEL_COLORS[model], label=model)

    ax.set_title(f"Daily Mean Prediction Comparison - Field {field_id} (Run10)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Daily mean value")
    ax.grid(alpha=0.25)
    ax.legend(ncols=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def create_panel_plot(field_df: pd.DataFrame, field_id: int, output_path: Path) -> list[dict[str, float | int | str]]:
    daily = (
        field_df.groupby(["model", "day"], as_index=False)[["y_true_level", "y_pred_level"]]
        .mean()
        .sort_values(["model", "day"])
    )

    models = [m for m in MODEL_ORDER if m in daily["model"].unique()]
    ncols = 2
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows), sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    metrics_records: list[dict[str, float | int | str]] = []
    for idx, model in enumerate(models):
        ax = axes[idx // ncols, idx % ncols]
        g = daily[daily["model"] == model]
        mae, rmse, r2 = compute_metrics(g)
        metrics_records.append(
            {
                "field_id": field_id,
                "model": model,
                "rows": len(g),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
        ax.plot(g["day"], g["y_true_level"], color="#1a1a1a", lw=2.0, label="Ground Truth")
        ax.plot(g["day"], g["y_pred_level"], color=MODEL_COLORS[model], lw=2.0, label="Prediction")
        ax.set_title(model)
        ax.set_xlabel("Day of year")
        ax.grid(alpha=0.25)
        ax.text(
            0.03,
            0.97,
            f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR2={r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
        )

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    axes[0, 0].set_ylabel("Daily mean value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Daily Mean Trajectories - Field {field_id} (Run10)", fontsize=15, y=1.06)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    return metrics_records


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        args.input_csv,
        usecols=["model", "field_id", "day", "y_true_level", "y_pred_level"],
    )

    field_ids = sorted(df["field_id"].dropna().astype(int).unique().tolist())
    metrics_records: list[dict[str, float | int | str]] = []
    for field_id in field_ids:
        field_df = df[df["field_id"] == field_id].copy()
        overlay_path = output_dir / f"field_{field_id:02d}_daily_mean_overlay_4models.png"
        panel_path = output_dir / f"field_{field_id:02d}_daily_mean_panels_4models.png"

        create_overlay_plot(field_df=field_df, field_id=field_id, output_path=overlay_path)
        metrics_records.extend(
            create_panel_plot(field_df=field_df, field_id=field_id, output_path=panel_path)
        )

    metrics_df = pd.DataFrame.from_records(metrics_records).sort_values(["field_id", "model"])
    metrics_df.to_csv(output_dir / "field_model_metrics.csv", index=False)
    print(f"Saved per-field plots to: {output_dir.resolve()}")
    print(f"Fields processed: {len(field_ids)}")


if __name__ == "__main__":
    main()
