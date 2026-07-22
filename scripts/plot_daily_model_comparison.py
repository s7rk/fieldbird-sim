from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create daily mean trajectory comparison for MLP, CNN, and GNN."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("~/Desktop/präsi/rebhuhn_h365_unified_predictions_FIXED.csv").expanduser(),
        help="Unified predictions CSV containing model/day/y_true_level/y_pred_level columns.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/presentation_plots/daily_mean_trajectories_mlp_cnn_gnn.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path: Path = args.output_png
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        args.input_csv,
        usecols=["model", "day", "y_true_level", "y_pred_level"],
    )

    model_order = ["MLP", "CNN", "GNN"]
    df = df[df["model"].isin(model_order)].copy()
    if df.empty:
        raise SystemExit("No rows found for MLP/CNN/GNN in the provided CSV.")

    grouped = (
        df.groupby(["model", "day"], as_index=False)[["y_true_level", "y_pred_level"]]
        .mean()
        .sort_values(["model", "day"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    colors = {"MLP": "#1f77b4", "CNN": "#d62728", "GNN": "#2ca02c"}

    y_min = float(
        np.nanmin([grouped["y_true_level"].min(), grouped["y_pred_level"].min()])
    )
    y_max = float(
        np.nanmax([grouped["y_true_level"].max(), grouped["y_pred_level"].max()])
    )
    y_pad = max(0.2, (y_max - y_min) * 0.05)

    for ax, model in zip(axes, model_order):
        g = grouped[grouped["model"] == model]
        ax.plot(g["day"], g["y_true_level"], color="#1f1f1f", lw=2.1, label="Ground Truth")
        ax.plot(g["day"], g["y_pred_level"], color=colors[model], lw=2.1, label="Prediction")
        ax.set_title(model)
        ax.set_xlabel("Day of year")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Daily mean value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Daily Mean Trajectories: MLP vs CNN vs GNN", fontsize=16, y=1.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
