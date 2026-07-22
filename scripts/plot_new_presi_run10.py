from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_COLS = ["species", "run_id", "field_id", "scenario", "split", "year", "day"]


def parse_args() -> argparse.Namespace:
    desktop_presi = Path.home() / "Desktop" / "präsi"
    parser = argparse.ArgumentParser(
        description="Create presentation plots from run10 baseline/MLP/CNN/GNN CSV files."
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=desktop_presi / "pred_baseline_Rebhuhn_h365_run10_baseline_test.csv",
    )
    parser.add_argument(
        "--mlp-csv",
        type=Path,
        default=desktop_presi / "pred_mlp_TRUE_Rebhuhn_h365_run10_baseline_test.csv",
    )
    parser.add_argument(
        "--cnn-csv",
        type=Path,
        default=desktop_presi / "pred_cnn_TRUE_Rebhuhn_h365_run10_baseline_test_debug.csv",
    )
    parser.add_argument(
        "--gnn-csv",
        type=Path,
        default=desktop_presi / "pred_gnn_TRUE_Rebhuhn_h365_run10_baseline_test_debug.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/presentation_plots/run10_new"),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Maximum points per model in scatter plot (sampled if needed).",
    )
    return parser.parse_args()


def load_baseline_or_mlp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["model", *KEY_COLS, "y_true_level", "y_pred_level"]).copy()
    return df


def load_cnn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[*KEY_COLS, "y_true", "y_pred"]).rename(
        columns={"y_true": "y_true_level", "y_pred": "y_pred_level"}
    )
    df["model"] = "CNN"
    return df[["model", *KEY_COLS, "y_true_level", "y_pred_level"]]


def load_gnn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[*KEY_COLS, "N_total", "N_total_tplus365", "y_pred_lvl"]).copy()
    # `y_pred_lvl` is log-delta output. Convert to count scale for fair model comparison.
    df["y_pred_level"] = df["N_total"] * np.exp(df["y_pred_lvl"])
    df["y_true_level"] = df["N_total_tplus365"]
    df["model"] = "GNN"
    return df[["model", *KEY_COLS, "y_true_level", "y_pred_level"]]


def dedup_by_key_mean(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", *KEY_COLS], as_index=False)[["y_true_level", "y_pred_level"]].mean()
    )
    return grouped


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str | int]] = []
    for model, g in df.groupby("model"):
        y_true = g["y_true_level"].to_numpy(dtype=float)
        y_pred = g["y_pred_level"].to_numpy(dtype=float)
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(np.square(err))))
        ss_res = float(np.sum(np.square(err)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
        records.append(
            {
                "model": model,
                "rows": len(g),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("model")


def create_daily_panels(df: pd.DataFrame, output_path: Path) -> None:
    daily = (
        df.groupby(["model", "day"], as_index=False)[["y_true_level", "y_pred_level"]]
        .mean()
        .sort_values(["model", "day"])
    )

    models = sorted(daily["model"].unique().tolist())
    ncols = 2
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    colors = {
        "Baseline": "#8c8c8c",
        "MLP": "#1f77b4",
        "CNN": "#d62728",
        "GNN": "#2ca02c",
    }

    for idx, model in enumerate(models):
        ax = axes[idx // ncols, idx % ncols]
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["y_true_level"], color="#1a1a1a", lw=2.0, label="Ground Truth")
        ax.plot(
            g["day"],
            g["y_pred_level"],
            color=colors.get(model, "#f15a24"),
            lw=2.0,
            label="Prediction",
        )
        ax.set_title(model)
        ax.set_xlabel("Day of year")
        ax.grid(alpha=0.25)

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    axes[0, 0].set_ylabel("Daily mean value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Daily Mean Trajectories (Run10, Baseline vs MLP vs CNN vs GNN)", fontsize=15, y=1.06)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def create_overlay_plot(df: pd.DataFrame, output_path: Path) -> None:
    daily = (
        df.groupby(["model", "day"], as_index=False)[["y_true_level", "y_pred_level"]]
        .mean()
        .sort_values(["model", "day"])
    )
    gt = daily.groupby("day", as_index=False)["y_true_level"].mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(gt["day"], gt["y_true_level"], color="black", lw=2.5, label="Ground Truth")

    colors = {
        "Baseline": "#8c8c8c",
        "MLP": "#1f77b4",
        "CNN": "#d62728",
        "GNN": "#2ca02c",
    }
    for model in sorted(daily["model"].unique().tolist()):
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["y_pred_level"], lw=2.0, color=colors.get(model, "#f15a24"), label=model)

    ax.set_title("Daily Mean Prediction Comparison (Run10)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Daily mean value")
    ax.grid(alpha=0.25)
    ax.legend(ncols=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def create_scatter_panels(df: pd.DataFrame, metrics: pd.DataFrame, output_path: Path, max_points: int) -> None:
    models = sorted(df["model"].unique().tolist())
    ncols = 2
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols, idx % ncols]
        g = df[df["model"] == model]
        if len(g) > max_points:
            g = g.sample(n=max_points, random_state=42)

        hb = ax.hexbin(
            g["y_true_level"],
            g["y_pred_level"],
            gridsize=60,
            mincnt=1,
            cmap="viridis",
        )

        min_xy = float(min(g["y_true_level"].min(), g["y_pred_level"].min()))
        max_xy = float(max(g["y_true_level"].max(), g["y_pred_level"].max()))
        ax.plot([min_xy, max_xy], [min_xy, max_xy], "--", color="tomato", lw=1.3)
        ax.set_title(model)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")

        m = metrics[metrics["model"] == model].iloc[0]
        ax.text(
            0.03,
            0.97,
            f"n={int(m['rows']):,}\nMAE={m['mae']:.3f}\nRMSE={m['rmse']:.3f}\nR2={m['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
        )
        fig.colorbar(hb, ax=ax, label="Point density")

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Prediction vs Ground Truth (Run10)", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_metrics_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["model"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(x, metrics["mae"], color="#2a9d8f")
    axes[0].set_title("MAE")
    axes[1].bar(x, metrics["rmse"], color="#457b9d")
    axes[1].set_title("RMSE")
    axes[2].bar(x, metrics["r2"], color="#e76f51")
    axes[2].set_title("R2")
    axes[2].axhline(0, color="black", lw=1)

    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Model Metrics (Run10)", fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def verify_ground_truth(df: pd.DataFrame) -> float:
    pivot = df.pivot_table(index=KEY_COLS, columns="model", values="y_true_level", aggfunc="first")
    cols = list(pivot.columns)
    max_diff = 0.0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            max_diff = max(max_diff, float((pivot[cols[i]] - pivot[cols[j]]).abs().max()))
    return max_diff


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = dedup_by_key_mean(load_baseline_or_mlp(args.baseline_csv))
    mlp = dedup_by_key_mean(load_baseline_or_mlp(args.mlp_csv))
    cnn = dedup_by_key_mean(load_cnn(args.cnn_csv))
    gnn = dedup_by_key_mean(load_gnn(args.gnn_csv))

    combined = pd.concat([baseline, mlp, cnn, gnn], ignore_index=True)
    combined = combined.sort_values(["model", "run_id", "field_id", "year", "day"])

    gt_max_diff = verify_ground_truth(combined)
    metrics = compute_metrics(combined)

    unified_csv = output_dir / "run10_unified_predictions_4models.csv"
    combined.to_csv(unified_csv, index=False)

    metrics_csv = output_dir / "run10_model_metrics_4models.csv"
    metrics.to_csv(metrics_csv, index=False)

    create_daily_panels(df=combined, output_path=output_dir / "run10_daily_mean_trajectories_4models.png")
    create_overlay_plot(df=combined, output_path=output_dir / "run10_daily_mean_overlay_4models.png")
    create_scatter_panels(
        df=combined,
        metrics=metrics,
        output_path=output_dir / "run10_prediction_vs_groundtruth_4models.png",
        max_points=args.max_points,
    )
    create_metrics_plot(metrics=metrics, output_path=output_dir / "run10_model_metrics_4models.png")

    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"Ground truth max difference between models: {gt_max_diff:.12f}")
    for _, row in metrics.iterrows():
        print(
            f"- {row['model']}: n={int(row['rows']):,}, "
            f"MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
