from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DatasetResult:
    name: str
    file_name: str
    truth_col: str
    pred_col: str
    scatter_frame: pd.DataFrame
    daily_frame: pd.DataFrame | None
    n_rows: int
    mae: float
    rmse: float
    r2: float


def model_name_from_file(path: Path) -> str:
    lower = path.name.lower()
    if "clean_predictions_gnn" in lower:
        return "GNN (Real Counts)"
    if "debug_single_predictions_gnn" in lower:
        return "GNN (Target Space)"
    if "debug_single_predictions_cnn" in lower:
        return "CNN"
    if "debug_single_predictions_mlp" in lower:
        return "MLP"
    return path.stem


def detect_columns(path: Path) -> tuple[str, str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    colset = set(cols)
    lower = path.name.lower()

    if "clean_predictions_gnn" in lower and {
        "N_true_tplus_from_logdelta",
        "N_pred_tplus",
    }.issubset(colset):
        return "N_true_tplus_from_logdelta", "N_pred_tplus"
    if "debug_single_predictions_cnn" in lower and {"y_true", "y_pred"}.issubset(colset):
        return "y_true", "y_pred"
    if "debug_single_predictions_mlp" in lower and {
        "y_true_lvl",
        "y_pred_lvl_mean",
    }.issubset(colset):
        return "y_true_lvl", "y_pred_lvl_mean"
    if "debug_single_predictions_gnn" in lower and {"y_true_lvl", "y_pred_lvl"}.issubset(colset):
        return "y_true_lvl", "y_pred_lvl"

    truth_candidates = [
        "y_true",
        "N_true_tplus_from_logdelta",
        "y_true_lvl",
        "N_total_tplus365",
    ]
    pred_candidates = [
        "y_pred",
        "N_pred_tplus",
        "y_pred_lvl_mean",
        "y_pred_lvl",
        "y_pred_target_mean",
    ]

    truth_col = next((c for c in truth_candidates if c in colset), None)
    pred_col = next((c for c in pred_candidates if c in colset), None)
    if not truth_col or not pred_col:
        raise ValueError(f"Could not detect target/prediction columns for: {path}")
    return truth_col, pred_col


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return mae, rmse, r2


def load_dataset(path: Path, max_points: int) -> DatasetResult:
    truth_col, pred_col = detect_columns(path)
    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [truth_col, pred_col]
    if "day" in header_cols:
        usecols.append("day")

    frame = pd.read_csv(path, usecols=usecols).rename(
        columns={truth_col: "y_true", pred_col: "y_pred"}
    )
    frame = frame.dropna(subset=["y_true", "y_pred"])

    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    mae, rmse, r2 = compute_metrics(y_true, y_pred)

    scatter_frame = sample_for_scatter(frame[["y_true", "y_pred"]], max_points=max_points)
    daily_frame: pd.DataFrame | None = None
    if "day" in frame.columns:
        daily_frame = (
            frame.groupby("day", as_index=False)[["y_true", "y_pred"]].mean().sort_values("day")
        )

    return DatasetResult(
        name=model_name_from_file(path),
        file_name=path.name,
        truth_col=truth_col,
        pred_col=pred_col,
        scatter_frame=scatter_frame,
        daily_frame=daily_frame,
        n_rows=len(frame),
        mae=mae,
        rmse=rmse,
        r2=r2,
    )


def sample_for_scatter(frame: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(n=max_points, random_state=seed)


def create_scatter_plot(results: list[DatasetResult], output_path: Path, max_points: int) -> None:
    n = len(results)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, result in enumerate(results):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        sampled = sample_for_scatter(result.scatter_frame, max_points=max_points)

        hb = ax.hexbin(
            sampled["y_true"],
            sampled["y_pred"],
            gridsize=60,
            mincnt=1,
            cmap="viridis",
        )
        min_xy = float(min(sampled["y_true"].min(), sampled["y_pred"].min()))
        max_xy = float(max(sampled["y_true"].max(), sampled["y_pred"].max()))
        ax.plot([min_xy, max_xy], [min_xy, max_xy], "--", color="tomato", lw=1.4)

        ax.set_title(result.name)
        ax.set_xlabel(f"Ground Truth ({result.truth_col})")
        ax.set_ylabel(f"Prediction ({result.pred_col})")
        ax.text(
            0.03,
            0.97,
            (
                f"n={result.n_rows:,}\n"
                f"MAE={result.mae:.3f}\n"
                f"RMSE={result.rmse:.3f}\n"
                f"R²={result.r2:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
        )
        fig.colorbar(hb, ax=ax, label="Point density")

    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    fig.suptitle("Prediction vs Ground Truth", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_daily_plot(results: list[DatasetResult], output_path: Path) -> None:
    with_day = [r for r in results if r.daily_frame is not None]
    if not with_day:
        return

    n = len(with_day)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, result in enumerate(with_day):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        grouped = result.daily_frame
        if grouped is None:
            continue
        ax.plot(grouped["day"], grouped["y_true"], lw=2.0, color="#1f1f1f", label="Ground Truth")
        ax.plot(grouped["day"], grouped["y_pred"], lw=2.0, color="#f15a24", label="Prediction")
        ax.set_title(result.name)
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Mean value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    fig.suptitle("Daily Mean Trajectories", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_metrics_plot(results: list[DatasetResult], output_path: Path) -> None:
    labels = [r.name for r in results]
    mae = [r.mae for r in results]
    rmse = [r.rmse for r in results]
    r2 = [r.r2 for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_mae, ax_rmse, ax_r2 = axes

    ax_mae.bar(x, mae, color="#2a9d8f")
    ax_mae.set_title("MAE")
    ax_mae.set_xticks(x, labels, rotation=20, ha="right")
    ax_mae.grid(axis="y", alpha=0.25)

    ax_rmse.bar(x, rmse, color="#457b9d")
    ax_rmse.set_title("RMSE")
    ax_rmse.set_xticks(x, labels, rotation=20, ha="right")
    ax_rmse.grid(axis="y", alpha=0.25)

    ax_r2.bar(x, r2, color="#e76f51")
    ax_r2.set_title("R²")
    ax_r2.set_xticks(x, labels, rotation=20, ha="right")
    ax_r2.axhline(0, color="black", linewidth=1)
    ax_r2.grid(axis="y", alpha=0.25)

    fig.suptitle("Model Metrics", fontsize=16, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(results: list[DatasetResult], output_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": r.name,
                "file": r.file_name,
                "ground_truth_column": r.truth_col,
                "prediction_column": r.pred_col,
                "rows": r.n_rows,
                "mae": r.mae,
                "rmse": r.rmse,
                "r2": r.r2,
            }
            for r in results
        ]
    )
    metrics.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ground truth vs prediction from CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("~/Desktop/präsi").expanduser(),
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/presentation_plots"),
        help="Directory for generated plots and metrics.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Maximum sampled points per plot panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    results: list[DatasetResult] = []
    for csv_file in csv_files:
        results.append(load_dataset(csv_file, max_points=args.max_points))

    create_scatter_plot(
        results=results,
        output_path=output_dir / "prediction_vs_groundtruth_scatter.png",
        max_points=args.max_points,
    )
    create_daily_plot(results=results, output_path=output_dir / "daily_mean_trajectories.png")
    create_metrics_plot(results=results, output_path=output_dir / "model_metrics.png")
    write_metrics_csv(results=results, output_path=output_dir / "model_metrics.csv")

    print(f"Saved outputs to: {output_dir.resolve()}")
    for result in results:
        print(
            f"- {result.name}: n={result.n_rows:,}, MAE={result.mae:.3f}, "
            f"RMSE={result.rmse:.3f}, R2={result.r2:.3f}"
        )


if __name__ == "__main__":
    main()
