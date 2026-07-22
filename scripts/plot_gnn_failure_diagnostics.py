from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


MODEL_ORDER = ["CNN", "MLP", "Baseline", "GNN"]
MODEL_COLORS = {
    "CNN": "#d62728",
    "MLP": "#1f77b4",
    "Baseline": "#7f7f7f",
    "GNN": "#2ca02c",
}

KEY_COLS = ["species", "run_id", "field_id", "scenario", "split", "year", "day"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create targeted diagnostics for GNN underperformance.")
    parser.add_argument(
        "--aligned-csv",
        type=Path,
        default=Path("outputs/final_ref/audit_results/aligned_predictions_long_corrected.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_ref/audit_results/presentation_pack/gnn_failure"),
    )
    return parser.parse_args()


def to_wide(aligned: pd.DataFrame) -> pd.DataFrame:
    base_cols = KEY_COLS + ["y_true"]
    wide = (
        aligned.pivot_table(index=base_cols, columns="model", values="y_pred", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    missing = [m for m in MODEL_ORDER if m not in wide.columns]
    if missing:
        raise ValueError(f"Missing models in aligned data: {missing}")
    return wide


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for m in MODEL_ORDER:
        out[f"err_{m}"] = out[m] - out["y_true"]
        out[f"ae_{m}"] = np.abs(out[f"err_{m}"])
        out[f"se_{m}"] = np.square(out[f"err_{m}"])
    out["delta_ae_gnn_vs_cnn"] = out["ae_GNN"] - out["ae_CNN"]
    out["delta_ae_gnn_vs_mlp"] = out["ae_GNN"] - out["ae_MLP"]
    out["delta_se_gnn_vs_cnn"] = out["se_GNN"] - out["se_CNN"]
    out["delta_se_gnn_vs_mlp"] = out["se_GNN"] - out["se_MLP"]
    return out


def plot_rmse_by_true_bin(df: pd.DataFrame, out_png: Path, out_csv: Path) -> pd.DataFrame:
    bins = [-0.1, 1, 3, 6, 10, 20, 50, 200]
    labels = ["0-1", "1-3", "3-6", "6-10", "10-20", "20-50", "50+"]
    data = df.copy()
    data["y_bin"] = pd.cut(data["y_true"], bins=bins, labels=labels)

    rows = []
    for m in MODEL_ORDER:
        g = (
            data.groupby("y_bin", as_index=False)
            .agg(n=("y_true", "size"), mse=(f"se_{m}", "mean"))
            .assign(model=m)
        )
        g["rmse"] = np.sqrt(g["mse"])
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(labels))
    width = 0.20
    offsets = {
        "CNN": -1.5 * width,
        "MLP": -0.5 * width,
        "Baseline": 0.5 * width,
        "GNN": 1.5 * width,
    }
    for m in MODEL_ORDER:
        g = out[out["model"] == m].set_index("y_bin").reindex(labels).reset_index()
        ax.bar(x + offsets[m], g["rmse"], width=width, color=MODEL_COLORS[m], label=m, alpha=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE within true-count bin")
    ax.set_xlabel("Ground-truth count bin")
    ax.set_title("Where GNN Loses: RMSE by Target Magnitude")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def _heatmap(ax: plt.Axes, table: pd.DataFrame, title: str) -> None:
    z = table.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(z), 98))
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
    ax.set_title(title)
    ax.set_ylabel("y_true bin")
    ax.set_yticks(np.arange(table.shape[0]), table.index.tolist())
    # show month-ish ticks
    day_ticks = [1, 60, 120, 180, 240, 300, 365]
    day_pos = [d - 1 for d in day_ticks]
    ax.set_xticks(day_pos, [str(d) for d in day_ticks])
    ax.set_xlabel("Day of year")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="GNN error - comparator error")


def plot_error_delta_heatmaps(df: pd.DataFrame, out_png: Path, out_csv: Path) -> pd.DataFrame:
    bins = [-0.1, 1, 3, 6, 10, 20, 50, 200]
    labels = ["0-1", "1-3", "3-6", "6-10", "10-20", "20-50", "50+"]
    data = df.copy()
    data["y_bin"] = pd.cut(data["y_true"], bins=bins, labels=labels)

    grouped = (
        data.groupby(["y_bin", "day"], as_index=False)
        .agg(
            delta_ae_gnn_vs_cnn=("delta_ae_gnn_vs_cnn", "mean"),
            delta_ae_gnn_vs_mlp=("delta_ae_gnn_vs_mlp", "mean"),
        )
        .sort_values(["y_bin", "day"])
    )
    grouped.to_csv(out_csv, index=False)

    piv_cnn = grouped.pivot(index="y_bin", columns="day", values="delta_ae_gnn_vs_cnn").reindex(labels)
    piv_mlp = grouped.pivot(index="y_bin", columns="day", values="delta_ae_gnn_vs_mlp").reindex(labels)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.5), sharex=True)
    _heatmap(axes[0], piv_cnn, "GNN minus CNN: Mean Absolute Error Delta")
    _heatmap(axes[1], piv_mlp, "GNN minus MLP: Mean Absolute Error Delta")
    fig.suptitle("Where and When GNN Underperforms (Red = GNN Worse)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return grouped


def plot_field_delta(df: pd.DataFrame, out_png: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    for fid, g in df.groupby("field_id"):
        rows.append(
            {
                "field_id": int(fid),
                "rmse_CNN": float(np.sqrt(np.mean(g["se_CNN"]))),
                "rmse_MLP": float(np.sqrt(np.mean(g["se_MLP"]))),
                "rmse_Baseline": float(np.sqrt(np.mean(g["se_Baseline"]))),
                "rmse_GNN": float(np.sqrt(np.mean(g["se_GNN"]))),
            }
        )
    out = pd.DataFrame(rows).sort_values("field_id")
    out["delta_rmse_gnn_minus_cnn"] = out["rmse_GNN"] - out["rmse_CNN"]
    out["delta_rmse_gnn_minus_mlp"] = out["rmse_GNN"] - out["rmse_MLP"]
    out = out.sort_values("delta_rmse_gnn_minus_cnn", ascending=False).reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    top = out.head(12).sort_values("delta_rmse_gnn_minus_cnn")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    y = np.arange(len(top))
    ax.barh(y - 0.18, top["delta_rmse_gnn_minus_cnn"], height=0.34, color="#c0392b", label="GNN - CNN")
    ax.barh(y + 0.18, top["delta_rmse_gnn_minus_mlp"], height=0.34, color="#8e44ad", label="GNN - MLP")
    ax.set_yticks(y, [str(int(x)) for x in top["field_id"]])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("RMSE delta (positive = GNN worse)")
    ax.set_ylabel("Field ID")
    ax.set_title("Fields Where GNN Falls Behind Most")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_worst_field_trajectory(df: pd.DataFrame, field_table: pd.DataFrame, out_png: Path) -> int:
    worst_field = int(field_table.iloc[0]["field_id"])
    g = df[df["field_id"] == worst_field].copy()

    # Use daily mean just for shape visualization, plus daily RMSE annotations for fairness.
    daily_pred = g.groupby("day", as_index=False)[["y_true", "CNN", "MLP", "Baseline", "GNN"]].mean()
    daily_rmse = (
        g.groupby("day", as_index=False)[["se_CNN", "se_MLP", "se_Baseline", "se_GNN"]]
        .mean()
        .assign(
            rmse_CNN=lambda d: np.sqrt(d["se_CNN"]),
            rmse_MLP=lambda d: np.sqrt(d["se_MLP"]),
            rmse_Baseline=lambda d: np.sqrt(d["se_Baseline"]),
            rmse_GNN=lambda d: np.sqrt(d["se_GNN"]),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1, ax2 = axes
    ax1.plot(daily_pred["day"], daily_pred["y_true"], color="black", lw=2.3, label="Ground Truth")
    for m in ["CNN", "MLP", "Baseline", "GNN"]:
        ax1.plot(daily_pred["day"], daily_pred[m], color=MODEL_COLORS[m], lw=1.8, label=m)
    ax1.set_ylabel("Daily mean level")
    ax1.set_title(f"Worst Field for GNN vs CNN (Field {worst_field})")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))

    for m in ["CNN", "MLP", "Baseline", "GNN"]:
        ax2.plot(daily_rmse["day"], daily_rmse[f"rmse_{m}"], color=MODEL_COLORS[m], lw=1.8, label=m)
    ax2.set_xlabel("Day of year")
    ax2.set_ylabel("Daily RMSE")
    ax2.grid(alpha=0.25)
    ax2.set_title("Same field as error time series (more reliable than level means)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return worst_field


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_csv(args.aligned_csv)
    wide = to_wide(aligned)
    df = add_error_columns(wide)

    plot_rmse_by_true_bin(
        df,
        args.output_dir / "plot_gnn_vs_models_rmse_by_true_bin.png",
        args.output_dir / "table_rmse_by_true_bin.csv",
    )
    plot_error_delta_heatmaps(
        df,
        args.output_dir / "plot_gnn_error_delta_heatmaps.png",
        args.output_dir / "table_gnn_error_delta_by_day_bin.csv",
    )
    field_table = plot_field_delta(
        df,
        args.output_dir / "plot_gnn_field_delta_top12.png",
        args.output_dir / "table_gnn_field_rmse_deltas.csv",
    )
    worst_field = plot_worst_field_trajectory(
        df,
        field_table,
        args.output_dir / "plot_gnn_worst_field_trajectory.png",
    )

    summary = pd.DataFrame(
        [
            {
                "note": "worst_field_by_gnn_minus_cnn_delta_rmse",
                "value": worst_field,
            },
            {
                "note": "max_delta_rmse_gnn_minus_cnn",
                "value": float(field_table.iloc[0]["delta_rmse_gnn_minus_cnn"]),
            },
            {
                "note": "max_delta_rmse_gnn_minus_mlp",
                "value": float(field_table.iloc[0]["delta_rmse_gnn_minus_mlp"]),
            },
        ]
    )
    summary.to_csv(args.output_dir / "summary_gnn_failure.csv", index=False)
    print(f"Saved GNN failure diagnostics to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
