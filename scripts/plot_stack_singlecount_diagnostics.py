from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["CNN", "MLP", "Baseline", "GNN"]
MODEL_COLORS = {
    "CNN": "#d62728",
    "MLP": "#1f77b4",
    "Baseline": "#7f7f7f",
    "GNN": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create single-count diagnostics to explain model differences per stack."
    )
    parser.add_argument(
        "--stack-results-root",
        type=Path,
        default=Path("outputs/final_ref/stack_audit_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_ref/stack_audit_results/singlecount_diagnostics"),
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=80000,
    )
    return parser.parse_args()


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _choose_example_field_year(wide: pd.DataFrame) -> tuple[int, int]:
    rows = []
    for (fid, year), g in wide.groupby(["field_id", "year"]):
        y = g["y_true"].to_numpy(dtype=float)
        rmse_gnn = float(np.sqrt(np.mean(np.square(g["GNN"].to_numpy(dtype=float) - y))))
        rmse_best = min(
            float(np.sqrt(np.mean(np.square(g["CNN"].to_numpy(dtype=float) - y)))),
            float(np.sqrt(np.mean(np.square(g["MLP"].to_numpy(dtype=float) - y)))),
        )
        rows.append({"field_id": int(fid), "year": int(year), "delta": rmse_gnn - rmse_best})
    tab = pd.DataFrame(rows).sort_values("delta", ascending=False)
    if tab.empty:
        return 1, 1
    top = tab.iloc[0]
    return int(top["field_id"]), int(top["year"])


def _plot_stack(aligned: pd.DataFrame, stack_name: str, species: str, run_id: str, out_png: Path, out_csv: Path, max_scatter_points: int) -> None:
    # Wide format on exact sample key.
    wide = (
        aligned.pivot_table(
            index=["field_id", "year", "day", "y_true"],
            columns="model",
            values="y_pred",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for m in MODEL_ORDER:
        wide[f"ae_{m}"] = np.abs(wide[m] - wide["y_true"])

    # Summary table
    rows = []
    for m in MODEL_ORDER:
        y = wide["y_true"].to_numpy(dtype=float)
        p = wide[m].to_numpy(dtype=float)
        e = p - y
        rows.append(
            {
                "model": m,
                "rmse": float(np.sqrt(np.mean(np.square(e)))),
                "mae": float(np.mean(np.abs(e))),
                "p95_abs_error": float(np.quantile(np.abs(e), 0.95)),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Figure: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (1) ECDF absolute error
    for m in MODEL_ORDER:
        x, y = _ecdf(wide[f"ae_{m}"].to_numpy(dtype=float))
        x_max = float(np.quantile(wide[f"ae_{m}"], 0.995))
        x_clip = x <= x_max
        ax1.plot(x[x_clip], y[x_clip], lw=2.0, color=MODEL_COLORS[m], label=m)
    ax1.set_title("A) Per-sample absolute error (ECDF)")
    ax1.set_xlabel("Absolute error")
    ax1.set_ylabel("Cumulative share")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)

    # (2) MAE by true-count bin
    bins = [-0.1, 1, 3, 6, 10, 20, 50, 200]
    labels = ["0-1", "1-3", "3-6", "6-10", "10-20", "20-50", "50+"]
    tmp = wide.copy()
    tmp["y_bin"] = pd.cut(tmp["y_true"], bins=bins, labels=labels)
    x = np.arange(len(labels))
    width = 0.2
    offsets = {"CNN": -1.5 * width, "MLP": -0.5 * width, "Baseline": 0.5 * width, "GNN": 1.5 * width}
    for m in MODEL_ORDER:
        g = tmp.groupby("y_bin", as_index=False)[f"ae_{m}"].mean().set_index("y_bin").reindex(labels).reset_index()
        ax2.bar(x + offsets[m], g[f"ae_{m}"], width=width, color=MODEL_COLORS[m], alpha=0.9, label=m)
    ax2.set_xticks(x, labels)
    ax2.set_title("B) MAE by ground-truth count bin")
    ax2.set_xlabel("Ground-truth bin")
    ax2.set_ylabel("MAE")
    ax2.grid(axis="y", alpha=0.25)

    # (3) Delta absolute error histogram
    d1 = wide["ae_GNN"] - wide["ae_CNN"]
    d2 = wide["ae_GNN"] - wide["ae_MLP"]
    q = float(np.quantile(np.abs(np.concatenate([d1.to_numpy(), d2.to_numpy()])), 0.99))
    bins_h = np.linspace(-q, q, 80)
    ax3.hist(d1, bins=bins_h, alpha=0.55, color="#c0392b", label="GNN - CNN", density=True)
    ax3.hist(d2, bins=bins_h, alpha=0.45, color="#8e44ad", label="GNN - MLP", density=True)
    ax3.axvline(0, color="black", lw=1)
    ax3.set_title("C) Distribution of |err| deltas")
    ax3.set_xlabel("Delta abs error (positive = GNN worse)")
    ax3.set_ylabel("Density")
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False, fontsize=8)

    # (4) One objective single-count trajectory example
    field_id, year = _choose_example_field_year(wide)
    ex = wide[(wide["field_id"] == field_id) & (wide["year"] == year)].sort_values("day")
    ax4.plot(ex["day"], ex["y_true"], color="black", lw=2.2, label="Ground Truth")
    for m in MODEL_ORDER:
        ax4.plot(ex["day"], ex[m], color=MODEL_COLORS[m], lw=1.8, label=m)
    ax4.set_title(f"D) Single-count trajectory (field {field_id}, year {year})")
    ax4.set_xlabel("Day of year")
    ax4.set_ylabel("Count")
    ax4.grid(alpha=0.25)
    # keep legend compact
    handles, labels_ = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels_, frameon=False, fontsize=8, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    fig.suptitle(
        f"{stack_name} | {species} | {run_id}\n"
        "Single-count view + error diagnostics (to avoid misleading daily-mean-only interpretation).",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stack_dirs = sorted([d for d in args.stack_results_root.iterdir() if d.is_dir() and d.name.startswith("stack")])
    for stack_dir in stack_dirs:
        aligned_path = stack_dir / "aligned_predictions_long_corrected.csv"
        if not aligned_path.exists():
            continue
        aligned = pd.read_csv(aligned_path)
        species = str(aligned["species"].iloc[0])
        run_id = str(aligned["run_id"].iloc[0])
        out_png = args.output_dir / f"{stack_dir.name}_singlecount_diagnostics.png"
        out_csv = args.output_dir / f"{stack_dir.name}_singlecount_summary.csv"
        _plot_stack(
            aligned,
            stack_dir.name,
            species,
            run_id,
            out_png,
            out_csv,
            max_scatter_points=args.max_scatter_points,
        )

    print(f"Saved single-count diagnostics to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
