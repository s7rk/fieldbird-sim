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
        description="Create explanatory panels showing why daily totals can disagree with sample RMSE."
    )
    parser.add_argument(
        "--stack-results-root",
        type=Path,
        default=Path("outputs/final_ref/stack_audit_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_ref/stack_audit_results/explainer_panels"),
    )
    return parser.parse_args()


def _compute_stack_tables(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    daily_rows = []

    for model in MODEL_ORDER:
        g = aligned[aligned["model"] == model].copy()
        y = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred"].to_numpy(dtype=float)
        e = p - y

        mse = float(np.mean(np.square(e)))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(e)))
        bias = float(np.mean(e))
        var = float(np.var(e, ddof=0))

        d = (
            g.assign(err=e)
            .groupby("day", as_index=False)
            .agg(
                pred_total=("y_pred", "sum"),
                true_total=("y_true", "sum"),
                err_sum=("err", "sum"),
                abs_err_sum=("err", lambda x: float(np.abs(x).sum())),
                n=("err", "size"),
            )
            .sort_values("day")
        )
        d["model"] = model
        daily_rows.append(d)

        daily_total_err = d["pred_total"].to_numpy(dtype=float) - d["true_total"].to_numpy(dtype=float)
        daily_total_rmse = float(np.sqrt(np.mean(np.square(daily_total_err))))
        daily_total_mae = float(np.mean(np.abs(daily_total_err)))
        cancel_ratio = float(np.mean(np.abs(d["err_sum"]) / (d["abs_err_sum"] + 1e-12)))

        metric_rows.append(
            {
                "model": model,
                "rows": len(g),
                "rmse_sample": rmse,
                "mae_sample": mae,
                "mse_sample": mse,
                "bias_sample": bias,
                "bias_sq": bias**2,
                "var_error": var,
                "daily_total_rmse": daily_total_rmse,
                "daily_total_mae": daily_total_mae,
                "daily_error_cancellation_ratio": cancel_ratio,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    return metrics, daily


def _plot_explainer(stack_name: str, species: str, run_id: str, metrics: pd.DataFrame, daily: pd.DataFrame, out_png: Path) -> None:
    m = metrics.copy()
    m["model"] = pd.Categorical(m["model"], categories=MODEL_ORDER, ordered=True)
    m = m.sort_values("model")

    gt_daily = daily.groupby("day", as_index=False)["true_total"].mean().sort_values("day")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    ax_tot, ax_rmse, ax_daily_rmse, ax_decomp = axes.ravel()

    # Panel 1: Daily total counts overlay.
    ax_tot.plot(gt_daily["day"], gt_daily["true_total"], color="black", lw=2.3, label="Ground Truth total")
    for model in MODEL_ORDER:
        g = daily[daily["model"] == model]
        ax_tot.plot(g["day"], g["pred_total"], color=MODEL_COLORS[model], lw=1.8, label=f"{model} total")
    ax_tot.set_title("A) Daily Total Counts")
    ax_tot.set_xlabel("Day of year")
    ax_tot.set_ylabel("Total count/day")
    ax_tot.grid(alpha=0.25)
    ax_tot.legend(frameon=False, ncols=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.17))

    # Panel 2: Sample RMSE ranking.
    ax_rmse.bar(m["model"], m["rmse_sample"], color=[MODEL_COLORS[x] for x in m["model"]])
    for i, v in enumerate(m["rmse_sample"]):
        ax_rmse.text(i, float(v) + 0.03, f"{float(v):.3f}", ha="center", va="bottom", fontsize=9)
    ax_rmse.set_title("B) Per-Sample RMSE (Ranking Metric)")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.grid(axis="y", alpha=0.25)

    # Panel 3: Daily-total RMSE (can suggest different ranking).
    ax_daily_rmse.bar(m["model"], m["daily_total_rmse"], color=[MODEL_COLORS[x] for x in m["model"]])
    for i, v in enumerate(m["daily_total_rmse"]):
        ax_daily_rmse.text(i, float(v) + 5, f"{float(v):.1f}", ha="center", va="bottom", fontsize=9)
    ax_daily_rmse.set_title("C) RMSE on Daily Totals")
    ax_daily_rmse.set_ylabel("RMSE (total counts/day)")
    ax_daily_rmse.grid(axis="y", alpha=0.25)

    # Panel 4: Error decomposition MSE = Bias^2 + Variance.
    ax_decomp.bar(m["model"], m["bias_sq"], color="#f4a261", label="Bias²")
    ax_decomp.bar(
        m["model"],
        m["var_error"],
        bottom=m["bias_sq"],
        color="#264653",
        label="Error variance",
    )
    for i, (_, row) in enumerate(m.iterrows()):
        ax_decomp.text(
            i,
            float(row["mse_sample"]) + 0.2,
            f"MSE={float(row['mse_sample']):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_decomp.set_title("D) Why This Happens: MSE = Bias² + Variance")
    ax_decomp.set_ylabel("MSE components")
    ax_decomp.grid(axis="y", alpha=0.25)
    ax_decomp.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"{stack_name} | {species} | {run_id}\n"
        "Daily totals can look close even when sample-level RMSE is poor (error cancellation).",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stack_dirs = sorted([d for d in args.stack_results_root.iterdir() if d.is_dir() and d.name.startswith("stack")])
    if not stack_dirs:
        raise SystemExit(f"No stack dirs found in {args.stack_results_root}")

    summary_rows = []
    for stack_dir in stack_dirs:
        aligned_path = stack_dir / "aligned_predictions_long_corrected.csv"
        if not aligned_path.exists():
            continue
        aligned = pd.read_csv(aligned_path)
        species = str(aligned["species"].iloc[0])
        run_id = str(aligned["run_id"].iloc[0])

        metrics, daily = _compute_stack_tables(aligned)
        metrics.to_csv(args.output_dir / f"{stack_dir.name}_error_metrics_decomp.csv", index=False)
        daily.to_csv(args.output_dir / f"{stack_dir.name}_daily_totals_by_model.csv", index=False)

        out_png = args.output_dir / f"{stack_dir.name}_explainer_panel.png"
        _plot_explainer(stack_dir.name, species, run_id, metrics, daily, out_png)

        for _, row in metrics.iterrows():
            summary_rows.append(
                {
                    "stack": stack_dir.name,
                    "species": species,
                    "run_id": run_id,
                    "model": row["model"],
                    "rmse_sample": row["rmse_sample"],
                    "daily_total_rmse": row["daily_total_rmse"],
                    "daily_error_cancellation_ratio": row["daily_error_cancellation_ratio"],
                    "bias_sample": row["bias_sample"],
                    "var_error": row["var_error"],
                }
            )

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "all_stacks_explainer_summary.csv", index=False)
    print(f"Saved explainer panels to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
