from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_COLS = ["species", "run_id", "field_id", "scenario", "split", "year", "day"]
MODEL_ORDER = ["CNN", "MLP", "Baseline", "GNN"]
MODEL_COLORS = {
    "CNN": "#d62728",
    "MLP": "#1f77b4",
    "Baseline": "#7f7f7f",
    "GNN": "#2ca02c",
}


@dataclass
class StackData:
    stack: str
    species: str
    run_id: str
    aligned: pd.DataFrame
    metrics: pd.DataFrame
    per_field: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multiple stack run folders for model comparison.")
    parser.add_argument(
        "--stack-root",
        type=Path,
        default=Path("~/Desktop/präsi").expanduser(),
        help="Directory containing stack1/stack2/stack3 subfolders.",
    )
    parser.add_argument(
        "--stack-prefix",
        type=str,
        default="stack",
        help="Subfolder prefix to include (default: stack).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_ref/stack_audit_results"),
        help="Output directory for summaries and plots.",
    )
    return parser.parse_args()


def _require_one(path_list: list[Path], label: str, stack_dir: Path) -> Path:
    if len(path_list) != 1:
        raise ValueError(f"{stack_dir}: expected exactly one {label} file, found {len(path_list)}")
    return path_list[0]


def _metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    e = p - y
    rmse = float(np.sqrt(np.mean(np.square(e))))
    mae = float(np.mean(np.abs(e)))
    ss_res = float(np.sum(np.square(e)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return rmse, mae, r2


def _load_stack(stack_dir: Path) -> StackData:
    baseline_path = _require_one(list(stack_dir.glob("pred_baseline_*.csv")), "baseline", stack_dir)
    mlp_path = _require_one(list(stack_dir.glob("pred_mlp_TRUE_*.csv")), "mlp", stack_dir)
    cnn_path = _require_one(list(stack_dir.glob("pred_cnn_TRUE_*_debug.csv")), "cnn", stack_dir)
    gnn_path = _require_one(list(stack_dir.glob("pred_gnn_TRUE_*_debug.csv")), "gnn", stack_dir)

    b = pd.read_csv(baseline_path, usecols=[*KEY_COLS, "y_true_level", "y_pred_level"])
    m = pd.read_csv(mlp_path, usecols=[*KEY_COLS, "y_true_level", "y_pred_level"])
    c = pd.read_csv(cnn_path, usecols=[*KEY_COLS, "y_true_lvl", "y_pred_lvl"])
    g = pd.read_csv(gnn_path, usecols=[*KEY_COLS, "N_total", "N_total_tplus365", "y_pred_lvl"])

    # Exact key alignment
    kb = set(map(tuple, b[KEY_COLS].to_numpy()))
    km = set(map(tuple, m[KEY_COLS].to_numpy()))
    kc = set(map(tuple, c[KEY_COLS].to_numpy()))
    kg = set(map(tuple, g[KEY_COLS].to_numpy()))
    if not (kb == km == kc == kg):
        raise ValueError(f"{stack_dir}: model files are not aligned on identical key sets")

    base = b[KEY_COLS].copy()
    base["y_true"] = b["y_true_level"].to_numpy(dtype=float)

    rows = []
    for model, pred in [
        ("Baseline", b["y_pred_level"].to_numpy(dtype=float)),
        ("MLP", m["y_pred_level"].to_numpy(dtype=float)),
        ("CNN", c["y_pred_lvl"].to_numpy(dtype=float)),
    ]:
        t = base.copy()
        t["model"] = model
        t["y_pred"] = pred
        rows.append(t)

    # Correct inverse transform for GNN (log-delta style output).
    gnn_pred = np.expm1(np.log1p(g["N_total"].to_numpy(dtype=float)) + g["y_pred_lvl"].to_numpy(dtype=float))
    gnn_pred = np.maximum(gnn_pred, 0.0)
    tg = base.copy()
    tg["model"] = "GNN"
    tg["y_pred"] = gnn_pred
    rows.append(tg)

    aligned = pd.concat(rows, ignore_index=True)
    species = str(aligned["species"].iloc[0])
    run_id = str(aligned["run_id"].iloc[0])

    # Metrics
    metric_rows: list[dict[str, float | str | int]] = []
    for model in MODEL_ORDER:
        gm = aligned[aligned["model"] == model]
        rmse, mae, r2 = _metrics(gm["y_true"].to_numpy(dtype=float), gm["y_pred"].to_numpy(dtype=float))
        metric_rows.append(
            {
                "stack": stack_dir.name,
                "species": species,
                "run_id": run_id,
                "model": model,
                "rows": len(gm),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )
    metrics = pd.DataFrame(metric_rows)

    # Per-field RMSE
    field_rows = []
    for fid, gf in aligned.groupby("field_id"):
        rec: dict[str, float | str | int] = {
            "stack": stack_dir.name,
            "species": species,
            "run_id": run_id,
            "field_id": int(fid),
        }
        best_model = None
        best_rmse = float("inf")
        for model in MODEL_ORDER:
            gfm = gf[gf["model"] == model]
            rmse, _, _ = _metrics(gfm["y_true"].to_numpy(dtype=float), gfm["y_pred"].to_numpy(dtype=float))
            rec[f"rmse_{model}"] = rmse
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
        rec["winner"] = best_model if best_model is not None else ""
        field_rows.append(rec)
    per_field = pd.DataFrame(field_rows).sort_values("field_id")

    return StackData(
        stack=stack_dir.name,
        species=species,
        run_id=run_id,
        aligned=aligned.sort_values(["model", "field_id", "year", "day"]),
        metrics=metrics.sort_values("rmse"),
        per_field=per_field,
    )


def _plot_stack_rmse_bar(stack_data: StackData, out_path: Path) -> None:
    m = stack_data.metrics.copy().sort_values("rmse")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(m["model"], m["rmse"], color=[MODEL_COLORS[x] for x in m["model"]])
    for i, v in enumerate(m["rmse"]):
        ax.text(i, float(v) + 0.03, f"{float(v):.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{stack_data.stack}: RMSE ({stack_data.species}, {stack_data.run_id})")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_stack_cumulative_rmse(stack_data: StackData, out_path: Path, out_csv: Path) -> None:
    df = stack_data.aligned.copy()
    df["se"] = np.square(df["y_pred"] - df["y_true"])
    daily = (
        df.groupby(["model", "day"], as_index=False)
        .agg(se_sum=("se", "sum"), n=("se", "size"))
        .sort_values(["model", "day"])
    )
    daily["cum_se"] = daily.groupby("model")["se_sum"].cumsum()
    daily["cum_n"] = daily.groupby("model")["n"].cumsum()
    daily["cum_rmse"] = np.sqrt(daily["cum_se"] / daily["cum_n"])
    daily.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for model in MODEL_ORDER:
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["cum_rmse"], lw=2.0, color=MODEL_COLORS[model], label=model)
    ax.set_title(f"{stack_data.stack}: Cumulative RMSE over Day")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Cumulative sample-weighted RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_stack_daily_rmse(stack_data: StackData, out_path: Path, out_csv: Path) -> None:
    df = stack_data.aligned.copy()
    df["se"] = np.square(df["y_pred"] - df["y_true"])
    daily = (
        df.groupby(["model", "day"], as_index=False)
        .agg(mse=("se", "mean"), n=("se", "size"))
        .sort_values(["model", "day"])
    )
    daily["rmse"] = np.sqrt(daily["mse"])
    daily["rmse_roll14"] = (
        daily.groupby("model", as_index=False)["rmse"]
        .transform(lambda s: s.rolling(window=14, min_periods=1, center=True).mean())
    )
    daily.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for model in MODEL_ORDER:
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["rmse_roll14"], lw=2.0, color=MODEL_COLORS[model], label=model)
    ax.set_title(f"{stack_data.stack}: Daily RMSE Time Series (14-day rolling)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Daily RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _stack_daily_means(stack_data: StackData) -> pd.DataFrame:
    return (
        stack_data.aligned.groupby(["model", "day"], as_index=False)[["y_true", "y_pred"]]
        .mean()
        .sort_values(["model", "day"])
        .reset_index(drop=True)
    )


def _stack_daily_totals(stack_data: StackData) -> pd.DataFrame:
    return (
        stack_data.aligned.groupby(["model", "day"], as_index=False)
        .agg(
            y_true_total=("y_true", "sum"),
            y_pred_total=("y_pred", "sum"),
            n_records=("y_true", "size"),
            y_true_mean=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
        )
        .sort_values(["model", "day"])
        .reset_index(drop=True)
    )


def _plot_stack_daily_mean_overlay(stack_data: StackData, out_path: Path, out_csv: Path) -> None:
    daily = _stack_daily_means(stack_data)
    daily.to_csv(out_csv, index=False)

    gt = daily.groupby("day", as_index=False)["y_true"].mean()
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(gt["day"], gt["y_true"], color="black", lw=2.4, label="Ground Truth")
    for model in MODEL_ORDER:
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["y_pred"], lw=2.0, color=MODEL_COLORS[model], label=model)
    ax.set_title(f"{stack_data.stack}: Daily Mean Trajectories ({stack_data.species}, {stack_data.run_id})")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Daily mean value")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_stack_daily_total_overlay(stack_data: StackData, out_path: Path, out_csv: Path) -> None:
    daily = _stack_daily_totals(stack_data)
    daily.to_csv(out_csv, index=False)

    # y_true_total and n_records are identical across models by construction.
    gt = (
        daily.groupby("day", as_index=False)
        .agg(y_true_total=("y_true_total", "mean"), n_records=("n_records", "mean"))
        .sort_values("day")
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    ax1, ax2 = axes
    ax1.plot(gt["day"], gt["y_true_total"], color="black", lw=2.4, label="Ground Truth total")
    for model in MODEL_ORDER:
        g = daily[daily["model"] == model]
        ax1.plot(g["day"], g["y_pred_total"], lw=2.0, color=MODEL_COLORS[model], label=f"{model} predicted total")
    ax1.set_title(f"{stack_data.stack}: Daily Total Counts ({stack_data.species}, {stack_data.run_id})")
    ax1.set_ylabel("Total count per day")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))

    ax2.plot(gt["day"], gt["n_records"], color="#4d4d4d", lw=1.8)
    ax2.set_xlabel("Day of year")
    ax2.set_ylabel("Records per day")
    ax2.set_title("Sample count per day (context for absolute totals)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_stack_daily_total_panels(stack_data: StackData, out_path: Path) -> None:
    daily = _stack_daily_totals(stack_data)
    gt = daily.groupby("day", as_index=False)["y_true_total"].mean().sort_values("day")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        g = daily[daily["model"] == model]
        ax.plot(gt["day"], gt["y_true_total"], color="black", lw=2.2, label="Ground Truth total")
        ax.plot(g["day"], g["y_pred_total"], color=MODEL_COLORS[model], lw=2.0, label=f"{model} total")
        ax.set_title(model)
        ax.grid(alpha=0.25)
        if i in (2, 3):
            ax.set_xlabel("Day of year")
        if i in (0, 2):
            ax.set_ylabel("Total count per day")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{stack_data.stack}: Daily Total Counts by Model", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_stack_daily_mean_panels(stack_data: StackData, out_path: Path) -> None:
    daily = _stack_daily_means(stack_data)
    gt = daily.groupby("day", as_index=False)["y_true"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        g = daily[daily["model"] == model]
        ax.plot(gt["day"], gt["y_true"], color="black", lw=2.2, label="Ground Truth")
        ax.plot(g["day"], g["y_pred"], color=MODEL_COLORS[model], lw=2.0, label=model)
        ax.set_title(model)
        ax.grid(alpha=0.25)
        if i in (2, 3):
            ax.set_xlabel("Day of year")
        if i in (0, 2):
            ax.set_ylabel("Daily mean value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{stack_data.stack}: Daily Mean Comparison by Model", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_stack_wins(stack_data: StackData, out_path: Path) -> pd.DataFrame:
    wins = (
        stack_data.per_field["winner"]
        .value_counts()
        .reindex(MODEL_ORDER, fill_value=0)
        .rename_axis("model")
        .reset_index(name="wins")
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(wins["model"], wins["wins"], color=[MODEL_COLORS[x] for x in wins["model"]])
    for i, v in enumerate(wins["wins"]):
        ax.text(i, int(v) + 0.2, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{stack_data.stack}: Field Wins")
    ax.set_ylabel("Number of fields won")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return wins


def _plot_rmse_grouped(all_metrics: pd.DataFrame, out_path: Path) -> None:
    stacks = sorted(all_metrics["stack"].unique().tolist())
    x = np.arange(len(stacks))
    width = 0.2
    offsets = {"CNN": -1.5 * width, "MLP": -0.5 * width, "Baseline": 0.5 * width, "GNN": 1.5 * width}

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for model in MODEL_ORDER:
        vals = []
        for s in stacks:
            row = all_metrics[(all_metrics["stack"] == s) & (all_metrics["model"] == model)]
            vals.append(float(row["rmse"].iloc[0]))
        ax.bar(x + offsets[model], vals, width=width, color=MODEL_COLORS[model], label=model)
    ax.set_xticks(x, stacks)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Stack and Model (Corrected GNN Transform)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_species_summary(all_metrics: pd.DataFrame, out_path: Path) -> None:
    sp = (
        all_metrics.groupby(["species", "model"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), n_stacks=("stack", "nunique"))
        .sort_values(["species", "rmse_mean"])
    )
    species = sorted(sp["species"].unique().tolist())
    fig, axes = plt.subplots(1, len(species), figsize=(6.2 * max(1, len(species)), 4.8), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, s in zip(axes, species):
        g = sp[sp["species"] == s]
        ax.bar(g["model"], g["rmse_mean"], color=[MODEL_COLORS[x] for x in g["model"]])
        for i, (_, row) in enumerate(g.iterrows()):
            err = 0.0 if np.isnan(float(row["rmse_std"])) else float(row["rmse_std"])
            if err > 0:
                ax.errorbar(i, float(row["rmse_mean"]), yerr=err, fmt="none", ecolor="black", capsize=3)
        ax.set_title(f"{s} (n={int(g['n_stacks'].max())} stack(s))")
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Species-Level RMSE Summary Across Stacks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stack_dirs = sorted([d for d in args.stack_root.iterdir() if d.is_dir() and d.name.startswith(args.stack_prefix)])
    if not stack_dirs:
        raise SystemExit(f"No stack directories found in {args.stack_root} with prefix '{args.stack_prefix}'")

    all_stack_data: list[StackData] = []
    for stack_dir in stack_dirs:
        sd = _load_stack(stack_dir)
        all_stack_data.append(sd)

        stack_out = output_dir / sd.stack
        stack_out.mkdir(parents=True, exist_ok=True)
        sd.aligned.to_csv(stack_out / "aligned_predictions_long_corrected.csv", index=False)
        sd.metrics.to_csv(stack_out / "subset_metrics_recomputed_corrected.csv", index=False)
        sd.per_field.to_csv(stack_out / "per_field_rmse_corrected.csv", index=False)

        _plot_stack_rmse_bar(sd, stack_out / "plot_stack_rmse_bar.png")
        _plot_stack_cumulative_rmse(
            sd,
            stack_out / "plot_stack_cumulative_rmse.png",
            stack_out / "table_stack_cumulative_rmse.csv",
        )
        _plot_stack_daily_rmse(
            sd,
            stack_out / "plot_stack_daily_rmse_timeseries.png",
            stack_out / "table_stack_daily_rmse_timeseries.csv",
        )
        _plot_stack_daily_mean_overlay(
            sd,
            stack_out / "plot_stack_daily_mean_overlay.png",
            stack_out / "table_stack_daily_mean_trajectories.csv",
        )
        _plot_stack_daily_mean_panels(sd, stack_out / "plot_stack_daily_mean_panels.png")
        _plot_stack_daily_total_overlay(
            sd,
            stack_out / "plot_stack_daily_total_counts_overlay.png",
            stack_out / "table_stack_daily_total_counts.csv",
        )
        _plot_stack_daily_total_panels(sd, stack_out / "plot_stack_daily_total_counts_panels.png")
        wins = _plot_stack_wins(sd, stack_out / "plot_stack_field_wins.png")
        wins.to_csv(stack_out / "table_stack_field_wins.csv", index=False)

    all_metrics = pd.concat([s.metrics for s in all_stack_data], ignore_index=True)
    all_fields = pd.concat([s.per_field for s in all_stack_data], ignore_index=True)
    all_metrics.to_csv(output_dir / "all_stacks_metrics.csv", index=False)
    all_fields.to_csv(output_dir / "all_stacks_per_field_rmse.csv", index=False)

    _plot_rmse_grouped(all_metrics, output_dir / "plot_all_stacks_rmse_grouped.png")
    _plot_species_summary(all_metrics, output_dir / "plot_species_rmse_summary.png")

    run_summary = (
        all_metrics.sort_values(["stack", "rmse"])
        .groupby(["stack", "species", "run_id"], as_index=False)
        .agg(best_model=("model", "first"), best_rmse=("rmse", "first"))
    )
    run_summary.to_csv(output_dir / "run_summary_best_model.csv", index=False)

    print(f"Saved stack analysis to: {output_dir.resolve()}")
    print("Stacks processed:", ", ".join([s.stack for s in all_stack_data]))


if __name__ == "__main__":
    main()
