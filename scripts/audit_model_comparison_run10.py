from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
class LoadedModel:
    model: str
    frame: pd.DataFrame
    n_rows: int
    n_unique_keys: int
    has_duplicate_keys: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and correct model-comparison plotting workflow for run10 subset."
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "outputs/final_ref/pred_baseline_Rebhuhn_h365_run10_baseline_test.csv"
        ),
    )
    parser.add_argument(
        "--mlp-csv",
        type=Path,
        default=Path("outputs/final_ref/pred_mlp_TRUE_Rebhuhn_h365_run10_baseline_test.csv"),
    )
    parser.add_argument(
        "--cnn-csv",
        type=Path,
        default=Path(
            "outputs/final_ref/pred_cnn_TRUE_Rebhuhn_h365_run10_baseline_test_debug.csv"
        ),
    )
    parser.add_argument(
        "--gnn-csv",
        type=Path,
        default=Path(
            "outputs/final_ref/pred_gnn_TRUE_Rebhuhn_h365_run10_baseline_test_debug.csv"
        ),
    )
    parser.add_argument(
        "--subset-metrics-csv",
        type=Path,
        default=Path(
            "outputs/final_ref/subset_metrics_Rebhuhn_h365_run10_baseline_test.csv"
        ),
    )
    parser.add_argument(
        "--daily-means-csv",
        type=Path,
        default=Path(
            "outputs/final_ref/daily_mean_trajectories_Rebhuhn_h365_run10_baseline_test.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_ref/audit_results"),
    )
    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=120000,
    )
    return parser.parse_args()


def require_columns(path: Path, required: Iterable[str]) -> None:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def load_baseline(path: Path) -> LoadedModel:
    required = [*KEY_COLS, "y_true_level", "y_pred_level"]
    require_columns(path, required)
    frame = pd.read_csv(path, usecols=required).copy()
    frame["model"] = "Baseline"
    frame = frame.rename(columns={"y_true_level": "y_true", "y_pred_level": "y_pred"})
    return finalize_loaded("Baseline", frame)


def load_mlp(path: Path) -> LoadedModel:
    required = [*KEY_COLS, "y_true_level", "y_pred_level"]
    require_columns(path, required)
    frame = pd.read_csv(path, usecols=required).copy()
    frame["model"] = "MLP"
    frame = frame.rename(columns={"y_true_level": "y_true", "y_pred_level": "y_pred"})
    return finalize_loaded("MLP", frame)


def load_cnn(path: Path) -> LoadedModel:
    required = [*KEY_COLS, "y_true_lvl", "y_pred_lvl"]
    require_columns(path, required)
    frame = pd.read_csv(path, usecols=required).copy()
    frame["model"] = "CNN"
    frame = frame.rename(columns={"y_true_lvl": "y_true", "y_pred_lvl": "y_pred"})
    return finalize_loaded("CNN", frame)


def load_gnn(path: Path) -> LoadedModel:
    required = [*KEY_COLS, "N_total", "N_total_tplus365", "y_pred_lvl"]
    require_columns(path, required)
    raw = pd.read_csv(path, usecols=required).copy()
    # Correct inverse transform from log-delta-like model output to count-level prediction.
    y_pred_level = np.expm1(np.log1p(raw["N_total"].to_numpy(dtype=float)) + raw["y_pred_lvl"].to_numpy(dtype=float))
    y_pred_level = np.maximum(y_pred_level, 0.0)
    frame = raw[KEY_COLS].copy()
    frame["model"] = "GNN"
    frame["y_true"] = raw["N_total_tplus365"].to_numpy(dtype=float)
    frame["y_pred"] = y_pred_level
    return finalize_loaded("GNN", frame)


def finalize_loaded(model: str, frame: pd.DataFrame) -> LoadedModel:
    frame = frame[["model", *KEY_COLS, "y_true", "y_pred"]].copy()
    n_rows = len(frame)
    n_unique_keys = frame[KEY_COLS].drop_duplicates().shape[0]
    has_dups = n_rows != n_unique_keys
    if has_dups:
        raise ValueError(
            f"{model} contains duplicate keys ({n_rows} rows but {n_unique_keys} unique keys). "
            "Refusing to continue with ambiguous sample alignment."
        )
    return LoadedModel(
        model=model,
        frame=frame,
        n_rows=n_rows,
        n_unique_keys=n_unique_keys,
        has_duplicate_keys=has_dups,
    )


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for model in MODEL_ORDER:
        g = df[df["model"] == model]
        if g.empty:
            continue
        y = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred"].to_numpy(dtype=float)
        err = p - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(np.square(err))))
        ss_res = float(np.sum(np.square(err)))
        ss_tot = float(np.sum(np.square(y - np.mean(y))))
        r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
        rows.append(
            {
                "model": model,
                "rows": len(g),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values("model").reset_index(drop=True)


def build_aligned_dataset(loaded: list[LoadedModel]) -> pd.DataFrame:
    key_base = loaded[0].frame[KEY_COLS].copy()
    for item in loaded[1:]:
        key_base = key_base.merge(item.frame[KEY_COLS], on=KEY_COLS, how="inner")
    aligned_keys = key_base.drop_duplicates().copy()

    aligned_frames = []
    for item in loaded:
        aligned = aligned_keys.merge(item.frame, on=KEY_COLS, how="left")
        aligned_frames.append(aligned)
    df = pd.concat(aligned_frames, ignore_index=True)
    return df.sort_values(["model", "run_id", "field_id", "year", "day"]).reset_index(drop=True)


def compare_ground_truth(df: pd.DataFrame) -> float:
    pivot = df.pivot_table(index=KEY_COLS, columns="model", values="y_true", aggfunc="first")
    max_diff = 0.0
    cols = [c for c in MODEL_ORDER if c in pivot.columns]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            max_diff = max(max_diff, float((pivot[cols[i]] - pivot[cols[j]]).abs().max()))
    return max_diff


def compute_per_field_rmse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field_id, gf in df.groupby("field_id"):
        entry: dict[str, float | int | str] = {"field_id": int(field_id)}
        best_model = None
        best_rmse = float("inf")
        for model in MODEL_ORDER:
            g = gf[gf["model"] == model]
            y = g["y_true"].to_numpy(dtype=float)
            p = g["y_pred"].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean(np.square(p - y))))
            entry[f"rmse_{model}"] = rmse
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
        entry["winner"] = best_model if best_model is not None else ""
        rows.append(entry)
    out = pd.DataFrame(rows).sort_values("field_id").reset_index(drop=True)
    return out


def compute_daily_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "day"], as_index=False)[["y_true", "y_pred"]]
        .mean()
        .sort_values(["model", "day"])
        .reset_index(drop=True)
    )


def compute_daily_mean_rmse(daily_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for model in MODEL_ORDER:
        g = daily_df[daily_df["model"] == model]
        if g.empty:
            continue
        rmse = float(np.sqrt(np.mean(np.square(g["y_pred"].to_numpy() - g["y_true"].to_numpy()))))
        rows.append({"model": model, "daily_mean_rmse": rmse, "daily_points": len(g)})
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values("model").reset_index(drop=True)


def compute_field_daily_winners(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field_id, gf in df.groupby("field_id"):
        entry: dict[str, float | int | str] = {"field_id": int(field_id)}
        best_model = None
        best_rmse = float("inf")
        for model in MODEL_ORDER:
            g = gf[gf["model"] == model]
            daily = g.groupby("day", as_index=False)[["y_true", "y_pred"]].mean()
            rmse = float(np.sqrt(np.mean(np.square(daily["y_pred"].to_numpy() - daily["y_true"].to_numpy()))))
            entry[f"daily_rmse_{model}"] = rmse
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
        entry["daily_winner"] = best_model if best_model is not None else ""
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("field_id").reset_index(drop=True)


def choose_example_fields(per_field: pd.DataFrame, daily_compare: pd.DataFrame) -> list[int]:
    merged = per_field.merge(daily_compare[["field_id", "daily_winner"]], on="field_id", how="left")

    # Field where CNN has its strongest margin over MLP (if any).
    cnn_margin = merged.assign(cnn_vs_mlp=merged["rmse_MLP"] - merged["rmse_CNN"]).sort_values(
        "cnn_vs_mlp", ascending=False
    )
    field_cnn = int(cnn_margin.iloc[0]["field_id"])

    # Field where daily averaging favors GNN while per-sample winner is not GNN.
    candidates = merged[merged["daily_winner"].eq("GNN") & ~merged["winner"].eq("GNN")].copy()
    if not candidates.empty:
        candidates["gnn_gap"] = candidates["rmse_GNN"] - candidates[
            ["rmse_CNN", "rmse_MLP", "rmse_Baseline"]
        ].min(axis=1)
        field_misleading = int(candidates.sort_values("gnn_gap", ascending=False).iloc[0]["field_id"])
    else:
        field_misleading = field_cnn

    return sorted({field_cnn, field_misleading})


def plot_rmse_bar(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    m = metrics.copy().sort_values("rmse")
    ax.bar(m["model"], m["rmse"], color=[MODEL_COLORS[x] for x in m["model"]])
    for i, val in enumerate(m["rmse"]):
        ax.text(i, float(val) + 0.03, f"{float(val):.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Subset RMSE by Model (Corrected, Level Scale)")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_wins_bar(per_field: pd.DataFrame, output_path: Path) -> None:
    counts = per_field["winner"].value_counts().reindex(MODEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(counts.index, counts.values, color=[MODEL_COLORS[x] for x in counts.index])
    for i, val in enumerate(counts.values):
        ax.text(i, int(val) + 0.2, str(int(val)), ha="center", va="bottom", fontsize=9)
    ax.set_title("Wins per Field (Lowest Per-Sample RMSE)")
    ax.set_ylabel("Number of Fields")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_panels(df: pd.DataFrame, metrics: pd.DataFrame, output_path: Path, max_points: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes_flat = axes.ravel()

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes_flat[idx]
        g = df[df["model"] == model]
        if len(g) > max_points:
            g = g.sample(n=max_points, random_state=42)
        hb = ax.hexbin(g["y_true"], g["y_pred"], gridsize=55, mincnt=1, cmap="viridis")
        min_xy = float(min(g["y_true"].min(), g["y_pred"].min()))
        max_xy = float(max(g["y_true"].max(), g["y_pred"].max()))
        ax.plot([min_xy, max_xy], [min_xy, max_xy], "--", color="tomato", lw=1.2)
        ax.set_title(model)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        m = metrics[metrics["model"] == model].iloc[0]
        ax.text(
            0.03,
            0.97,
            f"RMSE={float(m['rmse']):.3f}\nMAE={float(m['mae']):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
        )
        fig.colorbar(hb, ax=ax, label="Point density")

    fig.suptitle("Prediction vs Ground Truth (Per-Sample, Corrected)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_example_fields(df: pd.DataFrame, per_field: pd.DataFrame, field_ids: list[int], output_path: Path) -> None:
    if not field_ids:
        return
    n = len(field_ids)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.5 * n), sharex=True)
    axes = np.atleast_1d(axes)

    per_field_idx = per_field.set_index("field_id")
    for ax, field_id in zip(axes, field_ids):
        gf = df[df["field_id"] == field_id].copy()
        daily = gf.groupby(["model", "day"], as_index=False)[["y_true", "y_pred"]].mean()
        gt = daily.groupby("day", as_index=False)["y_true"].mean()
        ax.plot(gt["day"], gt["y_true"], color="black", lw=2.3, label="Ground Truth")
        for model in MODEL_ORDER:
            gm = daily[daily["model"] == model]
            ax.plot(gm["day"], gm["y_pred"], color=MODEL_COLORS[model], lw=1.9, label=f"{model}")
        if field_id in per_field_idx.index:
            row = per_field_idx.loc[field_id]
            label = (
                f"RMSE: CNN={row['rmse_CNN']:.3f}, MLP={row['rmse_MLP']:.3f}, "
                f"Baseline={row['rmse_Baseline']:.3f}, GNN={row['rmse_GNN']:.3f}"
            )
            ax.text(
                0.01,
                0.97,
                label,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
            )
        ax.set_title(f"Field {field_id}")
        ax.set_ylabel("Daily mean value")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Day of year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Example Field Trajectories (Daily Means for Visualization)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    files: dict[str, Path],
    schema_audit: pd.DataFrame,
    metrics: pd.DataFrame,
    provided_metrics: pd.DataFrame | None,
    compare_metrics: pd.DataFrame | None,
    daily_mean_rmse: pd.DataFrame,
    wins: pd.DataFrame,
    winner_compare: pd.DataFrame,
    gt_max_diff: float,
    example_fields: list[int],
    input_daily_rmse: pd.DataFrame | None = None,
    input_daily_diff: pd.DataFrame | None = None,
) -> None:
    def frame_block(df: pd.DataFrame) -> str:
        return "```\n" + df.to_string(index=False) + "\n```"

    lines: list[str] = []
    lines.append("# BirdSim Run10 Model-Comparison Audit")
    lines.append("")
    lines.append("## Files Used")
    for label, path in files.items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    lines.append("## Key Checks")
    lines.append(f"- Maximum ground-truth mismatch across models on aligned samples: `{gt_max_diff:.12f}`")
    lines.append(
        "- GNN prediction transformed with: `y_pred = max(expm1(log1p(N_total) + y_pred_lvl), 0)`"
    )
    lines.append("- All model files were aligned on identical sample keys before computing metrics.")
    lines.append("")
    lines.append("## Recomputed Per-Sample Metrics")
    lines.append(frame_block(metrics))
    lines.append("")
    if provided_metrics is not None:
        lines.append("## Provided Subset Metrics (Input File)")
        lines.append(frame_block(provided_metrics))
        lines.append("")
    if compare_metrics is not None:
        lines.append("## Recomputed vs Provided (Delta = recomputed - provided)")
        lines.append(frame_block(compare_metrics))
        lines.append("")
    lines.append("## Daily-Mean RMSE (Diagnostic Only, Potentially Misleading)")
    lines.append(frame_block(daily_mean_rmse))
    lines.append("")
    if input_daily_rmse is not None:
        lines.append("## Input Daily-Mean RMSE (From Provided Daily-Mean CSV)")
        lines.append(frame_block(input_daily_rmse))
        lines.append("")
    if input_daily_diff is not None:
        lines.append("## Input vs Corrected Daily-Mean Differences")
        lines.append(frame_block(input_daily_diff))
        lines.append("")
    lines.append("## Field Wins (Per-Sample RMSE)")
    lines.append(frame_block(wins))
    lines.append("")
    lines.append("## Sample Winner vs Daily-Mean Winner by Field")
    lines.append(frame_block(winner_compare))
    lines.append("")
    if example_fields:
        lines.append("## Suggested Example Fields")
        lines.append(f"- {', '.join(str(x) for x in example_fields)}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("- Original logic mixed comparable and non-comparable summaries (per-sample metrics vs daily-mean trajectory impressions).")
    lines.append("- GNN was previously transformed with `N_total * exp(y_pred_lvl)`; this is close but not identical to the required inverse transform.")
    lines.append("- Daily means hide large per-sample errors and can visually favor a model with worse RMSE.")
    lines.append("- For presentation ranking, use per-sample RMSE bar chart and per-field wins based on per-sample RMSE.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "baseline_csv": args.baseline_csv,
        "mlp_csv": args.mlp_csv,
        "cnn_csv": args.cnn_csv,
        "gnn_csv": args.gnn_csv,
        "subset_metrics_csv": args.subset_metrics_csv,
        "daily_means_csv": args.daily_means_csv,
    }
    for label, path in files.items():
        if label in {"subset_metrics_csv", "daily_means_csv"}:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    baseline = load_baseline(args.baseline_csv)
    mlp = load_mlp(args.mlp_csv)
    cnn = load_cnn(args.cnn_csv)
    gnn = load_gnn(args.gnn_csv)
    loaded = [baseline, mlp, cnn, gnn]

    schema_audit = pd.DataFrame(
        [
            {
                "model": item.model,
                "rows": item.n_rows,
                "unique_keys": item.n_unique_keys,
                "duplicate_keys": item.has_duplicate_keys,
            }
            for item in loaded
        ]
    )
    schema_audit.to_csv(output_dir / "schema_audit.csv", index=False)

    aligned = build_aligned_dataset(loaded)
    aligned.to_csv(output_dir / "aligned_predictions_long_corrected.csv", index=False)

    gt_max_diff = compare_ground_truth(aligned)
    metrics = compute_metrics(aligned)
    metrics.to_csv(output_dir / "subset_metrics_recomputed_corrected.csv", index=False)

    provided_metrics: pd.DataFrame | None = None
    compare_metrics: pd.DataFrame | None = None
    if args.subset_metrics_csv.exists():
        provided_metrics = pd.read_csv(args.subset_metrics_csv)
        provided_metrics.to_csv(output_dir / "subset_metrics_provided_input.csv", index=False)
        if {"model", "rmse", "mae", "r2"}.issubset(provided_metrics.columns):
            p = provided_metrics[["model", "rmse", "mae", "r2"]].copy()
            p = p.rename(columns={"rmse": "rmse_provided", "mae": "mae_provided", "r2": "r2_provided"})
            compare_metrics = metrics.merge(p, on="model", how="left")
            compare_metrics["rmse_delta"] = compare_metrics["rmse"] - compare_metrics["rmse_provided"]
            compare_metrics["mae_delta"] = compare_metrics["mae"] - compare_metrics["mae_provided"]
            compare_metrics["r2_delta"] = compare_metrics["r2"] - compare_metrics["r2_provided"]
            compare_metrics.to_csv(output_dir / "subset_metrics_compare_recomputed_vs_provided.csv", index=False)

    per_field = compute_per_field_rmse(aligned)
    per_field.to_csv(output_dir / "per_field_rmse_corrected.csv", index=False)
    wins = (
        per_field["winner"]
        .value_counts()
        .reindex(MODEL_ORDER, fill_value=0)
        .rename_axis("model")
        .reset_index(name="wins")
    )
    wins.to_csv(output_dir / "wins_per_field.csv", index=False)

    daily_means = compute_daily_means(aligned)
    daily_means.to_csv(output_dir / "daily_mean_trajectories_corrected.csv", index=False)
    daily_mean_rmse = compute_daily_mean_rmse(daily_means)
    daily_mean_rmse.to_csv(output_dir / "daily_mean_rmse_diagnostic.csv", index=False)

    input_daily_rmse: pd.DataFrame | None = None
    input_daily_diff: pd.DataFrame | None = None
    if args.daily_means_csv.exists():
        input_daily = pd.read_csv(args.daily_means_csv).copy()
        rename_map = {}
        if "y_true_level" in input_daily.columns:
            rename_map["y_true_level"] = "y_true"
        if "y_pred_level" in input_daily.columns:
            rename_map["y_pred_level"] = "y_pred"
        input_daily = input_daily.rename(columns=rename_map)
        if {"model", "day", "y_true", "y_pred"}.issubset(input_daily.columns):
            input_daily = input_daily[["model", "day", "y_true", "y_pred"]].copy()
            input_daily_rmse = compute_daily_mean_rmse(input_daily)
            input_daily_rmse.to_csv(output_dir / "daily_mean_rmse_input_file.csv", index=False)

            cmp = daily_means.merge(
                input_daily,
                on=["model", "day"],
                how="inner",
                suffixes=("_corrected", "_input"),
            )
            input_daily_diff = (
                cmp.groupby("model", as_index=False)
                .agg(
                    points=("day", "size"),
                    max_abs_y_true_diff=("y_true_corrected", lambda s: float(np.max(np.abs(s - cmp.loc[s.index, "y_true_input"])))),
                    max_abs_y_pred_diff=("y_pred_corrected", lambda s: float(np.max(np.abs(s - cmp.loc[s.index, "y_pred_input"])))),
                    mean_abs_y_pred_diff=("y_pred_corrected", lambda s: float(np.mean(np.abs(s - cmp.loc[s.index, "y_pred_input"])))),
                )
                .sort_values("model")
                .reset_index(drop=True)
            )
            input_daily_diff.to_csv(output_dir / "daily_mean_input_vs_corrected_diff.csv", index=False)

    field_daily_winners = compute_field_daily_winners(aligned)
    field_daily_winners.to_csv(output_dir / "per_field_daily_mean_rmse_diagnostic.csv", index=False)
    winner_compare = per_field[["field_id", "winner"]].merge(
        field_daily_winners[["field_id", "daily_winner"]],
        on="field_id",
        how="left",
    )
    winner_compare["winner_changed"] = winner_compare["winner"] != winner_compare["daily_winner"]
    winner_compare.to_csv(output_dir / "field_winner_sample_vs_daily_mean.csv", index=False)

    example_fields = choose_example_fields(per_field, field_daily_winners)

    plot_rmse_bar(metrics, output_dir / "plot_subset_rmse_bar_corrected.png")
    plot_wins_bar(per_field, output_dir / "plot_wins_per_field_corrected.png")
    plot_scatter_panels(
        aligned,
        metrics,
        output_dir / "plot_scatter_prediction_vs_ground_truth_corrected.png",
        max_points=args.scatter_max_points,
    )
    plot_example_fields(
        aligned,
        per_field,
        example_fields,
        output_dir / "plot_example_field_trajectories_corrected.png",
    )

    write_report(
        output_path=output_dir / "audit_report.md",
        files=files,
        schema_audit=schema_audit,
        metrics=metrics,
        provided_metrics=provided_metrics,
        compare_metrics=compare_metrics,
        daily_mean_rmse=daily_mean_rmse,
        wins=wins,
        winner_compare=winner_compare,
        gt_max_diff=gt_max_diff,
        example_fields=example_fields,
        input_daily_rmse=input_daily_rmse,
        input_daily_diff=input_daily_diff,
    )

    print(f"Saved audit outputs to: {output_dir.resolve()}")
    print(f"Aligned rows per model: {metrics[['model', 'rows']].to_dict(orient='records')}")
    print(f"Ground-truth max pairwise diff: {gt_max_diff:.12f}")
    print("Recomputed RMSE by model:")
    for _, row in metrics.iterrows():
        print(f"- {row['model']}: {row['rmse']:.6f}")


if __name__ == "__main__":
    main()
