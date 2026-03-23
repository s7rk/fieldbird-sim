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
        description="Generate presentation-friendly, non-misleading comparison plots."
    )
    parser.add_argument(
        "--aligned-csv",
        type=Path,
        default=Path("outputs/final_ref/audit_results/aligned_predictions_long_corrected.csv"),
    )
    parser.add_argument(
        "--per-field-csv",
        type=Path,
        default=Path("outputs/final_ref/audit_results/per_field_rmse_corrected.csv"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/final_ref/audit_results/subset_metrics_recomputed_corrected.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_ref/audit_results/presentation_pack"),
    )
    return parser.parse_args()


def plot_rmse_bar(metrics: pd.DataFrame, out: Path) -> None:
    m = metrics.copy()
    m["model"] = pd.Categorical(m["model"], categories=MODEL_ORDER, ordered=True)
    m = m.sort_values("rmse")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(m["model"], m["rmse"], color=[MODEL_COLORS[x] for x in m["model"]])
    for i, val in enumerate(m["rmse"]):
        ax.text(i, float(val) + 0.03, f"{float(val):.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Subset RMSE Ranking (Lower is Better)")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_abs_error_ecdf(aligned: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    all_abs = np.abs(aligned["y_pred"] - aligned["y_true"]).to_numpy()
    xmax = float(np.quantile(all_abs, 0.995))

    for model in MODEL_ORDER:
        g = aligned[aligned["model"] == model]
        err = np.sort(np.abs(g["y_pred"].to_numpy() - g["y_true"].to_numpy()))
        y = np.arange(1, len(err) + 1) / len(err)
        ax.plot(err, y, lw=2.0, color=MODEL_COLORS[model], label=model)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Cumulative share of samples")
    ax.set_title("Error Distribution (ECDF)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_field_rmse_strip(per_field: pd.DataFrame, out: Path) -> None:
    long = per_field.melt(
        id_vars=["field_id"],
        value_vars=[f"rmse_{m}" for m in MODEL_ORDER],
        var_name="model",
        value_name="rmse",
    )
    long["model"] = long["model"].str.replace("rmse_", "", regex=False)
    long["model"] = pd.Categorical(long["model"], categories=MODEL_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x_positions = {m: i for i, m in enumerate(MODEL_ORDER)}
    rng = np.random.default_rng(42)

    for model in MODEL_ORDER:
        g = long[long["model"] == model]
        x = np.full(len(g), x_positions[model], dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(g))
        ax.scatter(
            x + jitter,
            g["rmse"],
            s=26,
            alpha=0.75,
            color=MODEL_COLORS[model],
            edgecolors="none",
            label=model,
        )

    med = long.groupby("model", as_index=False)["rmse"].median()
    for _, row in med.iterrows():
        x = x_positions[str(row["model"])]
        ax.hlines(float(row["rmse"]), x - 0.22, x + 0.22, colors="black", linewidth=2.0)

    ax.set_xticks(list(x_positions.values()), list(x_positions.keys()))
    ax.set_ylabel("Field RMSE")
    ax.set_title("RMSE per Field (Dots) with Median Lines")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_wins_bar(per_field: pd.DataFrame, out: Path) -> pd.DataFrame:
    wins = (
        per_field["winner"].value_counts().reindex(MODEL_ORDER, fill_value=0).rename_axis("model").reset_index(name="wins")
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(wins["model"], wins["wins"], color=[MODEL_COLORS[m] for m in wins["model"]])
    for i, val in enumerate(wins["wins"]):
        ax.text(i, int(val) + 0.2, str(int(val)), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of fields won")
    ax.set_title("Field Wins (Lowest Per-Sample RMSE)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return wins


def plot_cnn_vs_mlp_mse_delta(aligned: pd.DataFrame, out: Path, out_csv: Path) -> None:
    df = aligned[aligned["model"].isin(["CNN", "MLP"])].copy()
    df["sq_err"] = np.square(df["y_pred"] - df["y_true"])
    mse = df.groupby(["field_id", "model"], as_index=False)["sq_err"].mean()
    piv = mse.pivot(index="field_id", columns="model", values="sq_err").reset_index()
    piv["delta_mse_cnn_minus_mlp"] = piv["CNN"] - piv["MLP"]
    piv = piv.sort_values("delta_mse_cnn_minus_mlp")
    piv.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = np.where(piv["delta_mse_cnn_minus_mlp"] <= 0, MODEL_COLORS["CNN"], MODEL_COLORS["MLP"])
    ax.bar(piv["field_id"].astype(str), piv["delta_mse_cnn_minus_mlp"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Field ID (sorted by delta)")
    ax.set_ylabel("MSE(CNN) - MSE(MLP)")
    ax.set_title("Fieldwise MSE Difference: CNN vs MLP")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_daily_rmse_timeseries(aligned: pd.DataFrame, out: Path, out_csv: Path) -> None:
    df = aligned.copy()
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

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for model in MODEL_ORDER:
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["rmse_roll14"], lw=2.2, color=MODEL_COLORS[model], label=model)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Daily RMSE (14-day rolling mean)")
    ax.set_title("Time-Series View: Daily RMSE by Model")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_rmse_timeseries(aligned: pd.DataFrame, out: Path, out_csv: Path) -> None:
    df = aligned.copy()
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

    piv = daily.pivot(index="day", columns="model", values="cum_rmse")
    crossing_day = None
    if {"CNN", "MLP"}.issubset(piv.columns):
        state = (piv["CNN"] < piv["MLP"]).astype(int)
        changes = state.diff().fillna(0)
        change_days = state.index[changes != 0].tolist()
        if change_days:
            crossing_day = int(change_days[0])

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for model in MODEL_ORDER:
        g = daily[daily["model"] == model]
        ax.plot(g["day"], g["cum_rmse"], lw=2.2, color=MODEL_COLORS[model], label=model)
    if crossing_day is not None:
        ax.axvline(crossing_day, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.text(
            crossing_day + 2,
            ax.get_ylim()[1] - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"CNN/MLP crossing at day {crossing_day}",
            fontsize=9,
            va="top",
        )
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Cumulative sample-weighted RMSE")
    ax.set_title("Time-Series View: Cumulative RMSE by Model")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned = pd.read_csv(args.aligned_csv)
    per_field = pd.read_csv(args.per_field_csv)
    metrics = pd.read_csv(args.metrics_csv)

    plot_rmse_bar(metrics, args.output_dir / "plot01_subset_rmse_bar.png")
    plot_abs_error_ecdf(aligned, args.output_dir / "plot02_abs_error_ecdf.png")
    plot_field_rmse_strip(per_field, args.output_dir / "plot03_field_rmse_strip.png")
    wins = plot_wins_bar(per_field, args.output_dir / "plot04_wins_per_field.png")
    plot_cnn_vs_mlp_mse_delta(
        aligned,
        args.output_dir / "plot05_cnn_vs_mlp_field_mse_delta.png",
        args.output_dir / "table_cnn_vs_mlp_field_mse_delta.csv",
    )
    plot_daily_rmse_timeseries(
        aligned,
        args.output_dir / "plot06_daily_rmse_timeseries_roll14.png",
        args.output_dir / "table_daily_rmse_timeseries.csv",
    )
    plot_cumulative_rmse_timeseries(
        aligned,
        args.output_dir / "plot07_cumulative_rmse_timeseries.png",
        args.output_dir / "table_cumulative_rmse_timeseries.csv",
    )

    wins.to_csv(args.output_dir / "table_wins_per_field.csv", index=False)
    metrics.to_csv(args.output_dir / "table_subset_metrics_used.csv", index=False)

    print(f"Saved presentation pack to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
