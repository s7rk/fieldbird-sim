from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLS = ["species", "run_id", "field_id", "scenario", "year", "day", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a comparable unified prediction file for MLP/CNN/GNN using only "
            "common keys."
        )
    )
    parser.add_argument(
        "--cnn-csv",
        type=Path,
        default=Path("~/Desktop/präsi/debug_single_predictions_cnn_Rebhuhn_h365.csv").expanduser(),
        help="CNN CSV path.",
    )
    parser.add_argument(
        "--mlp-csv",
        type=Path,
        default=Path("~/Desktop/präsi/debug_single_predictions_mlp_Rebhuhn_h365.csv").expanduser(),
        help="MLP CSV path.",
    )
    parser.add_argument(
        "--gnn-csv",
        type=Path,
        default=Path("~/Desktop/präsi/clean_predictions_gnn_Rebhuhn_h365_realcounts.csv").expanduser(),
        help="GNN CSV path (real-count predictions).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/presentation_plots/rebhuhn_h365_unified_predictions_COMPARABLE.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def aggregate_model(
    path: Path,
    model: str,
    truth_col: str,
    pred_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=KEY_COLS + [truth_col, pred_col]).rename(
        columns={truth_col: "y_true_level", pred_col: "y_pred_level"}
    )
    grouped = (
        df.groupby(KEY_COLS, as_index=False)
        .agg(
            y_true_level=("y_true_level", "mean"),
            y_pred_level=("y_pred_level", "mean"),
            source_rows=("y_pred_level", "size"),
        )
        .assign(model=model)
    )
    return grouped


def main() -> None:
    args = parse_args()
    output_csv: Path = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cnn = aggregate_model(args.cnn_csv, model="CNN", truth_col="y_true", pred_col="y_pred")
    mlp = aggregate_model(
        args.mlp_csv,
        model="MLP",
        truth_col="y_true_lvl",
        pred_col="y_pred_lvl_mean",
    )
    gnn = aggregate_model(
        args.gnn_csv,
        model="GNN",
        truth_col="N_true_tplus_from_logdelta",
        pred_col="N_pred_tplus",
    )

    common_keys = (
        cnn[KEY_COLS]
        .drop_duplicates()
        .merge(mlp[KEY_COLS].drop_duplicates(), on=KEY_COLS, how="inner")
        .merge(gnn[KEY_COLS].drop_duplicates(), on=KEY_COLS, how="inner")
    )
    if common_keys.empty:
        raise SystemExit("No common keys found across CNN/MLP/GNN.")

    cnn_c = common_keys.merge(cnn, on=KEY_COLS, how="left")
    mlp_c = common_keys.merge(mlp, on=KEY_COLS, how="left")
    gnn_c = common_keys.merge(gnn, on=KEY_COLS, how="left")

    combined = pd.concat([mlp_c, cnn_c, gnn_c], ignore_index=True)
    combined = combined[
        ["model"] + KEY_COLS + ["y_true_level", "y_pred_level", "source_rows"]
    ].sort_values(["model", "run_id", "field_id", "year", "day"])

    # Verify that ground truth aligns across models for each key.
    gt_check = combined.pivot_table(
        index=KEY_COLS,
        columns="model",
        values="y_true_level",
        aggfunc="first",
    )
    max_gt_pair_diff = 0.0
    cols = list(gt_check.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            diff = (gt_check[cols[i]] - gt_check[cols[j]]).abs().max()
            max_gt_pair_diff = max(max_gt_pair_diff, float(diff))

    combined.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv.resolve()}")
    print(f"Common keys: {len(common_keys):,}")
    print("Rows per model (after aggregation/intersection):")
    for model in ["MLP", "CNN", "GNN"]:
        print(f"- {model}: {len(combined[combined['model'] == model]):,}")
    print(f"Max GT difference between any model pair: {max_gt_pair_diff:.12f}")
    print("Day range:", int(common_keys["day"].min()), "to", int(common_keys["day"].max()))


if __name__ == "__main__":
    main()
