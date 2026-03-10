from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .globals import make_random_hedges
from .layout import build_hedge_geoms_on_shared_edges


def build_hedges_from_cfg(*, cfg, fields_gdf):
    """
    YAML-controlled hedges.

    If cfg.hedges.from_csv == True:
      - try cfg.inputs.hedges_csv
      - if file exists + has columns f1,f2 and rows -> use
      - else fallback to random

    Else:
      - always random

    Also supports cfg.hedges.p_edge for random hedges.

    Returns:
      hedges_gdf or None
    """
    hedges_section = getattr(cfg, "hedges", None)
    from_csv = bool(getattr(hedges_section, "from_csv", False))
    p_edge = float(getattr(hedges_section, "p_edge", 0.7))

    # ---- Try CSV ----
    if from_csv:
        inputs = getattr(cfg, "inputs", None)
        hedges_csv_raw = getattr(inputs, "hedges_csv", None) if inputs is not None else None
        hedges_csv = Path(str(hedges_csv_raw)) if hedges_csv_raw else None

        if hedges_csv and hedges_csv.exists():
            try:
                hedge_pairs = pd.read_csv(hedges_csv)
            except Exception as e:
                print(f"[warn] could not read hedges_csv={hedges_csv}: {e} -> fallback to random.")
                hedge_pairs = pd.DataFrame()

            if {"f1", "f2"}.issubset(hedge_pairs.columns) and len(hedge_pairs) > 0:
                hedges = build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)
                if hedges is not None and len(hedges) > 0:
                    return hedges
                print(
                    f"[warn] hedges_csv had no valid shared edges for current layout: {hedges_csv} -> fallback to random."
                )
            else:
                print(f"[warn] hedges.from_csv=true but hedges file empty/invalid: {hedges_csv} -> fallback to random.")
        else:
            print(f"[warn] hedges.from_csv=true but hedges_csv missing or not set: {hedges_csv_raw} -> fallback to random.")

    # ---- Random fallback ----
    hedge_pairs = make_random_hedges(fields_gdf, p_edge=p_edge, seed=int(cfg.seeds.hedges))
    return build_hedge_geoms_on_shared_edges(fields_gdf, hedge_pairs)


def attach_hedge_metrics_to_fields(*, fields_gdf: pd.DataFrame, hedges_gdf: pd.DataFrame | None) -> pd.DataFrame:
    """
    Derives per-field hedge metrics from hedge line geometries.
    Metrics are consumed by species-specific hedge effects in the simulator.
    """
    out = fields_gdf.copy()
    out["hedge_length_m"] = 0.0
    out["hedge_quality_weighted_m"] = 0.0
    out["hedge_quality_mean"] = 0.0

    if hedges_gdf is None or len(hedges_gdf) == 0:
        out["hedge_length_m_per_ha"] = 0.0
        out["hedge_quality_weighted_m_per_ha"] = 0.0
        return out

    if not {"f1", "f2"}.issubset(set(hedges_gdf.columns)):
        out["hedge_length_m_per_ha"] = 0.0
        out["hedge_quality_weighted_m_per_ha"] = 0.0
        return out

    length_by_field: dict[int, float] = {}
    qlen_by_field: dict[int, float] = {}

    for _, row in hedges_gdf.iterrows():
        try:
            f1 = int(row.get("f1"))
            f2 = int(row.get("f2"))
        except Exception:
            continue

        geom = row.get("geometry", None)
        try:
            length_m = float(getattr(geom, "length", 0.0) or 0.0)
        except Exception:
            length_m = 0.0
        if length_m <= 0.0:
            continue

        q = float(row.get("hedge_quality", 1.0) or 1.0)
        q = float(np.clip(q, 0.0, 1.0))
        qlen = length_m * q

        for fid in (f1, f2):
            length_by_field[fid] = length_by_field.get(fid, 0.0) + length_m
            qlen_by_field[fid] = qlen_by_field.get(fid, 0.0) + qlen

    out["hedge_length_m"] = (
        out["field_id"].astype(int).map(length_by_field).fillna(0.0).astype(float)
    )
    out["hedge_quality_weighted_m"] = (
        out["field_id"].astype(int).map(qlen_by_field).fillna(0.0).astype(float)
    )
    out["hedge_quality_mean"] = np.where(
        out["hedge_length_m"] > 0.0,
        out["hedge_quality_weighted_m"] / out["hedge_length_m"],
        0.0,
    )

    area_ha = pd.to_numeric(out.get("area", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["hedge_length_m_per_ha"] = np.where(area_ha > 0.0, out["hedge_length_m"] / area_ha, 0.0)
        out["hedge_quality_weighted_m_per_ha"] = np.where(
            area_ha > 0.0,
            out["hedge_quality_weighted_m"] / area_ha,
            0.0,
        )
    return out
