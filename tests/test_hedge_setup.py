from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon

from fieldbird_sim.legacy.hedge_setup import attach_hedge_metrics_to_fields, build_hedges_from_cfg


def _toy_fields() -> gpd.GeoDataFrame:
    # Two adjacent 1x1m polygons sharing one edge.
    geoms = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    ]
    return gpd.GeoDataFrame(
        {"field_id": [1, 2], "area": [1.0, 1.0], "x": [0.5, 1.5], "y": [0.5, 0.5]},
        geometry=geoms,
        crs="EPSG:3857",
    )


def test_attach_hedge_metrics_to_fields_basic():
    fields = _toy_fields()
    hedges = gpd.GeoDataFrame(
        {"f1": [1], "f2": [2], "hedge_quality": [0.5]},
        geometry=[LineString([(1, 0), (1, 1)])],
        crs="EPSG:3857",
    )

    out = attach_hedge_metrics_to_fields(fields_gdf=fields, hedges_gdf=hedges)

    assert list(out["hedge_length_m"]) == [1.0, 1.0]
    assert list(out["hedge_quality_weighted_m"]) == [0.5, 0.5]
    assert list(out["hedge_quality_mean"]) == [0.5, 0.5]
    assert list(out["hedge_length_m_per_ha"]) == [1.0, 1.0]


def test_build_hedges_from_cfg_prefers_csv(tmp_path):
    fields = _toy_fields()
    hedges_csv = tmp_path / "hedges.csv"
    pd.DataFrame({"f1": [1], "f2": [2], "hedge_quality": [0.8]}).to_csv(hedges_csv, index=False)

    cfg = SimpleNamespace(
        hedges=SimpleNamespace(from_csv=True, p_edge=0.0),
        inputs=SimpleNamespace(hedges_csv=str(hedges_csv)),
        seeds=SimpleNamespace(hedges=123),
    )

    out = build_hedges_from_cfg(cfg=cfg, fields_gdf=fields)

    assert out is not None
    assert len(out) == 1
    assert int(out.iloc[0]["f1"]) == 1
    assert int(out.iloc[0]["f2"]) == 2


def test_build_hedges_from_cfg_falls_back_to_random(tmp_path):
    fields = _toy_fields()
    missing_csv = tmp_path / "missing.csv"

    cfg = SimpleNamespace(
        hedges=SimpleNamespace(from_csv=True, p_edge=1.0),
        inputs=SimpleNamespace(hedges_csv=str(missing_csv)),
        seeds=SimpleNamespace(hedges=123),
    )

    out = build_hedges_from_cfg(cfg=cfg, fields_gdf=fields)

    assert out is not None
    assert len(out) >= 1
