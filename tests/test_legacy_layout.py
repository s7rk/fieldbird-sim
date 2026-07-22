import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from fieldbird_sim.legacy import layout


def test_day_and_window_parsers_accept_common_formats() -> None:
    np.testing.assert_array_equal(layout.parse_days("5; 3-1 | 366 / bad / 3"), [1, 2, 3, 5])
    np.testing.assert_array_equal(layout.parse_days([10, 12, 11]), [10, 11, 12])
    assert layout.parse_days(None).size == 0
    assert layout.parse_days(float("nan")).size == 0
    assert layout.parse_days("not-a-day").size == 0

    assert layout.parse_window("200–150") == (150, 200)
    assert layout.parse_window({"start": 0, "end": 400}) == (1, 365)
    assert layout.parse_window([10, "bad", 20]) == (10, 20)
    assert layout.parse_window("10 to 20") == (10, 20)
    assert layout.parse_window("bad") is None
    assert layout.in_window(15, (20, 10)) is True
    assert layout.in_window(0, (1, 10)) is False
    assert layout.in_window("bad", (1, 10)) is False
    assert layout.in_window(5, None) is False


def test_rectangular_layout_infers_missing_dimensions() -> None:
    fields = pd.DataFrame(
        {
            "field_id": [1, 2, 3],
            "area": [1.0, 2.0, 0.0],
            "width_m": [np.nan, np.nan, 20.0],
            "height_m": [np.nan, 100.0, 10.0],
        }
    )
    normalized = layout.ensure_rect_sides(fields)
    assert normalized.loc[0, "width_m"] == pytest.approx(100.0)
    assert normalized.loc[0, "height_m"] == pytest.approx(100.0)
    assert normalized.loc[1, "width_m"] == pytest.approx(200.0)
    assert normalized.loc[2, "area"] == 0.0

    packed = layout.rect_shelf_pack(fields, target_row_width=250.0)
    assert packed.crs.to_epsg() == 3857
    assert packed.geometry.is_valid.all()
    assert packed["area"].gt(0).all()
    assert packed["perimeter"].gt(0).all()


def test_shared_edges_and_layout_csv_exports(tmp_path) -> None:
    fields = gpd.GeoDataFrame(
        {"field_id": [1, 2], "area": [0.01, 0.01]},
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
        ],
        crs="EPSG:3857",
    )
    pairs = pd.DataFrame(
        {"f1": [1, 2, 1], "f2": [2, 1, 99], "hedge_quality": [0.2, 1.4, 0.5]}
    )
    hedges = layout.build_hedge_geoms_on_shared_edges(fields, pairs)
    assert hedges is not None
    assert len(hedges) == 1
    assert hedges.geometry.iloc[0].length == pytest.approx(10.0)
    assert hedges["hedge_quality"].iloc[0] == pytest.approx(0.6)
    assert layout.build_hedge_geoms_on_shared_edges(fields, pd.DataFrame()) is None

    bbox = pd.DataFrame(
        {"field_id": [2, 1], "x0": [10, 5], "y0": [5, 6], "x1": [0, 1], "y1": [0, 2]}
    )
    target = layout.save_fields_layout_csv(bbox, tmp_path)
    saved = pd.read_csv(target)
    assert saved["field_id"].tolist() == [1, 2]
    assert saved.loc[0, ["x", "y", "width_m", "height_m"]].tolist() == [1.0, 2.0, 4.0, 4.0]

    geom_target = layout.save_fields_layout_csv(fields, tmp_path, "geometry_layout.csv")
    assert pd.read_csv(geom_target)["width_m"].tolist() == [10.0, 10.0]
    with pytest.raises(ValueError, match="unknown field schema"):
        layout.save_fields_layout_csv(pd.DataFrame({"field_id": [1]}), tmp_path, "invalid.csv")
    with pytest.raises(ValueError, match="invalid or missing"):
        layout.save_fields_layout_csv(
            pd.DataFrame({"x": [0], "y": [0], "width_m": [0], "height_m": [2]}),
            tmp_path,
            "zero.csv",
        )


def test_centroid_layout_and_plotting_smokes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(layout, "USE_RECT_PACKING", False)
    fields = pd.DataFrame(
        {
            "field_id": [1, 2],
            "area": [1.0, 0.5],
            "flower_strip_share": [0.1, 0.0],
        }
    )
    placed = layout.rect_or_centroid_layout(fields)
    assert placed.geometry.is_valid.all()
    assert placed[["x", "y"]].notna().all().all()

    heatmap_path = tmp_path / "plots" / "heatmap.png"
    layout.heatmap(
        np.array([[0.0, 1.0], [2.0, 3.0], [np.nan, 4.0]]),
        field_ids=[99],
        title="Test heatmap",
        savepath=heatmap_path,
    )
    assert heatmap_path.stat().st_size > 0
    with pytest.raises(ValueError, match="2D"):
        layout.heatmap(np.array([1, 2]), [1, 2], "invalid")

    rotation = pd.DataFrame(
        {"year": [1, 1], "field_id": [1, 2], "crop": ["Winterweizen", "Neue Kultur"]}
    )
    hedge = gpd.GeoDataFrame(
        {"f1": [1], "f2": [2], "hedge_quality": [0.7]},
        geometry=[LineString([placed.geometry.iloc[0].centroid, placed.geometry.iloc[1].centroid])],
        crs=placed.crs,
    )
    map_path = tmp_path / "plots" / "map.png"
    layout.plot_map(
        placed,
        rotation,
        year=1,
        hedges_gdf=hedge,
        day_vals={1: 2.0, 2: 5.0},
        savepath=map_path,
    )
    assert map_path.stat().st_size > 0
