#!/usr/bin/env python
"""
03_sample_points.py
===================
Turn hand-digitized polygons into a per-pixel training table.

For each labeled polygon it draws N random sample points inside the polygon,
extracts the 64-dim AlphaEarth embedding at each point (plus NDVI, elevation
and LST for the later statistics), and assigns a train/val/test split
*by polygon* so spatially-correlated pixels never straddle the split
(random pixel-level splitting leaks and inflates accuracy — see plan Sec. 3).

INPUTS  (produced by scripts 01 & 02)
    data/labels/addis_labels.gpkg            layer `labels`
    data/exports/embedding_addis_2024.tif    64-band
    data/exports/ndvi_addis_2024.tif         (optional, for stats)
    data/exports/elevation_addis.tif         (optional, for stats)
    data/exports/lst_addis_2024.tif          (optional, for stats)

OUTPUT
    data/processed/samples_2024.parquet
        columns: polygon_id, class, split, lon, lat,
                 A00..A63 (embedding), ndvi, elevation, lst

USAGE
    python scripts/03_sample_points.py --points-per-polygon 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_GPKG = PROJECT_ROOT / "data" / "labels" / "addis_labels.gpkg"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
OUT = PROJECT_ROOT / "data" / "processed" / "samples_2024.parquet"

EMB_BANDS = 64
EMB_COLS = [f"A{i:02d}" for i in range(EMB_BANDS)]

RNG = np.random.default_rng(42)


def random_points_in_polygon(geom, n: int) -> list[Point]:
    """Rejection-sample n points uniformly within a (multi)polygon."""
    minx, miny, maxx, maxy = geom.bounds
    pts: list[Point] = []
    tries = 0
    max_tries = n * 200
    while len(pts) < n and tries < max_tries:
        x = RNG.uniform(minx, maxx)
        y = RNG.uniform(miny, maxy)
        p = Point(x, y)
        if geom.contains(p):
            pts.append(p)
        tries += 1
    return pts


def assign_splits(poly_gdf: gpd.GeoDataFrame,
                  frac=(0.7, 0.15, 0.15)) -> pd.Series:
    """Assign train/val/test at the POLYGON level, stratified by class."""
    split = pd.Series(index=poly_gdf.index, dtype="object")
    for cls, idx in poly_gdf.groupby("class").groups.items():
        idx = list(idx)
        RNG.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(frac[0] * n)))
        n_va = max(1, int(round(frac[1] * n))) if n >= 3 else 0
        for i, poly_i in enumerate(idx):
            if i < n_tr:
                split[poly_i] = "train"
            elif i < n_tr + n_va:
                split[poly_i] = "val"
            else:
                split[poly_i] = "test"
    return split


def sample_raster(path: Path, coords: np.ndarray, n_bands: int | None = None):
    """Sample raster band values at coords (lon,lat). Returns (N, bands)."""
    with rasterio.open(path) as src:
        vals = np.array(list(sample_gen(src, coords)), dtype="float64")
        nodata = src.nodata
    if nodata is not None:
        vals[vals == nodata] = np.nan
    if n_bands is not None:
        vals = vals[:, :n_bands]
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points-per-polygon", type=int, default=40)
    args = ap.parse_args()

    if not LABELS_GPKG.exists():
        raise SystemExit(f"Missing {LABELS_GPKG}. Run 02 and digitize polygons first.")
    emb_path = EXPORT_DIR / "embedding_addis_2024.tif"
    if not emb_path.exists():
        raise SystemExit(f"Missing {emb_path}. Run 01 to export the embedding first.")

    polys = gpd.read_file(LABELS_GPKG, layer="labels").to_crs("EPSG:4326")
    polys = polys[polys.geometry.notna()].reset_index(drop=True)
    if polys.empty:
        raise SystemExit("No polygons digitized yet in layer 'labels'.")
    if "class" not in polys.columns:
        raise SystemExit("Label layer has no 'class' field.")
    polys["class"] = polys["class"].astype(int)
    print(f"Loaded {len(polys)} polygons "
          f"({(polys['class']==1).sum()} informal / {(polys['class']==0).sum()} other)")

    splits = assign_splits(polys)

    # Build the point table
    recs = []
    for pid, row in polys.iterrows():
        pts = random_points_in_polygon(row.geometry, args.points_per_polygon)
        for p in pts:
            recs.append({
                "polygon_id": pid,
                "class": int(row["class"]),
                "split": splits[pid],
                "lon": p.x, "lat": p.y,
            })
    df = pd.DataFrame(recs)
    print(f"Sampled {len(df)} points "
          f"(train={sum(df.split=='train')}, val={sum(df.split=='val')}, "
          f"test={sum(df.split=='test')})")

    coords = df[["lon", "lat"]].to_numpy()

    # Embedding (required)
    emb = sample_raster(emb_path, coords, n_bands=EMB_BANDS)
    for i, c in enumerate(EMB_COLS):
        df[c] = emb[:, i]

    # Optional covariates for the statistics stage
    for name, fname in [("ndvi", "ndvi_addis_2024.tif"),
                        ("elevation", "elevation_addis.tif"),
                        ("lst", "lst_addis_2024.tif")]:
        fpath = EXPORT_DIR / fname
        if fpath.exists():
            df[name] = sample_raster(fpath, coords, n_bands=1)[:, 0]
        else:
            print(f"  note: {fname} not found — column '{name}' left NaN")
            df[name] = np.nan

    # Drop rows where the embedding failed to sample (outside raster / masked)
    before = len(df)
    polys_before = df["polygon_id"].nunique()
    df = df.dropna(subset=EMB_COLS).reset_index(drop=True)
    if len(df) < before:
        polys_after = df["polygon_id"].nunique()
        lost = polys_before - polys_after
        print(f"  dropped {before - len(df)} points with no embedding value")
        if lost:
            print(f"  WARNING: {lost} polygon(s) lost entirely — their pixels have "
                  "no embedding data. Almost always because the polygon lies "
                  "OUTSIDE the study-area boundary the rasters were clipped to. "
                  "Check those polygons, or expand the export boundary (script 01).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}  ({len(df)} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
