#!/usr/bin/env python
"""
01_export_ee_layers.py
=======================
Earth Engine export for the Addis Ababa thermal-inequity project.

Pulls and exports, clipped to the Addis Ababa administrative boundary, for
BOTH 2017 and 2024:

    * AlphaEarth Satellite Embedding V1 Annual   (64-dim, 10m)  -> classifier input
    * Landsat 8/9 C2 L2 Surface Temperature      (30m, degC)    -> outcome variable
    * Sentinel-2 L2A RGB (B4/B3/B2)              (10m)          -> visualization only
    * NDVI from Sentinel-2                        (10m)          -> regression covariate
    * SRTM elevation (static)                     (30m)          -> regression covariate

Everything is composited over the dry season (Dec of the prior year through
Feb) to minimise cloud/rain contamination from Ethiopia's rainy seasons.

Outputs are written as local GeoTIFFs under data/exports/ using
`geemap.download_ee_image`, which tiles under the hood and therefore is not
bound by the ~32-48 MB getDownloadURL limit that trips up naive downloads of
the 64-band embedding stack.

USAGE
-----
    # one-time, interactive (opens a browser for Google auth):
    python scripts/01_export_ee_layers.py --project YOUR_GCP_PROJECT_ID

    # or set it once in the environment:
    export EE_PROJECT=YOUR_GCP_PROJECT_ID      # PowerShell: $env:EE_PROJECT="..."
    python scripts/01_export_ee_layers.py

    # export just one layer / one year while iterating:
    python scripts/01_export_ee_layers.py --layers embedding --years 2024

PREREQUISITES
-------------
    pip install earthengine-api geemap
    A Google Cloud project with the Earth Engine API enabled
    (https://code.earthengine.google.com/ -> register a project).

The FIRST run will call ee.Authenticate() and open a browser. After that the
token is cached and runs are non-interactive.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ee
import geemap

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

YEARS = (2017, 2024)

# Native export scale (metres). AlphaEarth + Sentinel-2 are 10m; Landsat ST is
# resampled to the same grid so every raster shares pixel geometry, which makes
# the downstream per-pixel join trivial. Landsat's true information content is
# still ~30-100m — resampling does not create resolution that isn't there.
SCALE_M = 10

# Dry-season window: December of the prior year through end of February.
DRY_SEASON_START_MONTH_DAY = "-12-01"  # of year-1
DRY_SEASON_END_MONTH_DAY = "-02-28"    # of year

# Earth Engine asset ids
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
L8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
L9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SRTM_ASSET = "USGS/SRTMGL1_003"
GAUL_L1 = "FAO/GAUL/2015/level1"
# MODIS daily LST, 1 km native. Terra (MOD11A1) night ~22:30 / day ~10:30;
# Aqua (MYD11A1) night ~01:30 / day ~13:30. Both are merged for coverage —
# night retrievals are sparse, so two overpasses matter.
MODIS_LST_TERRA = "MODIS/061/MOD11A1"
MODIS_LST_AQUA = "MODIS/061/MYD11A1"

ADDIS_ADM1_NAME = "Addis Ababa"  # GAUL 2015 level-1 spelling (verified)

# --- Study-area definition -------------------------------------------------- #
# The GAUL admin boundary is tighter than the real built-up city and excludes
# the peri-urban expansion fringe (exactly where informal growth happens). So
# the study area = admin boundary buffered outward, THEN masked to actually
# built-up land (GHS_BUILT_S) so rural farmland doesn't contaminate the "other"
# class with cool non-urban pixels. The built-up mask uses the 2025 epoch so
# post-2020 expansion is retained for the 2017->2024 change analysis.
STUDY_BUFFER_KM = 10          # outward buffer on the admin boundary
BUILT_ASSET = "JRC/GHSL/P2023A/GHS_BUILT_S"
BUILT_EPOCH = 2025            # GHS_BUILT_S epoch used for the urban mask
BUILT_THRESH_M2 = 500         # min built surface per 100m cell to count as urban


# --------------------------------------------------------------------------- #
# Earth Engine init
# --------------------------------------------------------------------------- #
def init_ee(project: str | None) -> None:
    """Authenticate (first run only) and initialize Earth Engine."""
    project = project or os.environ.get("EE_PROJECT")
    if not project:
        sys.exit(
            "ERROR: no Earth Engine Cloud project supplied.\n"
            "Pass --project YOUR_GCP_PROJECT_ID or set $EE_PROJECT.\n"
            "Register one at https://code.earthengine.google.com/"
        )
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"Earth Engine initialized (project={project}).")


# --------------------------------------------------------------------------- #
# Study area
# --------------------------------------------------------------------------- #
def addis_boundary(buffer_km: float = 0.0) -> ee.Geometry:
    """Addis Ababa admin boundary (GAUL 2015 L1), optionally buffered outward."""
    fc = ee.FeatureCollection(GAUL_L1).filter(
        ee.Filter.eq("ADM1_NAME", ADDIS_ADM1_NAME)
    )
    geom = fc.geometry()
    if buffer_km and buffer_km > 0:
        geom = geom.buffer(buffer_km * 1000)
    return geom


def urban_mask(region: ee.Geometry, thresh_m2: float = BUILT_THRESH_M2) -> ee.Image:
    """Built-up mask (1 where built) from GHS_BUILT_S, to keep the study area
    urban and exclude rural farmland."""
    built = (
        ee.ImageCollection(BUILT_ASSET)
        .filterDate(f"{BUILT_EPOCH}-01-01", f"{BUILT_EPOCH}-12-31")
        .first()
        .select("built_surface")
    )
    return built.gte(thresh_m2).clip(region)


def dry_season_range(year: int) -> tuple[ee.Date, ee.Date]:
    """(start, end) covering Dec(year-1) .. Feb(year)."""
    start = ee.Date(f"{year - 1}{DRY_SEASON_START_MONTH_DAY}")
    end = ee.Date(f"{year}{DRY_SEASON_END_MONTH_DAY}")
    return start, end


# --------------------------------------------------------------------------- #
# Layer builders — each returns an ee.Image clipped to `region`
# --------------------------------------------------------------------------- #
def build_embedding(year: int, region: ee.Geometry) -> ee.Image:
    """64-band AlphaEarth annual embedding for `year`.

    The collection is tiled — there are several images per year over a city.
    Mosaic ALL tiles intersecting the region (not .first(), which returns a
    single tile and leaves most of the boundary as nodata).
    """
    col = (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(region)
    )
    return col.mosaic().clip(region)


def _mask_landsat_sr(img: ee.Image) -> ee.Image:
    """Cloud/shadow mask from the QA_PIXEL bitmask (C2 L2)."""
    qa = img.select("QA_PIXEL")
    # bit 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    return img.updateMask(mask)


def _landsat_st_celsius(img: ee.Image) -> ee.Image:
    """Convert ST_B10 DN to surface temperature in degrees Celsius."""
    st = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
    return st.rename("LST").copyProperties(img, ["system:time_start"])


def build_lst(year: int, region: ee.Geometry) -> ee.Image:
    """Median dry-season Landsat surface temperature (degC), L8 + L9 merged."""
    start, end = dry_season_range(year)
    l8 = ee.ImageCollection(L8_COLLECTION)
    # Landsat 9 launched late 2021 — only contributes to the 2024 composite.
    l9 = ee.ImageCollection(L9_COLLECTION)
    col = (
        l8.merge(l9)
        .filterDate(start, end)
        .filterBounds(region)
        .map(_mask_landsat_sr)
        .map(_landsat_st_celsius)
    )
    return col.median().clip(region).rename("LST")


def build_s2_dry(year: int, region: ee.Geometry) -> ee.ImageCollection:
    """Cloud-filtered Sentinel-2 SR collection for the dry season."""
    start, end = dry_season_range(year)
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )


def build_rgb(year: int, region: ee.Geometry) -> ee.Image:
    """Median dry-season Sentinel-2 RGB (B4,B3,B2), scaled to reflectance."""
    med = build_s2_dry(year, region).median()
    rgb = med.select(["B4", "B3", "B2"]).multiply(0.0001)
    return rgb.clip(region).rename(["R", "G", "B"])


def build_ndvi(year: int, region: ee.Geometry) -> ee.Image:
    """Median dry-season NDVI from Sentinel-2."""
    med = build_s2_dry(year, region).median()
    ndvi = med.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi.clip(region)


def build_elevation(region: ee.Geometry) -> ee.Image:
    """Static SRTM elevation (m)."""
    return ee.Image(SRTM_ASSET).select("elevation").clip(region)


def _modis_lst(year: int, region: ee.Geometry, band: str, out_name: str) -> ee.Image:
    """Dry-season median MODIS LST (degC) for the given band, Terra+Aqua merged.
    QC relaxed to 'LST produced' (mandatory-QA bits 0-1 <= 1) rather than
    good-quality-only, because night retrievals rarely earn the top flag and a
    strict filter leaves the composite empty. Native 1 km."""
    start, end = dry_season_range(year)
    qc_band = "QC_Night" if "Night" in band else "QC_Day"

    def scale_qc(img):
        lst = img.select(band).multiply(0.02).subtract(273.15)
        qc = img.select(qc_band)
        produced = qc.bitwiseAnd(0b11).lte(1)      # 00 good, 01 other quality
        fill = img.select(band).gt(0)              # raw 0 == not retrieved
        return lst.updateMask(produced.And(fill)).rename(out_name)

    col = (
        ee.ImageCollection(MODIS_LST_TERRA).merge(ee.ImageCollection(MODIS_LST_AQUA))
        .filterDate(start, end)
        .filterBounds(region)
        .map(scale_qc)
    )
    return col.median().clip(region).rename(out_name)


def build_modis_night(year: int, region: ee.Geometry) -> ee.Image:
    """Dry-season median MODIS Aqua NIGHT LST (~01:30), degC, 1 km."""
    return _modis_lst(year, region, "LST_Night_1km", "LST_night")


def build_modis_day(year: int, region: ee.Geometry) -> ee.Image:
    """Dry-season median MODIS Aqua DAY LST (~13:30), degC, 1 km."""
    return _modis_lst(year, region, "LST_Day_1km", "LST_day")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def download(img: ee.Image, region: ee.Geometry, out_path: Path, scale: int) -> None:
    """Download an ee.Image to a local GeoTIFF (tiled, no size cap)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"  SKIP (exists): {out_path.name}")
        return
    print(f"  downloading -> {out_path.name} @ {scale}m ...")
    geemap.download_ee_image(
        image=img,
        filename=str(out_path),
        region=region,
        scale=scale,
        crs="EPSG:4326",
    )
    print(f"  done: {out_path.name}")


# Layer registry: name -> (builder, scale, per_year?)
def layer_specs():
    return {
        "embedding": (build_embedding, SCALE_M, True),
        "lst": (build_lst, SCALE_M, True),
        "rgb": (build_rgb, SCALE_M, True),
        "ndvi": (build_ndvi, SCALE_M, True),
        "elevation": (build_elevation, 30, False),
        # Coarse (1 km) MODIS thermal — exported UNMASKED (see NO_URBAN_MASK)
        # so every built pixel still receives a night/day value on sampling.
        "modis_night": (build_modis_night, 1000, True),
        "modis_day": (build_modis_day, 1000, True),
    }


# Layers exported without the built-up mask (coarse grids the mask would punch
# holes in; downstream analysis samples them at already-in-mask class pixels).
NO_URBAN_MASK = {"modis_night", "modis_day"}


def run(layers: list[str], years: list[int],
        buffer_km: float = STUDY_BUFFER_KM,
        built_thresh: float = BUILT_THRESH_M2) -> None:
    region = addis_boundary(buffer_km)
    area_km2 = region.area(maxError=1).divide(1e6).getInfo()
    print(f"Study area: Addis Ababa admin + {buffer_km} km buffer "
          f"(~{area_km2:.0f} km^2 before urban mask)")
    if area_km2 < 100:
        sys.exit(
            f"ERROR: boundary area is only {area_km2:.1f} km^2 — the GAUL name "
            f"filter (ADM1_NAME == '{ADDIS_ADM1_NAME}') likely matched nothing. "
            "Aborting before exporting empty rasters."
        )

    mask = urban_mask(region, built_thresh)
    print(f"Urban mask: GHS_BUILT_S {BUILT_EPOCH} built_surface >= {built_thresh} m^2")

    specs = layer_specs()
    for name in layers:
        if name not in specs:
            print(f"  unknown layer '{name}', skipping")
            continue
        builder, scale, per_year = specs[name]
        apply_mask = name not in NO_URBAN_MASK

        if not per_year:  # static, e.g. elevation
            img = builder(region)
            out = EXPORT_DIR / f"{name}_addis.tif"
            download(img.updateMask(mask) if apply_mask else img, region, out, scale)
            continue

        for year in years:
            print(f"[{name}] {year}")
            img = builder(year, region)
            if apply_mask:
                img = img.updateMask(mask)
            out = EXPORT_DIR / f"{name}_addis_{year}.tif"
            download(img, region, out, scale)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=None,
                   help="Earth Engine Cloud project id (or set $EE_PROJECT).")
    p.add_argument("--years", nargs="+", type=int, default=list(YEARS),
                   help="Years to export (default: 2017 2024).")
    p.add_argument("--layers", nargs="+",
                   default=["embedding", "lst", "rgb", "ndvi", "elevation"],
                   help="Layers to export (default: all).")
    p.add_argument("--buffer-km", type=float, default=STUDY_BUFFER_KM,
                   help=f"Outward buffer on the admin boundary (default {STUDY_BUFFER_KM}).")
    p.add_argument("--built-thresh", type=float, default=BUILT_THRESH_M2,
                   help=f"Min built m^2/100m cell for urban mask (default {BUILT_THRESH_M2}).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    init_ee(args.project)
    run(args.layers, args.years, args.buffer_km, args.built_thresh)
    print(f"\nAll requested exports complete. Files under: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
