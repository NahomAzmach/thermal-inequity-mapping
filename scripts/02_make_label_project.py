#!/usr/bin/env python
"""
02_make_label_project.py
========================
Create the QGIS labeling project for hand-digitizing informal-settlement
polygons over 2024 basemap imagery.

Produces a single GeoPackage, data/labels/addis_labels.gpkg, with two layers:

  * `labels`          empty polygon layer with the digitizing schema
                      (class, class_name, notes) in EPSG:4326.
  * `hotspot_guide`   point layer of sub-city anchors flagged by the Addis
                      deprivation literature (MDPI 2071-1050/15/3/1934), each
                      tagged with the class you should EXPECT to find near it.
                      These are digitizing hints, NOT labels — you still draw
                      and judge every polygon yourself against the imagery.

HOW TO LABEL (in QGIS)
----------------------
  1. Add a high-res basemap: Browser panel -> XYZ Tiles -> add e.g.
       Esri World Imagery:
       https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
  2. Layer -> Add Layer -> Add Vector Layer -> addis_labels.gpkg -> both layers.
  3. Toggle editing on `labels`, digitize polygons. For each polygon set:
       class      = 1 (informal settlement)  or  0 (other / formal fabric)
       class_name = "informal" or "other"
  4. Aim for ~100-150 polygons total, reasonably balanced between classes,
     spread across the city (don't cluster them all in one sub-city).
  5. Save edits. That's the only input the rest of the pipeline needs.

Visual cues for "informal" fabric at basemap resolution: dense, irregular
small-footprint roofs; narrow/organic street pattern; little vegetation;
corrugated-metal roof texture. "Other": regular plots, apartment blocks,
villas, wide planned roads, green space, industrial sheds.

USAGE
-----
    python scripts/02_make_label_project.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_GPKG = PROJECT_ROOT / "data" / "labels" / "addis_labels.gpkg"

# Approximate sub-city anchor centroids (lon, lat) with the class the
# deprivation literature leads us to EXPECT nearby. Treat as digitizing hints.
# High-deprivation clustering: peri-urban Bole & Akaki Kaliti + inner-city
# slum pockets in Kirkos/Arada/Lideta. Lower-deprivation formal fabric:
# Gulele, Kolfe-Keranyo, parts of Yeka.
HOTSPOTS = [
    # name,               lon,     lat,   expected_class, class_name, note
    ("Bole (peri-urban)",   38.83, 8.95, 1, "informal", "peri-urban deprivation cluster"),
    ("Akaki Kaliti",        38.79, 8.88, 1, "informal", "peri-urban deprivation cluster"),
    ("Kirkos (Chirkos)",    38.76, 9.01, 1, "informal", "inner-city slum pockets"),
    ("Arada",               38.75, 9.04, 1, "informal", "inner-city slum pockets"),
    ("Lideta",              38.73, 9.01, 1, "informal", "inner-city slum pockets"),
    ("Addis Ketema",        38.72, 9.04, 1, "informal", "dense old-core fabric"),
    ("Gulele",              38.73, 9.07, 0, "other",    "lower-deprivation formal fabric"),
    ("Kolfe-Keranyo",       38.68, 9.03, 0, "other",    "lower-deprivation formal fabric"),
    ("Yeka",                38.82, 9.05, 0, "other",    "mixed / planned areas"),
    ("Bole (airport/CBD)",  38.79, 8.99, 0, "other",    "planned apartments/villas"),
    ("Nifas Silk-Lafto",    38.72, 8.95, 0, "other",    "mixed planned/industrial"),
]

CRS = "EPSG:4326"


def make_labels_layer() -> gpd.GeoDataFrame:
    """Empty polygon layer with the digitizing schema."""
    gdf = gpd.GeoDataFrame(
        {
            "class": pd.Series(dtype="int32"),
            "class_name": pd.Series(dtype="object"),
            "notes": pd.Series(dtype="object"),
        },
        geometry=gpd.GeoSeries([], crs=CRS),
        crs=CRS,
    )
    return gdf


def make_hotspot_layer() -> gpd.GeoDataFrame:
    rows = []
    geoms = []
    for name, lon, lat, cls, cname, note in HOTSPOTS:
        rows.append(
            {"name": name, "expected_class": cls, "class_name": cname, "note": note}
        )
        geoms.append(Point(lon, lat))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS)


def main() -> None:
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    labels = make_labels_layer()
    hotspots = make_hotspot_layer()

    # `labels` first (creates/overwrites the file), then append the guide layer.
    # Force an explicit MultiPolygon geometry type so QGIS's polygon digitizing
    # tool is unambiguous on the empty layer (otherwise it writes "Unknown").
    labels.to_file(OUT_GPKG, layer="labels", driver="GPKG",
                   geometry_type="MultiPolygon")
    hotspots.to_file(OUT_GPKG, layer="hotspot_guide", driver="GPKG")

    print(f"Wrote {OUT_GPKG}")
    print("  layer 'labels'        : empty polygon schema (class, class_name, notes)")
    print(f"  layer 'hotspot_guide' : {len(hotspots)} sub-city digitizing anchors")
    print("\nOpen it in QGIS and start digitizing polygons on the `labels` layer.")


if __name__ == "__main__":
    main()
