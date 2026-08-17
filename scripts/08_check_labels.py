#!/usr/bin/env python
"""
08_check_labels.py
==================
Sanity-check the hand-digitized labels as you go. Run it any time after saving
edits in QGIS — it never modifies your data, only reports.

Checks:
  * total polygon count and per-class balance
  * class values are valid (only 0 or 1)
  * CRS is set
  * geometry validity (self-intersections etc.)
  * suspiciously tiny polygons (< MIN_AREA_M2, likely mis-clicks)
  * overlaps between polygons (a pixel in two polygons of different classes
    is a contradiction the model can't learn from)
  * rough readiness verdict for the dry-run / full run

USAGE
    python scripts/08_check_labels.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_GPKG = PROJECT_ROOT / "data" / "labels" / "addis_labels.gpkg"

# Addis sits in UTM zone 37N — use it for real metres.
METRIC_CRS = "EPSG:32637"
MIN_AREA_M2 = 2000       # ~ a 45x45 m patch; smaller is probably a mis-click
DRYRUN_MIN = 15          # polygons needed for a meaningful dry-run
FULL_MIN = 100


def main() -> None:
    if not LABELS_GPKG.exists():
        raise SystemExit(f"Missing {LABELS_GPKG}. Run script 02 first.")

    g = gpd.read_file(LABELS_GPKG, layer="labels")
    g = g[g.geometry.notna()].copy()
    n = len(g)
    print(f"Labels file: {LABELS_GPKG}")
    print(f"Polygons digitized: {n}")
    if n == 0:
        raise SystemExit("No polygons yet — draw some in QGIS and save.")

    problems = []

    # --- class values -------------------------------------------------------
    if "class" not in g.columns:
        raise SystemExit("No 'class' field found.")
    g["class"] = g["class"]
    bad_class = g[~g["class"].isin([0, 1])]
    n_inf = int((g["class"] == 1).sum())
    n_oth = int((g["class"] == 0).sum())
    print(f"  class 1 (informal): {n_inf}")
    print(f"  class 0 (other)   : {n_oth}")
    if len(bad_class):
        problems.append(f"{len(bad_class)} polygon(s) have class not in {{0,1}} "
                        f"(values: {sorted(bad_class['class'].unique())}) — fix in QGIS.")
    if n_inf == 0 or n_oth == 0:
        problems.append("One class is empty — you need BOTH informal and other examples.")

    # --- balance ------------------------------------------------------------
    if n_inf and n_oth:
        ratio = max(n_inf, n_oth) / min(n_inf, n_oth)
        if ratio > 3:
            problems.append(f"Class balance is skewed ({n_inf} vs {n_oth}); "
                            "aim for roughly even numbers.")

    # --- CRS ----------------------------------------------------------------
    if g.crs is None:
        problems.append("Layer has no CRS set.")

    # --- geometry validity --------------------------------------------------
    invalid = g[~g.geometry.is_valid]
    if len(invalid):
        problems.append(f"{len(invalid)} invalid geometry(ies) (self-intersections?). "
                        "In QGIS: Vector > Geometry Tools > Fix Geometries, or redraw.")

    # --- tiny polygons ------------------------------------------------------
    gm = g.to_crs(METRIC_CRS)
    areas = gm.geometry.area
    tiny = g[areas < MIN_AREA_M2]
    print(f"  median polygon area: {areas.median():,.0f} m^2")
    if len(tiny):
        problems.append(f"{len(tiny)} polygon(s) smaller than {MIN_AREA_M2} m^2 "
                        "— likely mis-clicks; check or delete.")

    # --- overlaps between differing classes --------------------------------
    conflict = 0
    idx = list(gm.index)
    for a, b in combinations(idx, 2):
        ga, gb = gm.geometry[a], gm.geometry[b]
        if ga.intersects(gb) and ga.intersection(gb).area > 1:
            if g["class"][a] != g["class"][b]:
                conflict += 1
    if conflict:
        problems.append(f"{conflict} overlap(s) between polygons of DIFFERENT classes "
                        "— a pixel can't be both; trim them apart in QGIS.")

    # --- verdict ------------------------------------------------------------
    print("\n" + ("-" * 60))
    if problems:
        print("ISSUES TO ADDRESS:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("No structural problems found.")

    if n >= FULL_MIN and not problems:
        print(f"\nREADY for the FULL run ({n} polygons).")
    elif n >= DRYRUN_MIN and n_inf and n_oth:
        print(f"\nREADY for a DRY-RUN ({n} polygons). "
              f"Aim for ~{FULL_MIN}+ for final results.")
    else:
        need = max(0, DRYRUN_MIN - n)
        print(f"\nKeep going - ~{need} more polygon(s) for a dry-run "
              "(and both classes represented).")


if __name__ == "__main__":
    main()
