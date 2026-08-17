#!/usr/bin/env python
"""
10_core_vs_fringe.py
====================
Diagnostic for the nighttime result (script 09): does the nocturnal heat
penalty belong to the ESTABLISHED dense core, while the EXPANDING peri-urban
fringe hasn't developed it yet?

Method (all on the 1 km MODIS grid, 2024 night LST):
  * Aggregate the 10 m class maps of BOTH years to each MODIS cell -> fraction
    informal in 2017 and in 2024.
  * Label pure cells:
        OTHER      : <=30% informal in 2024
        CORE       : >=70% informal in 2024 AND >=50% informal already in 2017
        EXPANSION  : >=70% informal in 2024 AND <=20% informal in 2017
  * Compare 2024 MODIS NIGHT LST across groups with
        night ~ C(group) + elevation + ndvi
    (OTHER = reference). Positive coefficient = hotter at night than other.

If CORE is significantly hotter and EXPANSION is not, the "growth diluted the
2024 signal" reading in script 09 is supported.

OUTPUTS
    results/core_vs_fringe.json, results/core_vs_fringe.md
    figures/core_vs_fringe_night.png

USAGE
    python scripts/10_core_vs_fringe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"
MIN_10M = 300


def _read(path, as_float=True):
    with rasterio.open(path) as s:
        arr = s.read(1).astype("float64" if as_float else "float32")
        nd = s.nodata
        tr = s.transform
        shape = s.shape
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr, tr, shape


def build_cells() -> pd.DataFrame:
    cls24, ctr, (ch, cw) = _read(EXPORT_DIR / "class_2024.tif")
    cls17, _, _ = _read(EXPORT_DIR / "class_2017.tif")
    ndvi, _, _ = _read(EXPORT_DIR / "ndvi_addis_2024.tif")
    elev, etr, (eh, ew) = _read(EXPORT_DIR / "elevation_addis.tif")
    night, mtr, (mh, mw) = _read(EXPORT_DIR / "modis_night_addis_2024.tif")

    rows = []
    for r in range(mh):
        for c in range(mw):
            nv = night[r, c]
            if not np.isfinite(nv):
                continue
            left, top = mtr * (c, r)
            right, bottom = mtr * (c + 1, r + 1)
            r0, c0 = rasterio.transform.rowcol(ctr, left, top)
            r1, c1 = rasterio.transform.rowcol(ctr, right, bottom)
            r0, r1 = sorted((max(0, r0), min(ch, r1)))
            c0, c1 = sorted((max(0, c0), min(cw, c1)))
            if r1 <= r0 or c1 <= c0:
                continue
            s24 = cls24[r0:r1, c0:c1]
            s17 = cls17[r0:r1, c0:c1]
            v24 = np.isfinite(s24)
            k = int(v24.sum())
            if k < MIN_10M:
                continue
            frac24 = float(np.nansum(s24) / k)
            v17 = np.isfinite(s17)
            frac17 = float(np.nansum(s17) / v17.sum()) if v17.sum() else np.nan
            nsub = ndvi[r0:r1, c0:c1]
            er0, ec0 = rasterio.transform.rowcol(etr, left, top)
            er1, ec1 = rasterio.transform.rowcol(etr, right, bottom)
            er0, er1 = sorted((max(0, er0), min(eh, er1)))
            ec0, ec1 = sorted((max(0, ec0), min(ew, ec1)))
            esub = elev[er0:er1, ec0:ec1] if (er1 > er0 and ec1 > ec0) else np.array([np.nan])
            rows.append({
                "night": nv, "frac24": frac24, "frac17": frac17,
                "ndvi": float(np.nanmean(nsub)) if np.isfinite(nsub).any() else np.nan,
                "elevation": float(np.nanmean(esub)) if np.isfinite(esub).any() else np.nan,
            })
    return pd.DataFrame(rows)


def label_group(row):
    if row.frac24 <= 0.30:
        return "other"
    if row.frac24 >= 0.70 and row.frac17 >= 0.50:
        return "core"
    if row.frac24 >= 0.70 and row.frac17 <= 0.20:
        return "expansion"
    return None


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    cells = build_cells().dropna(subset=["night", "elevation", "ndvi", "frac17"])
    cells["group"] = cells.apply(label_group, axis=1)
    d = cells.dropna(subset=["group"]).copy()
    counts = d["group"].value_counts().to_dict()
    print("cell counts:", counts)

    if not {"other", "core", "expansion"}.issubset(counts) or \
       min(counts.get("core", 0), counts.get("expansion", 0)) < 10:
        print("WARNING: too few core/expansion cells for a stable contrast.")

    # night ~ C(group) + elevation + ndvi, other = reference
    d["group"] = pd.Categorical(d["group"], categories=["other", "core", "expansion"])
    fit = smf.ols("night ~ C(group, Treatment('other')) + elevation + ndvi", d).fit()

    def term(name):
        key = f"C(group, Treatment('other'))[T.{name}]"
        ci = fit.conf_int().loc[key].tolist()
        return {"effect": float(fit.params[key]), "ci95": [float(ci[0]), float(ci[1])],
                "p": float(fit.pvalues[key])}

    res = {"counts": counts, "core_vs_other": term("core"),
           "expansion_vs_other": term("expansion"), "r2": float(fit.rsquared),
           "raw_night_median": d.groupby("group")["night"].median().to_dict(),
           "median_elevation": d.groupby("group")["elevation"].median().to_dict()}

    core, exp = res["core_vs_other"], res["expansion_vs_other"]
    print(f"CORE  vs other (night, adj): {core['effect']:+.3f} degC "
          f"CI[{core['ci95'][0]:+.3f},{core['ci95'][1]:+.3f}] p={core['p']:.2e}")
    print(f"EXPAN vs other (night, adj): {exp['effect']:+.3f} degC "
          f"CI[{exp['ci95'][0]:+.3f},{exp['ci95'][1]:+.3f}] p={exp['p']:.2e}")

    (RESULTS_DIR / "core_vs_fringe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # markdown
    L = ["# Core vs fringe — where is the nighttime penalty? (2024 MODIS night)\n",
         "Adjusted night-LST difference vs 'other' cells "
         "(`night ~ C(group) + elevation + ndvi`). Positive = hotter at night.\n",
         "| Group | cells | median elev (m) | adjusted night Δ vs other (°C) | 95% CI | p |",
         "|---|---|---|---|---|---|"]
    for g in ("core", "expansion"):
        t = res[f"{g}_vs_other"]
        L.append(f"| {g} | {counts.get(g,0)} | {res['median_elevation'].get(g,float('nan')):.0f} | "
                 f"{t['effect']:+.3f} | [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['p']:.2e} |")
    L.append(f"| other (ref) | {counts.get('other',0)} | "
             f"{res['median_elevation'].get('other',float('nan')):.0f} | 0 (ref) | | |")
    L.append("\n## Read")
    if core["effect"] > 0 and core["p"] < 0.05 and (exp["effect"] < core["effect"]):
        L.append(f"- **Established CORE informal is hotter at night (+{core['effect']:.2f} °C, "
                 f"p={core['p']:.1e})**, while the EXPANSION fringe effect is weaker "
                 f"({exp['effect']:+.2f} °C). Supports the reading that the 2024 average was "
                 "diluted by newer, not-yet-heat-retaining peri-urban settlement.")
    else:
        L.append("- Core and fringe do not separate cleanly; the growth-dilution reading is "
                 "not supported by this split.")
    (RESULTS_DIR / "core_vs_fringe.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # figure
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = ["core\n(established)", "expansion\n(new fringe)"]
    effs = [core["effect"], exp["effect"]]
    errs = [core["effect"] - core["ci95"][0], exp["effect"] - exp["ci95"][0]]
    ax.bar([0, 1], effs, yerr=errs, capsize=4, color=["#7C1D1D", "#E45756"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(names)
    ax.set_ylabel("Adjusted night-LST vs other (°C)")
    ax.set_title("Nighttime heat penalty: established core vs expanding fringe (2024)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "core_vs_fringe_night.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {RESULTS_DIR/'core_vs_fringe.md'} and figure.")


if __name__ == "__main__":
    main()
