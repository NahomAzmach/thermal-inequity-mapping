#!/usr/bin/env python
"""
09_nighttime_analysis.py
========================
PART 2 — does the heat penalty appear at NIGHT?

Daytime Landsat LST (script 06) showed informal fabric slightly COOLER. Urban
heat inequity, though, is largely a nighttime, air-temperature phenomenon:
dense fabric releases stored heat after dark. This script tests that with MODIS
Aqua+Terra LST — NIGHT (~01:30/22:30) as the new outcome, DAY (~13:30/10:30) as
a MODIS-native cross-check — over the same predicted classes.

The catch: MODIS is ~1 km, and informal patches are often smaller, so a MODIS
pixel usually MIXES classes. Two analyses handle this:

  (A) Per-pixel adjusted model  night_LST ~ class + NDVI + elevation
      (directly comparable to the Landsat-day model in script 06; but MODIS
       values repeat across the ~100x100 10 m pixels inside each 1 km cell).

  (B) Purity-restricted cell test  — aggregate the 10 m class map to the 1 km
      MODIS grid, keep only cells that are >=70% one class, and compare night
      LST between mostly-informal and mostly-other cells (raw and elevation/
      NDVI-adjusted at the cell level). This is the cleaner contrast.

INPUTS  (data/exports/)
    class_2017.tif, class_2024.tif
    modis_night_addis_{year}.tif, modis_day_addis_{year}.tif
    lst_addis_{year}.tif, ndvi_addis_{year}.tif, elevation_addis.tif
    results/statistics.json   (for the Landsat-day effect, for comparison)
OUTPUTS
    results/nighttime.json, results/nighttime.md
    figures/day_vs_night_effect.png
    figures/night_lst_by_class.png

USAGE
    python scripts/09_nighttime_analysis.py --n-per-year 60000 --purity 0.70
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import rasterio
from rasterio.sample import sample_gen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"
RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def random_valid_coords(class_path: Path, n: int) -> np.ndarray:
    with rasterio.open(class_path) as src:
        arr = src.read(1)
        transform = src.transform
    rows, cols = np.where((arr == 0) | (arr == 1))
    if len(rows) > n:
        pick = RNG.choice(len(rows), size=n, replace=False)
        rows, cols = rows[pick], cols[pick]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return np.column_stack([np.asarray(xs), np.asarray(ys)])


def sample1(path: Path, coords: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        vals = np.array(list(sample_gen(src, coords)), dtype="float64")[:, 0]
        nd = src.nodata
    if nd is not None:
        vals[vals == nd] = np.nan
    return vals


def adjusted_effect(d: pd.DataFrame, outcome: str) -> dict:
    """OLS  outcome ~ class + ndvi + elevation ; return class coef + CI."""
    sub = d.dropna(subset=[outcome, "ndvi", "elevation", "class"]).copy()
    sub["class"] = sub["class"].astype(int)
    if sub["class"].nunique() < 2 or len(sub) < 50:
        return {"n": int(len(sub)), "effect": None}
    fit = smf.ols(f"{outcome} ~ Q('class') + ndvi + elevation", sub).fit()
    ci = fit.conf_int().loc["Q('class')"].tolist()
    inf = sub.loc[sub["class"] == 1, outcome]
    oth = sub.loc[sub["class"] == 0, outcome]
    u, p = mannwhitneyu(inf, oth, alternative="two-sided")
    return {
        "n": int(len(sub)),
        "raw_median_diff": float(inf.median() - oth.median()),
        "effect": float(fit.params["Q('class')"]),
        "ci95": [float(ci[0]), float(ci[1])],
        "p": float(fit.pvalues["Q('class')"]),
        "mw_p": float(p),
    }


# --------------------------------------------------------------------------- #
# (B) purity-restricted 1 km cell aggregation
# --------------------------------------------------------------------------- #
def cell_table(year: int, min_10m: int = 300) -> pd.DataFrame:
    """One row per valid MODIS 1 km cell: night/day LST + fraction-informal +
    mean elevation/NDVI from the 10 m layers within the cell."""
    with rasterio.open(EXPORT_DIR / f"class_{year}.tif") as cs:
        cls = cs.read(1).astype("float32")
        ctrans = cs.transform
        cnod = cs.nodata
    cls[cls == cnod] = np.nan
    with rasterio.open(EXPORT_DIR / f"ndvi_addis_{year}.tif") as ns:
        ndvi = ns.read(1).astype("float32")
        nnod = ns.nodata
    ndvi[ndvi == nnod] = np.nan
    with rasterio.open(EXPORT_DIR / "elevation_addis.tif") as es:
        elev = es.read(1).astype("float32")
        etrans = es.transform
        enod = es.nodata
    elev[elev == enod] = np.nan

    night_p = EXPORT_DIR / f"modis_night_addis_{year}.tif"
    day_p = EXPORT_DIR / f"modis_day_addis_{year}.tif"
    with rasterio.open(night_p) as ms:
        night = ms.read(1).astype("float64")
        mtrans = ms.transform
        mnod = ms.nodata
        mh, mw = ms.shape
    with rasterio.open(day_p) as ms:
        day = ms.read(1).astype("float64")
        dnod = ms.nodata

    ch, cw = cls.shape
    eh, ew = elev.shape
    rows = []
    for r in range(mh):
        for c in range(mw):
            nv = night[r, c]
            if mnod is not None and nv == mnod:
                continue
            # geographic bounds of this MODIS cell
            left, top = mtrans * (c, r)
            right, bottom = mtrans * (c + 1, r + 1)
            # -> class/ndvi (10 m) index window
            r0, c0 = rasterio.transform.rowcol(ctrans, left, top)
            r1, c1 = rasterio.transform.rowcol(ctrans, right, bottom)
            r0, r1 = sorted((max(0, r0), min(ch, r1)))
            c0, c1 = sorted((max(0, c0), min(cw, c1)))
            if r1 <= r0 or c1 <= c0:
                continue
            sub = cls[r0:r1, c0:c1]
            valid = np.isfinite(sub)
            k = int(valid.sum())
            if k < min_10m:
                continue
            frac = float(np.nansum(sub) / k)
            nsub = ndvi[r0:r1, c0:c1]
            # elevation grid is coarser/separate -> its own window
            er0, ec0 = rasterio.transform.rowcol(etrans, left, top)
            er1, ec1 = rasterio.transform.rowcol(etrans, right, bottom)
            er0, er1 = sorted((max(0, er0), min(eh, er1)))
            ec0, ec1 = sorted((max(0, ec0), min(ew, ec1)))
            esub = elev[er0:er1, ec0:ec1] if (er1 > er0 and ec1 > ec0) else np.array([np.nan])
            dv = day[r, c]
            rows.append({
                "year": year, "frac_informal": frac, "n10m": k,
                "night": nv,
                "day": (np.nan if (dnod is not None and dv == dnod) else dv),
                "ndvi": float(np.nanmean(nsub)) if np.isfinite(nsub).any() else np.nan,
                "elevation": float(np.nanmean(esub)) if np.isfinite(esub).any() else np.nan,
            })
    return pd.DataFrame(rows)


def purity_test(cells: pd.DataFrame, thresh: float) -> dict:
    d = cells.dropna(subset=["night", "elevation", "ndvi"]).copy()
    inf = d[d.frac_informal >= thresh]
    oth = d[d.frac_informal <= (1 - thresh)]
    out = {"thresh": thresh, "n_informal_cells": int(len(inf)),
           "n_other_cells": int(len(oth))}
    if len(inf) < 10 or len(oth) < 10:
        out["note"] = "too few pure cells for a stable contrast"
        return out
    out["raw_night_diff"] = float(inf.night.median() - oth.night.median())
    u, p = mannwhitneyu(inf.night, oth.night, alternative="two-sided")
    out["mw_p"] = float(p)
    # cell-level adjusted: night ~ frac_informal + elevation + ndvi on pure cells
    pure = pd.concat([inf.assign(grp=1), oth.assign(grp=0)])
    fit = smf.ols("night ~ grp + elevation + ndvi", pure).fit()
    ci = fit.conf_int().loc["grp"].tolist()
    out["adjusted_night_diff"] = float(fit.params["grp"])
    out["adjusted_ci95"] = [float(ci[0]), float(ci[1])]
    out["adjusted_p"] = float(fit.pvalues["grp"])
    return out


# --------------------------------------------------------------------------- #
def make_figures(perpix, night_frames):
    # (1) day-vs-night adjusted effect bars
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels, vals, errs, colors = [], [], [], []
    palette = {"Landsat day": "#F58518", "MODIS day": "#E45756", "MODIS night": "#4C78A8"}
    for year in (2017, 2024):
        for src in ("Landsat day", "MODIS day", "MODIS night"):
            e = perpix[year][src]
            if e.get("effect") is None:
                continue
            labels.append(f"{year}\n{src}")
            vals.append(e["effect"])
            errs.append((e["effect"] - e["ci95"][0]))
            colors.append(palette[src])
    x = np.arange(len(vals))
    ax.bar(x, vals, yerr=errs, color=colors, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Adjusted informal effect (°C)")
    ax.set_title("Informal thermal effect: negative = cooler, positive = hotter")
    fig.tight_layout(); fig.savefig(FIG_DIR / "day_vs_night_effect.png", dpi=150)
    plt.close(fig)

    # (2) night LST by class violin (per-pixel)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups, ticks, pos = [], [], []
    p = 0
    for year in (2017, 2024):
        d = night_frames[year]
        for cls in (0, 1):
            vals = d[d["class"] == cls]["night"].dropna().values
            if len(vals):
                groups.append(vals); ticks.append(f"{year}\n{'informal' if cls else 'other'}")
                pos.append(p)
            p += 1
        p += 0.6
    parts = ax.violinplot(groups, positions=pos, showmedians=True, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor("#E45756" if "informal" in ticks[i] else "#4C78A8")
        b.set_alpha(0.7)
    ax.set_xticks(pos); ax.set_xticklabels(ticks)
    ax.set_ylabel("MODIS night LST (°C)")
    ax.set_title("Nighttime surface temperature by predicted fabric class")
    fig.tight_layout(); fig.savefig(FIG_DIR / "night_lst_by_class.png", dpi=150)
    plt.close(fig)


def to_md(perpix, purity, landsat) -> str:
    L = ["# Nighttime thermal test (MODIS)\n",
         "## Adjusted informal effect by time-of-day (per-pixel model)\n",
         "Negative = informal cooler; positive = informal hotter. "
         "`effect` = class coefficient in `LST ~ class + NDVI + elevation`.\n",
         "| Year | Source | Effect (°C) | 95% CI | Raw ΔLST |",
         "|---|---|---|---|---|"]
    for year in (2017, 2024):
        for src in ("Landsat day", "MODIS day", "MODIS night"):
            e = perpix[year][src]
            if e.get("effect") is None:
                continue
            L.append(f"| {year} | {src} | {e['effect']:+.3f} | "
                     f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}] | "
                     f"{e['raw_median_diff']:+.2f} |")
    L += ["\n## Purity-restricted 1 km-cell test (cleaner contrast)\n",
          "Only MODIS cells that are >=70% one class; compares NIGHT LST.\n",
          "| Year | informal cells | other cells | raw night Δ (°C) | adjusted Δ (°C) | 95% CI | p |",
          "|---|---|---|---|---|---|---|"]
    for year in (2017, 2024):
        pt = purity[year]
        if "adjusted_night_diff" in pt:
            L.append(f"| {year} | {pt['n_informal_cells']} | {pt['n_other_cells']} | "
                     f"{pt['raw_night_diff']:+.2f} | {pt['adjusted_night_diff']:+.3f} | "
                     f"[{pt['adjusted_ci95'][0]:+.3f}, {pt['adjusted_ci95'][1]:+.3f}] | "
                     f"{pt['adjusted_p']:.2e} |")
        else:
            L.append(f"| {year} | {pt.get('n_informal_cells','?')} | "
                     f"{pt.get('n_other_cells','?')} | {pt.get('note','')} | | | |")
    # verdict
    n17 = perpix[2017]["MODIS night"].get("effect")
    n24 = perpix[2024]["MODIS night"].get("effect")
    L.append("\n## Read")
    if n17 is not None and n24 is not None:
        flipped = (n17 > 0 and n24 > 0)
        L.append(f"- Daytime Landsat effect: {landsat.get(2017,'?')} / {landsat.get(2024,'?')} °C (cooler).")
        L.append(f"- MODIS night effect: {n17:+.3f} / {n24:+.3f} °C.")
        if flipped:
            L.append("- **The sign FLIPS at night: informal fabric is HOTTER after dark** — "
                     "consistent with the hypothesis that daytime LST was the wrong proxy.")
        elif n17 < 0 and n24 < 0:
            L.append("- Informal stays cooler even at night — the daytime-proxy explanation is "
                     "NOT supported; the cooler-informal pattern is robust across time-of-day.")
        else:
            L.append("- Mixed/ambiguous across years — no clean night reversal.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-year", type=int, default=60000)
    ap.add_argument("--purity", type=float, default=0.70)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

    # Landsat-day effect for reference
    landsat = {}
    sp = RESULTS_DIR / "statistics.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        for r in s.get("per_year", []):
            landsat[r["year"]] = round(r.get("adjusted_class_effect_degC", float("nan")), 3)

    perpix, night_frames, purity = {}, {}, {}
    for year in (2017, 2024):
        coords = random_valid_coords(EXPORT_DIR / f"class_{year}.tif", args.n_per_year)
        d = pd.DataFrame({
            "class": sample1(EXPORT_DIR / f"class_{year}.tif", coords),
            "night": sample1(EXPORT_DIR / f"modis_night_addis_{year}.tif", coords),
            "day": sample1(EXPORT_DIR / f"modis_day_addis_{year}.tif", coords),
            "lst": sample1(EXPORT_DIR / f"lst_addis_{year}.tif", coords),
            "ndvi": sample1(EXPORT_DIR / f"ndvi_addis_{year}.tif", coords),
            "elevation": sample1(EXPORT_DIR / "elevation_addis.tif", coords),
        })
        d = d[d["class"].isin([0, 1])]
        night_frames[year] = d
        perpix[year] = {
            "Landsat day": adjusted_effect(d, "lst"),
            "MODIS day": adjusted_effect(d, "day"),
            "MODIS night": adjusted_effect(d, "night"),
        }
        cells = cell_table(year)
        purity[year] = purity_test(cells, args.purity)
        e = perpix[year]["MODIS night"]
        print(f"[{year}] MODIS-night per-pixel effect = "
              f"{e['effect']:+.3f} degC | purity-cell adjusted = "
              f"{purity[year].get('adjusted_night_diff', float('nan')):+.3f} "
              f"(inf={purity[year].get('n_informal_cells')} oth={purity[year].get('n_other_cells')})")

    (RESULTS_DIR / "nighttime.json").write_text(
        json.dumps({"per_pixel": perpix, "purity": purity, "landsat_day": landsat},
                   indent=2), encoding="utf-8")
    (RESULTS_DIR / "nighttime.md").write_text(to_md(perpix, purity, landsat), encoding="utf-8")
    make_figures(perpix, night_frames)
    print(f"Wrote {RESULTS_DIR/'nighttime.md'} and figures.")


if __name__ == "__main__":
    main()
