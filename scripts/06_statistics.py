#!/usr/bin/env python
"""
06_statistics.py
================
The analytical core (plan Sec. 6). For each year, sample a large random set of
valid pixels and, for each pixel, read predicted class + LST + NDVI + elevation
by coordinate (so rasters on different native grids still align).

Then:
  1. Mann-Whitney U test of LST between predicted classes (LST is not normal,
     so a non-parametric test rather than a raw mean difference).
  2. OLS regression  LST ~ class + NDVI + elevation  per year, to isolate the
     informal-settlement effect independent of vegetation and elevation. The
     `class` coefficient (with 95% CI) is the adjusted thermal penalty in degC.
  3. Pooled model  LST ~ class * year + NDVI + elevation. The class:year
     interaction term tests directly whether the penalty widened 2017->2024.

INPUTS  (data/exports/)
    class_2017.tif, class_2024.tif
    lst_addis_2017.tif, lst_addis_2024.tif
    ndvi_addis_2017.tif, ndvi_addis_2024.tif
    elevation_addis.tif
OUTPUTS
    results/statistics.json
    results/statistics.md
    data/processed/pixels_pooled.parquet
    figures/lst_by_class_year.png   (violin/box)

USAGE
    python scripts/06_statistics.py --n-per-year 100000
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
PROC_DIR = PROJECT_ROOT / "data" / "processed"

RNG = np.random.default_rng(42)


def random_valid_coords(class_path: Path, n: int) -> np.ndarray:
    """Pick up to n random valid (class in {0,1}) pixel-center coords."""
    with rasterio.open(class_path) as src:
        arr = src.read(1)
        transform = src.transform
    rows, cols = np.where((arr == 0) | (arr == 1))
    if len(rows) == 0:
        raise SystemExit(f"No valid class pixels in {class_path}")
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


def build_year_frame(year: int, n: int) -> pd.DataFrame:
    coords = random_valid_coords(EXPORT_DIR / f"class_{year}.tif", n)
    df = pd.DataFrame({
        "year": year,
        "class": sample1(EXPORT_DIR / f"class_{year}.tif", coords),
        "lst": sample1(EXPORT_DIR / f"lst_addis_{year}.tif", coords),
        "ndvi": sample1(EXPORT_DIR / f"ndvi_addis_{year}.tif", coords),
        "elevation": sample1(EXPORT_DIR / "elevation_addis.tif", coords),
    })
    df = df.dropna(subset=["class", "lst", "ndvi", "elevation"])
    df["class"] = df["class"].astype(int)
    df = df[df["class"].isin([0, 1])]
    return df


def per_year_stats(df: pd.DataFrame, year: int) -> dict:
    d = df[df.year == year]
    inf = d.loc[d["class"] == 1, "lst"]
    oth = d.loc[d["class"] == 0, "lst"]
    out = {
        "year": year, "n": int(len(d)),
        "n_informal": int(len(inf)), "n_other": int(len(oth)),
        "lst_median_informal": float(inf.median()) if len(inf) else None,
        "lst_median_other": float(oth.median()) if len(oth) else None,
        "raw_median_diff": (float(inf.median() - oth.median())
                            if len(inf) and len(oth) else None),
    }
    if len(inf) and len(oth):
        u, p = mannwhitneyu(inf, oth, alternative="two-sided")
        out["mannwhitney_U"] = float(u)
        out["mannwhitney_p"] = float(p)
        # rank-biserial effect size
        out["rank_biserial"] = float(1 - 2 * u / (len(inf) * len(oth)))

    # adjusted effect: OLS LST ~ class + ndvi + elevation
    fit = smf.ols("lst ~ Q('class') + ndvi + elevation", data=d).fit()
    coef = fit.params["Q('class')"]
    ci = fit.conf_int().loc["Q('class')"].tolist()
    out["adjusted_class_effect_degC"] = float(coef)
    out["adjusted_class_effect_ci95"] = [float(ci[0]), float(ci[1])]
    out["adjusted_class_p"] = float(fit.pvalues["Q('class')"])
    out["ols_r2"] = float(fit.rsquared)
    return out


def pooled_interaction(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["year_c"] = (d["year"] == 2024).astype(int)  # 0=2017, 1=2024
    # Interact the covariates with year too: the two dry seasons differ in
    # absolute temperature and in how NDVI/elevation map to LST. Forcing shared
    # covariate slopes (class*year + ndvi + elevation) makes the class:year term
    # absorb that mismatch and disagree with the per-year fits. Letting every
    # term vary by year isolates the class-effect *change* cleanly.
    fit = smf.ols("lst ~ Q('class') * year_c + ndvi * year_c + elevation * year_c",
                  data=d).fit()
    term = "Q('class'):year_c"
    ci = fit.conf_int().loc[term].tolist()
    return {
        "class_effect_2017_degC": float(fit.params["Q('class')"]),
        "interaction_class_x_2024_degC": float(fit.params[term]),
        "interaction_ci95": [float(ci[0]), float(ci[1])],
        "interaction_p": float(fit.pvalues[term]),
        "interpretation": (
            "positive interaction => penalty WIDENED 2017->2024; "
            "negative => narrowed; CI crossing 0 => not distinguishable"),
        "r2": float(fit.rsquared),
    }


def make_plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    groups, labels, positions = [], [], []
    pos = 0
    colors = {0: "#4C78A8", 1: "#E45756"}
    for year in (2017, 2024):
        for cls in (0, 1):
            vals = df[(df.year == year) & (df["class"] == cls)]["lst"].dropna()
            if len(vals):
                groups.append(vals.values)
                labels.append(f"{year}\n{'informal' if cls else 'other'}")
                positions.append(pos)
            pos += 1
        pos += 0.6
    parts = ax.violinplot(groups, positions=positions, showmedians=True,
                          widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        cls = 1 if "informal" in labels[i] else 0
        b.set_facecolor(colors[cls]); b.set_alpha(0.7)
    ax.set_xticks(positions); ax.set_xticklabels(labels)
    ax.set_ylabel("Land surface temperature (degC)")
    ax.set_title("LST by predicted fabric class and year")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def to_markdown(per_year: list[dict], pooled: dict) -> str:
    lines = ["# Thermal-inequity statistics\n"]
    lines.append("## Adjusted informal-settlement thermal penalty\n")
    lines.append("| Year | n | Raw median ΔLST (°C) | Adjusted effect (°C) | 95% CI | p |")
    lines.append("|---|---|---|---|---|---|")
    for r in per_year:
        ci = r.get("adjusted_class_effect_ci95", [float('nan'), float('nan')])
        lines.append(
            f"| {r['year']} | {r['n']:,} | "
            f"{r.get('raw_median_diff', float('nan')):.2f} | "
            f"{r.get('adjusted_class_effect_degC', float('nan')):.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{r.get('adjusted_class_p', float('nan')):.2e} |")
    lines.append("\n*Adjusted effect = coefficient on predicted class in "
                 "`LST ~ class + NDVI + elevation`, i.e. the LST gap attributable "
                 "to fabric type after removing vegetation and elevation.*\n")
    lines.append("## Did the penalty widen? (pooled class×year model)\n")
    lines.append(f"- Class effect in 2017: **{pooled['class_effect_2017_degC']:.3f} °C**")
    lines.append(f"- Additional effect in 2024 (interaction): "
                 f"**{pooled['interaction_class_x_2024_degC']:.3f} °C** "
                 f"(95% CI [{pooled['interaction_ci95'][0]:.3f}, "
                 f"{pooled['interaction_ci95'][1]:.3f}], p={pooled['interaction_p']:.2e})")
    lines.append(f"- {pooled['interpretation']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-year", type=int, default=100000)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    frames = [build_year_frame(y, args.n_per_year) for y in (2017, 2024)]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(PROC_DIR / "pixels_pooled.parquet", index=False)
    print(f"Pooled pixel sample: {len(df):,} rows")

    per_year = [per_year_stats(df, y) for y in (2017, 2024)]
    for r in per_year:
        print(f"[{r['year']}] adjusted penalty = "
              f"{r.get('adjusted_class_effect_degC', float('nan')):.3f} degC "
              f"(MW p={r.get('mannwhitney_p', float('nan')):.1e})")

    pooled = pooled_interaction(df)
    print(f"[pooled] interaction (class x 2024) = "
          f"{pooled['interaction_class_x_2024_degC']:.3f} degC, "
          f"p={pooled['interaction_p']:.1e}")

    results = {"per_year": per_year, "pooled_interaction": pooled}
    (RESULTS_DIR / "statistics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "statistics.md").write_text(
        to_markdown(per_year, pooled), encoding="utf-8")
    make_plot(df, FIG_DIR / "lst_by_class_year.png")
    print(f"Wrote results to {RESULTS_DIR} and figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
