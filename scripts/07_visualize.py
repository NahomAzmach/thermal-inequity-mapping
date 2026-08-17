#!/usr/bin/env python
"""
07_visualize.py
===============
Render the video-ready visuals (plan Sec. 7.3):

  * figures/panels_<year>.png   3-panel Sentinel-2 RGB | class map | LST heatmap
  * figures/change_map.png      settlement-growth overlay (2017->2024)
  * figures/class_toggle.gif    2-frame GIF toggling class 2017 <-> 2024

The LST-by-class violin plot is produced by script 06.

INPUTS  (data/exports/)  — whatever exists is used; missing panels are skipped.
    rgb_addis_<year>.tif, class_<year>.tif, lst_addis_<year>.tif, change_map.tif

USAGE
    python scripts/07_visualize.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
FIG_DIR = PROJECT_ROOT / "figures"

CLASS_CMAP = ListedColormap(["#B8B8B8", "#E45756"])          # other, informal
CHANGE_CMAP = ListedColormap(["#DADADA", "#E45756", "#7C1D1D", "#4C78A8"])
CHANGE_LABELS = ["stable other", "expansion", "stable informal", "informal→other"]


def read1(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr


def read_rgb(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")  # (3,H,W)
        nd = src.nodata
    arr = np.moveaxis(arr, 0, -1)
    # nodata here is -inf; force all non-finite (and nodata) to NaN so the
    # percentile stretch isn't dragged to a blank image.
    arr[~np.isfinite(arr)] = np.nan
    if nd is not None and np.isfinite(nd):
        arr[arr == nd] = np.nan
    out = np.zeros(arr.shape, dtype="float32")
    for i in range(3):
        ch = arr[..., i]
        lo, hi = np.nanpercentile(ch, [2, 98])
        out[..., i] = np.clip((ch - lo) / (hi - lo + 1e-9), 0, 1)
    # paint nodata pixels white so the city shape reads cleanly
    nanmask = np.isnan(arr).any(axis=-1)
    out[nanmask] = 1.0
    return out


def panel_year(year: int) -> None:
    rgb_p = EXPORT_DIR / f"rgb_addis_{year}.tif"
    cls_p = EXPORT_DIR / f"class_{year}.tif"
    lst_p = EXPORT_DIR / f"lst_addis_{year}.tif"
    have = [p.exists() for p in (rgb_p, cls_p, lst_p)]
    if not any(have):
        print(f"  {year}: no rasters found, skipping panel")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(f"Addis Ababa — {year}", fontsize=14, weight="bold")

    if rgb_p.exists():
        axes[0].imshow(read_rgb(rgb_p))
    axes[0].set_title("Sentinel-2 RGB")

    if cls_p.exists():
        cls = read1(cls_p)
        axes[1].imshow(cls, cmap=CLASS_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[1].legend(handles=[Patch(color="#B8B8B8", label="other"),
                                Patch(color="#E45756", label="informal")],
                       loc="lower right", fontsize=8)
    axes[1].set_title("Predicted fabric class")

    if lst_p.exists():
        lst = read1(lst_p)
        im = axes[2].imshow(lst, cmap="inferno")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="LST (°C)")
    axes[2].set_title("Land surface temperature")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / f"panels_{year}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def change_figure() -> None:
    p = EXPORT_DIR / "change_map.tif"
    if not p.exists():
        print("  no change_map.tif, skipping")
        return
    arr = read1(p)
    fig, ax = plt.subplots(figsize=(7, 7))
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], CHANGE_CMAP.N)
    ax.imshow(arr, cmap=CHANGE_CMAP, norm=norm, interpolation="nearest")
    ax.legend(handles=[Patch(color=CHANGE_CMAP(i), label=CHANGE_LABELS[i])
                       for i in range(4)], loc="lower right", fontsize=8)
    ax.set_title("Settlement change 2017→2024", weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = FIG_DIR / "change_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def class_toggle_gif() -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter
    paths = {y: EXPORT_DIR / f"class_{y}.tif" for y in (2017, 2024)}
    if not all(p.exists() for p in paths.values()):
        print("  need both class rasters for the GIF, skipping")
        return
    imgs = {y: read1(p) for y, p in paths.items()}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(imgs[2017], cmap=CLASS_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    title = ax.set_title("2017")
    years = [2017, 2024]

    def update(i):
        y = years[i % 2]
        im.set_data(imgs[y])
        title.set_text(str(y))
        return im, title

    anim = FuncAnimation(fig, update, frames=4, interval=900, blit=False)
    out = FIG_DIR / "class_toggle.gif"
    anim.save(out, writer=PillowWriter(fps=1))
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for year in (2017, 2024):
        panel_year(year)
    change_figure()
    class_toggle_gif()
    print("Visualization done.")


if __name__ == "__main__":
    main()
