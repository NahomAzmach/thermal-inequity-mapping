#!/usr/bin/env python
"""
05_apply_and_change.py
======================
Apply the FROZEN 2024-trained classifier to BOTH years' embedding rasters
(train once, apply twice — plan Sec. 5), then derive the settlement-change map.

Because the decision function never changes, any 2017->2024 difference in the
output reflects real land-cover change, not labeling drift.

INPUTS
    models/scaler.joblib, models/<model>.joblib
    data/exports/embedding_addis_2017.tif
    data/exports/embedding_addis_2024.tif
OUTPUTS  (data/exports/)
    prob_informal_2017.tif, prob_informal_2024.tif    P(informal), 0..1
    class_2017.tif, class_2024.tif                     0=other, 1=informal
    change_map.tif                                     see CHANGE_CODES below

CHANGE_CODES
    0 = other in both years (stable non-informal)
    1 = other(2017) -> informal(2024)   == settlement EXPANSION
    2 = informal in both years          (stable informal)
    3 = informal(2017) -> other(2024)   (rare; redevelopment / label noise)

Processing is windowed by raster blocks so the 64-band stack never has to sit
in memory all at once.

USAGE
    python scripts/05_apply_and_change.py --model mlp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
EMB_BANDS = 64


def apply_to_raster(emb_path: Path, scaler, model,
                    prob_out: Path, class_out: Path,
                    block_rows: int = 256) -> None:
    with rasterio.open(emb_path) as src:
        prof = src.profile
        H, W = src.height, src.width
        nodata = src.nodata

        prob_prof = prof.copy()
        prob_prof.update(count=1, dtype="float32", nodata=-1.0)
        cls_prof = prof.copy()
        cls_prof.update(count=1, dtype="uint8", nodata=255)

        with rasterio.open(prob_out, "w", **prob_prof) as pdst, \
             rasterio.open(class_out, "w", **cls_prof) as cdst:
            for row in range(0, H, block_rows):
                h = min(block_rows, H - row)
                win = Window(0, row, W, h)
                arr = src.read(window=win).astype("float32")  # (64, h, W)
                bands, hh, ww = arr.shape
                flat = arr.reshape(bands, hh * ww).T          # (N, 64)

                valid = np.isfinite(flat).all(axis=1)
                if nodata is not None:
                    valid &= ~(flat == nodata).any(axis=1)

                prob = np.full(hh * ww, -1.0, dtype="float32")
                cls = np.full(hh * ww, 255, dtype="uint8")
                if valid.any():
                    Xs = scaler.transform(flat[valid])
                    p = model.predict_proba(Xs)[:, 1].astype("float32")
                    prob[valid] = p
                    cls[valid] = (p >= 0.5).astype("uint8")

                pdst.write(prob.reshape(hh, ww)[None, :, :], window=win)
                cdst.write(cls.reshape(hh, ww)[None, :, :], window=win)
    print(f"  wrote {prob_out.name}, {class_out.name}")


def build_change_map(c2017: Path, c2024: Path, out: Path) -> None:
    with rasterio.open(c2017) as a, rasterio.open(c2024) as b:
        prof = a.profile.copy()
        A = a.read(1)
        B = b.read(1)
    valid = (A != 255) & (B != 255)
    change = np.full(A.shape, 255, dtype="uint8")
    o2017, i2017 = (A == 0), (A == 1)
    o2024, i2024 = (B == 0), (B == 1)
    change[valid & o2017 & o2024] = 0  # stable other
    change[valid & o2017 & i2024] = 1  # expansion
    change[valid & i2017 & i2024] = 2  # stable informal
    change[valid & i2017 & o2024] = 3  # informal -> other
    prof.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(change[None, :, :])

    tot = valid.sum()
    if tot:
        exp = (change == 1).sum()
        print(f"  change map: expansion pixels = {exp:,} "
              f"({100*exp/tot:.2f}% of valid); stable-informal = "
              f"{(change==2).sum():,}")
    print(f"  wrote {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="auto",
                    choices=["auto", "logreg", "mlp", "xgb"],
                    help="'auto' uses the primary model chosen by script 04.")
    ap.add_argument("--block-rows", type=int, default=256)
    args = ap.parse_args()

    model_name = args.model
    if model_name == "auto":
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
        model_name = metrics.get("primary", "logreg")

    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    print(f"Loaded scaler + {model_name} model.")

    for year in (2017, 2024):
        emb = EXPORT_DIR / f"embedding_addis_{year}.tif"
        if not emb.exists():
            raise SystemExit(f"Missing {emb}. Run script 01 for {year} first.")
        print(f"[{year}] applying classifier ...")
        apply_to_raster(
            emb, scaler, model,
            EXPORT_DIR / f"prob_informal_{year}.tif",
            EXPORT_DIR / f"class_{year}.tif",
            block_rows=args.block_rows,
        )

    print("Building change map ...")
    build_change_map(
        EXPORT_DIR / "class_2017.tif",
        EXPORT_DIR / "class_2024.tif",
        EXPORT_DIR / "change_map.tif",
    )
    print("Done.")


if __name__ == "__main__":
    main()
