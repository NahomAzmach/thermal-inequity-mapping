#!/usr/bin/env python
"""
04_train_classifier.py
======================
Train the per-pixel informal-vs-other classifier on the 64-dim AlphaEarth
embedding, evaluated with POLYGON-GROUPED cross-validation.

Why grouped CV rather than one fixed test split: pixels within a polygon are
spatially correlated, so the honest unit of evaluation is the polygon. With a
modest number of hand-drawn polygons a single held-out split is dominated by
*which* few polygons happen to land in it (it can read as pure noise). Grouped
k-fold gives every polygon a turn in the held-out fold and aggregates
out-of-fold (OOF) predictions for a stable estimate.

Three models (mirroring AlphaEarth's shallow-probe protocol):
    * logreg  — linear probe (usually best on these embeddings; least overfit)
    * mlp     — small neural probe
    * xgb     — gradient-boosted trees baseline
The primary model = highest grouped-CV AUC.

Final models are REFIT on ALL labeled data (CV is only for the honest metric;
the deployed model should use every label). The scaler is refit on all data too.

INPUT   data/processed/samples_2024.parquet
OUTPUTS
    models/scaler.joblib
    models/{logreg,mlp,xgb}.joblib
    models/metrics.json          (grouped-CV metrics + 'primary' model name)
    figures/roc_cv.png, figures/confusion_cv.png, figures/polygon_separation.png

USAGE
    python scripts/04_train_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "processed" / "samples_2024.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "figures"

EMB_COLS = [f"A{i:02d}" for i in range(64)]
SEED = 42


def make_models():
    return {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                     max_iter=500, random_state=SEED,
                                     early_stopping=True, n_iter_no_change=20),
        "xgb": lambda: XGBClassifier(n_estimators=300, max_depth=4,
                                     learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric="logloss",
                                     random_state=SEED, n_jobs=-1),
    }


def grouped_oof(make, X, y, groups, n_splits):
    """Out-of-fold P(informal) via GroupKFold."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = make()
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def score_oof(y, oof, groups):
    pred = (oof >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    # per-polygon AUC: mean OOF prob per polygon vs its (single) class
    pp = pd.DataFrame({"g": groups, "y": y, "p": oof}).groupby("g").agg(
        y=("y", "first"), p=("p", "mean"))
    poly_auc = (roc_auc_score(pp.y, pp.p) if pp.y.nunique() > 1 else float("nan"))
    return {
        "cv_auc": float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan"),
        "cv_accuracy": float(accuracy_score(y, pred)),
        "cv_precision": float(pr), "cv_recall": float(rc), "cv_f1": float(f1),
        "per_polygon_auc": float(poly_auc),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main() -> None:
    if not SAMPLES.exists():
        raise SystemExit(f"Missing {SAMPLES}. Run script 03 first.")
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df = pd.read_parquet(SAMPLES)
    X = df[EMB_COLS].to_numpy()
    y = df["class"].to_numpy().astype(int)
    groups = df["polygon_id"].to_numpy()
    n_poly = df["polygon_id"].nunique()
    n_splits = int(min(5, n_poly))
    print(f"{len(df)} points across {n_poly} polygons; grouped {n_splits}-fold CV")
    if n_poly < 3 or len(np.unique(y)) < 2:
        raise SystemExit("Need >=3 polygons spanning both classes. Label more.")

    models = make_models()
    results, oofs = {}, {}
    for name, make in models.items():
        oof = grouped_oof(make, X, y, groups, n_splits)
        oofs[name] = oof
        results[name] = score_oof(y, oof, groups)
        r = results[name]
        print(f"[{name:6}] CV AUC={r['cv_auc']:.3f}  acc={r['cv_accuracy']:.3f}  "
              f"f1={r['cv_f1']:.3f}  per-polygon AUC={r['per_polygon_auc']:.3f}")

    primary = max(results, key=lambda k: (results[k]["cv_auc"]
                                          if np.isfinite(results[k]["cv_auc"]) else -1))
    print(f"Primary model (highest CV AUC): {primary}")

    # ---- refit each model on ALL data, save --------------------------------
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    for name, make in models.items():
        m = make(); m.fit(Xs, y)
        joblib.dump(m, MODEL_DIR / f"{name}.joblib")

    metrics_out = {"primary": primary, "n_points": int(len(df)),
                   "n_polygons": int(n_poly), "n_splits": n_splits,
                   "models": results}
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"Saved 3 models + scaler + metrics to {MODEL_DIR}")

    # ---- figures -----------------------------------------------------------
    # ROC (OOF) for all models
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(np.unique(y)) > 1:
        for name in models:
            fpr, tpr, _ = roc_curve(y, oofs[name])
            ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['cv_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Grouped-CV ROC — informal classifier"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "roc_cv.png", dpi=150); plt.close(fig)

    # Confusion (OOF) for primary
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=np.array(results[primary]["confusion_matrix"]),
        display_labels=["other", "informal"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{primary} — grouped-CV (OOF)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_cv.png", dpi=150); plt.close(fig)

    # Per-polygon separation strip
    pp = pd.DataFrame({"g": groups, "y": y, "p": oofs[primary]}).groupby("g").agg(
        y=("y", "first"), p=("p", "mean"))
    fig, ax = plt.subplots(figsize=(5, 4))
    for cls, color in [(0, "#4C78A8"), (1, "#E45756")]:
        sub = pp[pp.y == cls]
        ax.scatter(np.full(len(sub), cls) + np.random.uniform(-0.05, 0.05, len(sub)),
                   sub.p, color=color, alpha=0.8,
                   label=("informal" if cls else "other"))
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["other", "informal"])
    ax.set_ylabel("mean OOF P(informal)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-polygon separation"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "polygon_separation.png", dpi=150)
    plt.close(fig)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
