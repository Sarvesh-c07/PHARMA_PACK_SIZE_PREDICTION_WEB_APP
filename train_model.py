"""
Train / evaluate the pack-dimension model from the command line.

Usage:
    python train_model.py            # cross-validated accuracy, then (re)build cache
    python train_model.py --eval     # cross-validated accuracy only, no cache write
    python train_model.py --rebuild  # force a fresh model cache, skip the CV report

The cross-validation here is the honest measure of accuracy: every reported number
is from packs held out of training. Use it to check whether a data change actually
helped (lower average MAE = better).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import app  # reuses the app's data loading + feature parsing
import ml_engine
from ml_engine import TARGETS, DimensionModel, _trainable

UNITS = {"length_mm": "mm", "width_mm": "mm", "height_mm": "mm", "weight_g": "g"}
LABELS = {"length_mm": "Length", "width_mm": "Width", "height_mm": "Height", "weight_g": "Weight"}


def cross_validate(records, folds: int = 5) -> None:
    usable = [r for r in records if _trainable(r)]
    if len(usable) < 50:
        print(f"Only {len(usable)} usable rows — need more measured data to evaluate.")
        return

    print(f"\nCross-validating on {len(usable):,} measured packs ({folds}-fold)…\n")
    base = DimensionModel()
    X = base.featurize(usable, fit=True)

    rng = np.random.default_rng(0)
    fold_id = rng.integers(0, folds, size=len(usable))

    print(f"{'Dimension':<10}{'baseline':>11}{'model MAE':>12}{'R²':>8}{'improvement':>14}")
    print("-" * 55)
    avg_impr = []
    for target in TARGETS:
        y = np.array(
            [(np.nan if ml_engine._missing(r.get(target)) else float(r.get(target))) for r in usable]
        )
        mask = ~np.isnan(y)
        if mask.sum() < 50:
            continue
        preds = np.full(len(usable), np.nan)
        for f in range(folds):
            tr = np.where(mask & (fold_id != f))[0]
            te = np.where(mask & (fold_id == f))[0]
            if len(tr) < 40 or len(te) == 0:
                continue
            m = ml_engine.ExtraTreesRegressor(
                n_estimators=ml_engine.N_TREES,
                max_features=0.5,
                min_samples_leaf=ml_engine.MIN_LEAF,
                max_depth=ml_engine.MAX_DEPTH,
                n_jobs=-1,
                random_state=0,
            )
            m.fit(X[tr], np.log1p(np.clip(y[tr], 0, None)))
            preds[te] = np.expm1(m.predict(X[te]))
        ok = ~np.isnan(preds) & mask
        yt, yp = y[ok], preds[ok]
        mae = float(np.mean(np.abs(yt - yp)))
        base_mae = float(np.mean(np.abs(yt - np.median(yt))))
        denom = np.sum((yt - yt.mean()) ** 2)
        r2 = float(1 - np.sum((yt - yp) ** 2) / denom) if denom else 0.0
        impr = (base_mae - mae) / base_mae * 100
        avg_impr.append(impr)
        u = UNITS[target]
        print(f"{LABELS[target]:<10}{base_mae:>9.1f}{u:<2}{mae:>10.1f}{u:<2}{r2:>8.2f}{impr:>12.0f}%")
    if avg_impr:
        print("-" * 55)
        print(f"Average error reduction vs. a naive guess: {np.mean(avg_impr):.0f}%")
    print(
        "\n'baseline' = always guessing the median. 'improvement' = how much the "
        "model beats that guess.\nLower MAE is better; compare these numbers before "
        "and after any data change.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate the pack-dimension model.")
    ap.add_argument("--eval", action="store_true", help="cross-validate only, don't write cache")
    ap.add_argument("--rebuild", action="store_true", help="force fresh cache, skip CV report")
    args = ap.parse_args()

    dataset = app.get_dataset()
    if dataset is None:
        print("No dataset found. Put data in data/ or run build_master_data.py first.")
        sys.exit(1)
    print(f"Loaded {len(dataset.records):,} records from {dataset.path.name}")

    if not args.rebuild:
        cross_validate(dataset.records)

    if not args.eval:
        cache = app.MODEL_CACHE_FILE
        for p in (cache, cache.with_suffix(".sig")):
            if p.exists():
                p.unlink()
        t = time.time()
        model, notes = ml_engine.train_or_load(dataset.records, cache, str(dataset.path))
        size_mb = cache.stat().st_size / 1e6 if cache.exists() else 0
        print(f"Built model cache in {time.time() - t:.0f}s ({size_mb:.0f} MB): {', '.join(notes)}")
        print("The web app will now start instantly and use this model.")


if __name__ == "__main__":
    main()
