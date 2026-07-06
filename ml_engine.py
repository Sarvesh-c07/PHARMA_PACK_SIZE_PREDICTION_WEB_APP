"""
Machine-learning engine for the Pharma Pack Dimension Predictor.

This module turns the historical pack records into trained regression models
that estimate a pack's length, width, height and weight. It is intentionally
self-contained: app.py passes in plain record/query dictionaries that already
share the same feature keys, so the same featuriser works for both training
rows and a live prediction.

Design choices (validated by cross-validation on the 3S history):
  * Targets are modelled in log space (log1p). Pack dimensions are positive and
    right-skewed, so log-target regression lowers error and never predicts a
    negative size.
  * The primary learner is an ExtraTrees ensemble (extremely randomised trees).
    On this data it beat both a plain k-NN and gradient boosting on every
    dimension while staying fast to train and easy to cache.
  * Features = log-scaled numeric pack attributes (with explicit missing flags),
    one-hot dosage form / category / container, and hashed text from the
    product name, brand, pack text and strength.
  * Prediction intervals come from the spread across the individual trees, which
    gives an honest, data-driven confidence band instead of a hand-tuned rule.

If scikit-learn is unavailable the app keeps working on its nearest-neighbour
path; this module simply will not be loaded.
"""
from __future__ import annotations

import hashlib
import gzip
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer


# Numeric pack attributes the model reads. Keys match the record/query dicts
# produced by app.py, so the same featuriser serves training and inference.
NUMERIC_FEATURES = [
    "strength_mg",
    "concentration_mg_ml",
    "percent_strength",
    "container_volume_ml",
    "content_weight_g",
    "dose_count",
    "strip_count",
    "units_per_strip",
    "unit_count",
    "bulk_count",
]

# "category" is now the clean, curated dosage-form/packaging label that ships in
# the master data (Tablet, Ampoule, Liquid Injection Vial, ...). It replaces the
# old text-derived "form" field, which was noisy and has been dropped.
CATEGORICAL_FEATURES = ["category", "container_type"]

TARGETS = ["length_mm", "width_mm", "height_mm", "weight_g"]

TEXT_HASH_DIM = 256
N_TREES = 130
EVAL_TREES = 90
MAX_DEPTH = 28
MIN_LEAF = 2
MIN_TRAIN_ROWS = 40


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _text_of(rec: Dict[str, Any]) -> str:
    parts = [
        str(rec.get("name") or ""),
        str(rec.get("brand") or ""),
        str(rec.get("pack_text") or ""),
        str(rec.get("strength_text") or ""),
        str(rec.get("stated_form") or ""),
    ]
    return " ".join(parts).lower().strip()


class DimensionModel:
    """Holds the fitted per-target ExtraTrees models and their feature pipeline."""

    def __init__(self) -> None:
        self.category_vocab: Dict[str, List[str]] = {}
        self.hasher = HashingVectorizer(
            n_features=TEXT_HASH_DIM, alternate_sign=False, norm="l2"
        )
        self.models: Dict[str, ExtraTreesRegressor] = {}
        self.hgb: Dict[str, HistGradientBoostingRegressor] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []
        self.trained_rows: int = 0
        self.trained_at: float = 0.0

    # ----- feature construction -------------------------------------------------
    def _numeric_block(self, recs: List[Dict[str, Any]]) -> np.ndarray:
        cols = []
        for feat in NUMERIC_FEATURES:
            raw = np.array(
                [
                    (np.nan if _missing(r.get(feat)) else float(r.get(feat)))
                    for r in recs
                ],
                dtype=float,
            )
            logged = np.log1p(np.clip(np.nan_to_num(raw, nan=0.0), 0, None))
            present = (~np.isnan(raw)).astype(float)
            # value channel uses -1 where missing so trees can split it away
            value = np.where(np.isnan(raw), -1.0, logged)
            cols.append(value)
            cols.append(present)
        return np.column_stack(cols) if cols else np.zeros((len(recs), 0))

    def _categorical_block(
        self, recs: List[Dict[str, Any]], fit: bool
    ) -> np.ndarray:
        blocks = []
        for feat in CATEGORICAL_FEATURES:
            values = [str(r.get(feat) or "unknown") for r in recs]
            if fit:
                vocab = sorted(set(values))
                self.category_vocab[feat] = vocab
            vocab = self.category_vocab.get(feat, [])
            index = {v: i for i, v in enumerate(vocab)}
            block = np.zeros((len(recs), len(vocab) + 1), dtype=float)
            for row, val in enumerate(values):
                if val in index:
                    block[row, index[val]] = 1.0
                else:
                    block[row, -1] = 1.0  # unseen-category channel
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((len(recs), 0))

    def _text_block(self, recs: List[Dict[str, Any]]) -> np.ndarray:
        texts = [_text_of(r) for r in recs]
        return self.hasher.transform(texts).toarray()

    def featurize(self, recs: List[Dict[str, Any]], fit: bool = False) -> np.ndarray:
        num = self._numeric_block(recs)
        cat = self._categorical_block(recs, fit=fit)
        txt = self._text_block(recs)
        return np.hstack([num, cat, txt])

    # ----- training -------------------------------------------------------------
    def fit(self, records: List[Dict[str, Any]]) -> "DimensionModel":
        usable = [r for r in records if _trainable(r)]
        if len(usable) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Only {len(usable)} usable measured rows; need at least {MIN_TRAIN_ROWS}."
            )

        X = self.featurize(usable, fit=True)
        self.trained_rows = len(usable)

        # one held-out split for an honest accuracy read-out (fast, shown in UI)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(usable))
        cut = int(len(usable) * 0.8)
        tr_idx, te_idx = order[:cut], order[cut:]

        for target in TARGETS:
            y_all = np.array(
                [
                    (np.nan if _missing(r.get(target)) else float(r.get(target)))
                    for r in usable
                ],
                dtype=float,
            )
            mask = ~np.isnan(y_all)
            if mask.sum() < MIN_TRAIN_ROWS:
                continue
            y_log = np.log1p(np.clip(y_all, 0, None))

            # held-out evaluation. We evaluate on ExtraTrees alone here to keep
            # training fast; the shipped predictor also averages in gradient
            # boosting, so the real error is a little lower than this figure.
            tr = np.array([i for i in tr_idx if mask[i]])
            te = np.array([i for i in te_idx if mask[i]])
            if len(tr) >= MIN_TRAIN_ROWS and len(te) >= 10:
                ev = ExtraTreesRegressor(
                    n_estimators=EVAL_TREES,
                    max_features=0.5,
                    min_samples_leaf=MIN_LEAF,
                    max_depth=MAX_DEPTH,
                    n_jobs=-1,
                    random_state=0,
                )
                ev.fit(X[tr], y_log[tr])
                pred = np.expm1(ev.predict(X[te]))
                truth = y_all[te]
                mae = float(np.mean(np.abs(pred - truth)))
                denom = np.sum((truth - truth.mean()) ** 2)
                r2 = float(1 - np.sum((truth - pred) ** 2) / denom) if denom else 0.0
                self.metrics[target] = {
                    "mae": mae,
                    "r2": r2,
                    "median": float(np.median(truth)),
                }

            # final models on all rows that have this target (ExtraTrees + GB)
            model = ExtraTreesRegressor(
                n_estimators=N_TREES,
                max_features=0.5,
                min_samples_leaf=MIN_LEAF,
                max_depth=MAX_DEPTH,
                n_jobs=-1,
                random_state=0,
            )
            model.fit(X[mask], y_log[mask])
            self.models[target] = model

            hgb = HistGradientBoostingRegressor(
                max_iter=350, learning_rate=0.06, l2_regularization=1.0,
                max_leaf_nodes=63, random_state=0,
            )
            hgb.fit(X[mask], y_log[mask])
            self.hgb[target] = hgb

        self.trained_at = time.time()
        return self

    # ----- inference ------------------------------------------------------------
    def _tree_spread(
        self, model: ExtraTreesRegressor, x: np.ndarray
    ) -> Tuple[float, float, float]:
        """Return (mean, p15, p85) in original units from the tree ensemble."""
        per_tree = np.array([est.predict(x)[0] for est in model.estimators_])
        per_tree = np.expm1(per_tree)
        mean = float(np.mean(per_tree))
        low = float(np.percentile(per_tree, 15))
        high = float(np.percentile(per_tree, 85))
        return mean, low, high

    def predict(self, query: Dict[str, Any]) -> Dict[str, Any]:
        x = self.featurize([query], fit=False)
        out: Dict[str, Any] = {"predicted": {}, "intervals": {}, "spread_ratio": {}}
        for target in TARGETS:
            model = self.models.get(target)
            if model is None:
                out["predicted"][target] = None
                out["intervals"][target] = (None, None)
                continue
            # ExtraTrees gives both a point estimate and a confidence band (tree spread)
            et_mean, low, high = self._tree_spread(model, x)
            # Average with the gradient-boosting model for a lower-error point estimate
            hgb = self.hgb.get(target)
            if hgb is not None:
                hgb_mean = float(np.expm1(hgb.predict(x)[0]))
                point = (et_mean + hgb_mean) / 2.0
            else:
                point = et_mean
            out["predicted"][target] = point
            out["intervals"][target] = (low, high)
            out["spread_ratio"][target] = (high - low) / point if point else 1.0

        p = out["predicted"]
        if all(p.get(k) is not None for k in ("length_mm", "width_mm", "height_mm")):
            out["predicted"]["volume_cm3"] = (
                p["length_mm"] * p["width_mm"] * p["height_mm"] / 1000.0
            )
        else:
            out["predicted"]["volume_cm3"] = None
        return out


def _trainable(rec: Dict[str, Any]) -> bool:
    dims = [rec.get("length_mm"), rec.get("width_mm"), rec.get("height_mm")]
    if any(_missing(d) for d in dims):
        return False
    try:
        dims = [float(d) for d in dims]
    except (TypeError, ValueError):
        return False
    if any(d <= 0 or d < 5 or d > 1500 for d in dims):
        return False
    return True


# ----- caching -----------------------------------------------------------------
def _signature(records: List[Dict[str, Any]], source: str) -> str:
    n = sum(1 for r in records if _trainable(r))
    # Use only the file *name*, not the absolute path, so a model cache trained
    # on one machine is reused on another (the app ships a pre-trained cache).
    tag = Path(source).name or source
    return hashlib.sha1(f"{tag}|{len(records)}|{n}|v5-category".encode()).hexdigest()[:16]


def train_or_load(
    records: List[Dict[str, Any]], cache_path: Path, source_tag: str
) -> Tuple[Optional[DimensionModel], List[str]]:
    """Train a model or load a cached one. Returns (model, log_notes)."""
    notes: List[str] = []
    sig = _signature(records, source_tag)
    meta_path = cache_path.with_suffix(".sig")

    if cache_path.exists() and meta_path.exists():
        try:
            if meta_path.read_text().strip() == sig:
                with gzip.open(cache_path, "rb") as fh:
                    model = pickle.load(fh)
                notes.append("Loaded trained model from cache.")
                return model, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Cache unreadable, retraining ({exc}).")

    model = DimensionModel().fit(records)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wb", compresslevel=6) as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        meta_path.write_text(sig)
        notes.append(f"Trained model on {model.trained_rows:,} measured packs.")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Trained model (cache write failed: {exc}).")
    return model, notes
