# Improving prediction accuracy

This app now predicts each pack's length, width, height and weight with a trained
ExtraTrees machine-learning model, blended 65/35 with the older nearest-neighbour
estimate. On packs the model never saw during training, current typical error is
roughly:

| Dimension | Typical error (MAE) | Variance explained (R²) |
|-----------|--------------------:|------------------------:|
| Length    | ± ~23 mm            | ~0.50                   |
| Width     | ± ~13 mm            | ~0.53                   |
| Height    | ± ~10 mm            | ~0.55                   |
| Weight    | ± ~63 g             | ~0.50                   |

Those numbers are the honest starting point. Everything below is aimed at pushing
them down. They are ordered by how much impact they usually have on *your* data.

## 1. Feed real measurements back in (highest impact)

Every pack you actually measure and record becomes a training example. The model
improves the most when you add measured rows for products or forms it currently
guesses on.

- After you weigh/measure a real pack, add a row to a workbook in `data/raw/`
  with its true L, W, H and weight, then rebuild (`python build_master_data.py`).
  Delete the model cache (`work/model_cache.pkl`) or just restart — the app
  retrains automatically.
- Prioritise measuring items where the app showed **Low** or **Medium**
  confidence, or where the closest reference packs were all a poor match. Those
  are the gaps in the model's knowledge.
- Aim for coverage, not volume: 30 well-chosen new forms help more than 3,000
  more repeats of tablets you already predict well.

## 2. Fix the vial-vs-ampoule and form classification

Injectables are your largest category (over 2,000 vials + ampoules), and a vial
carton and an ampoule tray have very different dimensions. The classification
preview on the home page flags rows where the app was unsure (e.g. *"verify vial
vs ampoule"*). Each misclassified row teaches the model the wrong shape.

- Skim the classification preview and the `Review Notes` column in
  `data/master_training.csv` for `verify…` notes.
- Where the source form is ambiguous, make the raw workbook explicit: write
  "injection vial" or "injection ampoule" rather than just "inj" or "amp".
- Consistent, explicit dosage-form text is the single cheapest accuracy win after
  adding measurements, because form is the model's strongest categorical signal.

## 3. Capture the numeric pack fields more completely

The model's strongest numeric predictors are the ones that describe how much is in
the pack. When these are blank the model has to fall back on text alone.

- **Container fill volume (mL)** for vials, ampoules, bottles, respules.
- **Total unit count** and **strips × units per strip** for blisters.
- **Content weight (g)** for tubes, jars and sachets.
- **Bulk / master-pack count** when the row is a shipper, not a single pack.

If your raw workbooks have these in separate columns, keep them separate rather
than merging everything into one free-text "pack size" cell — the builder maps
named columns far more reliably than it parses prose.

## 4. Keep units and orientation consistent

- Record all dimensions in the same unit (mm is assumed). The builder tries to
  detect cm/inch, but an unlabelled "7.5" that means 75 mm becomes noise.
- The app already normalises every pack to *longest × middle × shortest*, so you
  don't need a fixed L/W/H orientation — but do make sure the three numbers are
  the three edges of the same pack, not, say, two edges plus a diagonal.
- Watch for Excel turning "10 x 2" into a date. The builder filters obvious date
  artefacts, but clean source data is better than filtered source data.

## 5. Retrain regularly, and watch the model card

- The model retrains whenever the underlying data changes (the cache is keyed to
  the row count and file). After any bulk edit, confirm the home-page **model
  card** still shows sensible error figures — a sudden jump means a data problem
  slipped in.
- Run `python train_model.py` any time to print fresh cross-validated accuracy
  without starting the web app. Use it to check whether a change actually helped.

## 6. Advanced levers (when the basics are exhausted)

These need code changes but can each shave a few more millimetres:

- **Per-category models.** Injectables, solid-oral blisters and tubes follow very
  different geometry. Training a separate model per `Form Category` (and routing a
  query to its category's model) usually beats one global model once you have a
  few hundred measured rows per category.
- **Richer text features.** The name/brand/pack text is currently hashed into 256
  buckets. Swapping in a TF-IDF vocabulary built from your actual corpus, or
  adding explicit flags ("lyophilized", "pre-filled", "multidose"), gives the
  trees cleaner signals.
- **Predict volume directly, then solve edges.** For some forms, pack volume is
  more predictable than individual edges; predicting volume + two aspect ratios
  and deriving L/W/H can tighten results.
- **Quantile models for the band.** The likely-range shown per metric comes from
  the spread across trees. Training explicit 10th/90th-percentile gradient-boosting
  models would give calibrated bands you could trust for min/max carton planning.
- **Outlier review.** A handful of physically impossible rows (a "tablet" 400 mm
  long) drag the trees. The builder already excludes the worst, but periodically
  sorting `master_training.csv` by each dimension and eyeballing the extremes pays
  off.

## How to tell if a change helped

Don't judge by a single prediction — one item can look better or worse by luck.
Always compare the cross-validated MAE from `train_model.py` before and after a
change. If the average MAE across the four targets went down, the change helped;
if it went up, revert it. That discipline is what separates real improvement from
moving numbers around.
