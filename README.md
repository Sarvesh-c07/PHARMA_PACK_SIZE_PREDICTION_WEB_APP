# Pharmaceutical Pack Dimension Predictor

This local web app estimates an individual pharmaceutical product pack's:

- length, width and height (mm);
- calculated pack volume (cm³);
- approximate gross weight (g);
- confidence score; and
- closest historical reference packs with match reasons and source rows.

## Data it is built on

The app loads `data/master_training.csv` automatically. It is generated from the
curated master workbook `data/3S_Master_ML_Ready.xlsx` by `prepare_data.py`.

Two things are deliberately different from earlier versions:

1. **Category replaces Form.** The noisy free-text *Form* column (VIAL, pack,
   Amps, STRIP …) has been **dropped**. The single categorical field the model
   and the UI now use is the clean, curated **Category** (Tablet, Ampoule,
   Liquid Injection Vial, Dry Powder Injection Vial, Capsule, …). It drives
   neighbour matching, is a model feature, and is shown in the "closest
   reference packs" table and the prediction form's dropdown.
2. **Millimetres internally.** The workbook stores L/W/H in centimetres; the app
   works in millimetres, so `prepare_data.py` multiplies by 10 on the way in.

### Two trained builds

This project ships in two variants, differing only in which sheet trained it:

| Build          | Trained on   | Rows  | What it is                                             |
|----------------|--------------|-------|--------------------------------------------------------|
| **Train_Clean**| `Train_Clean`| 3,210 | Hand-vetted CLEAN + REVIEW rows, original measurements |
| **Full_Imputed**| `Full_Imputed`| 4,809 | Every usable row; missing/outlier values imputed      |

To rebuild either from the workbook:

```powershell
python prepare_data.py --sheet Train_Clean  --out data/master_training.csv
# or
python prepare_data.py --sheet Full_Imputed --out data/master_training.csv
python train_model.py            # cross-validate, then rebuild the model cache
```

## Run the app

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

On Windows you can also double-click **Start Pharma Predictor.bat**.

## Inputs considered

- product and brand name
- **category** (chosen from the dropdown of real data categories)
- strength or concentration
- strips and units per strip
- total pack units
- bottle, vial, ampoule or respule fill volume
- content weight for tubes, jars and sachets
- bulk or master quantity when identified

The predictor uses a **trained machine-learning model** (an ExtraTrees +
gradient-boosting ensemble, one pair per dimension, learned on the log of each
measurement) blended 65/35 with a transparent weighted nearest-neighbour
method. The neighbour list under each prediction shows the real historical packs
that support the estimate. Estimates are operational approximations and should
be replaced with actual measurements when available.

### How the model behaves

- On first launch it trains in the background (~60 s) while the app stays usable
  on the neighbour estimate, then upgrades automatically. The trained model is
  cached in `work/model_cache.pkl` (already included), so launches are instant.
- The home page shows a **model card** with honest held-out accuracy for length,
  width, height and weight.
- Each prediction shows a confidence score, a likely range per dimension, an
  interactive true-to-scale 3D pack box and 2D dimensioned drawings.

### Check or rebuild the model

```powershell
python train_model.py          # cross-validated accuracy, then rebuild the cache
python train_model.py --eval   # accuracy report only
python train_model.py --rebuild # force a fresh model cache
```

If the model ever looks wrong, delete `work/model_cache.pkl` and restart — it
retrains from the current data. Machine learning requires `scikit-learn`; if it
is not installed the app still runs on the nearest-neighbour method alone.
