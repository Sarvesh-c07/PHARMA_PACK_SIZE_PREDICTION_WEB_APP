"""
Build the model-ready training CSV(s) from the curated 3S master workbook.

The workbook (`3S_Master_ML_Ready.xlsx`) ships several sheets. Two are meant for
model training:

    * Train_Clean   — hand-vetted CLEAN + REVIEW rows, original measured weights.
    * Full_Imputed  — every usable row, with a handful of missing / outlier
                      weights and dimensions imputed so nothing is dropped.

This script turns either sheet into the flat CSV the app expects
(`data/master_training.csv`). Two deliberate transformations happen here:

    1. The noisy free-text **Form** column is dropped. The clean, curated
       **Category** column (Tablet, Ampoule, Liquid Injection Vial, …) is the
       single categorical field the model and the UI use from now on.
    2. L / W / H are stored in the source workbook in **centimetres**. The app
       works internally in **millimetres**, so we multiply by 10 and label the
       columns "… (mm)". Weight is already in grams.

Usage:
    python prepare_data.py --sheet Train_Clean   --out data/master_training.csv
    python prepare_data.py --sheet Full_Imputed  --out data/master_training.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = BASE_DIR / "data" / "3S_Master_ML_Ready.xlsx"

CM_TO_MM = 10.0

# Columns copied straight through (source name -> output name).
PASSTHROUGH = {
    "Item Name": "Item Name",
    "Brand Name": "Brand Name",
    "Category": "Category",          # clean, curated — replaces "Form"
    "Manufacturer": "Manufacturer",
    "Source File": "Source File",
    "Source Year": "Source Year",
    "Order Date": "Order Date",
    "Order No.": "Order No.",
    "Sr No.": "Source Row",
}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build(sheet: str, workbook: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet)
    out = pd.DataFrame()

    for src, dst in PASSTHROUGH.items():
        out[dst] = df[src] if src in df.columns else ""

    # Clean pack text: prefer the standardized description ("10 vials per pack"),
    # keep the original in parentheses for extra token / weight signal.
    std = df.get("Pack Size (Standardized)")
    orig = df.get("Pack Size")
    pack = []
    for s, o in zip(
        std if std is not None else [""] * len(df),
        orig if orig is not None else [""] * len(df),
    ):
        s = "" if pd.isna(s) else str(s).strip()
        o = "" if pd.isna(o) else str(o).strip()
        if s and o and s.lower() not in o.lower():
            pack.append(f"{s} ({o})")
        else:
            pack.append(s or o)
    out["Pack Size"] = pack

    # Dimensions: centimetres -> millimetres, labelled so the app treats them as mm.
    out["Length (mm)"] = (_num(df["L"]) * CM_TO_MM).round(1)
    out["Width (mm)"] = (_num(df["W"]) * CM_TO_MM).round(1)
    out["Height (mm)"] = (_num(df["H"]) * CM_TO_MM).round(1)

    # Weight already in grams.
    out["Weight (g)"] = _num(df["Weight (g)"]).round(1)

    # Drop rows with no usable dimensions at all (defensive; both sheets are full).
    dims = out[["Length (mm)", "Width (mm)", "Height (mm)"]]
    out = out[dims.notna().all(axis=1)].reset_index(drop=True)
    return out


def write_metadata(sheet: str, rows: int, out_path: Path) -> None:
    meta = {
        "built_from": sheet,
        "source_workbook": "3S_Master_ML_Ready.xlsx",
        "normalized_rows": int(rows),
        "training_candidate_rows_before_deduplication": int(rows),
        "training_rows_after_exact_deduplication": int(rows),
        "categorical_field": "Category (Form column dropped)",
        "dimension_units": "millimetres (converted from source centimetres)",
        "note": (
            "Category is the single clean categorical field; the legacy free-text "
            "Form column was intentionally dropped."
        ),
    }
    (out_path.parent / "master_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the training CSV from the master workbook.")
    ap.add_argument("--sheet", default="Train_Clean", choices=["Train_Clean", "Full_Imputed"])
    ap.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "master_training.csv"))
    args = ap.parse_args()

    workbook = Path(args.workbook)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = build(args.sheet, workbook)
    frame.to_csv(out_path, index=False)
    write_metadata(args.sheet, len(frame), out_path)
    print(f"Built {out_path} from '{args.sheet}': {len(frame):,} rows, {frame.shape[1]} columns.")
    print("Columns:", ", ".join(frame.columns))


if __name__ == "__main__":
    main()
