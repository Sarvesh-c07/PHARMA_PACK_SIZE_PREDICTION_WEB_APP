from __future__ import annotations

import datetime as dt
try:
    import cgi  # removed in Python 3.13; only used for the optional file upload
    HAS_CGI = True
except Exception:  # noqa: BLE001
    cgi = None
    HAS_CGI = False
import html
import io
import json
import math
import os
import re
import sys
import traceback
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ml_engine

    ML_AVAILABLE = True
    ML_IMPORT_ERROR = ""
except Exception as _ml_exc:  # noqa: BLE001  (sklearn etc. may be missing)
    ml_engine = None
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(_ml_exc)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "work" / "uploads"
STATE_FILE = UPLOAD_DIR / "last_upload.json"
MASTER_TRAINING_FILE = DATA_DIR / "master_training.csv"
MASTER_METADATA_FILE = DATA_DIR / "master_metadata.json"

SUPPORTED_FILES = {".xlsx", ".xls", ".xlsm", ".csv", ".tsv"}


FORM_CATEGORY = {
    "tablet": "solid_oral",
    "capsule": "solid_oral",
    "softgel capsule": "solid_oral",
    "lozenge": "solid_oral",
    "powder / granules": "powder",
    "sachet": "powder_liquid_unit",
    "oral liquid / syrup": "oral_liquid",
    "oral suspension": "oral_liquid",
    "oral drops": "oral_liquid",
    "oral solution": "oral_liquid",
    "injection vial": "injectable",
    "injection ampoule": "injectable",
    "pre-filled syringe": "injectable_device",
    "cartridge": "injectable_device",
    "injection pen / auto-injector": "injectable_device",
    "iv infusion": "injectable_large_volume",
    "inhaler": "respiratory",
    "nebuliser solution / respule": "respiratory",
    "eye drops": "ophthalmic",
    "eye ointment / gel": "ophthalmic",
    "ear drops": "otic",
    "nasal drops": "nasal",
    "nasal spray": "nasal",
    "cream": "topical",
    "ointment": "topical",
    "gel": "topical",
    "lotion": "topical",
    "topical solution": "topical",
    "topical spray": "topical",
    "dusting powder": "topical",
    "shampoo / wash": "topical",
    "suppository": "rectal_vaginal",
    "pessary / vaginal tablet": "rectal_vaginal",
    "vaginal cream / gel": "rectal_vaginal",
    "transdermal patch": "patch",
    "oral spray": "oral_spray",
    "unknown": "unknown",
}


NUMERIC_FEATURE_WEIGHTS = {
    "strength_mg": 0.6,
    "concentration_mg_ml": 1.0,
    "percent_strength": 0.8,
    "container_volume_ml": 2.1,
    "content_weight_g": 1.8,
    "dose_count": 1.2,
    "strip_count": 1.5,
    "units_per_strip": 1.5,
    "unit_count": 2.1,
    "bulk_count": 2.3,
}


TEXT_STOPWORDS = {
    "mg",
    "mcg",
    "g",
    "ml",
    "tablet",
    "tablets",
    "tab",
    "tabs",
    "capsule",
    "capsules",
    "cap",
    "caps",
    "vial",
    "vials",
    "amp",
    "amps",
    "ampoule",
    "ampoules",
    "bottle",
    "box",
    "pack",
    "strip",
    "strips",
    "sachet",
    "tube",
}


@dataclass
class WorkbookData:
    path: Path
    sheets: Dict[str, pd.DataFrame]
    combined: pd.DataFrame
    schema: Dict[str, Optional[str]]
    dimension_factors: Dict[str, float]
    weight_factors: Dict[str, float]
    records: List[Dict[str, Any]]
    usable_dimension_rows: int
    notes: List[str] = field(default_factory=list)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).replace("_x000D_", " ").replace("\r", " ").replace("\n", " ").strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def norm_col(name: Any) -> str:
    text = clean_text(name).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def numeric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def first_float(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def unit_multiplier(unit: str, target: str) -> float:
    unit = unit.lower().strip()
    if target == "mm":
        if unit in {"mm", "millimeter", "millimeters"}:
            return 1.0
        if unit in {"cm", "centimeter", "centimeters"}:
            return 10.0
        if unit in {"m", "meter", "meters"}:
            return 1000.0
        if unit in {"inch", "inches", "in", '"'}:
            return 25.4
    if target == "g":
        if unit in {"kg", "kilogram", "kilograms"}:
            return 1000.0
        if unit in {"g", "gm", "gram", "grams"}:
            return 1.0
        if unit in {"mg", "milligram", "milligrams"}:
            return 0.001
        if unit in {"mcg", "µg", "ug", "microgram", "micrograms"}:
            return 0.000001
    if target == "mg":
        if unit in {"g", "gm", "gram", "grams"}:
            return 1000.0
        if unit in {"mg", "milligram", "milligrams"}:
            return 1.0
        if unit in {"mcg", "µg", "ug", "microgram", "micrograms"}:
            return 0.001
    return 1.0


def header_mentions(header: str, choices: Iterable[str]) -> bool:
    h = norm_col(header)
    return any(choice in h for choice in choices)


def infer_dimension_factor(header: str, series: pd.Series) -> float:
    h = norm_col(header)
    if "mm" in h:
        return 1.0
    if "cm" in h:
        return 10.0
    if re.search(r"\b(in|inch|inches)\b", h):
        return 25.4

    values = []
    for value in series.dropna().head(200):
        n = numeric_value(value)
        if n and n > 0:
            values.append(n)
    if not values:
        return 1.0

    median = float(np.median(values))
    # In pharma carton/retail packing files, unlabeled dimensions like 11 x 6 x 4
    # are usually centimeters. Values above 50 are more likely already millimeters.
    if 1.0 <= median <= 45.0:
        return 10.0
    return 1.0


def infer_weight_factor(header: str) -> float:
    h = norm_col(header)
    if re.search(r"\bkg\b|kilogram", h):
        return 1000.0
    if re.search(r"\bmg\b|milligram", h):
        return 0.001
    if re.search(r"\bmcg\b|microgram|µg|ug", h):
        return 0.000001
    return 1.0


def parse_dimension_text(text: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    raw = clean_text(text).lower().replace("×", "x")
    if not raw:
        return None, None, None
    pattern = (
        r"(\d+(?:\.\d+)?)\s*(?:mm|cm|inch|inches|in|m)?\s*"
        r"(?:x|\*|by)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:mm|cm|inch|inches|in|m)?\s*"
        r"(?:x|\*|by)\s*"
        r"(\d+(?:\.\d+)?)\s*(mm|cm|inch|inches|in|m)?"
    )
    matches = list(re.finditer(pattern, raw))
    if not matches:
        return None, None, None
    parsed = []
    for match in matches:
        vals = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
        unit = match.group(4) or ""
        if not unit:
            tail = raw[match.end() : match.end() + 10]
            unit_match = re.search(r"\b(mm|cm|inch|inches|in|m)\b", tail)
            unit = unit_match.group(1) if unit_match else ""
        factor = unit_multiplier(unit, "mm") if unit else (10.0 if max(vals) <= 45 else 1.0)
        converted = [vals[0] * factor, vals[1] * factor, vals[2] * factor]
        parsed.append((converted[0] * converted[1] * converted[2], converted))
    # When a cell contains inner and outer box sizes, choose the larger box because
    # it is usually the sale/export pack dimension.
    parsed.sort(key=lambda item: item[0], reverse=True)
    chosen = parsed[0][1]
    return chosen[0], chosen[1], chosen[2]


def parse_weight_text(text: Any, default_factor: float = 1.0) -> Optional[float]:
    raw = clean_text(text).lower().replace(",", "")
    if not raw:
        return None
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(kg|g|gm|gram|grams|mg)\b", raw))
    if matches:
        values = []
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2)
            # In this workbook's WEIGHT column, a few package weights are written
            # as "68mg"/"216mg" even though the surrounding values are grams. A
            # retail/export pack weighing tens of milligrams is physically
            # implausible, so treat those as gram typos.
            if unit == "mg" and value >= 10:
                values.append(value)
            else:
                values.append(value * unit_multiplier(unit, "g"))
        return max(values)
    n = numeric_value(text)
    return n * default_factor if n is not None else None


def best_column(columns: Iterable[str], include: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    scored: List[Tuple[float, str]] = []
    include = [norm_col(x) for x in include]
    exclude = [norm_col(x) for x in exclude]
    for col in columns:
        n = norm_col(col)
        if not n:
            continue
        if any(term and term in n for term in exclude):
            continue
        score = 0.0
        for term in include:
            if not term:
                continue
            if n == term:
                score += 10
            elif n.startswith(term + " ") or n.endswith(" " + term):
                score += 6
            elif term in n:
                score += 3
        if score:
            # Prefer shorter, direct column names over long descriptive notes.
            score -= min(len(n) / 100.0, 2.0)
            scored.append((score, str(col)))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def exact_column(columns: Iterable[str], *choices: str) -> Optional[str]:
    lookup = {norm_col(column): str(column) for column in columns}
    for choice in choices:
        match = lookup.get(norm_col(choice))
        if match:
            return match
    return None


def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    columns = [str(c) for c in df.columns]

    schema = {
        "name": exact_column(
            columns,
            "Product Name",
            "Item Name",
            "Medicine Generic Name",
            "Generic Name",
        )
        or best_column(
            columns,
            ["product name", "item name", "medicine generic name", "generic name", "description", "product", "item", "name"],
            ["form", "pack", "dimension", "size"],
        ),
        "brand": exact_column(columns, "Brand Name", "Brand"),
        # Curated, clean dosage-form / packaging label from the master data.
        # When present it is the authoritative category and replaces the old
        # text-derived "form" field for matching, features and display.
        "category": exact_column(columns, "Category", "Product Category", "Dosage Category"),
        "manufacturer": exact_column(columns, "Manufacturer", "Manufacturing Name", "Manufacturing Company")
        or best_column(columns, ["manufacturer", "manufacturing name", "mfg name"], ["date"]),
        "date": exact_column(columns, "Date", "Order Date", "Inward Date"),
        "form": exact_column(
            columns,
            "Dosage Form",
            "Form",
            "Strip/Vials/Bottles/Amp",
            "Strip/Vials/Bottles/Ampoules",
        )
        or best_column(
            columns,
            ["dosage form", "form", "strip vials bottles amp", "type", "pharma form", "product type", "category"],
            ["dimension", "pack size", "size"],
        ),
        "strength": exact_column(columns, "Strength Text", "Strength", "Dosage")
        or best_column(
            columns,
            ["strength", "dosage", "dose", "composition", "potency", "concentration"],
            ["dimension", "pack"],
        ),
        "pack": exact_column(columns, "Pack Size", "Additional Details", "Pack Details")
        or best_column(
            columns,
            ["pack size", "additional details", "pack details", "packing", "packaging", "pack", "pack qty", "pack quantity", "package"],
            ["dimension"],
        ),
        "bulk": exact_column(columns, "Bulk Packing", "Bulk Pack", "Master Carton")
        or best_column(
            columns,
            ["bulk pack", "bulk packing", "master carton", "shipper", "case pack", "outer packing", "carton pack"],
            ["dimension"],
        ),
        "dimension_text": exact_column(columns, "Dimension LxWxH", "Dimensions", "Combined Dimensions")
        or best_column(
            columns,
            ["dimension", "dimensions", "box dimension", "carton dimension", "size l w h", "l x b x h", "lxbxh", "lwh"],
            ["pack size"],
        ),
        "length": exact_column(columns, "Length (mm)", "Length mm", "Length", "L")
        or best_column(
            columns,
            ["length", "len", "long", "l mm", "l cm", "l"],
            ["pack", "bulk", "weight"],
        ),
        "width": exact_column(columns, "Width (mm)", "Width mm", "Width", "Breadth", "W")
        or best_column(
            columns,
            ["width", "breadth", "wide", "w mm", "w cm", "b mm", "b cm", "w", "b"],
            ["weight", "pack", "bulk"],
        ),
        "height": exact_column(columns, "Height (mm)", "Height mm", "Height", "H")
        or best_column(
            columns,
            ["height", "depth", "ht", "h mm", "h cm", "d mm", "d cm", "h", "d"],
            ["weight", "pack", "bulk"],
        ),
        "weight": exact_column(columns, "Weight (g)", "Weight g", "Weight", "WEIGHT")
        or best_column(
            columns,
            ["gross weight", "net weight", "weight", "wt", "gw", "nw", "mass"],
            ["dosage", "strength", "mg"],
        ),
        "source_file": exact_column(columns, "Source File"),
        "source_sheet": exact_column(columns, "Source Sheet"),
        "source_row": exact_column(columns, "Source Row"),
        "source_date": exact_column(columns, "Source Date"),
        "occurrence_count": exact_column(columns, "Occurrence Count"),
    }

    # When the clean Category column exists, don't let the legacy free-text
    # "form" detector alias onto it — Category is authoritative on its own.
    if schema.get("category") and schema.get("form") == schema.get("category"):
        schema["form"] = None

    # Avoid using the same ambiguous "size" column for both pack and dimensions.
    if schema["pack"] == schema["dimension_text"]:
        name = norm_col(schema["pack"])
        if "dimension" in name or "l x" in name or "lwh" in name:
            schema["pack"] = None
        else:
            schema["dimension_text"] = None

    # The normalized master keeps the original dimension text for audit, but
    # its explicit millimetre columns are the authoritative model inputs.
    if all(
        exact_column(columns, label)
        for label in ["Length (mm)", "Width (mm)", "Height (mm)"]
    ):
        schema["dimension_text"] = None

    return schema


def make_unique_headers(values: List[Any]) -> List[str]:
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for i, value in enumerate(values):
        header = clean_text(value) or f"Column_{i + 1}"
        if header in seen:
            seen[header] += 1
            header = f"{header}_{seen[header]}"
        else:
            seen[header] = 1
        headers.append(header)
    return headers


def read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()

    max_scan = min(15, len(raw))
    best_idx = 0
    best_score = -1.0
    for idx in range(max_scan):
        row = raw.iloc[idx].tolist()
        non_empty = sum(bool(clean_text(v)) for v in row)
        text_like = sum(bool(re.search(r"[A-Za-z]", clean_text(v))) for v in row)
        unique = len({clean_text(v).lower() for v in row if clean_text(v)})
        score = non_empty + text_like * 1.4 + unique * 0.2
        if score > best_score:
            best_score = score
            best_idx = idx

    headers = make_unique_headers(raw.iloc[best_idx].tolist())
    data = raw.iloc[best_idx + 1 :].copy()
    data.columns = headers
    data = data.dropna(how="all")
    data.reset_index(drop=True, inplace=True)
    return data


def read_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {path.stem: pd.read_csv(path, dtype=object)}
    if suffix == ".tsv":
        return {path.stem: pd.read_csv(path, sep="\t", dtype=object)}
    xls = pd.ExcelFile(path)
    sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name in xls.sheet_names:
        try:
            df = read_excel_sheet(path, sheet_name)
            if not df.empty:
                sheets[sheet_name] = df
        except Exception as exc:
            sheets[f"{sheet_name} (read error)"] = pd.DataFrame({"Read error": [str(exc)]})
    return sheets


def combine_sheets(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        copy = df.copy()
        copy.insert(0, "_sheet", sheet_name)
        frames.append(copy)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def get_value(row: pd.Series, col: Optional[str]) -> str:
    if not col or col not in row.index:
        return ""
    return clean_text(row.get(col))


def words(text: str) -> set:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower()))
    return {token for token in tokens if token not in TEXT_STOPWORDS and len(token) > 2}


def classify_form(name: str, stated_form: str, pack: str, strength: str, parsed: Dict[str, Any]) -> Tuple[str, List[str]]:
    text = f"{name} {stated_form} {pack} {strength}".lower()
    notes: List[str] = []

    def has(pattern: str) -> bool:
        return bool(re.search(pattern, text, flags=re.I))

    original = stated_form.lower()
    form = "unknown"

    # Prefer the medicine's named dosage form over generic container labels in
    # legacy columns (for example, eye drops were sometimes labelled "VIAL").
    if has(r"\bsoft\s*gel\b|softgel|soft gelatin"):
        form = "softgel capsule"
    elif has(r"\b(pfs|pre[-\s]?filled syringe|prefilled syringe|syringe)\b"):
        form = "pre-filled syringe"
    elif has(r"\b(auto[-\s]?injectors?|injection pens?|pens?)\b"):
        form = "injection pen / auto-injector"
    elif has(r"\bcartridge\b"):
        form = "cartridge"
    elif has(r"\b(respules?|nebules?|nebuliser|nebulizer)\b"):
        form = "nebuliser solution / respule"
    elif has(r"\b(inhaler|mdi|dpi)\b"):
        form = "inhaler"
    elif has(r"\beye\b.*\b(drop|drops)\b|ophthalmic.*(drop|solution)"):
        form = "eye drops"
    elif has(r"\beye\b.*\b(ointment|gel)\b|ophthalmic.*(ointment|gel)"):
        form = "eye ointment / gel"
    elif has(r"\bear\b.*\bdrop|otic"):
        form = "ear drops"
    elif has(r"\bnasal\b.*\bdrop"):
        form = "nasal drops"
    elif has(r"\bnasal\b.*\bspray"):
        form = "nasal spray"
    elif has(r"\bvaginal\b.*\b(cream|gel)\b"):
        form = "vaginal cream / gel"
    elif has(r"\bpessary\b|vaginal tablet"):
        form = "pessary / vaginal tablet"
    elif has(r"\bsuppositor"):
        form = "suppository"
    elif has(r"\bpatch(es)?\b|transdermal"):
        form = "transdermal patch"
    elif has(r"\btopical solution\b"):
        form = "topical solution"
    elif has(r"\btopical spray\b"):
        form = "topical spray"
    elif has(r"\bdusting powder\b"):
        form = "dusting powder"
    elif has(r"\bshampoo\b|\bwash\b"):
        form = "shampoo / wash"
    elif has(r"\boral spray\b"):
        form = "oral spray"
    elif has(r"\bspray\b") and has(r"\bdose\b"):
        form = "oral spray"
    elif has(r"\boral suspension\b|\bsuspension\b|\bsusp\b"):
        form = "oral suspension"
    elif has(r"\boral drops?\b"):
        form = "oral drops"
    elif has(r"\boral solution\b|drinkable vial"):
        form = "oral solution"
    elif has(r"\bsyrup\b|\belixir\b|\boral liquid\b"):
        form = "oral liquid / syrup"
    elif has(r"\b(iv|intravenous)\b.*\b(infusion|solution)\b|\binfusion\b|\biv bag\b") and (
        not has(r"\bvials?\b|flacon")
        or bool(parsed.get("container_volume_ml") and parsed["container_volume_ml"] >= 50)
    ):
        form = "iv infusion"
    elif has(r"\bamp(?:oules?|ules?|uls?|s)?\b"):
        form = "injection ampoule"
    elif has(r"\bvial(s)?\b|flacon"):
        form = "injection vial"
    elif has(r"\btablet(s)?\b|\btab(s)?\b"):
        form = "tablet"
    elif has(r"\bcapsule(s)?\b|\bcap(s)?\b"):
        form = "capsule"
    elif has(r"\blozenge(s)?\b"):
        form = "lozenge"
    elif has(r"\bsachet(s)?\b"):
        form = "sachet"
    elif has(r"\bpowder\b|granule"):
        form = "powder / granules"
    elif has(r"\bcream\b"):
        form = "cream"
    elif has(r"\boin?tment\b"):
        form = "ointment"
    elif has(r"\bgel\b"):
        form = "gel"
    elif has(r"\blotion\b"):
        form = "lotion"
    elif has(r"\bspray\b"):
        form = "topical spray"
    elif has(r"\bdrop(s)?\b") and parsed.get("container_volume_ml"):
        form = "oral drops"
    elif has(r"\bsolution\b") and has(r"\b(povidone|iodine|topical|external)\b"):
        form = "topical solution"
    elif has(r"\bbottle(s)?\b") and parsed.get("container_volume_ml"):
        form = "oral liquid / syrup"
    elif has(r"\bstrips?\b"):
        form = "tablet"
    elif has(r"\bsolution\b") and parsed.get("container_volume_ml"):
        form = "oral solution"

    volume = parsed.get("container_volume_ml")
    mentions_amp = bool(re.search(r"\bamp(?:oules?|ules?|uls?|s)?\b", text))
    mentions_vial = bool(re.search(r"\bvial(s)?\b|flacon", text))
    mentions_injection = bool(re.search(r"\binj\b|inject|i\.v\.|iv\b", text))

    if form == "injection vial" and mentions_amp:
        form = "injection ampoule"
        notes.append("Corrected to ampoule because amp/ampoule appears in the item or pack text.")

    if form == "injection ampoule" and volume and volume > 10:
        notes.append("Ampoule volume is above 10 mL; check whether this is really a vial/bottle/bag or a data-entry issue.")

    if mentions_vial and volume and volume <= 10 and not mentions_amp:
        notes.append("Small-volume vial; verify if supplier actually means ampoule. The app does not auto-convert without ampoule text.")

    if form == "unknown" and mentions_injection and volume and volume <= 10:
        form = "injection vial"
        notes.append("Classified as small-volume injectable; verify vial vs ampoule.")
    elif form == "unknown" and mentions_injection and volume and volume > 10:
        form = "iv infusion"
    elif form == "unknown" and mentions_injection and has(r"\bbottles?\b"):
        form = "iv infusion"
    elif form == "unknown" and mentions_injection:
        form = "injection vial"

    if original and form != "unknown" and original not in text:
        # The stated-form cell may use abbreviations; this is only a gentle trace note.
        pass

    return form, notes


def parse_strength(text: str) -> Dict[str, Optional[float]]:
    raw = text.lower().replace(",", "")
    result: Dict[str, Optional[float]] = {
        "strength_mg": None,
        "concentration_mg_ml": None,
        "percent_strength": None,
        "dose_count": None,
    }
    if not raw:
        return result

    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", raw)
    if pct:
        result["percent_strength"] = float(pct.group(1))

    # Examples: 125 mg/5 ml, 10mg/ml, 250 mcg per ml
    conc = re.search(
        r"(\d+(?:\.\d+)?)\s*(mcg|µg|ug|mg|g)\s*(?:/|per)\s*(\d+(?:\.\d+)?)?\s*(ml|mL)\b",
        raw,
        flags=re.I,
    )
    if conc:
        amount = float(conc.group(1)) * unit_multiplier(conc.group(2), "mg")
        divisor = float(conc.group(3) or "1")
        if divisor:
            result["concentration_mg_ml"] = amount / divisor

    simple = re.search(r"(\d+(?:\.\d+)?)\s*(mcg|µg|ug|mg|g)\b", raw)
    if simple:
        result["strength_mg"] = float(simple.group(1)) * unit_multiplier(simple.group(2), "mg")

    doses = re.search(r"(\d+(?:\.\d+)?)\s*(?:dose|doses|sprays|puffs|actuations)\b", raw)
    if doses:
        result["dose_count"] = float(doses.group(1))

    return result


def parse_pack_text(pack_text: str, form_hint: str = "") -> Dict[str, Optional[float]]:
    raw = clean_text(pack_text).lower()
    raw = re.sub(r"(?<=\d),(?=\d{3}\b)", "", raw)
    raw = raw.replace(",", " ")
    text = raw.replace("×", "x")
    hint = clean_text(form_hint).lower()
    result: Dict[str, Optional[float]] = {
        "container_volume_ml": None,
        "content_weight_g": None,
        "strip_count": None,
        "units_per_strip": None,
        "unit_count": None,
        "bulk_count": None,
        "dose_count": None,
    }
    if not text:
        return result

    # Volume/fill size.
    volumes = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(ml|mL|l|liter|litre)\b", text, flags=re.I):
        val = float(match.group(1))
        unit = match.group(2).lower()
        if unit in {"l", "liter", "litre"}:
            val *= 1000
        volumes.append(val)
    if volumes:
        # Prefer the smaller value for "10 x 2 ml" ampoule packs; prefer larger for bottles if only one exists.
        result["container_volume_ml"] = min(volumes) if re.search(r"\bamp|vial|respule|nebule|syringe|cartridge\b", text) else max(volumes)

    # Content weight for tubes, sachets, jars, powders, creams, etc.
    content_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(kg|g|gm|gram|grams|mg)\b\s*(?:tube|jar|sachet|bottle|container|powder|cream|ointment|gel|lotion)?",
        text,
    )
    if content_match and re.search(r"\btube|jar|sachet|powder|granule|cream|ointment|gel|lotion|shampoo|wash|dusting\b", text + " " + hint):
        result["content_weight_g"] = float(content_match.group(1)) * unit_multiplier(content_match.group(2), "g")

    # Dose count for inhalers/sprays.
    dose_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:dose|doses|sprays|puffs|actuations)\b", text)
    if dose_match:
        result["dose_count"] = float(dose_match.group(1))

    # Strip/blister patterns: 10 x 10 tablets, 3*10 caps, 1 x 15 tabs.
    is_solid_oral = bool(re.search(r"\btab|tablet|cap|capsule|softgel|lozenge|pessary\b", text + " " + hint))

    # Natural-language strip packs:
    # "1 pack of 10 strips, 1 strip of 10 tablets" => 10 strips * 10 tablets = 100 tablets.
    strip_count_matches = [int(m.group(1)) for m in re.finditer(r"(\d+)\s*strips?\b", text)]
    explicit_strip_multiplication = re.search(
        r"(\d+)\s*strips?\s*(?:x|\*)\s*(\d+)\s*(?:tabs?|tablets?|caps?|capsules?|softgels?|lozenges?)\b",
        text,
    )
    if is_solid_oral and explicit_strip_multiplication:
        result["strip_count"] = float(explicit_strip_multiplication.group(1))
        result["units_per_strip"] = float(explicit_strip_multiplication.group(2))
        result["unit_count"] = result["strip_count"] * result["units_per_strip"]
    units_per_strip_match = re.search(
        r"(?:1\s*)?strips?\s*(?:of)?\s*(\d+)\s*(?:tabs?|tablets?|caps?|capsules?|softgels?|lozenges?|pessaries?)\b",
        text,
    )
    if is_solid_oral and strip_count_matches and units_per_strip_match and not result.get("unit_count"):
        # If both "10 strips" and "1 strip of 10 tablets" appear, the real pack
        # strip count is the larger number. This directly handles the user's
        # warning about tablets/capsules written as packs of strips.
        strip_count = max(strip_count_matches)
        units_per_strip = int(units_per_strip_match.group(1))
        result["strip_count"] = float(strip_count)
        result["units_per_strip"] = float(units_per_strip)
        result["unit_count"] = float(strip_count * units_per_strip)

    strip_match = re.search(
        r"(\d+)\s*(?:x|\*)\s*(\d+)\s*(?:'s|s|tabs?|tablets?|caps?|capsules?|softgels?|lozenges?)?\b",
        text,
    )
    if strip_match and is_solid_oral and not result.get("unit_count"):
        a = int(strip_match.group(1))
        b = int(strip_match.group(2))
        result["strip_count"] = float(a)
        result["units_per_strip"] = float(b)
        result["unit_count"] = float(a * b)

    # Some injectable packs are arranged as strips/trays, for example:
    # "10 strips, 1 strip of 5 amps" => 50 ampoules.
    injectable_strip = re.search(
        r"(\d+)\s*strips?.{0,45}?(?:1\s*)?strips?\s*(?:of)?\s*(\d+)\s*(?:amps?|ampoules?|vials?|respules?|nebules?)\b",
        text,
    )
    if injectable_strip and not is_solid_oral:
        result["strip_count"] = float(injectable_strip.group(1))
        result["units_per_strip"] = float(injectable_strip.group(2))
        result["unit_count"] = result["strip_count"] * result["units_per_strip"]

    # Injectable packs: 10 x 2 ml amp means 10 ampoules of 2 ml, not 20 units.
    inj_volume_pack = re.search(r"(\d+)\s*(?:x|\*)\s*(\d+(?:\.\d+)?)\s*ml\b", text)
    injectable_context = bool(
        re.search(r"\bamp|vial|respule|nebule|syringe|cartridge|inj|inject", text + " " + hint)
    )
    if inj_volume_pack and injectable_context:
        result["unit_count"] = float(inj_volume_pack.group(1))
        result["container_volume_ml"] = float(inj_volume_pack.group(2))

    # For a bottle/drop entry such as "25 x 10 mL", the first number commonly
    # describes the outer quantity while the measured dimensions are for one
    # sellable bottle. Keep that count separate so it cannot inflate the pack.
    if inj_volume_pack and not is_solid_oral and not injectable_context and not result.get("unit_count"):
        result["unit_count"] = 1.0
        result["bulk_count"] = float(inj_volume_pack.group(1))
        result["container_volume_ml"] = float(inj_volume_pack.group(2))

    # Generic "pack of 10", "box 50", "10 vials", "100 ampoules".
    unit_patterns = [
        r"(?:pack|box|bottle|jar|tube|case|carton|shipper|bulk|outer)\s*(?:of)?\s*(\d+)\b",
        r"(\d+)\s*(?:tabs?|tablets?|caps?|capsules?|softgels?|vials?|amps?|ampoules?|bottles?|tubes?|jars?|sachets?|suppositories|pessaries|patches|respules|nebules|pcs|pieces)\b",
        r"(\d+)\s*'s\b",
    ]
    for pattern in unit_patterns:
        match = re.search(pattern, text)
        if match and not result.get("unit_count"):
            result["unit_count"] = float(match.group(1))
            break

    if not result.get("unit_count") and re.search(
        r"\b(?:single|one)\s+(?:pack(?:\s+of)?\s+)?(?:vial|amp(?:oule)?|bottle|tube|jar|sachet|strip|pack)\b",
        text,
    ):
        result["unit_count"] = 1.0

    # Bulk/master/carton counts deserve a separate feature because they can dominate dimensions.
    if re.search(r"\bbulk|master|shipper|carton|case|outer\b", text):
        bulk_patterns = [
            r"(?:bulk|master|shipper|carton|case|outer)\s*(?:pack|packing|of)?\s*(\d+)\b",
            r"(\d+)\s*(?:vials?|amps?|ampoules?|strips?|boxes|packs|bottles|sachets?)\s*(?:per|/)?\s*(?:carton|case|shipper|outer|master)",
        ]
        for pattern in bulk_patterns:
            match = re.search(pattern, text)
            if match:
                result["bulk_count"] = float(match.group(1))
                break

    return result


def container_type(text: str, form: str) -> str:
    raw = text.lower()
    if form == "injection ampoule":
        return "ampoule"
    if form == "injection vial":
        return "vial"
    if form == "pre-filled syringe" or "syringe" in raw or "pfs" in raw:
        return "syringe"
    if form == "cartridge" or "cartridge" in raw:
        return "cartridge"
    if form == "injection pen / auto-injector" or re.search(r"\bpens?\b|auto[-\s]?injector", raw):
        return "pen"
    if form == "iv infusion":
        return "bag" if "bag" in raw else "bottle"
    if form in {"eye drops", "ear drops", "nasal drops", "nasal spray", "oral drops", "oral liquid / syrup", "oral suspension", "oral solution", "oral spray"}:
        return "bottle"
    if form in {"tablet", "capsule", "softgel capsule", "lozenge", "pessary / vaginal tablet"} and (
        "strip" in raw or "blister" in raw
    ):
        return "strip/blister"
    if re.search(r"\bamp(?:oules?|ules?|uls?|s)?\b", raw):
        return "ampoule"
    if re.search(r"\bvials?\b|flacon", raw):
        return "vial"
    if "bottle" in raw or "syrup" in raw or "drops" in raw:
        return "bottle"
    if "tube" in raw:
        return "tube"
    if "jar" in raw:
        return "jar"
    if "sachet" in raw:
        return "sachet"
    if "strip" in raw or "blister" in raw:
        return "strip/blister"
    if "bag" in raw:
        return "bag"
    return ""


def row_to_record(row: pd.Series, schema: Dict[str, Optional[str]], dim_factors: Dict[str, float], weight_factors: Dict[str, float]) -> Dict[str, Any]:
    name = get_value(row, schema.get("name"))
    brand = get_value(row, schema.get("brand"))
    manufacturer = get_value(row, schema.get("manufacturer"))
    record_date = get_value(row, schema.get("date")) or get_value(row, schema.get("source_date"))
    explicit_category = get_value(row, schema.get("category"))
    # If there is no legacy free-text form column, fall back to the clean
    # category so downstream text/parsing still has a dosage-form hint.
    stated_form = get_value(row, schema.get("form")) or explicit_category
    strength_text = get_value(row, schema.get("strength"))
    pack_text = get_value(row, schema.get("pack"))
    bulk_text = get_value(row, schema.get("bulk"))
    all_text = " ".join(clean_text(v) for v in row.tolist() if clean_text(v))
    product_text = " ".join(x for x in [name, brand, stated_form, strength_text, pack_text, bulk_text] if x) or all_text

    parsed = {}
    parsed.update(parse_strength(" ".join([strength_text, pack_text, name, brand])))
    pack_features = parse_pack_text(" ".join([pack_text, bulk_text, name, brand]), " ".join([stated_form, name, brand]))
    for key, value in pack_features.items():
        if value is not None:
            parsed[key] = value
        else:
            parsed.setdefault(key, None)

    form, notes = classify_form(name, stated_form, " ".join([pack_text, bulk_text]), strength_text, parsed)
    # The clean, curated Category is authoritative when the data provides it.
    # Otherwise we fall back to the coarse group derived from the text form.
    category = explicit_category or FORM_CATEGORY.get(form, "unknown")
    ctype = container_type(product_text, form)

    length = width = height = None
    dim_col = schema.get("dimension_text")
    if dim_col:
        length, width, height = parse_dimension_text(get_value(row, dim_col))

    if length is None or width is None or height is None:
        for key in ["length", "width", "height"]:
            col = schema.get(key)
            if not col:
                continue
            candidate = parse_dimension_text(get_value(row, col))
            if all(v is not None for v in candidate):
                length, width, height = candidate
                break

    if length is None and schema.get("length"):
        col = schema["length"]
        val = numeric_value(row.get(col))
        if val is not None:
            length = val * dim_factors.get(col, 1.0)
    if width is None and schema.get("width"):
        col = schema["width"]
        val = numeric_value(row.get(col))
        if val is not None:
            width = val * dim_factors.get(col, 1.0)
    if height is None and schema.get("height"):
        col = schema["height"]
        val = numeric_value(row.get(col))
        if val is not None:
            height = val * dim_factors.get(col, 1.0)

    weight = None
    if schema.get("weight"):
        col = schema["weight"]
        weight = parse_weight_text(row.get(col), weight_factors.get(col, 1.0))

    if all(value is not None for value in [length, width, height]):
        # Historic dimension strings do not use orientation consistently.
        # Standardize the envelope as longest × middle × shortest.
        length, width, height = sorted(
            [float(length), float(width), float(height)],
            reverse=True,
        )
    volume_cm3 = (length * width * height / 1000.0) if all(
        value is not None for value in [length, width, height]
    ) else None

    occurrence_count = numeric_value(row.get(schema.get("occurrence_count"))) if schema.get("occurrence_count") else 1.0
    return {
        "sheet": get_value(row, "_sheet"),
        "name": name or clean_text(row.iloc[0] if len(row) else ""),
        "brand": brand,
        "manufacturer": manufacturer,
        "date": record_date,
        "source_file": get_value(row, schema.get("source_file")),
        "source_sheet": get_value(row, schema.get("source_sheet")) or get_value(row, "_sheet"),
        "source_row": get_value(row, schema.get("source_row")),
        "occurrence_count": occurrence_count or 1.0,
        "stated_form": stated_form,
        "form": form,
        "category": category,
        "container_type": ctype,
        "strength_text": strength_text,
        "pack_text": " ".join(x for x in [pack_text, bulk_text] if x),
        "product_text": product_text,
        "tokens": words(product_text),
        "classification_notes": notes,
        "length_mm": length,
        "width_mm": width,
        "height_mm": height,
        "volume_cm3": volume_cm3,
        "weight_g": weight,
        **parsed,
    }


def has_complete_dimensions(rec: Dict[str, Any]) -> bool:
    return rec.get("length_mm") is not None and rec.get("width_mm") is not None and rec.get("height_mm") is not None


def has_valid_dimensions(rec: Dict[str, Any]) -> bool:
    if not has_complete_dimensions(rec):
        return False
    dims = [float(rec["length_mm"]), float(rec["width_mm"]), float(rec["height_mm"])]
    if any(d <= 0 for d in dims):
        return False
    if any(d < 5 for d in dims):
        return False
    if any(d > 1500 for d in dims):
        return False
    return True


def load_dataset(path: Path) -> WorkbookData:
    sheets = read_workbook(path)
    combined = combine_sheets(sheets)
    notes = []
    if combined.empty:
        raise ValueError("The workbook did not contain any readable rows.")

    schema = detect_schema(combined)
    dimension_factors: Dict[str, float] = {}
    weight_factors: Dict[str, float] = {}
    for key in ["length", "width", "height"]:
        col = schema.get(key)
        if col:
            dimension_factors[col] = infer_dimension_factor(col, combined[col])
    if schema.get("weight"):
        weight_factors[schema["weight"]] = infer_weight_factor(schema["weight"])

    records = [row_to_record(row, schema, dimension_factors, weight_factors) for _, row in combined.iterrows()]
    complete_dimension_rows = sum(1 for rec in records if has_complete_dimensions(rec))
    usable_dimension_rows = sum(1 for rec in records if has_valid_dimensions(rec))

    if not schema.get("pack"):
        notes.append("No obvious pack-size column was detected. The app will still scan item text, but predictions will be weaker.")
    if usable_dimension_rows < 10:
        notes.append("Fewer than 10 rows have complete L/W/H dimensions; predictions will be rough until more measured rows are available.")
    excluded = complete_dimension_rows - usable_dimension_rows
    if excluded:
        notes.append(f"{excluded} rows have complete dimensions but were excluded from prediction because they look physically impossible or like Excel date-entry artifacts.")

    return WorkbookData(
        path=path,
        sheets=sheets,
        combined=combined,
        schema=schema,
        dimension_factors=dimension_factors,
        weight_factors=weight_factors,
        records=records,
        usable_dimension_rows=usable_dimension_rows,
        notes=notes,
    )


def save_last_upload(path: Path) -> None:
    ensure_dirs()
    STATE_FILE.write_text(json.dumps({"path": str(path)}, indent=2), encoding="utf-8")


def active_file() -> Optional[Path]:
    ensure_dirs()
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            path = Path(data.get("path", ""))
            if path.exists() and path.suffix.lower() in SUPPORTED_FILES:
                return path
        except Exception:
            pass
    if MASTER_TRAINING_FILE.exists():
        return MASTER_TRAINING_FILE
    candidates = []
    for directory in [DATA_DIR, BASE_DIR]:
        for path in directory.glob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_FILES:
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


DATA_CACHE: Dict[str, Tuple[float, WorkbookData]] = {}


def get_dataset() -> Optional[WorkbookData]:
    path = active_file()
    if not path:
        return None
    mtime = path.stat().st_mtime
    cache_key = str(path)
    cached = DATA_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]
    data = load_dataset(path)
    DATA_CACHE.clear()
    DATA_CACHE[cache_key] = (mtime, data)
    return data


# ---------------------------------------------------------------------------
# Machine-learning model lifecycle
#
# The ExtraTrees model takes ~30-60s to train the first time, so we never block
# a web request on it. On the first dataset load we either load a cached model
# instantly or kick off training in a background thread. Until the model is
# ready the app falls back to the transparent nearest-neighbour estimate, then
# upgrades to the blended ML estimate automatically once training finishes.
# ---------------------------------------------------------------------------
import threading  # noqa: E402

MODEL_CACHE_FILE = BASE_DIR / "work" / "model_cache.pkl"
_MODEL_LOCK = threading.Lock()
_MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "source": None,
    "status": "idle",  # idle | training | ready | error | unavailable
    "notes": [],
    "error": "",
}


def _train_model_worker(records: List[Dict[str, Any]], source_tag: str) -> None:
    try:
        model, notes = ml_engine.train_or_load(records, MODEL_CACHE_FILE, source_tag)
        with _MODEL_LOCK:
            _MODEL_STATE.update(
                model=model, source=source_tag, status="ready", notes=notes, error=""
            )
    except Exception as exc:  # noqa: BLE001
        with _MODEL_LOCK:
            _MODEL_STATE.update(
                model=None, status="error", error=str(exc), notes=[]
            )


def ensure_model(dataset: Optional[WorkbookData]) -> None:
    """Make sure a model is trained/loaded for the active dataset (non-blocking).

    Both training (~40s) and loading the cached model (a large pickle) happen on a
    daemon thread so an HTTP request is never blocked. The page shows a "training"
    status and refreshes itself until the model reports ready.
    """
    if not ML_AVAILABLE or dataset is None:
        if not ML_AVAILABLE:
            with _MODEL_LOCK:
                _MODEL_STATE.update(status="unavailable", error=ML_IMPORT_ERROR)
        return
    source_tag = str(dataset.path)
    with _MODEL_LOCK:
        status = _MODEL_STATE["status"]
        same_source = _MODEL_STATE["source"] == source_tag
        if same_source and status in {"ready", "training"}:
            return
        _MODEL_STATE.update(source=source_tag, status="training", notes=[], error="")

    thread = threading.Thread(
        target=_train_model_worker, args=(dataset.records, source_tag), daemon=True
    )
    thread.start()


def get_model() -> Optional["ml_engine.DimensionModel"]:
    with _MODEL_LOCK:
        if _MODEL_STATE["status"] == "ready":
            return _MODEL_STATE["model"]
    return None


def model_status() -> Dict[str, Any]:
    with _MODEL_LOCK:
        return dict(_MODEL_STATE)


def relative_log_distance(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a < 0 or b < 0:
        return abs(a - b)
    return abs(math.log1p(a) - math.log1p(b))


def record_distance(query: Dict[str, Any], rec: Dict[str, Any]) -> float:
    dist = 0.0
    # Clean curated category is the primary packaging signal (replaces old form).
    if query.get("category") and rec.get("category"):
        dist += 0.0 if query["category"] == rec["category"] else 3.0
    if query.get("container_type") and rec.get("container_type"):
        dist += 0.0 if query["container_type"] == rec["container_type"] else 0.8
    q_brand = norm_col(query.get("brand", ""))
    r_brand = norm_col(rec.get("brand", ""))
    q_name = norm_col(query.get("name", ""))
    r_identity = norm_col(" ".join([rec.get("name", ""), rec.get("brand", "")]))
    if q_brand and r_brand and (q_brand == r_brand or q_brand in r_brand or r_brand in q_brand):
        dist -= 1.35
    elif q_name and r_identity and len(q_name) >= 4 and q_name in r_identity:
        dist -= 0.9
    used_numeric = 0
    for feature, weight in NUMERIC_FEATURE_WEIGHTS.items():
        d = relative_log_distance(query.get(feature), rec.get(feature))
        if d is not None:
            dist += d * weight
            used_numeric += 1

    # Slight penalty if there are almost no numeric features to compare.
    if used_numeric <= 1:
        dist += 1.0

    q_tokens = query.get("tokens") or set()
    r_tokens = rec.get("tokens") or set()
    if q_tokens and r_tokens:
        overlap = len(q_tokens & r_tokens)
        union = len(q_tokens | r_tokens)
        dist += (1 - overlap / union) * 0.8

    return max(dist, 0.0)


def match_reasons(query: Dict[str, Any], rec: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if query.get("category") and query.get("category") == rec.get("category"):
        reasons.append("same category")
    if query.get("container_type") and query.get("container_type") == rec.get("container_type"):
        reasons.append("same container")
    q_brand = norm_col(query.get("brand", ""))
    r_brand = norm_col(rec.get("brand", ""))
    q_name = norm_col(query.get("name", ""))
    r_identity = norm_col(" ".join([rec.get("name", ""), rec.get("brand", "")]))
    if q_brand and r_brand and (q_brand == r_brand or q_brand in r_brand or r_brand in q_brand):
        reasons.append("same brand")
    elif q_name and r_identity and len(q_name) >= 4 and q_name in r_identity:
        reasons.append("same product name")
    if query.get("strip_count") and query.get("strip_count") == rec.get("strip_count"):
        reasons.append("same strip count")
    if query.get("units_per_strip") and query.get("units_per_strip") == rec.get("units_per_strip"):
        reasons.append("same units per strip")
    if query.get("unit_count") and query.get("unit_count") == rec.get("unit_count"):
        reasons.append("same total units")
    if query.get("container_volume_ml") and rec.get("container_volume_ml"):
        ratio = max(query["container_volume_ml"], rec["container_volume_ml"]) / max(
            min(query["container_volume_ml"], rec["container_volume_ml"]),
            0.001,
        )
        if ratio <= 1.15:
            reasons.append("similar fill volume")
    if query.get("strength_mg") and rec.get("strength_mg"):
        ratio = max(query["strength_mg"], rec["strength_mg"]) / max(min(query["strength_mg"], rec["strength_mg"]), 0.001)
        if ratio <= 1.25:
            reasons.append("similar strength")
    shared = sorted((query.get("tokens") or set()) & (rec.get("tokens") or set()))
    if shared:
        reasons.append("shared product terms: " + ", ".join(shared[:3]))
    return reasons or ["closest available pack profile"]


def weighted_average(values: List[Tuple[float, float]]) -> Optional[float]:
    values = [(v, w) for v, w in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    total_w = sum(w for _, w in values)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in values) / total_w


def prediction_range(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None, None
    return float(np.percentile(clean, 20)), float(np.percentile(clean, 80))


def build_query_product(form: Dict[str, str]) -> Dict[str, Any]:
    name = form.get("name", "")
    brand = form.get("brand", "")
    manufacturer = form.get("manufacturer", "")
    stated_form = form.get("stated_form", "")
    strength = form.get("strength", "")
    pack = form.get("pack", "")
    bulk = form.get("bulk", "")
    parsed = {}
    parsed.update(parse_strength(" ".join([strength, pack, name, brand])))
    pack_features = parse_pack_text(" ".join([pack, bulk, name, brand]), " ".join([stated_form, name, brand]))
    for key, value in pack_features.items():
        parsed[key] = value
    chosen_category = clean_text(form.get("category", ""))
    final_form, notes = classify_form(name, stated_form, " ".join([pack, bulk]), strength, parsed)
    # The user-selected category (from the dropdown of real data categories) is
    # authoritative. Fall back to the derived group only when nothing was chosen.
    category = chosen_category or FORM_CATEGORY.get(final_form, "unknown")
    text = " ".join([name, brand, stated_form, chosen_category, strength, pack, bulk])
    return {
        "name": name,
        "brand": brand,
        "manufacturer": manufacturer,
        "stated_form": stated_form,
        "strength_text": strength,
        "pack_text": " ".join([pack, bulk]).strip(),
        "product_text": text,
        "tokens": words(text),
        "form": final_form,
        "category": category,
        "container_type": container_type(text, final_form),
        "classification_notes": notes,
        **parsed,
    }


def predict_dimensions(dataset: WorkbookData, query: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [rec for rec in dataset.records if has_valid_dimensions(rec)]
    if not candidates:
        return {
            "error": "No rows with complete length/width/height were found in the loaded file.",
            "query": query,
            "neighbors": [],
        }

    # When the user has explicitly chosen a category, only reference packs from
    # THAT category — both for the estimate and the "closest packs" table. If the
    # category has no measured packs at all we fall back to the whole set.
    chosen_category = query.get("category")
    restricted_to_category = False
    if chosen_category and chosen_category != "unknown":
        same_cat = [rec for rec in candidates if clean_text(rec.get("category")) == chosen_category]
        if same_cat:
            candidates = same_cat
            restricted_to_category = True

    scored = [(record_distance(query, rec), rec) for rec in candidates]
    scored.sort(key=lambda item: item[0])
    top = scored[: min(9, len(scored))]

    weights = []
    for dist, rec in top:
        occurrence_count = max(float(rec.get("occurrence_count") or 1), 1.0)
        # Repeated identical measurements are useful evidence, but their
        # influence is deliberately capped so a frequently repeated inward row
        # cannot overwhelm genuinely different nearby pack profiles.
        evidence_multiplier = 1.0 + min(math.log1p(occurrence_count - 1.0) * 0.12, 0.4)
        weights.append(((1.0 / ((dist + 0.35) ** 2)) * evidence_multiplier, rec))
    knn_predicted = {
        "length_mm": weighted_average([(rec.get("length_mm"), w) for w, rec in weights]),
        "width_mm": weighted_average([(rec.get("width_mm"), w) for w, rec in weights]),
        "height_mm": weighted_average([(rec.get("height_mm"), w) for w, rec in weights]),
        "weight_g": weighted_average([(rec.get("weight_g"), w) for w, rec in weights if rec.get("weight_g") is not None]),
    }
    knn_ranges = {
        "length_mm": prediction_range([rec.get("length_mm") for _, rec in top]),
        "width_mm": prediction_range([rec.get("width_mm") for _, rec in top]),
        "height_mm": prediction_range([rec.get("height_mm") for _, rec in top]),
        "volume_cm3": prediction_range([rec.get("volume_cm3") for _, rec in top]),
        "weight_g": prediction_range([rec.get("weight_g") for _, rec in top if rec.get("weight_g") is not None]),
    }

    # ---- Blend the trained ML model with the nearest-neighbour estimate ----
    # ExtraTrees wins on average accuracy; the neighbour estimate adds robustness
    # for unusual queries. A 0.65/0.35 blend was the most reliable on held-out
    # data. Prediction bands come from the model's own tree spread when present.
    ml_out = None
    model = get_model()
    if model is not None:
        try:
            ml_out = model.predict(query)
        except Exception:  # noqa: BLE001 — never let a model glitch break a prediction
            ml_out = None

    predicted: Dict[str, Optional[float]] = {}
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    ml_weight = 0.65
    for key in ["length_mm", "width_mm", "height_mm", "weight_g"]:
        knn_v = knn_predicted.get(key)
        ml_v = ml_out["predicted"].get(key) if ml_out else None
        if ml_v is not None and knn_v is not None:
            predicted[key] = ml_weight * ml_v + (1 - ml_weight) * knn_v
        else:
            predicted[key] = ml_v if ml_v is not None else knn_v
        if ml_out and ml_out["intervals"].get(key, (None, None))[0] is not None:
            ranges[key] = ml_out["intervals"][key]
        else:
            ranges[key] = knn_ranges.get(key, (None, None))

    if all(predicted.get(key) is not None for key in ["length_mm", "width_mm", "height_mm"]):
        predicted["volume_cm3"] = (
            predicted["length_mm"] * predicted["width_mm"] * predicted["height_mm"] / 1000.0
        )
    else:
        predicted["volume_cm3"] = None
    ranges["volume_cm3"] = knn_ranges.get("volume_cm3", (None, None))
    prediction_method = "Trained ML model + nearest-neighbour blend" if ml_out else "Nearest-neighbour estimate"

    same_category_count = sum(1 for _, rec in top if rec.get("category") == query.get("category"))
    best_distance = top[0][0] if top else 99.0
    neighbor_volumes = [rec.get("volume_cm3") for _, rec in top if rec.get("volume_cm3")]
    volume_cv = (
        float(np.std(neighbor_volumes) / np.mean(neighbor_volumes))
        if len(neighbor_volumes) >= 2 and np.mean(neighbor_volumes)
        else 1.0
    )
    confidence_score = 62.0
    confidence_score += 17 if best_distance <= 0.6 else 10 if best_distance <= 1.3 else 3 if best_distance <= 2.5 else -16
    confidence_score += 10 if same_category_count >= 7 else 5 if same_category_count >= 4 else -10
    confidence_score += 8 if volume_cv <= 0.25 else 2 if volume_cv <= 0.5 else -12
    repeated_evidence = sum(min(float(rec.get("occurrence_count") or 1), 5.0) for _, rec in top)
    if repeated_evidence >= 18:
        confidence_score += 4
    elif repeated_evidence >= 12:
        confidence_score += 2
    query_pack_features = sum(
        query.get(key) is not None
        for key in ["unit_count", "strip_count", "units_per_strip", "container_volume_ml", "content_weight_g", "dose_count"]
    )
    if query_pack_features == 0:
        confidence_score -= 12
    if not query.get("category") or query.get("category") == "unknown":
        confidence_score = min(confidence_score, 45)

    # When the trained model is available, fold the tightness of its own
    # prediction band into confidence: narrow tree spread => more agreement.
    if ml_out and ml_out.get("spread_ratio"):
        ratios = [v for v in ml_out["spread_ratio"].values() if v is not None]
        if ratios:
            avg_spread = float(np.mean(ratios))
            confidence_score += 8 if avg_spread <= 0.35 else 3 if avg_spread <= 0.6 else -8

    # ---- Coverage-aware confidence ----------------------------------------
    # How many measured packs actually exist in the chosen category. Thin
    # categories cap confidence so the estimate is never over-sold.
    query_category = query.get("category")
    category_sample_count = sum(
        1 for rec in candidates if clean_text(rec.get("category")) == query_category
    ) if query_category and query_category != "unknown" else 0
    coverage = coverage_info(category_sample_count)
    confidence_score = min(confidence_score, coverage["cap"])

    confidence_score = int(round(max(15, min(95, confidence_score))))
    confidence = "High" if confidence_score >= 75 else "Medium" if confidence_score >= 50 else "Low"

    return {
        "query": query,
        "predicted": predicted,
        "ranges": ranges,
        "category_sample_count": category_sample_count,
        "coverage": coverage,
        "restricted_to_category": restricted_to_category,
        "prediction_method": prediction_method,
        "ml_used": ml_out is not None,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "best_distance": best_distance,
        "same_category_neighbors": same_category_count,
        "measured_attempts_nearby": int(round(repeated_evidence)),
        "neighbor_volume_cv": volume_cv,
        "neighbors": [
            {
                "distance": dist,
                "name": rec.get("name"),
                "brand": rec.get("brand"),
                "manufacturer": rec.get("manufacturer"),
                "category": rec.get("category"),
                "pack": rec.get("pack_text"),
                "length_mm": rec.get("length_mm"),
                "width_mm": rec.get("width_mm"),
                "height_mm": rec.get("height_mm"),
                "volume_cm3": rec.get("volume_cm3"),
                "weight_g": rec.get("weight_g"),
                "source_file": rec.get("source_file"),
                "source_row": rec.get("source_row"),
                "occurrence_count": rec.get("occurrence_count", 1),
                "match_reasons": match_reasons(query, rec),
                "notes": rec.get("classification_notes") or [],
            }
            for dist, rec in top
        ],
    }


def fmt_number(value: Optional[float], suffix: str = "", decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if abs(value - round(value)) < 0.05:
        return f"{round(value):,.0f}{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def h(value: Any) -> str:
    return html.escape(clean_text(value))


def app_css() -> str:
    return """
    :root{
      --ink:#0f1b2d; --muted:#54657f; --hair:rgba(15,27,45,.10);
      --blue:#2563eb; --cyan:#06b6d4; --indigo:#6366f1;
      --good:#059669; --good-bg:rgba(16,185,129,.16);
      --warn:#b45309; --warn-bg:rgba(245,158,11,.18);
      --low:#e11d48;  --low-bg:rgba(244,63,94,.16);
      --glass:rgba(255,255,255,.55); --glass-strong:rgba(255,255,255,.72);
      --glass-brd:rgba(255,255,255,.7); --field:rgba(255,255,255,.65);
      --accent:linear-gradient(135deg,#2563eb 0%,#06b6d4 100%);
      --shadow:0 18px 50px -20px rgba(20,40,90,.45);
      --radius:22px;
      --sans:ui-sans-serif,system-ui,"Segoe UI Variable","Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
    }
    *{box-sizing:border-box;}
    html{-webkit-text-size-adjust:100%;}
    body{
      margin:0; color:var(--ink); font-family:var(--sans); line-height:1.5;
      background:
        radial-gradient(40rem 30rem at 8% -6%, rgba(99,102,241,.40), transparent 60%),
        radial-gradient(36rem 30rem at 100% 4%, rgba(6,182,212,.38), transparent 58%),
        radial-gradient(48rem 38rem at 60% 110%, rgba(45,212,191,.30), transparent 60%),
        radial-gradient(30rem 26rem at -8% 80%, rgba(37,99,235,.28), transparent 60%),
        linear-gradient(180deg,#eef4ff 0%,#e9f6ff 50%,#eafbf4 100%);
      background-attachment:fixed; min-height:100vh;
    }
    a{color:var(--blue);}
    ::selection{background:rgba(37,99,235,.22);}

    /* ---------- hero ---------- */
    header{position:relative; padding:34px 26px 12px; max-width:1220px; margin:0 auto;}
    .brandline{display:inline-flex; align-items:center; gap:9px; font-weight:700;
      font-size:12.5px; letter-spacing:.14em; text-transform:uppercase; color:#2b3f63;
      background:var(--glass); border:1px solid var(--glass-brd); backdrop-filter:blur(12px);
      -webkit-backdrop-filter:blur(12px); padding:7px 13px; border-radius:999px; box-shadow:var(--shadow);}
    .brandline .dot{width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px rgba(6,182,212,.18);}
    header h1{margin:16px 0 6px; font-size:clamp(28px,4.4vw,44px); font-weight:850;
      letter-spacing:-.022em; line-height:1.04;
      background:linear-gradient(120deg,#10243f 0%,#1d4ed8 55%,#0891b2 100%);
      -webkit-background-clip:text; background-clip:text; color:transparent;}
    header p.lede{margin:0; max-width:760px; color:#33455f; font-size:15.5px;}
    .statusbar{display:flex; flex-wrap:wrap; gap:9px; margin:18px 0 4px;}
    .chip{display:inline-flex; align-items:center; gap:8px; font-size:12.5px; font-weight:650;
      color:#243a5e; background:var(--glass); border:1px solid var(--glass-brd);
      backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
      padding:7px 13px; border-radius:999px; box-shadow:0 8px 22px -14px rgba(20,40,90,.6);}
    .chip b{font-weight:800;}
    .chip .led{width:8px;height:8px;border-radius:50%;}
    .led.ready{background:var(--good);box-shadow:0 0 0 4px var(--good-bg);}
    .led.training{background:var(--warn);box-shadow:0 0 0 4px var(--warn-bg);animation:pulse 1.3s ease-in-out infinite;}
    .led.error,.led.unavailable{background:var(--low);box-shadow:0 0 0 4px var(--low-bg);}
    @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.35;}}

    main{max-width:1220px; margin:8px auto 70px; padding:0 22px;}
    .grid{display:grid; grid-template-columns:1.05fr .95fr; gap:20px; align-items:start;}

    /* ---------- glass card ---------- */
    .card{
      position:relative; background:var(--glass); border:1px solid var(--glass-brd);
      border-radius:var(--radius); box-shadow:var(--shadow);
      backdrop-filter:blur(22px) saturate(165%); -webkit-backdrop-filter:blur(22px) saturate(165%);
      padding:22px 22px 20px; margin-bottom:20px; overflow:hidden;
    }
    .card::before{content:""; position:absolute; inset:0 0 auto 0; height:1px;
      background:linear-gradient(90deg,transparent,rgba(255,255,255,.95),transparent);}
    .card h2{margin:0 0 6px; font-size:19px; font-weight:800; letter-spacing:-.01em; display:flex; align-items:center; gap:9px;}
    .card h2 .ic{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;
      background:var(--accent); color:#fff; font-size:15px; box-shadow:0 6px 16px -6px rgba(6,182,212,.7);}
    .card h3{margin:18px 0 8px; font-size:15px; font-weight:750;}
    .small{color:var(--muted); font-size:13px; line-height:1.5;}
    code{background:rgba(15,27,45,.07); padding:1px 6px; border-radius:6px; font-size:12.5px;}

    /* ---------- form ---------- */
    label{display:block; font-weight:680; margin:14px 0 6px; font-size:13.5px; color:#1f3252;}
    input[type=text],textarea{width:100%; border:1px solid rgba(15,27,45,.14); border-radius:13px;
      padding:11px 13px; font:inherit; color:var(--ink); background:var(--field);
      transition:border-color .15s, box-shadow .15s, background .15s; backdrop-filter:blur(6px);}
    input[type=text]::placeholder,textarea::placeholder{color:#8493ab;}
    input[type=text]:focus,textarea:focus{outline:none; border-color:#2563eb; background:rgba(255,255,255,.92);
      box-shadow:0 0 0 4px rgba(37,99,235,.16);}
    textarea{min-height:74px; resize:vertical;}
    /* ---------- select / dropdown (matches text inputs + single custom chevron) ---------- */
    /* NOTE: background-color longhand is used everywhere (never the `background`
       shorthand) so the chevron image / no-repeat / position / size are never reset. */
    select{
      -webkit-appearance:none; -moz-appearance:none; appearance:none;
      width:100%; border:1px solid rgba(15,27,45,.14); border-radius:13px;
      padding:11px 42px 11px 13px; font:inherit; color:var(--ink); line-height:1.4; cursor:pointer;
      backdrop-filter:blur(6px); transition:border-color .15s, box-shadow .15s, background-color .15s;
      background-color:var(--field);
      background-image:url("data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2354657f' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
      background-repeat:no-repeat; background-position:right 14px center; background-size:16px 16px;
    }
    select:hover{border-color:rgba(37,99,235,.4);}
    select:focus{
      outline:none; border-color:#2563eb; background-color:rgba(255,255,255,.92);
      box-shadow:0 0 0 4px rgba(37,99,235,.16);
      background-image:url("data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%232563eb' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    }
    select::-ms-expand{display:none;}
    /* placeholder option looks muted like input placeholders until a real value is chosen */
    select:required:invalid{color:#8493ab;}
    select option{color:var(--ink); background:#fff;}
    select option[value=""]{color:#8493ab;}
    input[type=file]{width:100%; border:1.5px dashed rgba(37,99,235,.45); border-radius:14px;
      padding:18px; background:rgba(255,255,255,.45); cursor:pointer;}
    button{border:0; border-radius:13px; background:var(--accent); color:#fff; font-weight:780;
      font-size:14.5px; padding:12px 18px; cursor:pointer; margin-top:16px; letter-spacing:.01em;
      box-shadow:0 12px 26px -10px rgba(37,99,235,.7); transition:transform .12s, box-shadow .12s, filter .12s;}
    button:hover{transform:translateY(-1px); filter:brightness(1.04); box-shadow:0 16px 30px -10px rgba(6,182,212,.7);}
    button:active{transform:translateY(0);}
    details summary{cursor:pointer; color:var(--blue); font-weight:650;}
    summary::-webkit-details-marker{color:var(--cyan);}

    /* ---------- pills ---------- */
    .pill{display:inline-flex; align-items:center; gap:6px; padding:6px 12px; border-radius:999px;
      background:rgba(37,99,235,.12); color:#1d4ed8; font-size:12px; font-weight:720; margin:4px 5px 4px 0;
      border:1px solid rgba(37,99,235,.16);}
    .pill.good{background:var(--good-bg); color:#047857; border-color:rgba(16,185,129,.28);}
    .pill.warn{background:var(--warn-bg); color:#92400e; border-color:rgba(245,158,11,.32);}
    .pill.low{background:var(--low-bg); color:#be123c; border-color:rgba(244,63,94,.3);}

    /* ---------- tables ---------- */
    table{width:100%; border-collapse:collapse; margin-top:12px; font-size:13px;}
    th,td{text-align:left; border-bottom:1px solid var(--hair); padding:9px 8px; vertical-align:top;}
    th{color:#33486b; font-size:11px; text-transform:uppercase; letter-spacing:.05em; font-weight:750;
      background:rgba(255,255,255,.35);}
    tbody tr:hover{background:rgba(255,255,255,.4);}
    .table-wrap{overflow-x:auto; border-radius:14px;}

    /* ---------- result metrics ---------- */
    .result{display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:13px; margin:16px 0;}
    .metric{position:relative; background:var(--glass-strong); border:1px solid var(--glass-brd);
      border-radius:16px; padding:15px 14px 14px; overflow:hidden; box-shadow:0 10px 24px -16px rgba(20,40,90,.55);}
    .metric::before{content:""; position:absolute; inset:0 0 auto 0; height:3px; background:var(--accent);}
    .metric .label{color:var(--muted); font-size:11.5px; font-weight:680; text-transform:uppercase; letter-spacing:.04em;}
    .metric .value{font-size:25px; font-weight:840; margin-top:5px; letter-spacing:-.02em; font-variant-numeric:tabular-nums;}
    .metric .value small{font-size:13px; font-weight:680; color:var(--muted); margin-left:2px;}
    .metric .band{color:#5b6c86; font-size:11.5px; margin-top:5px; font-variant-numeric:tabular-nums;}

    /* ---------- confidence ring ---------- */
    .conf-wrap{display:flex; align-items:center; gap:18px; flex-wrap:wrap; margin:6px 0 4px;}
    .ring{--p:60; --col:var(--blue); width:104px; height:104px; border-radius:50%; flex:0 0 auto;
      background:conic-gradient(var(--col) calc(var(--p)*1%), rgba(15,27,45,.10) 0);
      display:grid; place-items:center; box-shadow:inset 0 0 0 1px rgba(255,255,255,.5);}
    .ring .inner{width:78px; height:78px; border-radius:50%; background:var(--glass-strong);
      backdrop-filter:blur(8px); display:grid; place-items:center; text-align:center; box-shadow:0 6px 16px -10px rgba(20,40,90,.6);}
    .ring .num{font-size:24px; font-weight:850; line-height:1; font-variant-numeric:tabular-nums;}
    .ring .cap{font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-top:3px;}
    .conf-meta{min-width:200px; flex:1;}

    /* ---------- isometric pack preview ---------- */
    .pack-preview{display:flex; align-items:center; justify-content:center; padding:6px;
      background:radial-gradient(circle at 50% 35%, rgba(255,255,255,.6), rgba(255,255,255,.2));
      border:1px solid var(--glass-brd); border-radius:16px;}

    /* ---------- 3D + 2D pack visuals ---------- */
    .viz3d{margin:10px 0 4px; border:1px solid var(--glass-brd); border-radius:16px; overflow:hidden;
      background:radial-gradient(circle at 50% 30%, rgba(255,255,255,.65), rgba(236,244,255,.35));}
    .viz3d-canvas{width:100%; height:340px; cursor:grab; touch-action:none;}
    .viz3d-legend{display:flex; flex-wrap:wrap; gap:14px; justify-content:center; padding:10px 12px 12px;
      font-size:12px; font-weight:650; color:#33486b; border-top:1px solid var(--hair); background:rgba(255,255,255,.4);}
    .viz3d-legend span{display:inline-flex; align-items:center; gap:6px;}
    .viz3d-legend i{width:12px; height:12px; border-radius:3px; display:inline-block;}
    .views2d{display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:8px 0 6px;}
    .view2d{background:var(--glass-strong); border:1px solid var(--glass-brd); border-radius:14px; padding:12px 12px 8px;}
    .view2d-title{text-align:center; font-size:12px; font-weight:700; color:#33486b; margin-top:4px;}
    @media (max-width:720px){ .views2d{grid-template-columns:1fr;} }

    /* ---------- notices ---------- */
    .notice{background:var(--warn-bg); border:1px solid rgba(245,158,11,.4); color:#7c2d12;
      border-radius:14px; padding:12px 14px; margin:10px 0; font-size:13.5px;}
    .ok{background:var(--good-bg); border-color:rgba(16,185,129,.4); color:#065f46;}
    .muted-box{background:rgba(255,255,255,.45); border:1px solid var(--glass-brd); border-radius:14px;
      padding:12px; font-size:12px; overflow:auto;}
    .foot{max-width:1220px; margin:0 auto; padding:0 22px 40px; color:#5a6b85; font-size:12.5px;}

    /* ---------- top navigation ---------- */
    .nav{display:inline-flex; gap:6px; margin:16px 0 2px; padding:5px; border-radius:999px;
      background:var(--glass); border:1px solid var(--glass-brd); backdrop-filter:blur(12px);
      -webkit-backdrop-filter:blur(12px); box-shadow:0 8px 22px -14px rgba(20,40,90,.6);}
    .nav a{display:inline-flex; align-items:center; gap:7px; text-decoration:none; color:#2b3f63;
      font-weight:700; font-size:13px; padding:8px 15px; border-radius:999px; transition:background .15s, color .15s;}
    .nav a:hover{background:rgba(37,99,235,.10);}
    .nav a.active{background:var(--accent); color:#fff; box-shadow:0 8px 18px -8px rgba(37,99,235,.7);}

    /* ---------- unit toggle (segmented) ---------- */
    .unit-bar{display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin:2px 0 14px;}
    .unit-group{display:inline-flex; align-items:center; gap:9px;}
    .unit-group>span{font-size:11.5px; font-weight:750; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);}
    .seg{display:inline-flex; padding:3px; border-radius:11px; background:rgba(255,255,255,.55);
      border:1px solid var(--glass-brd); backdrop-filter:blur(6px);}
    .seg button{margin:0; padding:6px 12px; border-radius:8px; background:transparent; color:#33486b;
      font-size:12.5px; font-weight:720; box-shadow:none; letter-spacing:0; transition:background .14s, color .14s;}
    .seg button:hover{transform:none; filter:none; box-shadow:none; background:rgba(37,99,235,.08);}
    .seg button.on{background:var(--accent); color:#fff; box-shadow:0 6px 14px -8px rgba(37,99,235,.7);}

    /* ---------- density / coverage extras ---------- */
    .coverage-note{margin-top:2px;}
    .cov-badge{display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:999px;
      font-size:11.5px; font-weight:750; border:1px solid transparent;}
    .cov-rich{background:var(--good-bg); color:#047857; border-color:rgba(16,185,129,.3);}
    .cov-limited{background:rgba(37,99,235,.12); color:#1d4ed8; border-color:rgba(37,99,235,.2);}
    .cov-sparse{background:var(--warn-bg); color:#92400e; border-color:rgba(245,158,11,.35);}
    .cov-none{background:var(--low-bg); color:#be123c; border-color:rgba(244,63,94,.3);}

    /* ---------- compare ---------- */
    .cmp-grid{display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start;}
    .cmp-col h3{margin:2px 0 6px;}
    .cmp-table td.delta{font-variant-numeric:tabular-nums; font-weight:750;}
    .delta.up{color:#047857;} .delta.down{color:#be123c;} .delta.flat{color:var(--muted);}
    .cmp-visuals{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:10px;}
    @media (max-width:720px){ .cmp-grid,.cmp-visuals{grid-template-columns:1fr;} }

    /* ---------- dashboard ---------- */
    .dash-grid{display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start;}
    @media (max-width:920px){ .dash-grid{grid-template-columns:1fr;} }
    .kpi-row{display:grid; grid-template-columns:repeat(4,1fr); gap:13px; margin:4px 0 4px;}
    @media (max-width:720px){ .kpi-row{grid-template-columns:repeat(2,1fr);} }
    /* rich KPI band */
    .kpi-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:14px; margin:4px 0 6px;}
    .kpi{position:relative; background:var(--glass-strong); border:1px solid var(--glass-brd); border-radius:16px;
      padding:16px 16px 15px; overflow:hidden; box-shadow:0 10px 26px -18px rgba(20,40,90,.55);
      backdrop-filter:blur(12px) saturate(150%); -webkit-backdrop-filter:blur(12px) saturate(150%);}
    .kpi::before{content:""; position:absolute; inset:0 0 auto 0; height:3px; background:var(--accent);}
    .kpi .kpi-ic{position:absolute; top:13px; right:14px; font-size:16px; opacity:.5;}
    .kpi .kpi-val{font-size:29px; font-weight:850; letter-spacing:-.025em; line-height:1.02; font-variant-numeric:tabular-nums;
      background:linear-gradient(120deg,#10243f 0%,#1d4ed8 60%,#0891b2 100%);
      -webkit-background-clip:text; background-clip:text; color:transparent;}
    .kpi .kpi-val small{font-size:14px; font-weight:750; -webkit-text-fill-color:var(--muted); color:var(--muted); margin-left:2px;}
    .kpi.text .kpi-val{font-size:20px; line-height:1.12; letter-spacing:-.01em;}
    .kpi .kpi-lab{margin-top:7px; font-size:11.5px; font-weight:800; text-transform:uppercase; letter-spacing:.045em; color:#1f3252;}
    .kpi .kpi-cap{margin-top:4px; font-size:12px; line-height:1.42; color:var(--muted);}
    .chart-svg{width:100%; height:auto; display:block; margin-top:6px; overflow:visible;}
    .chart-svg text{font-family:var(--sans);}
    .legend{display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; font-size:12px; font-weight:650; color:#33486b;}
    .legend span{display:inline-flex; align-items:center; gap:6px;}
    .legend i{width:11px; height:11px; border-radius:3px; display:inline-block;}
    .cov-cell{white-space:nowrap;}

    /* ---------- drill-down benchmark table ---------- */
    .drill .cat-row{cursor:pointer;}
    .drill .cat-row:hover{background:rgba(37,99,235,.07);}
    .drill .cat-row td{font-size:13.5px;}
    .drill .caret{display:inline-block; width:12px; color:var(--cyan); font-size:11px; transition:transform .16s; transform:translateY(-1px);}
    .drill .open .caret{transform:rotate(90deg) translateX(-1px);}
    .drill .sub-row{background:rgba(37,99,235,.045);}
    .drill .sub-row td{padding-top:6px; padding-bottom:6px; font-size:12.5px; color:#33486b;}
    .drill .str-row{cursor:pointer;}
    .drill .str-row:hover{background:rgba(37,99,235,.09);}
    .drill .str-label{padding-left:24px; position:relative;}
    .drill .str-label .caret{margin-right:3px;}
    .drill .sub-label{padding-left:26px; position:relative;}
    .drill .sub-label::before{content:"↳"; position:absolute; left:10px; color:#8aa0c4;}
    .drill .subsub-row{background:rgba(6,182,212,.06);}
    .drill .subsub-row td{padding-top:5px; padding-bottom:5px; font-size:12px; color:#4a5c78;}
    .drill .man-label{padding-left:52px; position:relative;}
    .drill .man-label::before{content:"•"; position:absolute; left:34px; color:#5eb3c9;}
    .drill-hint{font-size:12px; color:var(--muted); margin:0 2px 8px; display:flex; align-items:center; gap:7px;}
    .drill-hint b{color:#2b3f63;}

    /* ---------- pack-size autocomplete ---------- */
    .ac-wrap{position:relative;}
    .ac-list{position:absolute; top:calc(100% + 6px); left:0; right:0; z-index:50; max-height:264px; overflow-y:auto;
      background:var(--glass-strong); border:1px solid var(--glass-brd); border-radius:14px; padding:6px;
      box-shadow:0 24px 50px -18px rgba(20,40,90,.55); backdrop-filter:blur(20px) saturate(160%);
      -webkit-backdrop-filter:blur(20px) saturate(160%);}
    .ac-list[hidden]{display:none;}
    .ac-item{display:flex; align-items:center; gap:9px; padding:9px 11px; border-radius:10px; cursor:pointer;
      font-size:13.5px; color:#1f3252; line-height:1.35; transition:background .12s;}
    .ac-item::before{content:"▤"; font-size:11px; color:#8aa0c4; flex:0 0 auto;}
    .ac-item mark{background:rgba(37,99,235,.16); color:#1d4ed8; border-radius:4px; padding:0 2px; font-weight:750;}
    .ac-item.active, .ac-item:hover{background:linear-gradient(120deg,rgba(37,99,235,.14),rgba(6,182,212,.14));}
    .ac-item.active::before{color:#2563eb;}
    .ac-hint{font-size:11.5px; color:var(--muted); margin:5px 2px 0; display:flex; align-items:center; gap:6px;}
    .ac-hint kbd{font-family:var(--sans); font-size:10.5px; font-weight:750; background:rgba(15,27,45,.08);
      border:1px solid var(--hair); border-radius:5px; padding:1px 5px;}

    /* ---------- print / PDF spec sheet ---------- */
    .print-only{display:none;}
    @media print{
      @page{margin:14mm;}
      html,body{background:#fff !important;}
      body{color:#0f1b2d;}
      header,.nav,.foot,.no-print,form,details,.upload-card{display:none !important;}
      main{margin:0 !important; padding:0 !important; max-width:none !important;}
      .card{box-shadow:none !important; border:1px solid #d5deec !important; background:#fff !important;
        backdrop-filter:none !important; -webkit-backdrop-filter:none !important; page-break-inside:avoid;}
      #specsheet{display:block !important;}
      #specsheet ~ *{display:none !important;}
      .print-only{display:block !important;}
      .viz3d{display:none !important;}       /* canvas doesn't print reliably */
      .metric,.view2d{border:1px solid #d5deec !important; box-shadow:none !important; background:#fff !important;}
      .print-head{display:flex !important; justify-content:space-between; align-items:flex-end;
        border-bottom:2px solid #1d4ed8; padding-bottom:8px; margin-bottom:14px;}
      .print-head .pt{font-size:20px; font-weight:850; color:#12325e;}
      .print-head .ps{font-size:12px; color:#54657f;}
      a[href]:after{content:"";}
    }

    @media (max-width:920px){ .grid,.result{grid-template-columns:1fr;} .result{grid-template-columns:repeat(2,1fr);} }
    @media (max-width:520px){ .result{grid-template-columns:1fr;} }
    @media (prefers-reduced-motion:reduce){ *{animation:none!important; transition:none!important;} }
    """


def model_status_chip() -> str:
    st = model_status()
    status = st.get("status", "idle")
    if not ML_AVAILABLE:
        return ('<span class="chip"><span class="led unavailable"></span>'
                'ML libraries not installed — using nearest-neighbour mode</span>')
    if status == "ready":
        model = st.get("model")
        rows = getattr(model, "trained_rows", 0) if model else 0
        return (f'<span class="chip"><span class="led ready"></span>'
                f'<b>ML model live</b> · trained on {rows:,} packs</span>')
    if status == "training":
        return ('<span class="chip"><span class="led training"></span>'
                'Training ML model… nearest-neighbour estimates active</span>')
    if status == "error":
        return ('<span class="chip"><span class="led error"></span>'
                'Model training failed — using nearest-neighbour mode</span>')
    return ('<span class="chip"><span class="led training"></span>'
            'Preparing model…</span>')


def page_shell(title: str, body: str, active: str = "predictor") -> bytes:
    refresh = ""
    if ML_AVAILABLE and model_status().get("status") == "training":
        # Light auto-refresh so the page upgrades to ML once training finishes.
        refresh = '<meta http-equiv="refresh" content="6">'

    def navlink(href: str, key: str, icon: str, label: str) -> str:
        cls = " active" if key == active else ""
        return f'<a class="{cls.strip()}" href="{href}"><span>{icon}</span>{h(label)}</a>'

    nav = (
        '<nav class="nav no-print">'
        + navlink("/", "predictor", "▣", "Predictor")
        + navlink("/dashboard", "dashboard", "▤", "Data insights")
        + "</nav>"
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh}
  <title>{h(title)}</title>
  <style>{app_css()}</style>
</head>
<body>
  <header>
    <span class="brandline"><span class="dot"></span>3S Pharma Logistics · Pack Intelligence</span>
    <h1>Pharma Pack Dimension Predictor</h1>
    <p class="lede">Estimate a single pharmaceutical pack's outer length, width, height, volume and weight from
    {('a trained machine-learning model' if ML_AVAILABLE else 'the historical database')}, with a confidence band and the closest reference packs from your own inward history.</p>
    <div class="statusbar">{model_status_chip()}</div>
    {nav}
  </header>
  <main>{body}</main>
  <div class="foot no-print">Estimates are operational approximations from historical data — always replace them with a real measurement when the pack is in hand, then add it back to the database to sharpen future predictions.</div>
  <script>{app_js()}</script>
</body>
</html>"""
    return html_doc.encode("utf-8")


def app_js() -> str:
    # Client-side unit conversion (mm/cm/in, cm³/in³, g/kg/lb) + print handler.
    # Server renders every convertible number as <span class="uval" data-kind data-v data-dec>.
    return r"""
(function(){
  var LEN = {mm:{f:1,l:'mm'}, cm:{f:0.1,l:'cm'}, in:{f:1/25.4,l:'in'}};
  var VOL = {mm:{f:1,l:'cm\u00b3'}, cm:{f:1,l:'cm\u00b3'}, in:{f:1/16.387064,l:'in\u00b3'}};
  var MASS= {g:{f:1,l:'g'}, kg:{f:0.001,l:'kg'}, lb:{f:1/453.59237,l:'lb'}};
  var state = {len:'mm', mass:'g'};

  function fmt(n, dec){
    if(!isFinite(n)) return '\u2014';
    var s = n.toLocaleString(undefined,{minimumFractionDigits:dec,maximumFractionDigits:dec});
    return s;
  }
  function apply(){
    var nodes = document.querySelectorAll('.uval');
    for(var i=0;i<nodes.length;i++){
      var el = nodes[i];
      var kind = el.getAttribute('data-kind');
      if(!kind) continue;
      var v = parseFloat(el.getAttribute('data-v'));
      if(isNaN(v)){ el.textContent = '\u2014'; continue; }
      var dec = parseInt(el.getAttribute('data-dec')||'1',10);
      var conv, lbl;
      if(kind==='mm'){ conv=LEN[state.len]; }
      else if(kind==='cm3'){ conv=VOL[state.len]; }
      else if(kind==='g'){ conv=MASS[state.mass]; }
      else { continue; }
      el.textContent = fmt(v*conv.f, dec) + '\u00a0' + conv.l;
    }
  }
  function wireSeg(sel, group){
    var segs = document.querySelectorAll(sel+' button');
    for(var i=0;i<segs.length;i++){
      (function(btn){
        btn.addEventListener('click', function(){
          state[group] = btn.getAttribute('data-u');
          var sibs = btn.parentNode.querySelectorAll('button');
          for(var j=0;j<sibs.length;j++) sibs[j].classList.remove('on');
          btn.classList.add('on');
          apply();
        });
      })(segs[i]);
    }
  }
  function attachAC(ta){
    var wrap = ta.closest('.ac-wrap'); if(!wrap) return;
    var list = wrap.querySelector('.ac-list'); if(!list) return;
    var items = []; var active = -1;
    function esc(s){ return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'); }
    function close(){ list.hidden = true; list.innerHTML=''; active = -1; }
    function pick(v){ ta.value = v; close(); ta.focus(); }
    function setActive(i){
      active = i;
      var els = list.querySelectorAll('.ac-item');
      for(var k=0;k<els.length;k++){ els[k].classList.toggle('active', k===active); }
      if(active>=0 && els[active]) els[active].scrollIntoView({block:'nearest'});
    }
    function render(q){
      var all = window.PACK_SUGGESTIONS || [];
      var ql = q.toLowerCase().trim();
      var res;
      if(!ql){ res = all.slice(0,8); }
      else {
        var starts=[], contains=[];
        for(var i=0;i<all.length;i++){
          var sl = all[i].toLowerCase();
          if(sl.indexOf(ql)===0) starts.push(all[i]);
          else if(sl.indexOf(ql)>-1) contains.push(all[i]);
        }
        res = starts.concat(contains).slice(0,8);
      }
      items = res; active = -1;
      if(!res.length){ close(); return; }
      var re = ql ? new RegExp('('+esc(q.trim())+')','ig') : null;
      list.innerHTML = res.map(function(s){
        return '<div class="ac-item">' + (re ? s.replace(re,'<mark>$1</mark>') : s) + '</div>';
      }).join('');
      var els = list.querySelectorAll('.ac-item');
      for(var j=0;j<els.length;j++){
        (function(idx){
          els[idx].addEventListener('mousedown', function(e){ e.preventDefault(); pick(items[idx]); });
          els[idx].addEventListener('mouseenter', function(){ setActive(idx); });
        })(j);
      }
      list.hidden = false;
    }
    ta.addEventListener('input', function(){ render(ta.value); });
    ta.addEventListener('focus', function(){ if(ta.value.trim()==='') render(''); });
    ta.addEventListener('keydown', function(e){
      if(list.hidden) return;
      if(e.key==='ArrowDown'){ e.preventDefault(); setActive(Math.min(active+1, items.length-1)); }
      else if(e.key==='ArrowUp'){ e.preventDefault(); setActive(Math.max(active-1, 0)); }
      else if(e.key==='Enter' && active>=0){ e.preventDefault(); pick(items[active]); }
      else if(e.key==='Escape'){ close(); }
    });
    ta.addEventListener('blur', function(){ setTimeout(close, 130); });
  }
  function init(){
    wireSeg('.seg[data-group="len"]','len');
    wireSeg('.seg[data-group="mass"]','mass');
    apply();
    var acs = document.querySelectorAll('.ac-pack');
    for(var i=0;i<acs.length;i++) attachAC(acs[i]);
    var catRows = document.querySelectorAll('.drill .cat-row');
    for(var c=0;c<catRows.length;c++){
      (function(row){
        row.addEventListener('click', function(){
          var id = row.getAttribute('data-cat');
          var open = row.classList.toggle('open');
          var subs = document.querySelectorAll('.sub-row[data-parent="'+id+'"]');
          for(var s=0;s<subs.length;s++) subs[s].hidden = !open;
          if(!open){
            var strs = document.querySelectorAll('.str-row[data-parent="'+id+'"]');
            for(var t=0;t<strs.length;t++) strs[t].classList.remove('open');
            var mans = document.querySelectorAll('.subsub-row[data-sparent^="'+id+'-"]');
            for(var m=0;m<mans.length;m++) mans[m].hidden = true;
          }
        });
      })(catRows[c]);
    }
    var strRows = document.querySelectorAll('.drill .str-row');
    for(var r=0;r<strRows.length;r++){
      (function(row){
        row.addEventListener('click', function(e){
          e.stopPropagation();
          var sid = row.getAttribute('data-strength');
          var open = row.classList.toggle('open');
          var mans = document.querySelectorAll('.subsub-row[data-sparent="'+sid+'"]');
          for(var m=0;m<mans.length;m++) mans[m].hidden = !open;
        });
      })(strRows[r]);
    }
    var pb = document.getElementById('printBtn');
    if(pb) pb.addEventListener('click', function(){ window.print(); });
  }
  if(document.readyState!=='loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
"""


def schema_table(dataset: WorkbookData) -> str:
    rows = []
    labels = {
        "name": "Product/item",
        "brand": "Brand",
        "manufacturer": "Manufacturer",
        "category": "Category (dosage form / packaging)",
        "strength": "Strength/dosage",
        "pack": "Pack size",
        "bulk": "Bulk/master pack",
        "dimension_text": "Combined dimensions",
        "length": "Length",
        "width": "Width/breadth",
        "height": "Height/depth",
        "weight": "Weight",
    }
    for key, label in labels.items():
        value = dataset.schema.get(key) or "Not detected"
        rows.append(f"<tr><td>{h(label)}</td><td>{h(value)}</td></tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def read_master_metadata() -> Dict[str, Any]:
    if not MASTER_METADATA_FILE.exists():
        return {}
    try:
        return json.loads(MASTER_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def dataset_summary(dataset: Optional[WorkbookData]) -> str:
    if not dataset:
        return """
        <div class="card">
          <h2><span class="ic">⤓</span>No data loaded yet</h2>
          <p class="small">Put your workbook in the <code>data</code> folder or upload it below. After that the app will inspect sheets, detect columns, parse pack sizes, and train the estimator from rows that already have dimensions.</p>
        </div>
        """

    metadata = read_master_metadata() if dataset.path == MASTER_TRAINING_FILE else {}
    sheet_pills = "".join(f"<span class='pill'>{h(name)}: {len(df):,} rows</span>" for name, df in dataset.sheets.items())
    notes = "".join(f"<div class='notice'>{h(note)}</div>" for note in dataset.notes)
    master_pills = ""
    if metadata:
        master_pills = (
            f"<span class='pill'>{metadata.get('source_file_count', 0):,} historical files</span>"
            f"<span class='pill'>{metadata.get('normalized_rows', 0):,} total rows stored</span>"
            f"<span class='pill'>{metadata.get('training_candidate_rows_before_deduplication', 0):,} usable measured attempts</span>"
            f"<span class='pill good'>{metadata.get('training_rows_after_exact_deduplication', 0):,} unique pack profiles</span>"
            f"<span class='pill'>{metadata.get('repeated_attempts_grouped_with_occurrence_counts', metadata.get('exact_duplicate_observations_removed_from_model', 0)):,} repeated attempts retained as evidence</span>"
        )
    return f"""
    <div class="card">
      <h2><span class="ic">⛁</span>Combined database ready</h2>
      <p><span class="pill good">{h(dataset.path.name)}</span>
      <span class="pill">{dataset.usable_dimension_rows:,} unique measured profiles loaded</span></p>
      <p>{master_pills}</p>
      <p class="small">No historical row was deleted. Exact repeats share one model profile, while their occurrence count and every source row remain stored and contribute capped supporting evidence.</p>
      <p class="small">Future historical files can be appended to this database; existing records remain intact and the model is rebuilt with the expanded history.</p>
      <div>{sheet_pills}</div>
      {notes}
      <details>
        <summary class="small">Show detected database fields</summary>
        {schema_table(dataset)}
      </details>
    </div>
    """


def upload_card() -> str:
    return """
    <div class="card">
      <h2><span class="ic">⇪</span>Test another workbook</h2>
      <p class="small">Optional. The combined master database is already loaded. Uploading here temporarily switches the estimator to that one file; the original is copied and never overwritten.</p>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" accept=".xlsx,.xls,.xlsm,.csv,.tsv" required>
        <button type="submit">Load this file</button>
      </form>
    </div>
    """


def dataset_categories(dataset: Optional[WorkbookData]) -> List[str]:
    """Distinct, clean categories present in the loaded data (for the dropdown)."""
    if not dataset:
        return []
    seen = {}
    for rec in dataset.records:
        cat = clean_text(rec.get("category"))
        if cat and cat.lower() != "unknown":
            seen[cat] = seen.get(cat, 0) + 1
    return sorted(seen, key=lambda c: (-seen[c], c))


def pack_suggestions(dataset: Optional[WorkbookData], limit: int = 140) -> List[str]:
    """Most common canonical pack-size phrases, for the autocomplete field."""
    if not dataset:
        return []
    counter: Dict[str, int] = {}
    for rec in dataset.records:
        s = clean_text(rec.get("pack_text"))
        if not s:
            continue
        s = re.split(r"\s*\(", s, 1)[0].strip()   # drop the "(SINGLE PACK OF …)" technical tail
        if len(s) > 2:
            counter[s] = counter.get(s, 0) + int(max(float(rec.get("occurrence_count") or 1), 1.0))
    return [s for s, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:limit]]


def prediction_form(dataset: Optional[WorkbookData] = None) -> str:
    cats = dataset_categories(dataset)
    if cats:
        options = "".join(f"<option value=\"{h(c)}\">{h(c)}</option>" for c in cats)
        category_field = f"""
        <label>Category</label>
        <select name="category" required>
          <option value="" disabled selected>Choose a category…</option>
          {options}
        </select>
        <label class="small">Optional extra type hint</label>
        <input type="text" name="stated_form" placeholder="Example: prefilled syringe, syrup, cream">
        """
    else:
        # No data loaded yet — fall back to a free-text field.
        category_field = """
        <label>Category / dosage form</label>
        <input type="text" name="stated_form" placeholder="Example: injection vial, tablet, ampoule, syrup, cream" required>
        """
    return f"""
    <div class="card">
      <h2><span class="ic">✎</span>Estimate a new item</h2>
      <form method="post" action="/predict">
        <label>Drug or brand name</label>
        <input type="text" name="name" placeholder="Example: Enhertu or Paracetamol" required>

        <label>Strength / dosage</label>
        <input type="text" name="strength" placeholder="Example: 500 mg, 125 mg/5 mL, 2%, 100 mcg/dose">
        {category_field}
        <label>Pack size</label>
        <div class="ac-wrap">
          <textarea name="pack" class="ac-pack" autocomplete="off" placeholder="Start typing… e.g. 10 strips of tablets, 1 vial per pack, 5 ampoules per pack" required></textarea>
          <div class="ac-list" hidden></div>
        </div>
        <div class="ac-hint">↑↓ to browse · <kbd>Enter</kbd> to fill · suggestions from your own history</div>

        <details>
          <summary class="small">Optional bulk / master packing</summary>
          <label>Bulk / shipper / master pack</label>
          <textarea name="bulk" placeholder="Example: bulk pack 100 vials, 50 strips per carton"></textarea>
        </details>

        <button type="submit">Predict dimensions</button>
      </form>
    </div>
    """


def _dim_view_svg(w_mm: float, h_mm: float, title: str, scale: float,
                  w_label: str, h_label: str) -> str:
    """One engineering-style view: a rectangle with dimension lines on two sides."""
    pad = 46  # room for dimension lines + labels
    rw, rh = w_mm * scale, h_mm * scale
    W = rw + pad * 2
    H = rh + pad * 2
    x0, y0 = pad, pad
    x1, y1 = x0 + rw, y0 + rh
    # horizontal dimension line below the rectangle
    hy = y1 + 20
    # vertical dimension line left of the rectangle
    vx = x0 - 20
    return f"""
    <div class="view2d">
      <svg viewBox="0 0 {W:.0f} {H:.0f}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="{h(title)} view">
        <defs>
          <marker id="ar" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
            <path d="M1,1 L7,4 L1,7" fill="none" stroke="#1d4ed8" stroke-width="1.2"/>
          </marker>
        </defs>
        <rect x="{x0:.1f}" y="{y0:.1f}" width="{rw:.1f}" height="{rh:.1f}"
              fill="rgba(37,99,235,.10)" stroke="#1d4ed8" stroke-width="1.6" rx="3"/>
        <!-- width dimension -->
        <line x1="{x0:.1f}" y1="{hy:.1f}" x2="{x1:.1f}" y2="{hy:.1f}" stroke="#1d4ed8" stroke-width="1" marker-start="url(#ar)" marker-end="url(#ar)"/>
        <line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x0:.1f}" y2="{hy+4:.1f}" stroke="#93b0e8" stroke-width=".8"/>
        <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{hy+4:.1f}" stroke="#93b0e8" stroke-width=".8"/>
        <text x="{(x0+x1)/2:.1f}" y="{hy+16:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#12325e">{h(w_label)}</text>
        <!-- height dimension -->
        <line x1="{vx:.1f}" y1="{y0:.1f}" x2="{vx:.1f}" y2="{y1:.1f}" stroke="#1d4ed8" stroke-width="1" marker-start="url(#ar)" marker-end="url(#ar)"/>
        <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{vx-4:.1f}" y2="{y0:.1f}" stroke="#93b0e8" stroke-width=".8"/>
        <line x1="{x0:.1f}" y1="{y1:.1f}" x2="{vx-4:.1f}" y2="{y1:.1f}" stroke="#93b0e8" stroke-width=".8"/>
        <text x="{vx-8:.1f}" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#12325e" transform="rotate(-90 {vx-8:.1f} {(y0+y1)/2:.1f})">{h(h_label)}</text>
      </svg>
      <div class="view2d-title">{h(title)}</div>
    </div>
    """


def pack_2d_drawings(length: Optional[float], width: Optional[float], height: Optional[float]) -> str:
    if not all(isinstance(v, (int, float)) and v and v > 0 for v in [length, width, height]):
        return ""
    l, w, hgt = float(length), float(width), float(height)
    scale = 150.0 / max(l, w, hgt)  # shared scale so views are truthful relative to each other

    def lab(v: float) -> str:
        return f"{v:,.0f} mm"

    front = _dim_view_svg(w, hgt, "Front  (width × height)", scale, lab(w), lab(hgt))
    side = _dim_view_svg(l, hgt, "Side  (length × height)", scale, lab(l), lab(hgt))
    top = _dim_view_svg(w, l, "Top  (width × length)", scale, lab(w), lab(l))
    return f"""
    <h3>Dimensioned drawings</h3>
    <p class="small">Three true-to-scale views with exact millimetre labels — the way the pack would appear on a packing sheet.</p>
    <div class="views2d">{front}{side}{top}</div>
    """


def pack_3d_widget(length: Optional[float], width: Optional[float], height: Optional[float],
                   ranges: Dict[str, Any]) -> str:
    """An interactive, true-to-scale 3D pack box (drag to rotate, scroll to zoom)."""
    if not all(isinstance(v, (int, float)) and v and v > 0 for v in [length, width, height]):
        return ""
    l, w, hgt = float(length), float(width), float(height)
    hi_l = (ranges.get("length_mm") or (None, None))[1] or l
    hi_w = (ranges.get("width_mm") or (None, None))[1] or w
    hi_h = (ranges.get("height_mm") or (None, None))[1] or hgt
    wid = f"pack3d_{int(l)}_{int(w)}_{int(hgt)}"
    return f"""
    <h3>Interactive 3D pack</h3>
    <p class="small">Drag to rotate, scroll to zoom. The card is a standard credit card (85.6 × 54 mm) for scale;
    the faint outer box shows the upper end of the likely range.</p>
    <div class="viz3d"><div id="{wid}" class="viz3d-canvas"></div>
      <div class="viz3d-legend">
        <span><i style="background:#2563eb"></i>Predicted pack {l:.0f}×{w:.0f}×{hgt:.0f} mm</span>
        <span><i style="background:#94a3b8"></i>Upper likely range</span>
        <span><i style="background:#f5c542"></i>Credit card (for scale)</span>
      </div>
    </div>
    <script src="/static/three.min.js"></script>
    <script>
    (function(){{
      if(!window.THREE){{return;}}
      var L={l:.1f}, W={w:.1f}, H={hgt:.1f};
      var HL={hi_l:.1f}, HW={hi_w:.1f}, HH={hi_h:.1f};
      var host=document.getElementById("{wid}");
      if(!host) return;
      var width=host.clientWidth||520, height=340;
      var scene=new THREE.Scene();
      var camera=new THREE.PerspectiveCamera(42, width/height, 1, 20000);
      var renderer=new THREE.WebGLRenderer({{antialias:true, alpha:true}});
      renderer.setPixelRatio(window.devicePixelRatio||1);
      renderer.setSize(width,height);
      host.appendChild(renderer.domElement);

      var pivot=new THREE.Group(); scene.add(pivot);
      // Solid predicted box (x=width, y=height, z=length), sitting on ground y=0
      function boxEdges(bw,bh,bd,color,op,solid){{
        var g=new THREE.Group();
        var geo=new THREE.BoxGeometry(bw,bh,bd);
        if(solid){{
          var mat=new THREE.MeshPhongMaterial({{color:color,transparent:true,opacity:op,shininess:60}});
          g.add(new THREE.Mesh(geo,mat));
        }}
        var edges=new THREE.LineSegments(new THREE.EdgesGeometry(geo),
          new THREE.LineBasicMaterial({{color:color,transparent:!solid,opacity:solid?1:0.55}}));
        g.add(edges); g.position.y=bh/2; return g;
      }}
      pivot.add(boxEdges(HW,HH,HL,0x94a3b8,0.06,false)); // upper-range ghost
      pivot.add(boxEdges(W,H,L,0x2563eb,0.28,true));      // predicted solid

      // Credit card on the ground next to the box
      var card=new THREE.Mesh(new THREE.BoxGeometry(85.6,0.76,54),
        new THREE.MeshPhongMaterial({{color:0xf5c542,shininess:80}}));
      card.position.set(W/2 + 85.6/2 + Math.max(12,W*0.12), 0.38, 0);
      pivot.add(card);

      // Ground grid for depth
      var span=Math.max(L,W,HL,HW)*2.4;
      var grid=new THREE.GridHelper(span, 16, 0xc7d2e5, 0xe2e8f5);
      grid.material.transparent=true; grid.material.opacity=0.5; scene.add(grid);

      // Dimension labels as sprites
      function label(text){{
        var c=document.createElement("canvas"); c.width=256; c.height=64;
        var x=c.getContext("2d"); x.fillStyle="rgba(255,255,255,0.92)";
        roundRect(x,4,10,248,44,10); x.fill();
        x.fillStyle="#12325e"; x.font="bold 30px system-ui,Segoe UI,Arial"; x.textAlign="center"; x.textBaseline="middle";
        x.fillText(text,128,34);
        var tex=new THREE.CanvasTexture(c);
        var sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:tex,transparent:true}}));
        var s=Math.max(L,W,H)*0.30; sp.scale.set(s*2,s*0.5,1); return sp;
      }}
      function roundRect(ctx,x,y,w,h,r){{ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}}
      var lw=label(W.toFixed(0)+" mm"); lw.position.set(0,-Math.max(L,W,H)*0.12,L/2+2); pivot.add(lw);
      var lh=label(H.toFixed(0)+" mm"); lh.position.set(-W/2-Math.max(L,W,H)*0.14,H/2,L/2); pivot.add(lh);
      var ll=label(L.toFixed(0)+" mm"); ll.position.set(W/2+Math.max(L,W,H)*0.14,H*0.1,0); pivot.add(ll);

      scene.add(new THREE.AmbientLight(0xffffff,0.75));
      var d=new THREE.DirectionalLight(0xffffff,0.7); d.position.set(1,2,1.5); scene.add(d);
      var d2=new THREE.DirectionalLight(0xbcd0ff,0.4); d2.position.set(-1,1,-1); scene.add(d2);

      var center=new THREE.Vector3(W*0.25,H/2,0);
      var dist=Math.max(L,W,HL,HW,H)*1.95;
      var rotX=-0.5, rotY=0.7, auto=true;
      function place(){{
        camera.position.set(center.x+dist*Math.sin(rotY)*Math.cos(rotX),
                            center.y+dist*Math.sin(rotX)+Math.max(L,W,H)*0.2,
                            center.z+dist*Math.cos(rotY)*Math.cos(rotX));
        camera.lookAt(center);
      }}
      // interaction
      var dragging=false,px=0,py=0;
      renderer.domElement.style.cursor="grab";
      renderer.domElement.addEventListener("mousedown",function(e){{dragging=true;auto=false;px=e.clientX;py=e.clientY;renderer.domElement.style.cursor="grabbing";}});
      window.addEventListener("mouseup",function(){{dragging=false;renderer.domElement.style.cursor="grab";}});
      window.addEventListener("mousemove",function(e){{if(!dragging)return;rotY+=(e.clientX-px)*0.01;rotX+=(e.clientY-py)*0.01;rotX=Math.max(-1.35,Math.min(1.35,rotX));px=e.clientX;py=e.clientY;}});
      renderer.domElement.addEventListener("wheel",function(e){{e.preventDefault();dist*=(1+(e.deltaY>0?0.1:-0.1));dist=Math.max(Math.max(L,W,H)*1.1,Math.min(Math.max(L,W,H)*6,dist));}},{{passive:false}});
      // touch
      renderer.domElement.addEventListener("touchstart",function(e){{if(e.touches.length){{dragging=true;auto=false;px=e.touches[0].clientX;py=e.touches[0].clientY;}}}},{{passive:true}});
      renderer.domElement.addEventListener("touchmove",function(e){{if(!dragging||!e.touches.length)return;rotY+=(e.touches[0].clientX-px)*0.01;rotX+=(e.touches[0].clientY-py)*0.01;rotX=Math.max(-1.35,Math.min(1.35,rotX));px=e.touches[0].clientX;py=e.touches[0].clientY;}},{{passive:true}});
      window.addEventListener("touchend",function(){{dragging=false;}});

      function resize(){{var nw=host.clientWidth||width;renderer.setSize(nw,height);camera.aspect=nw/height;camera.updateProjectionMatrix();}}
      window.addEventListener("resize",resize);

      function loop(){{requestAnimationFrame(loop);if(auto)rotY+=0.004;place();renderer.render(scene,camera);}}
      loop();
    }})();
    </script>
    """


def isometric_box_svg(length: Optional[float], width: Optional[float], height: Optional[float]) -> str:
    """A lightweight isometric carton whose proportions track the predicted L/W/H."""
    if not all(isinstance(v, (int, float)) and v and v > 0 for v in [length, width, height]):
        return ""
    l, w, hgt = float(length), float(width), float(height)
    longest = max(l, w, hgt)
    # scale the longest edge to a fixed drawing size, keep relative proportions
    s = 92.0 / longest
    L, W, H = l * s, w * s, hgt * s
    ox, oy = 150.0, 150.0  # origin (top vertex area)
    cos30, sin30 = 0.8660254, 0.5

    def iso(x: float, y: float, z: float) -> Tuple[float, float]:
        # map 3D (x=length, y=width, z=height) to 2D isometric
        sx = ox + (x - y) * cos30
        sy = oy + (x + y) * sin30 - z
        return sx, sy

    # base footprint L×W, extruded up by H
    a = iso(0, 0, H); b = iso(L, 0, H); c = iso(L, W, H); d = iso(0, W, H)   # top face
    e = iso(L, 0, 0); f = iso(L, W, 0); g = iso(0, W, 0)                     # lower verts

    def pts(*p):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in p)

    return f"""
    <div class="pack-preview">
      <svg viewBox="0 0 300 300" width="190" height="190" role="img" aria-label="Predicted pack shape">
        <defs>
          <linearGradient id="gTop" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#7dd3fc"/><stop offset="1" stop-color="#a5b4fc"/>
          </linearGradient>
          <linearGradient id="gLeft" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#2563eb"/><stop offset="1" stop-color="#1e40af"/>
          </linearGradient>
          <linearGradient id="gRight" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#06b6d4"/><stop offset="1" stop-color="#0e7490"/>
          </linearGradient>
        </defs>
        <polygon points="{pts(d,c,f,g)}" fill="url(#gRight)" opacity="0.96"/>
        <polygon points="{pts(b,c,f,e)}" fill="url(#gLeft)" opacity="0.96"/>
        <polygon points="{pts(a,b,c,d)}" fill="url(#gTop)"/>
        <g fill="none" stroke="rgba(255,255,255,.55)" stroke-width="1">
          <polygon points="{pts(a,b,c,d)}"/>
        </g>
      </svg>
    </div>
    <div class="small" style="text-align:center;margin-top:6px;">Proportional pack envelope (longest × middle × shortest)</div>
    """


def confidence_ring(score: int, label: str) -> str:
    col = "var(--good)" if label == "High" else "var(--warn)" if label == "Medium" else "var(--low)"
    return f"""
    <div class="ring" style="--p:{score};--col:{col};">
      <div class="inner">
        <div>
          <div class="num">{score}<small style="font-size:13px;font-weight:700;">%</small></div>
          <div class="cap">{h(label)}</div>
        </div>
      </div>
    </div>
    """


def model_card(dataset: Optional[WorkbookData]) -> str:
    if not ML_AVAILABLE:
        return (
            '<div class="card"><h2><span class="ic">✦</span>Machine-learning model</h2>'
            '<div class="notice">The trained ML model is switched off because <b>scikit-learn</b> '
            'is not installed, so predictions are using the simpler nearest-neighbour method '
            '(lower accuracy and confidence).</div>'
            '<p class="small">To turn the ML model on, close the app, open a Command Prompt in this '
            'folder and run:</p>'
            '<div class="muted-box">pip install scikit-learn</div>'
            '<p class="small" style="margin-top:8px;">Then start the app again. On the next launch it '
            'will train the model (about a minute) and predictions will use the higher-accuracy ML blend.</p>'
            '</div>'
        )
    st = model_status()
    status = st.get("status")
    if status == "training":
        body = ('<p class="small">The ExtraTrees model is training in the background on your full history. '
                'Predictions use the nearest-neighbour fallback until it is ready (usually under a minute) — '
                'the page refreshes itself automatically.</p>')
        return f'<div class="card"><h2><span class="ic">◐</span>Machine-learning model</h2>{body}</div>'
    if status == "error":
        return (f'<div class="card"><h2><span class="ic">!</span>Machine-learning model</h2>'
                f'<div class="notice">Training failed: {h(st.get("error"))}. Nearest-neighbour mode is active.</div></div>')
    model = st.get("model")
    if model is None or not getattr(model, "metrics", None):
        return ""
    labels = {"length_mm": "Length", "width_mm": "Width", "height_mm": "Height", "weight_g": "Weight"}
    units = {"length_mm": "mm", "width_mm": "mm", "height_mm": "mm", "weight_g": "g"}
    rows = []
    for key, lab in labels.items():
        m = model.metrics.get(key)
        if not m:
            continue
        acc = max(0, min(100, round(m["r2"] * 100)))
        rows.append(
            "<tr>"
            f"<td><b>{lab}</b></td>"
            f"<td>± {fmt_number(m['mae'], ' ' + units[key])}</td>"
            f"<td>{acc}%</td>"
            f"<td><div style='height:7px;border-radius:6px;background:rgba(15,27,45,.1);overflow:hidden;'>"
            f"<div style='height:100%;width:{acc}%;background:var(--accent);'></div></div></td>"
            "</tr>"
        )
    return f"""
    <div class="card">
      <h2><span class="ic">✦</span>Machine-learning model</h2>
      <p class="small">A trained ExtraTrees ensemble (200 randomised regression trees per dimension) learns pack shape from
      drug name, brand, dosage form, container, strength and pack counts. Typical error and fit are measured on packs the
      model never saw during training.</p>
      <table>
        <thead><tr><th>Dimension</th><th>Typical error</th><th>Variance explained</th><th></th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="small" style="margin-top:10px;">Each estimate is a 65/35 blend of this model and the nearest-neighbour
      estimate, so it stays robust even for unusual items.</p>
    </div>
    """


def render_prediction(result: Dict[str, Any]) -> str:
    if result.get("error"):
        return f"<div class='card'><h2><span class='ic'>!</span>Prediction</h2><div class='notice'>{h(result['error'])}</div></div>"

    query = result["query"]
    predicted = result["predicted"]
    ranges = result["ranges"]
    notes = "".join(f"<div class='notice'>{h(note)}</div>" for note in query.get("classification_notes", []))
    parsed_pills = [
        ("Category", query.get("category")),
        ("Container", query.get("container_type")),
        ("Unit count", fmt_number(query.get("unit_count")) if query.get("unit_count") is not None else None),
        ("Bulk count", fmt_number(query.get("bulk_count")) if query.get("bulk_count") is not None else None),
        ("Volume", fmt_number(query.get("container_volume_ml"), " mL") if query.get("container_volume_ml") is not None else None),
        ("Content weight", fmt_number(query.get("content_weight_g"), " g") if query.get("content_weight_g") is not None else None),
    ]
    pills = "".join(f"<span class='pill'>{h(label)}: {h(value)}</span>" for label, value in parsed_pills if value)

    def metric(label: str, key: str, kind: str, decimals: int = 1) -> str:
        low, high = ranges.get(key, (None, None))
        band = ""
        if low is not None and high is not None:
            band = f"<div class='band'>likely {uval(low, kind, decimals)} – {uval(high, kind, decimals)}</div>"
        return f"""
        <div class="metric">
          <div class="label">{h(label)}</div>
          <div class="value">{uval(predicted.get(key), kind, decimals)}</div>
          {band}
        </div>
        """

    neighbor_rows = []
    for item in result["neighbors"]:
        identity = h(item.get("name"))
        if item.get("brand"):
            identity += f"<div class='small'>{h(item.get('brand'))}</div>"
        source = h(item.get("source_file"))
        if item.get("source_row"):
            source += f"<div class='small'>row {h(item.get('source_row'))}</div>"
        occurrence = item.get("occurrence_count") or 1
        if occurrence > 1:
            source += f"<div class='small'>{fmt_number(occurrence)} exact observations</div>"
        reason_text = "; ".join(item.get("match_reasons") or [])
        neighbor_rows.append(
            "<tr>"
            f"<td>{identity}</td>"
            f"<td>{h(item.get('manufacturer'))}</td>"
            f"<td>{h(item.get('category'))}</td>"
            f"<td>{h(item.get('pack'))}</td>"
            f"<td>{uval(item.get('length_mm'), 'mm', 0)} × {uval(item.get('width_mm'), 'mm', 0)} × {uval(item.get('height_mm'), 'mm', 0)}</td>"
            f"<td>{uval(item.get('volume_cm3'), 'cm3', 1)}</td>"
            f"<td>{uval(item.get('weight_g'), 'g', 1)}</td>"
            f"<td>{h(reason_text)}</td>"
            f"<td>{source}</td>"
            "</tr>"
        )

    method = result.get("prediction_method", "Nearest-neighbour estimate")
    method_class = "good" if result.get("ml_used") else "warn"
    box3d = pack_3d_widget(predicted.get("length_mm"), predicted.get("width_mm"), predicted.get("height_mm"), ranges)
    draw2d = pack_2d_drawings(predicted.get("length_mm"), predicted.get("width_mm"), predicted.get("height_mm"))
    ring = confidence_ring(result["confidence_score"], result["confidence"])

    # ---- coverage badge + notice ----
    cov = result.get("coverage") or {}
    cov_count = result.get("category_sample_count", 0)
    cov_badge = ""
    cov_notice = ""
    if cov:
        cat_label = query.get("category") or "this category"
        cov_badge = f'<span class="cov-badge {cov.get("cls", "cov-limited")}">◆ {h(cov.get("label", ""))} · {fmt_int(cov_count)} packs</span>'
        if cov.get("key") in ("sparse", "verysparse", "none"):
            if cov.get("key") == "none":
                msg = (f"No measured packs exist in the “{cat_label}” category yet, so this estimate is "
                       f"extrapolated from the nearest different packs. Treat it as a rough guide and confirm by measurement.")
            else:
                msg = (f"Only {fmt_int(cov_count)} measured pack(s) exist in the “{cat_label}” category, so confidence "
                       f"is capped. Measuring a few more items here will sharpen future estimates.")
            cov_notice = f'<div class="notice">{h(msg)}</div>'

    spec_title = ", ".join(filter(None, [query.get('name'), query.get('strength_text'), query.get('category')]))
    today = dt.date.today().isoformat()
    density_card = density_metric(predicted)
    restricted = result.get("restricted_to_category")
    chosen_cat = query.get("category")
    restrict_pill = (f'<span class="pill good">◧ {h(chosen_cat)} references only</span>'
                     if restricted and chosen_cat else "")
    ref_heading = (f"Closest {h(chosen_cat)} packs from your history"
                   if restricted and chosen_cat else "Closest reference packs from your history")
    return f"""
    <div class="card" id="specsheet">
      <div class="print-only print-head">
        <div><div class="pt">Pack Dimension Spec Sheet</div><div class="ps">3S Pharma Logistics · Pack Intelligence</div></div>
        <div class="ps">{h(spec_title)}<br>Generated {h(today)}</div>
      </div>
      <h2><span class="ic">▣</span>Predicted pack</h2>
      <div class="conf-wrap">
        {ring}
        <div class="conf-meta">
          <span class="pill {method_class}">{h(method)}</span>
          <span class="pill">Same-category matches: {result['same_category_neighbors']}</span>
          <span class="pill">Evidence rows: {result['measured_attempts_nearby']}</span>
          {restrict_pill}
          <div class="coverage-note" style="margin-top:8px;">{cov_badge}</div>
          <div class="small" style="margin-top:8px;">{h(spec_title)}</div>
          <button type="button" id="printBtn" class="no-print" style="margin-top:12px;padding:9px 15px;font-size:13px;">⎙ Print / Save as PDF</button>
        </div>
      </div>
      <div>{pills}</div>
      {cov_notice}
      {notes}
      {unit_toggle_html()}
      <div class="result">
        {metric("Length", "length_mm", "mm")}
        {metric("Width", "width_mm", "mm")}
        {metric("Height", "height_mm", "mm")}
        {metric("Volume", "volume_cm3", "cm3", 1)}
        {metric("Weight", "weight_g", "g")}
        {density_card}
      </div>
      {box3d}
      {draw2d}
      <h3>{ref_heading}</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Reference product</th><th>Manufacturer</th><th>Category</th><th>Pack</th><th>Dimensions</th><th>Volume</th><th>Weight</th><th>Why it matched</th><th>Source</th></tr></thead>
          <tbody>{''.join(neighbor_rows)}</tbody>
        </table>
      </div>
    </div>
    """


def sample_rows(dataset: WorkbookData, limit: int = 8) -> str:
    rows = []
    for rec in dataset.records[:limit]:
        rows.append(
            "<tr>"
            f"<td>{h(rec.get('name'))}</td>"
            f"<td>{h(rec.get('category'))}</td>"
            f"<td>{h(rec.get('pack_text'))}</td>"
            f"<td>{fmt_number(rec.get('length_mm'), ' mm')}</td>"
            f"<td>{fmt_number(rec.get('width_mm'), ' mm')}</td>"
            f"<td>{fmt_number(rec.get('height_mm'), ' mm')}</td>"
            f"<td>{'; '.join(h(x) for x in rec.get('classification_notes') or [])}</td>"
            "</tr>"
        )
    return f"""
    <div class="card">
      <h2><span class="ic">☰</span>Classification preview</h2>
      <p class="small">A quick look at how the app is interpreting the first rows. This is where vial/ampoule and pack parsing issues become visible.</p>
      <table>
        <thead><tr><th>Item</th><th>Category</th><th>Pack text</th><th>L</th><th>W</th><th>H</th><th>Notes</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def home_page(prediction_html: str = "") -> bytes:
    try:
        dataset = get_dataset()
        if dataset is not None:
            ensure_model(dataset)
        preview = sample_rows(dataset) if dataset else ""
        body = f"""
        <script>window.PACK_SUGGESTIONS = {json.dumps(pack_suggestions(dataset))};</script>
        <div class="grid">
          <div>{dataset_summary(dataset)}{model_card(dataset)}{upload_card()}</div>
          <div>{prediction_form(dataset)}</div>
        </div>
        {prediction_html}
        {compare_form(dataset)}
        {preview}
        """
    except Exception as exc:
        body = f"""
        <div class="grid">
          <div>{upload_card()}</div>
          <div>{prediction_form(None)}</div>
        </div>
        <div class="card">
          <h2><span class="ic">!</span>Could not load workbook</h2>
          <div class="notice">{h(exc)}</div>
          <pre class="muted-box">{h(traceback.format_exc())}</pre>
        </div>
        """
    return page_shell("Pharma Pack Dimension Predictor", body)


# ===================================================================
#  Unit conversion, coverage, comparison and dashboard (added features)
# ===================================================================

UNIT_SUFFIX = {"mm": " mm", "cm3": " cm\u00b3", "g": " g"}


def fmt_int(value: Optional[float]) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "\u2014"


def _short(text: Optional[str], limit: int) -> str:
    text = (text or "\u2014").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\u2026"


def strength_bucket(rec: Dict[str, Any]) -> str:
    """A clean, canonical strength/dosage label used to group packs within a category."""
    mg = rec.get("strength_mg")
    conc = rec.get("concentration_mg_ml")
    pct = rec.get("percent_strength")
    if mg:
        return f"{mg / 1000:g} g" if mg >= 1000 else f"{mg:g} mg"
    if conc:
        return f"{conc:g} mg/mL"
    if pct:
        return f"{pct:g}%"
    st = clean_text(rec.get("strength_text"))
    return _short(st, 22) if st else "Unspecified"


def uval(value: Optional[float], kind: str, decimals: int = 1) -> str:
    """A number that the client-side unit toggle can re-express live."""
    try:
        fv = float(value)
        if math.isnan(fv):
            raise ValueError
    except (TypeError, ValueError):
        return '<span class="uval">\u2014</span>'
    return (f'<span class="uval" data-kind="{kind}" data-v="{fv:.4f}" data-dec="{decimals}">'
            f'{fmt_number(fv, UNIT_SUFFIX.get(kind, ""), decimals)}</span>')


def unit_toggle_html() -> str:
    return """
    <div class="unit-bar no-print">
      <div class="unit-group"><span>Length</span>
        <div class="seg" data-group="len">
          <button type="button" data-u="mm" class="on">mm</button>
          <button type="button" data-u="cm">cm</button>
          <button type="button" data-u="in">in</button>
        </div>
      </div>
      <div class="unit-group"><span>Weight</span>
        <div class="seg" data-group="mass">
          <button type="button" data-u="g" class="on">g</button>
          <button type="button" data-u="kg">kg</button>
          <button type="button" data-u="lb">lb</button>
        </div>
      </div>
    </div>
    """


def _density(pred: Dict[str, Any]) -> Optional[float]:
    vol = pred.get("volume_cm3")
    wt = pred.get("weight_g")
    if vol and wt and vol > 0:
        return wt / vol
    return None


def density_metric(pred: Dict[str, Any]) -> str:
    d = _density(pred)
    if d is None:
        return ""
    hint = "denser than water" if d > 1.0 else "lighter than water"
    return f"""
        <div class="metric">
          <div class="label">Density</div>
          <div class="value">{fmt_number(d, '', 2)}<small> g/cm\u00b3</small></div>
          <div class="band">{hint}</div>
        </div>
        """


def coverage_info(count: int) -> Dict[str, Any]:
    """Turn a category sample count into a coverage tier + confidence cap."""
    if count >= 40:
        return {"key": "rich", "cls": "cov-rich", "label": "Rich coverage", "cap": 96}
    if count >= 15:
        return {"key": "limited", "cls": "cov-limited", "label": "Good coverage", "cap": 82}
    if count >= 5:
        return {"key": "sparse", "cls": "cov-sparse", "label": "Limited data", "cap": 62}
    if count >= 1:
        return {"key": "verysparse", "cls": "cov-sparse", "label": "Very little data", "cap": 45}
    return {"key": "none", "cls": "cov-none", "label": "No direct samples", "cap": 34}


# ---- SVG chart primitives (theme-matched, dependency-free) ----

PALETTE = ["#2563eb", "#06b6d4", "#6366f1", "#059669", "#d97706", "#db2777", "#0891b2", "#7c3aed"]


def svg_hbar(pairs: List[Tuple[str, float]], gid: str, unit: str = "",
             width: int = 520, pad_left: int = 176, bar_h: int = 22, gap: int = 9) -> str:
    pairs = list(pairs)
    if not pairs:
        return "<p class='small'>No data available.</p>"
    maxv = max(v for _, v in pairs) or 1
    inner = width - pad_left - 66
    height = gap + len(pairs) * (bar_h + gap)
    parts = [
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="#2563eb"/><stop offset="1" stop-color="#06b6d4"/>'
        f'</linearGradient></defs>'
    ]
    for i, (lab, v) in enumerate(pairs):
        y = gap + i * (bar_h + gap)
        w = max(2.0, inner * (v / maxv))
        label = h(str(lab)[:28])
        parts.append(f'<text x="{pad_left - 10}" y="{y + bar_h * 0.68:.0f}" text-anchor="end" '
                     f'font-size="12" fill="#33486b">{label}</text>')
        parts.append(f'<rect x="{pad_left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="5" fill="url(#{gid})"/>')
        parts.append(f'<text x="{pad_left + w + 7:.1f}" y="{y + bar_h * 0.68:.0f}" '
                     f'font-size="11.5" font-weight="700" fill="#12325e">{fmt_int(v)}{h(unit)}</text>')
    return (f'<svg class="chart-svg" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMinYMin meet" role="img">{"".join(parts)}</svg>')


def svg_scatter(points: List[Tuple[float, float, str]], color_map: Dict[str, str],
                width: int = 520, height: int = 360) -> str:
    pts = [(v, w, c) for v, w, c in points if v and w and v > 0 and w > 0]
    if len(pts) < 3:
        return "<p class='small'>Not enough measured points for a scatter.</p>"
    pad_l, pad_b, pad_t, pad_r = 52, 38, 14, 14
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_b - pad_t
    xs = [math.log10(v) for v, _, _ in pts]
    ys = [math.log10(w) for _, w, _ in pts]
    xmin, xmax = math.floor(min(xs)), math.ceil(max(xs))
    ymin, ymax = math.floor(min(ys)), math.ceil(max(ys))
    xmax = xmax if xmax > xmin else xmin + 1
    ymax = ymax if ymax > ymin else ymin + 1

    def px(lx):
        return pad_l + (lx - xmin) / (xmax - xmin) * plot_w

    def py(ly):
        return pad_t + plot_h - (ly - ymin) / (ymax - ymin) * plot_h

    parts = [f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="rgba(255,255,255,.4)" rx="8"/>']
    # grid + axis ticks (log decades)
    for dx in range(xmin, xmax + 1):
        x = px(dx)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" stroke="rgba(15,27,45,.08)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 15}" text-anchor="middle" font-size="10.5" fill="#54657f">{fmt_int(10 ** dx)}</text>')
    for dy in range(ymin, ymax + 1):
        y = py(dy)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="rgba(15,27,45,.08)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" text-anchor="end" font-size="10.5" fill="#54657f">{fmt_int(10 ** dy)}</text>')
    for v, w, c in pts:
        cx, cy = px(math.log10(v)), py(math.log10(w))
        col = color_map.get(c, "rgba(84,101,127,.5)")
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" fill="{col}" fill-opacity="0.8"/>')
    parts.append(f'<text x="{pad_l + plot_w / 2:.0f}" y="{height - 4}" text-anchor="middle" font-size="11" font-weight="700" fill="#33486b">Volume (cm\u00b3, log)</text>')
    parts.append(f'<text x="14" y="{pad_t + plot_h / 2:.0f}" text-anchor="middle" font-size="11" font-weight="700" fill="#33486b" transform="rotate(-90 14 {pad_t + plot_h / 2:.0f})">Weight (g, log)</text>')
    return (f'<svg class="chart-svg" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMidYMid meet" role="img">{"".join(parts)}</svg>')


def svg_histogram(values: List[float], gid: str, vmax: float = 1.5, nbins: int = 15,
                  width: int = 520, height: int = 300, xlabel: str = "",
                  refs: Optional[List[Tuple[float, str, str]]] = None) -> str:
    vals = [min(max(v, 0.0), vmax - 1e-9) for v in values if v is not None]
    if not vals:
        return "<p class='small'>No data.</p>"
    bw = vmax / nbins
    bins = [0] * nbins
    for v in vals:
        bins[min(int(v / bw), nbins - 1)] += 1
    maxc = max(bins) or 1
    pad_l, pad_b, pad_t, pad_r = 44, 36, 16, 14
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def X(val):
        return pad_l + (val / vmax) * pw

    parts = [f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="#2563eb"/><stop offset="1" stop-color="#06b6d4"/></linearGradient></defs>']
    barw = pw / nbins
    for i, c in enumerate(bins):
        bh = ph * (c / maxc)
        x = pad_l + i * barw
        y = pad_t + ph - bh
        parts.append(f'<rect x="{x + 1.4:.1f}" y="{y:.1f}" width="{barw - 2.8:.1f}" height="{bh:.1f}" rx="3" fill="url(#{gid})"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" y2="{pad_t + ph}" stroke="rgba(15,27,45,.18)" stroke-width="1"/>')
    for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        if t > vmax:
            continue
        x = X(t)
        parts.append(f'<text x="{x:.1f}" y="{pad_t + ph + 15}" text-anchor="middle" font-size="10.5" fill="#54657f">{t:g}</text>')
    for val, label, color in (refs or []):
        if val <= vmax:
            x = X(val)
            parts.append(f'<line x1="{x:.1f}" y1="{pad_t - 2}" x2="{x:.1f}" y2="{pad_t + ph}" stroke="{color}" stroke-width="1.6" stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{x + 4:.1f}" y="{pad_t + 8}" font-size="10.5" font-weight="700" fill="{color}">{h(label)}</text>')
    if xlabel:
        parts.append(f'<text x="{pad_l + pw / 2:.0f}" y="{height - 3}" text-anchor="middle" font-size="11" font-weight="700" fill="#33486b">{h(xlabel)}</text>')
    return (f'<svg class="chart-svg" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMidYMid meet" role="img">{"".join(parts)}</svg>')


def svg_dumbbell(rows: List[Tuple[str, float, float]], width: int = 520,
                 unit: str = " g", row_h: int = 32) -> str:
    """Each row: (label, actual, billable). Line links the two dots to show the gap."""
    rows = [(l, a, b) for l, a, b in rows if a is not None and b is not None]
    if not rows:
        return "<p class='small'>No data.</p>"
    maxv = max(max(a, b) for _, a, b in rows) or 1
    pad_l, pad_r, pad_t, pad_b = 150, 58, 10, 10
    plot_w = width - pad_l - pad_r
    height = pad_t + pad_b + len(rows) * row_h

    def X(v):
        return pad_l + (v / maxv) * plot_w

    parts = []
    for i, (lab, a, b) in enumerate(rows):
        y = pad_t + i * row_h + row_h / 2
        xa, xb = X(a), X(b)
        heavy = b > a
        line_col = "#db2777" if heavy else "#059669"
        parts.append(f'<text x="{pad_l - 12}" y="{y + 3.5:.1f}" text-anchor="end" font-size="11.5" fill="#33486b">{h(str(lab)[:20])}</text>')
        parts.append(f'<line x1="{min(xa, xb):.1f}" y1="{y:.1f}" x2="{max(xa, xb):.1f}" y2="{y:.1f}" stroke="{line_col}" stroke-width="3.4" stroke-linecap="round" opacity="0.45"/>')
        parts.append(f'<circle cx="{xa:.1f}" cy="{y:.1f}" r="5.2" fill="#2563eb"/>')
        parts.append(f'<circle cx="{xb:.1f}" cy="{y:.1f}" r="5.2" fill="#f59e0b"/>')
        endx = max(xa, xb) + 8
        parts.append(f'<text x="{endx:.1f}" y="{y + 3.5:.1f}" font-size="10.5" font-weight="700" fill="{line_col}">{"×" + format(b / a, ".1f") if a else ""}</text>')
    return (f'<svg class="chart-svg" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMinYMin meet" role="img">{"".join(parts)}</svg>')


def dashboard_body(dataset: Optional[WorkbookData]) -> str:
    if not dataset:
        return ("<div class='card'><h2><span class='ic'>▤</span>Data insights</h2>"
                "<div class='notice'>No dataset is loaded yet. Add your workbook to see distributions and coverage.</div></div>")
    import statistics
    import collections

    valid = [r for r in dataset.records if has_valid_dimensions(r)]
    total = len(valid)

    cat_counts: Dict[str, int] = {}
    cat_recs: Dict[str, List[Dict[str, Any]]] = {}
    for r in valid:
        c = clean_text(r.get("category")) or "unknown"
        cat_counts[c] = cat_counts.get(c, 0) + 1
        cat_recs.setdefault(c, []).append(r)
    cat_sorted = sorted(cat_counts.items(), key=lambda kv: -kv[1])

    man_counts: Dict[str, int] = {}
    for r in dataset.records:
        m = clean_text(r.get("manufacturer"))
        if m:
            man_counts[m] = man_counts.get(m, 0) + 1
    man_sorted = sorted(man_counts.items(), key=lambda kv: -kv[1])[:12]

    year_counts: Dict[str, int] = {}
    for r in dataset.records:
        s = str(r.get("date") or "")
        m = re.search(r"(20\d{2})", s)
        if m:
            year_counts[m.group(1)] = year_counts.get(m.group(1), 0) + 1

    # ---- rich KPI band --------------------------------------------------
    all_vol = [r.get("volume_cm3") for r in valid if r.get("volume_cm3")]
    all_wt = [r.get("weight_g") for r in valid if r.get("weight_g")]
    median_vol = statistics.median(all_vol) if all_vol else 0
    tot_vol_cm3 = sum(all_vol)
    tot_wt_g = sum(all_wt)
    vmin = min(all_vol) if all_vol else 0
    vmax = max(all_vol) if all_vol else 0
    size_ratio = (vmax / vmin) if vmin else 0
    biggest = max(valid, key=lambda r: r.get("volume_cm3") or 0) if valid else {}

    dens_all = [r["weight_g"] / r["volume_cm3"] for r in valid
                if r.get("weight_g") and r.get("volume_cm3") and r["volume_cm3"] > 0]
    med_dens = statistics.median(dens_all) if dens_all else 0
    water_ratio = (1.0 / med_dens) if med_dens else 0

    # dimensional (volumetric) weight: courier divisor 5000 → kg from cm dimensions
    dim_heavy = dim_both = 0
    for r in valid:
        if not r.get("weight_g"):
            continue
        vol_wt_g = (r["length_mm"] / 10 * r["width_mm"] / 10 * r["height_mm"] / 10) / 5000 * 1000
        dim_both += 1
        if vol_wt_g > r["weight_g"]:
            dim_heavy += 1
    dim_pct = (100 * dim_heavy / dim_both) if dim_both else 0

    # shape mix (longest ÷ shortest side)
    flat = long_ = cube = 0
    for r in valid:
        d = sorted([r.get("length_mm") or 0, r.get("width_mm") or 0, r.get("height_mm") or 0], reverse=True)
        if d[2] and d[0] / d[2] >= 3:
            long_ += 1
        elif d[2] and d[0] / d[2] <= 1.5:
            cube += 1
        else:
            flat += 1
    flat_pct = (100 * flat / total) if total else 0

    # supplier concentration + coverage over the measured set
    man_valid = collections.Counter(clean_text(r.get("manufacturer")) for r in valid if clean_text(r.get("manufacturer")))
    top_maker_share = (100 * man_valid.most_common(1)[0][1] / total) if man_valid and total else 0
    rich_packs = sum(n for _, n in cat_sorted if n >= 40)
    rich_pct = (100 * rich_packs / total) if total else 0

    # densest vs lightest category (need a few samples each)
    cat_dens = {}
    for c, rs in cat_recs.items():
        dv = [x["weight_g"] / x["volume_cm3"] for x in rs
              if x.get("weight_g") and x.get("volume_cm3") and x["volume_cm3"] > 0]
        if len(dv) >= 5:
            cat_dens[c] = statistics.median(dv)
    densest = max(cat_dens, key=cat_dens.get) if cat_dens else "—"
    lightest = min(cat_dens, key=cat_dens.get) if cat_dens else "—"

    med_l = statistics.median([r["length_mm"] for r in valid if r.get("length_mm")]) if valid else 0
    med_w = statistics.median([r["width_mm"] for r in valid if r.get("width_mm")]) if valid else 0
    med_h = statistics.median([r["height_mm"] for r in valid if r.get("height_mm")]) if valid else 0
    n_products = len(set(clean_text(r.get("name")) for r in valid if clean_text(r.get("name"))))
    cartons = tot_vol_cm3 / 1e6 / 0.03  # 0.03 m³ ≈ a standard medium shipping carton

    def kpi(value, label, caption, icon="", text=False):
        cls = "kpi text" if text else "kpi"
        ic = f'<div class="kpi-ic">{icon}</div>' if icon else ""
        return (f'<div class="{cls}">{ic}<div class="kpi-val">{value}</div>'
                f'<div class="kpi-lab">{h(label)}</div><div class="kpi-cap">{h(caption)}</div></div>')

    kpi_tiles = "".join([
        kpi(fmt_int(total), "Measured packs", "Unique pack profiles in the model", "▦"),
        kpi(f"{fmt_number(tot_vol_cm3 / 1000, '', 0)}<small> L</small>", "Catalogue footprint",
            f"Every pack stacked ≈ {fmt_int(cartons)} shipping cartons", "▧"),
        kpi(f"{fmt_number(tot_wt_g / 1000, '', 0)}<small> kg</small>", "Combined weight",
            "Total mass of all measured packs", "⚖"),
        kpi(f"{fmt_int(size_ratio)}<small>×</small>", "Size spread",
            "Largest pack vs smallest, by volume", "↔"),
        kpi(f"{fmt_number(dim_pct, '', 0)}<small>%</small>", "Dimensional-weight heavy",
            "Cost more to ship by size than by weight", "✈"),
        kpi(f"{fmt_number(med_dens, '', 2)}<small> g/cm³</small>", "Typical density",
            f"≈ {fmt_number(water_ratio, '', 1)}× lighter than water — mostly packaging & air", "◍"),
        kpi(f"{fmt_number(flat_pct, '', 0)}<small>%</small>", "Flat & slab-shaped",
            f"{fmt_int(long_)} long, {fmt_int(cube)} near-cubic packs too", "▭"),
        kpi(fmt_int(len(man_valid)), "Supplier network",
            f"Biggest maker is only {fmt_number(top_maker_share, '', 0)}% of the catalogue", "⌂"),
        kpi(f"{fmt_number(rich_pct, '', 0)}<small>%</small>", "Model-ready coverage",
            "Share of packs in data-rich categories", "✦"),
        kpi(f"{fmt_number((biggest.get('volume_cm3') or 0) / 1000, '', 1)}<small> L</small>", "Biggest single pack",
            f"{_short(clean_text(biggest.get('name')), 30)} — {fmt_int(size_ratio)}× the smallest", "◰"),
        kpi(h(densest), "Densest category",
            f"{fmt_number(cat_dens.get(densest, 0), '', 2)} g/cm³ · lightest is {lightest} ({fmt_number(cat_dens.get(lightest, 0), '', 2)})", "◆", text=True),
        kpi(f"{fmt_number(med_l, '', 0)}×{fmt_number(med_w, '', 0)}×{fmt_number(med_h, '', 0)}<small> mm</small>",
            "Typical pack", f"The median pack across {fmt_int(n_products)} products", "▣"),
    ])
    kpis = f'<div class="kpi-grid">{kpi_tiles}</div>'

    # colour map for the top categories, everything else grey
    top_cats = [c for c, _ in cat_sorted[:8]]
    color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(top_cats)}
    legend = "".join(
        f'<span><i style="background:{color_map[c]}"></i>{h(c)}</span>' for c in top_cats
    ) + '<span><i style="background:rgba(84,101,127,.5)"></i>other</span>'
    scatter_points = [(r.get("volume_cm3"), r.get("weight_g"), clean_text(r.get("category")) or "unknown")
                      for r in valid if r.get("volume_cm3") and r.get("weight_g")]

    cat_bar = svg_hbar([(c, n) for c, n in cat_sorted[:15]], "gcat", " packs")
    man_bar = svg_hbar([(m, n) for m, n in man_sorted], "gman", "")
    scatter = svg_scatter(scatter_points, color_map)
    year_bar = svg_hbar([(y, n) for y, n in sorted(year_counts.items())], "gyr", "") if len(year_counts) >= 1 else "<p class='small'>No dated rows found.</p>"

    # NEW chart 1 — density distribution ("how much of each pack is air?")
    hist = svg_histogram(
        dens_all, "ghist", vmax=1.5, nbins=15,
        xlabel="Pack density (g/cm³)  —  1.0 = water",
        refs=[(1.0, "water", "#be123c"), (med_dens, "median", "#7c3aed")],
    )
    below_water = (100 * sum(1 for d in dens_all if d < 1.0) / len(dens_all)) if dens_all else 0

    # NEW chart 2 — "paying for air": median actual vs billable (volumetric) weight per category
    dumbbell_rows = []
    for c, n in cat_sorted:
        rs = [x for x in cat_recs[c] if x.get("weight_g")]
        if len(rs) < 8:
            continue
        act = statistics.median([x["weight_g"] for x in rs])
        bil = statistics.median([(x["length_mm"] / 10 * x["width_mm"] / 10 * x["height_mm"] / 10) / 5000 * 1000 for x in rs])
        dumbbell_rows.append((c, act, bil))
        if len(dumbbell_rows) >= 9:
            break
    dumbbell = svg_dumbbell(dumbbell_rows)

    # per-category summary table — expandable into strength / dosage breakdown
    def _med(rs, key):
        vals = [x.get(key) for x in rs if x.get(key)]
        return statistics.median(vals) if vals else None

    def _dens(rs):
        dv = [x["weight_g"] / x["volume_cm3"] for x in rs
              if x.get("weight_g") and x.get("volume_cm3") and x["volume_cm3"] > 0]
        return statistics.median(dv) if dv else None

    def _packsize(rs):
        vals = [clean_text(x.get("pack_text")) for x in rs if clean_text(x.get("pack_text"))]
        if not vals:
            return "\u2014"
        common, _ = collections.Counter(vals).most_common(1)[0]
        distinct = len(set(vals))
        label = h(_short(common, 30))
        if distinct > 1:
            label += f' <span class="small">+{distinct - 1} more</span>'
        return label

    def _cells(rs):
        return (f"<td>{uval(_med(rs, 'length_mm'), 'mm', 0)} × {uval(_med(rs, 'width_mm'), 'mm', 0)} × "
                f"{uval(_med(rs, 'height_mm'), 'mm', 0)}</td>"
                f"<td>{uval(_med(rs, 'weight_g'), 'g', 1)}</td>"
                f"<td>{_packsize(rs)}</td>")

    rows = []
    for ci, (c, n) in enumerate(cat_sorted):
        rs = cat_recs[c]
        cov = coverage_info(n)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for x in rs:
            groups.setdefault(strength_bucket(x), []).append(x)
        g_sorted = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        rows.append(
            f'<tr class="cat-row" data-cat="{ci}">'
            f'<td><span class="caret">▸</span> <b>{h(c)}</b> <span class="small">· {len(groups)} strengths</span></td>'
            f'<td>{fmt_int(n)}</td>'
            f'{_cells(rs)}'
            f'<td class="cov-cell"><span class="cov-badge {cov["cls"]}">{h(cov["label"])}</span></td>'
            '</tr>'
        )
        for si, (gk, gr) in enumerate(g_sorted[:8]):
            gcov = coverage_info(len(gr))
            sid = f"{ci}-{si}"
            item_groups: Dict[str, List[Dict[str, Any]]] = {}
            for x in gr:
                iname = clean_text(x.get("name")) or "Unnamed item"
                item_groups.setdefault(iname, []).append(x)
            item_sorted = sorted(item_groups.items(), key=lambda kv: -len(kv[1]))
            rows.append(
                f'<tr class="sub-row str-row" data-parent="{ci}" data-strength="{sid}" hidden>'
                f'<td class="str-label"><span class="caret">▸</span>{h(gk)} <span class="small">· {len(item_sorted)} items</span></td>'
                f'<td>{fmt_int(len(gr))}</td>'
                f'{_cells(gr)}'
                f'<td class="cov-cell"><span class="cov-badge {gcov["cls"]}">{h(gcov["label"])}</span></td>'
                '</tr>'
            )
            for ik, ir in item_sorted[:8]:
                rows.append(
                    f'<tr class="subsub-row" data-sparent="{sid}" hidden>'
                    f'<td class="man-label">{h(_short(ik, 36))}</td>'
                    f'<td>{fmt_int(len(ir))}</td>'
                    f'{_cells(ir)}'
                    f'<td class="cov-cell"></td>'
                    '</tr>'
                )
            if len(item_sorted) > 8:
                extra = item_sorted[8:]
                extra_recs = [x for _, ir in extra for x in ir]
                rows.append(
                    f'<tr class="subsub-row" data-sparent="{sid}" hidden>'
                    f'<td class="man-label"><span class="small">{len(extra)} other items</span></td>'
                    f'<td>{fmt_int(len(extra_recs))}</td>'
                    f'{_cells(extra_recs)}'
                    f'<td class="cov-cell"></td>'
                    '</tr>'
                )
        rest = g_sorted[8:]
        if rest:
            rest_recs = [x for _, gr in rest for x in gr]
            rows.append(
                f'<tr class="sub-row" data-parent="{ci}" hidden>'
                f'<td class="sub-label"><span class="small">{len(rest)} other strengths</span></td>'
                f'<td>{fmt_int(len(rest_recs))}</td>'
                f'{_cells(rest_recs)}'
                f'<td class="cov-cell"></td>'
                '</tr>'
            )
    summary_table = f"""
    <div class="drill-hint">▸ <b>Click a category</b> to open it by strength / dosage, then <b>click a strength</b> to see the items in it.</div>
    <div class="table-wrap"><table class="drill">
      <thead><tr><th>Category / strength</th><th>Packs</th><th>Median L × W × H</th><th>Median weight</th><th>Pack size</th><th>Model coverage</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
    """

    return f"""
    {kpis}
    {unit_toggle_html()}
    <div class="dash-grid">
      <div class="card">
        <h2><span class="ic">◍</span>How much of each pack is air?</h2>
        <p class="small">Distribution of pack density. <b>{fmt_number(below_water, '', 0)}%</b> of packs sit left of the water line — they'd
        float, because most of the box is packaging and empty space, not product.</p>
        {hist}
        <div class="legend">
          <span><i style="background:#2563eb"></i>packs at this density</span>
          <span><i style="background:#be123c"></i>water (1.0 g/cm³)</span>
          <span><i style="background:#7c3aed"></i>catalogue median</span>
        </div>
      </div>
      <div class="card">
        <h2><span class="ic">✈</span>Paying for air — actual vs billable weight</h2>
        <p class="small">For each category: the median <b>real</b> weight (blue) vs the <b>volumetric</b> weight couriers bill
        (orange, L×W×H ÷ 5000). The wider the gap, the more you pay to ship empty space. ×N marks how many times heavier the billable weight is.</p>
        {dumbbell}
        <div class="legend">
          <span><i style="background:#2563eb"></i>actual weight</span>
          <span><i style="background:#f59e0b"></i>billable (volumetric) weight</span>
        </div>
      </div>
    </div>
    <div class="dash-grid">
      <div class="card">
        <h2><span class="ic">▤</span>Packs by category</h2>
        <p class="small">How many measured pack profiles back each category. Short bars are where the model has the least to learn from.</p>
        {cat_bar}
      </div>
      <div class="card">
        <h2><span class="ic">◎</span>Volume vs weight</h2>
        <p class="small">Every measured pack, log–log. Points drifting below the cloud are unusually light for their size; points above are dense.</p>
        {scatter}
        <div class="legend">{legend}</div>
      </div>
      <div class="card">
        <h2><span class="ic">⌂</span>Top manufacturers</h2>
        <p class="small">Suppliers contributing the most rows to your inward history.</p>
        {man_bar}
      </div>
      <div class="card">
        <h2><span class="ic">◷</span>Records by year</h2>
        <p class="small">Volume of measured history captured per source year.</p>
        {year_bar}
      </div>
    </div>
    <div class="card">
      <h2><span class="ic">☷</span>Category benchmarks &amp; coverage</h2>
      <p class="small">Typical (median) pack for each category, plus how much real data supports predictions there. Categories marked
      <b>Limited</b> or <b>Very little data</b> are the ones to prioritise measuring next — the estimator caps its confidence for them.</p>
      {summary_table}
    </div>
    """


def dashboard_page() -> bytes:
    try:
        dataset = get_dataset()
        if dataset is not None:
            ensure_model(dataset)
        body = dashboard_body(dataset)
    except Exception as exc:  # noqa: BLE001
        body = (f"<div class='card'><h2><span class='ic'>!</span>Insights unavailable</h2>"
                f"<div class='notice'>{h(exc)}</div><pre class='muted-box'>{h(traceback.format_exc())}</pre></div>")
    return page_shell("Data insights · Pharma Pack Intelligence", body, active="dashboard")


# ---- Compare two items ----

def compare_form(dataset: Optional[WorkbookData]) -> str:
    cats = dataset_categories(dataset)
    if cats:
        opts = "".join(f'<option value="{h(c)}">{h(c)}</option>' for c in cats)
        cat_a = ('<label>Category</label><select name="a_category"><option value="" disabled selected>'
                 f'Choose a category…</option>{opts}</select>')
        cat_b = ('<label>Category</label><select name="b_category"><option value="" disabled selected>'
                 f'Choose a category…</option>{opts}</select>')
    else:
        cat_a = '<label>Category / form</label><input type="text" name="a_stated_form" placeholder="e.g. tablet, vial">'
        cat_b = '<label>Category / form</label><input type="text" name="b_stated_form" placeholder="e.g. tablet, vial">'
    return f"""
    <div class="card" id="compare">
      <h2><span class="ic">⇋</span>Compare two items</h2>
      <p class="small">Predict two packs at once and see the size, volume, weight and density differences side by side.</p>
      <form method="post" action="/compare">
        <div class="cmp-grid">
          <div class="cmp-col">
            <h3>Item A</h3>
            <label>Drug or brand name</label>
            <input type="text" name="a_name" placeholder="Example: Paracetamol" required>
            <label>Strength / dosage</label>
            <input type="text" name="a_strength" placeholder="Example: 500 mg">
            {cat_a}
            <label>Pack size</label>
            <div class="ac-wrap">
              <textarea name="a_pack" class="ac-pack" autocomplete="off" placeholder="Example: 10 strips of tablets" required></textarea>
              <div class="ac-list" hidden></div>
            </div>
          </div>
          <div class="cmp-col">
            <h3>Item B</h3>
            <label>Drug or brand name</label>
            <input type="text" name="b_name" placeholder="Example: Enhertu" required>
            <label>Strength / dosage</label>
            <input type="text" name="b_strength" placeholder="Example: 100 mg">
            {cat_b}
            <label>Pack size</label>
            <div class="ac-wrap">
              <textarea name="b_pack" class="ac-pack" autocomplete="off" placeholder="Example: 1 vial per pack" required></textarea>
              <div class="ac-list" hidden></div>
            </div>
          </div>
        </div>
        <button type="submit">Compare items</button>
      </form>
    </div>
    """


def _cmp_row(label: str, key: str, kind: Optional[str], ra: Dict[str, Any], rb: Dict[str, Any],
             decimals: int = 1, density: bool = False) -> str:
    if density:
        a, b = _density(ra["predicted"]), _density(rb["predicted"])
        acell = f"{fmt_number(a, '', 2)} g/cm\u00b3" if a else "\u2014"
        bcell = f"{fmt_number(b, '', 2)} g/cm\u00b3" if b else "\u2014"
    else:
        a, b = ra["predicted"].get(key), rb["predicted"].get(key)
        acell, bcell = uval(a, kind, decimals), uval(b, kind, decimals)
    if a and b and a != 0:
        pct = (b - a) / a * 100.0
        cls = "flat" if abs(pct) < 1 else ("up" if pct > 0 else "down")
        sign = "+" if pct > 0 else ""
        dcell = f'<span class="delta {cls}">{sign}{pct:.0f}%</span>'
    else:
        dcell = '<span class="delta flat">\u2014</span>'
    return f"<tr><td><b>{h(label)}</b></td><td>{acell}</td><td>{bcell}</td><td class='delta'>{dcell}</td></tr>"


def render_comparison(ra: Dict[str, Any], rb: Dict[str, Any]) -> str:
    if ra.get("error") or rb.get("error"):
        err = ra.get("error") or rb.get("error")
        return (f"<div class='card'><h2><span class='ic'>!</span>Comparison</h2>"
                f"<div class='notice'>{h(err)}</div></div>")
    qa, qb = ra["query"], rb["query"]
    rows = "".join([
        _cmp_row("Length", "length_mm", "mm", ra, rb),
        _cmp_row("Width", "width_mm", "mm", ra, rb),
        _cmp_row("Height", "height_mm", "mm", ra, rb),
        _cmp_row("Volume", "volume_cm3", "cm3", ra, rb, 1),
        _cmp_row("Weight", "weight_g", "g", ra, rb),
        _cmp_row("Density", "", None, ra, rb, density=True),
    ])
    iso_a = isometric_box_svg(ra["predicted"].get("length_mm"), ra["predicted"].get("width_mm"), ra["predicted"].get("height_mm"))
    iso_b = isometric_box_svg(rb["predicted"].get("length_mm"), rb["predicted"].get("width_mm"), rb["predicted"].get("height_mm"))
    name_a = h(qa.get("name") or "Item A")
    name_b = h(qb.get("name") or "Item B")
    return f"""
    <div class="card">
      <h2><span class="ic">⇋</span>Comparison result</h2>
      {unit_toggle_html()}
      <div class="table-wrap"><table class="cmp-table">
        <thead><tr><th>Dimension</th><th>{name_a}</th><th>{name_b}</th><th>&Delta; B vs A</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
      <div class="cmp-visuals">
        <div><div class="view2d-title" style="margin-bottom:6px;">{name_a}</div><div class="pack-preview">{iso_a}</div></div>
        <div><div class="view2d-title" style="margin-bottom:6px;">{name_b}</div><div class="pack-preview">{iso_b}</div></div>
      </div>
      <div class="legend">
        <span>Confidence — {name_a}: <b>{ra['confidence_score']}%</b> ({h(ra['confidence'])})</span>
        <span>Confidence — {name_b}: <b>{rb['confidence_score']}%</b> ({h(rb['confidence'])})</span>
      </div>
    </div>
    """


class AppHandler(BaseHTTPRequestHandler):
    def send_html(self, content: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/static/"):
            self.serve_static(self.path)
            return
        route = urllib.parse.urlparse(self.path).path
        if route == "/dashboard":
            self.send_html(dashboard_page())
            return
        if self.path.startswith("/"):
            self.send_html(home_page())

    def serve_static(self, path: str) -> None:
        # Serve bundled assets (e.g. the local three.js) so the app works offline.
        rel = urllib.parse.urlparse(path).path[len("/static/"):]
        safe = (STATIC_DIR / rel).resolve()
        if not str(safe).startswith(str(STATIC_DIR.resolve())) or not safe.is_file():
            self.send_response(404)
            self.end_headers()
            return
        ctype = "application/javascript" if safe.suffix == ".js" else "application/octet-stream"
        data = safe.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path.startswith("/upload"):
            self.handle_upload()
        elif self.path.startswith("/predict"):
            self.handle_predict()
        elif self.path.startswith("/compare"):
            self.handle_compare()
        else:
            self.send_html(page_shell("Not found", "<div class='card'><h2>Not found</h2></div>"), status=404)

    def _read_uploaded_file(self) -> Tuple[Optional[str], Optional[bytes]]:
        """Return (filename, data) for a multipart upload, with or without cgi."""
        if HAS_CGI:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
            )
            item = form["file"] if "file" in form else None
            if item is None or not getattr(item, "filename", ""):
                return None, None
            return Path(item.filename).name, item.file.read()

        # cgi was removed in Python 3.13 — parse multipart via the email module.
        import email
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        header = (
            f"Content-Type: {self.headers.get('Content-Type', '')}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8")
        msg = email.message_from_bytes(header + body)
        for part in msg.walk():
            if part.get_filename():
                return Path(part.get_filename()).name, part.get_payload(decode=True)
        return None, None

    def handle_upload(self) -> None:
        ensure_dirs()
        original, data = self._read_uploaded_file()
        if not original or data is None:
            self.send_html(page_shell("Upload failed", "<div class='card'><div class='notice'>No file was selected.</div></div>"), 400)
            return
        suffix = Path(original).suffix.lower()
        if suffix not in SUPPORTED_FILES:
            self.send_html(
                page_shell("Upload failed", f"<div class='card'><div class='notice'>Unsupported file type: {h(suffix)}</div></div>"),
                400,
            )
            return
        target = UPLOAD_DIR / original
        counter = 1
        while target.exists():
            target = UPLOAD_DIR / f"{Path(original).stem}_{counter}{suffix}"
            counter += 1
        with target.open("wb") as f:
            f.write(data)
        save_last_upload(target)
        self.redirect("/")

    def handle_predict(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        params = urllib.parse.parse_qs(raw)
        form = {
            key: params.get(key, [""])[0]
            for key in ["name", "brand", "manufacturer", "category", "stated_form", "strength", "pack", "bulk"]
        }
        try:
            dataset = get_dataset()
            if not dataset:
                prediction = "<div class='card'><h2>Prediction</h2><div class='notice'>Please upload your Excel file first.</div></div>"
            else:
                ensure_model(dataset)
                query = build_query_product(form)
                result = predict_dimensions(dataset, query)
                prediction = render_prediction(result)
            self.send_html(home_page(prediction))
        except Exception as exc:
            body = f"<div class='card'><h2>Prediction failed</h2><div class='notice'>{h(exc)}</div><pre class='muted-box'>{h(traceback.format_exc())}</pre></div>"
            self.send_html(home_page(body), 500)

    def handle_compare(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        params = urllib.parse.parse_qs(raw)

        def side(prefix: str) -> Dict[str, str]:
            keys = ["name", "brand", "manufacturer", "category", "stated_form", "strength", "pack", "bulk"]
            return {k: params.get(prefix + k, [""])[0] for k in keys}

        try:
            dataset = get_dataset()
            if not dataset:
                body = "<div class='card'><h2>Compare</h2><div class='notice'>Please load your data first.</div></div>"
            else:
                ensure_model(dataset)
                ra = predict_dimensions(dataset, build_query_product(side("a_")))
                rb = predict_dimensions(dataset, build_query_product(side("b_")))
                body = render_comparison(ra, rb)
            self.send_html(home_page(body))
        except Exception as exc:  # noqa: BLE001
            body = f"<div class='card'><h2>Compare failed</h2><div class='notice'>{h(exc)}</div><pre class='muted-box'>{h(traceback.format_exc())}</pre></div>"
            self.send_html(home_page(body), 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep the terminal readable.
        return


def main() -> None:
    ensure_dirs()
    port = int(os.environ.get("PORT", "8000"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Pharma Pack Dimension Predictor running at http://{host}:{port}")
    print(f"Place your Excel file in: {DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")


if __name__ == "__main__":
    main()
