# Combined historical data study

## Source files

Seven workbooks were inspected independently before normalization:

| Source file | Normalized rows | Complete dimensions | Training candidates | Layout |
|---|---:|---:|---:|---|
| 2023 INWARD 3S CORPORATION Dimensions.xlsx | 1,786 | 1,624 | 1,609 | Combined L×W×H |
| 2025 3S Corporation Dimensions 1.xlsx | 1,905 | 1,709 | 1,687 | Separate L/W/H |
| 2025 3S Pharma Dimensions.xlsx | 1,786 | 1,609 | 1,533 | Separate L/W/H |
| 3S CORPORATION INWARD 2024.xlsx | 195 | 158 | 149 | Combined L×W×H |
| 3S Pharmaceuticals Dimensions 3.xlsx | 362 | 346 | 336 | Combined L×W×H |
| APRIL 2023 INWARD 3S PHARMACEUTICAL 1.xlsx | 2,225 | 0 | 0 | Companion copy without dimensions |
| APRIL 2023 INWARD 3S PHARMACEUTICAL.xlsx | 2,225 | 2,164 | 2,081 | Combined L×W×H |

## Result

- 10,484 substantive source rows preserved
- 7,395 usable measured attempts
- 4,524 unique measured pack profiles used by the app
- 2,871 exact repeated attempts retained with occurrence counts
- 215 rows placed in the review queue

## Interpretation rules

- Original workbooks are never modified.
- Every normalized row retains source file, sheet and original Excel row.
- Dimensions are converted to millimetres and ordered longest × middle × shortest.
- Volume is calculated in cubic centimetres.
- A different pack description remains separate, even for the same product and manufacturer.
- Only exact identity + manufacturer + pack + measurement repeats share one model profile; the original rows and occurrence count remain stored.
- Tablet/capsule patterns preserve strip count, units per strip and total units.
- Injectable strip packs such as `10 strips × 5 ampoules` become 50 ampoules.
- `10 × 2 mL ampoules` becomes 10 ampoules of 2 mL each.
- Bottle/drop text such as `25 × 10 mL` keeps 10 mL as the sellable unit fill and stores 25 separately as a bulk count.
- Eye drops and other named dosage forms take priority over legacy container labels such as `VIAL`.
- Powder/concentrate products supplied in a vial remain injection-vial packs even when the medicine is later prepared for infusion.
- The 2023 companion workbook without dimensions remains in the full audit database but does not train dimensions.

## Model

The app uses weighted nearest neighbours so each prediction can show its historical references. Matching considers:

- dosage form and container type;
- strength, concentration or fill;
- strip and unit configuration;
- total unit count and bulk quantity;
- brand/product terms; and
- agreement among nearby historical pack volumes.

Manufacturer is retained for audit and reference display, but it is not required and does not influence the prediction. Repeated identical measurements add capped supporting evidence. The confidence score is an evidence score, not a guaranteed accuracy percentage.
