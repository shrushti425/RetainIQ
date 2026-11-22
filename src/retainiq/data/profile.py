"""Reproducible source-data profiling for UCI Online Retail."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_workbook(path: Path) -> Path:
    """Return an xlsx input or extract the sole workbook from a UCI zip."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Expected .zip or .xlsx input, got: {path}")
    with ZipFile(path) as archive:
        workbooks = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
        if len(workbooks) != 1:
            raise ValueError(f"Expected one workbook in {path}; found {len(workbooks)}")
        extracted = path.parent / Path(workbooks[0]).name
        if not extracted.exists():
            archive.extract(workbooks[0], path.parent)
            original = path.parent / workbooks[0]
            if original != extracted:
                original.replace(extracted)
        return extracted


def profile_dataset(input_path: Path) -> dict:
    """Read the source and return a machine-readable, non-destructive profile."""
    workbook = locate_workbook(input_path)
    frame = pd.read_excel(workbook, engine="openpyxl")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected source schema. "
            f"Expected {EXPECTED_COLUMNS}; received {list(frame.columns)}"
        )

    quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    unit_price = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    invoice = frame["InvoiceNo"].astype("string").str.strip()
    cancellation = invoice.str.upper().str.startswith("C", na=False)
    profile = {
        "source": {
            "file": workbook.name,
            "sha256": sha256_file(workbook),
            "dataset_url": DATASET_URL,
            "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "shape": {"rows": int(len(frame)), "columns": int(len(frame.columns))},
        "columns": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
                "missing_count": int(frame[column].isna().sum()),
                "missing_fraction": float(frame[column].isna().mean()),
                "distinct_non_null": int(frame[column].nunique(dropna=True)),
            }
            for column in frame.columns
        ],
        "quality_checks": {
            "exact_duplicate_rows": int(frame.duplicated().sum()),
            "cancellation_invoice_rows": int(cancellation.sum()),
            "negative_quantity_rows": int((quantity < 0).sum()),
            "zero_quantity_rows": int((quantity == 0).sum()),
            "negative_unit_price_rows": int((unit_price < 0).sum()),
            "zero_unit_price_rows": int((unit_price == 0).sum()),
            "invalid_quantity_rows": int(quantity.isna().sum()),
            "invalid_unit_price_rows": int(unit_price.isna().sum()),
            "unparseable_invoice_date_rows": int(
                pd.to_datetime(frame["InvoiceDate"], errors="coerce").isna().sum()
            ),
            "missing_customer_id_rows": int(frame["CustomerID"].isna().sum()),
            "customer_id_distinct_non_null": int(frame["CustomerID"].nunique()),
        },
        "date_range": {
            "minimum": str(pd.to_datetime(frame["InvoiceDate"], errors="coerce").min()),
            "maximum": str(pd.to_datetime(frame["InvoiceDate"], errors="coerce").max()),
        },
        "interpretation_notes": [
            "A cancellation is flagged from an InvoiceNo beginning with C; it is not silently dropped.",
            "Negative quantity/price and zero-value rows are counted for review, not automatically removed.",
            "Rows without CustomerID remain in source profiling and may be excluded only from customer-level analyses.",
        ],
    }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UCI .zip or extracted .xlsx source")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/profiling/source_profile.json")
    )
    args = parser.parse_args()
    result = profile_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
