"""Unit tests for source archive handling and schema validation."""

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from retainiq.data.profile import EXPECTED_COLUMNS, locate_workbook, profile_dataset


class ProfileTests(unittest.TestCase):
    def _write_workbook(self, path: Path, columns: list[str] | None = None) -> None:
        names = columns or EXPECTED_COLUMNS
        row = {
            "InvoiceNo": "536365",
            "StockCode": "85123A",
            "Description": "Sample item",
            "Quantity": 2,
            "InvoiceDate": "2010-12-01 08:26:00",
            "UnitPrice": 2.55,
            "CustomerID": 17850,
            "Country": "United Kingdom",
        }
        pd.DataFrame([{name: row.get(name) for name in names}]).to_excel(path, index=False)

    def test_zip_extracts_single_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            workbook = folder / "Online Retail.xlsx"
            self._write_workbook(workbook)
            archive = folder / "source.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.write(workbook, arcname="Online Retail.xlsx")
            workbook.unlink()
            extracted = locate_workbook(archive)
            self.assertTrue(extracted.exists())
            self.assertEqual(extracted.name, "Online Retail.xlsx")

    def test_profile_reports_source_quality_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "small.xlsx"
            self._write_workbook(workbook)
            result = profile_dataset(workbook)
            self.assertEqual(result["shape"]["rows"], 1)
            self.assertEqual(result["quality_checks"]["cancellation_invoice_rows"], 0)
            self.assertEqual(result["quality_checks"]["missing_customer_id_rows"], 0)
            self.assertEqual(len(result["source"]["sha256"]), 64)

    def test_unexpected_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "bad.xlsx"
            self._write_workbook(workbook, columns=EXPECTED_COLUMNS[:-1])
            with self.assertRaisesRegex(ValueError, "Unexpected source schema"):
                profile_dataset(workbook)


if __name__ == "__main__":
    unittest.main()
