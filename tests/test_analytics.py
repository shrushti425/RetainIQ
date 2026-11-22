"""Small fixtures for analytical and data-cleaning invariants."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from retainiq.analytics.eda import build_eda
from retainiq.market_basket import apriori_rules
from retainiq.prediction import make_snapshot
from retainiq.preprocessing import prepare_tables
from retainiq.retention import cohort_analysis
from retainiq.segmentation import segment_customers


class AnalyticsTests(unittest.TestCase):
    def test_cleaning_separates_returns_cancellations_duplicates(self):
        columns = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
        rows = [
            ["100", "A", "item a", 2, "2011-01-01 00:00", 3.0, 10, "UK"],
            ["100", "A", "item a", 2, "2011-01-01 00:00", 3.0, 10, "UK"],
            ["100", "B", "item b", 1, "2011-01-01 12:00", 1.0, 10, "UK"],
            ["C101", "A", "item a", -2, "2011-01-02 00:00", 3.0, 10, "UK"],
            ["102", "A", "item a", -1, "2011-01-03 00:00", 3.0, 10, "UK"],
            ["103", "B", "free", 1, "2011-01-04 00:00", 0.0, None, "UK"],
            ["104", "B", "anon", 1, "2011-01-05 00:00", 4.0, None, "UK"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xlsx"
            pd.DataFrame(rows, columns=columns).to_excel(source, index=False)
            result = prepare_tables(source, root / "out")
            sales = pd.read_csv(root / "out/sales_lines.csv")
            audit = pd.read_csv(root / "out/cleaning_audit.csv")
            self.assertEqual(result["source_rows"], 7)
            self.assertEqual(result["sales_line_rows"], 3)
            self.assertEqual(result["anonymous_sale_rows_retained_for_aggregate_analysis"], 1)
            orders = pd.read_csv(root / "out/orders.csv")
            self.assertEqual(orders["InvoiceNo"].nunique(), len(orders))
            self.assertEqual(len(orders), 2)
            self.assertEqual(set(audit["quality_status"]), {"exact_duplicate", "cancellation_invoice", "return_negative_quantity", "zero_unit_price"})
            self.assertAlmostEqual(sales["LineRevenue"].sum(), 11.0)

    def test_temporal_snapshot_uses_only_future_outcome_window(self):
        orders = pd.DataFrame([
            {"CustomerID": 123.0, "InvoiceNo": "1", "InvoiceDate": "2011-05-01", "revenue": 20, "units": 2},
            {"CustomerID": 123.0, "InvoiceNo": "2", "InvoiceDate": "2011-07-01", "revenue": 10, "units": 1},
            {"CustomerID": 456.0, "InvoiceNo": "3", "InvoiceDate": "2011-05-01", "revenue": 5, "units": 1},
        ])
        lines = pd.DataFrame([
            {"CustomerID": 123.0, "InvoiceDate": "2011-05-01", "StockCode": "A"},
            {"CustomerID": 456.0, "InvoiceDate": "2011-05-01", "StockCode": "B"},
        ])
        snap = make_snapshot(orders, lines, "2011-05-31")
        labels = snap.set_index("CustomerID")["label"].to_dict()
        self.assertEqual(labels, {"123": 1, "456": 0})
        active_days = snap.set_index("CustomerID")["active_days_180d"].to_dict()
        self.assertEqual(active_days, {"123": 1, "456": 1})

    def test_cohort_retention_and_eda_metrics(self):
        orders = pd.DataFrame([
            {"CustomerID": 1, "InvoiceNo": "a", "InvoiceDate": "2011-01-10", "OrderMonth": "2011-01", "revenue": 100, "units": 1, "line_count": 1, "Country": "UK"},
            {"CustomerID": 1, "InvoiceNo": "b", "InvoiceDate": "2011-02-10", "OrderMonth": "2011-02", "revenue": 50, "units": 1, "line_count": 1, "Country": "UK"},
            {"CustomerID": 2, "InvoiceNo": "c", "InvoiceDate": "2011-01-20", "OrderMonth": "2011-01", "revenue": 25, "units": 1, "line_count": 1, "Country": "UK"},
            {"CustomerID": 3, "InvoiceNo": "d", "InvoiceDate": "2011-02-10", "OrderMonth": "2011-02", "revenue": 50, "units": 1, "line_count": 1, "Country": "UK"},
        ])
        lines = pd.DataFrame([
            {"InvoiceNo": "a", "InvoiceDate": "2011-01-10", "CustomerID": 1, "Country": "UK", "LineRevenue": 100, "Quantity": 1, "StockCode": "A", "Description": "a"},
            {"InvoiceNo": "b", "InvoiceDate": "2011-02-10", "CustomerID": 1, "Country": "UK", "LineRevenue": 50, "Quantity": 1, "StockCode": "B", "Description": "b"},
            {"InvoiceNo": "c", "InvoiceDate": "2011-01-20", "CustomerID": 2, "Country": "UK", "LineRevenue": 25, "Quantity": 1, "StockCode": "A", "Description": "a"},
            {"InvoiceNo": "d", "InvoiceDate": "2011-02-10", "CustomerID": 3, "Country": "UK", "LineRevenue": 50, "Quantity": 1, "StockCode": "B", "Description": "b"},
        ])
        retention, revenue, lifecycle = cohort_analysis(orders)
        self.assertEqual(retention.loc["2011-01", 1], 0.5)
        self.assertTrue(pd.isna(retention.loc["2011-02", 1]))
        self.assertEqual(float(revenue["revenue"].sum()), 225.0)
        self.assertEqual(int(lifecycle.loc[0, "repeat_customers"]), 1)
        eda = build_eda(lines, orders, pd.DataFrame())
        self.assertEqual(eda["metrics"]["total_orders"], 4)
        self.assertAlmostEqual(eda["metrics"]["total_revenue_gbp"], 225.0)

    def test_apriori_emits_supported_associations(self):
        rows = []
        for invoice in range(10):
            rows.append({"InvoiceNo": str(invoice), "StockCode": "C", "Description": "gamma"})
            if invoice < 8:
                rows.append({"InvoiceNo": str(invoice), "StockCode": "A", "Description": "alpha"})
            if invoice < 6:
                rows.append({"InvoiceNo": str(invoice), "StockCode": "B", "Description": "beta"})
        frequent, rules = apriori_rules(pd.DataFrame(rows), top_products=3, min_support=0.2, min_confidence=0.2)
        self.assertGreaterEqual(len(frequent), 3)
        self.assertGreater(len(rules), 0)
        self.assertTrue((rules["lift"] > 1).all())

    def test_segmentation_selects_cluster_count_and_profiles_customers(self):
        rows = []
        for customer in range(12):
            for order in range(1 + customer % 5):
                rows.append({"CustomerID": str(customer), "InvoiceNo": f"{customer}-{order}", "InvoiceDate": pd.Timestamp("2011-11-01") - pd.Timedelta(customer * 7 + order, unit="D"), "revenue": (customer + 1) * (order + 1) * 4})
        segmented, profiles, metadata = segment_customers(pd.DataFrame(rows))
        self.assertEqual(len(segmented), 12)
        self.assertGreaterEqual(metadata["selected_k"], 2)
        self.assertEqual(int(profiles["customers"].sum()), 12)
        self.assertTrue(segmented["segment"].notna().all())


if __name__ == "__main__":
    unittest.main()
