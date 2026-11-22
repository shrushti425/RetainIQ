"""Auditable preparation of invoice-line and customer-level analytical tables."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from retainiq.data.profile import EXPECTED_COLUMNS, locate_workbook

LOG = logging.getLogger("retainiq.preprocessing")


def _normalise_customer_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.map(lambda value: str(int(value)) if pd.notna(value) else pd.NA).astype("string")


def prepare_tables(source: Path, output_dir: Path) -> dict:
    """Create sales lines, invoice orders, customers, and a reason-coded audit."""
    workbook = locate_workbook(source)
    LOG.info("Reading source workbook %s", workbook)
    frame = pd.read_excel(workbook, engine="openpyxl")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected source schema: {list(frame.columns)}")

    frame.insert(0, "source_row_id", range(2, len(frame) + 2))  # Excel row (header is row 1)
    frame["InvoiceNo"] = frame["InvoiceNo"].astype("string").str.strip()
    frame["StockCode"] = frame["StockCode"].astype("string").str.strip()
    frame["Description"] = frame["Description"].astype("string").str.strip()
    frame["Country"] = frame["Country"].astype("string").str.strip()
    frame["CustomerID"] = _normalise_customer_id(frame["CustomerID"])
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    frame["LineRevenue"] = frame["Quantity"] * frame["UnitPrice"]

    reason = pd.Series("valid_sale", index=frame.index, dtype="string")
    # First-match assignment makes categories mutually exclusive and audit totals reconcilable.
    duplicate = frame[EXPECTED_COLUMNS].duplicated(keep="first")
    rules = [
        (duplicate, "exact_duplicate"),
        (frame["InvoiceNo"].str.upper().str.startswith("C", na=False), "cancellation_invoice"),
        (frame["Quantity"].lt(0), "return_negative_quantity"),
        (frame["InvoiceDate"].isna(), "invalid_invoice_date"),
        (frame["InvoiceNo"].isna() | frame["InvoiceNo"].eq(""), "missing_invoice_number"),
        (frame["StockCode"].isna() | frame["StockCode"].eq(""), "missing_stock_code"),
        (frame["Quantity"].isna() | frame["Quantity"].le(0), "invalid_quantity"),
        (frame["UnitPrice"].isna() | frame["UnitPrice"].lt(0), "invalid_unit_price"),
        (frame["UnitPrice"].eq(0), "zero_unit_price"),
    ]
    for mask, label in rules:
        reason.loc[reason.eq("valid_sale") & mask.fillna(False)] = label
    frame["quality_status"] = reason
    sales = frame.loc[reason.eq("valid_sale")].copy()
    # Currency values are GBP in the source; report nominal source-period values without inflation adjustment.
    sales["OrderMonth"] = sales["InvoiceDate"].dt.to_period("M").astype(str)
    audit = frame.loc[~reason.eq("valid_sale"), [
        "source_row_id", "InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice",
        "LineRevenue", "InvoiceDate", "CustomerID", "Country", "quality_status",
    ]].copy()
    audit["action"] = audit["quality_status"].map({
        "exact_duplicate": "exclude_duplicate_copy_keep_first",
        "cancellation_invoice": "segregate_cancellation",
        "return_negative_quantity": "segregate_return",
        "zero_unit_price": "exclude_from_revenue_analysis_review_free_item",
    }).fillna("exclude_invalid_record")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.loc[audit["quality_status"].eq("cancellation_invoice")].to_csv(output_dir / "cancellations.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    audit.loc[audit["quality_status"].eq("return_negative_quantity")].to_csv(output_dir / "returns.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    audit.loc[audit["quality_status"].isin(["invalid_invoice_date", "missing_invoice_number", "missing_stock_code", "invalid_quantity", "invalid_unit_price", "zero_unit_price"])].to_csv(output_dir / "invalid_transactions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    # A sale line can be anonymous; retain it for aggregate/product analysis.
    sales_cols = [
        "source_row_id", "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "LineRevenue", "CustomerID", "Country", "OrderMonth",
    ]
    sales = sales[sales_cols].sort_values(["InvoiceDate", "InvoiceNo", "source_row_id"])
    # Some source invoices have more than one InvoiceDate timestamp. Collapse by
    # invoice number so each completed transaction is counted once.
    orders = sales.groupby("InvoiceNo", as_index=False, dropna=False).agg(
        InvoiceDate=("InvoiceDate", "min"),
        CustomerID=("CustomerID", "first"),
        Country=("Country", "first"),
        revenue=("LineRevenue", "sum"),
        units=("Quantity", "sum"),
        line_count=("StockCode", "size"),
    )
    orders["OrderMonth"] = orders["InvoiceDate"].dt.to_period("M").astype(str)
    customer_lines = sales.loc[sales["CustomerID"].notna()].copy()
    customer_orders = orders.loc[orders["CustomerID"].notna()].copy()
    customers = (
        customer_orders.groupby("CustomerID", as_index=False)
        .agg(
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("revenue", "sum"),
            average_order_value=("revenue", "mean"),
        )
    )
    customer_products = customer_lines.groupby("CustomerID").agg(
        units=("Quantity", "sum"),
        distinct_products=("StockCode", "nunique"),
        country=("Country", lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"),
    ).reset_index()
    customers = customers.merge(customer_products, on="CustomerID", how="left")

    sales.to_csv(output_dir / "sales_lines.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    orders.to_csv(output_dir / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    customers.to_csv(output_dir / "customers.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    audit.to_csv(output_dir / "cleaning_audit.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    reason_counts = reason.value_counts().sort_index().astype(int).to_dict()
    summary = {
        "source_file": workbook.name,
        "source_rows": int(len(frame)),
        "sales_line_rows": int(len(sales)),
        "sale_rows_with_customer_id": int(sales["CustomerID"].notna().sum()),
        "anonymous_sale_rows_retained_for_aggregate_analysis": int(sales["CustomerID"].isna().sum()),
        "order_rows": int(len(orders)),
        "customers": int(customers["CustomerID"].nunique()),
        "sales_revenue_gbp": float(sales["LineRevenue"].sum()),
        "audit_reason_counts": reason_counts,
        "excluded_or_segregated_rows": int(len(audit)),
        "method": {
            "duplicates": "Exact duplicates across all eight original fields are excluded after first occurrence.",
            "cancellations": "InvoiceNo beginning with C is segregated and excluded from positive sales tables.",
            "returns": "Negative quantity lines are segregated; no synthetic original-sale matching is attempted.",
            "revenue": "Positive quantity multiplied by positive unit price; zero-price lines are excluded from sales tables.",
            "anonymous_customers": "Retained in line and order tables for aggregate analysis; excluded from customer-level table.",
            "money": "Nominal GBP values during the source period; no inflation adjustment.",
        },
    }
    (output_dir / "data_quality.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOG.info("Wrote %s sales lines, %s customers, %s audited exclusions", len(sales), len(customers), len(audit))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(prepare_tables(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
