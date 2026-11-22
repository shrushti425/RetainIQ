"""Descriptive commercial metrics and summaries."""

from __future__ import annotations

import pandas as pd


def build_eda(lines: pd.DataFrame, orders: pd.DataFrame, customers: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    lines = lines.copy()
    orders = orders.copy()
    customers = customers.copy()
    lines["InvoiceDate"] = pd.to_datetime(lines["InvoiceDate"])
    orders["InvoiceDate"] = pd.to_datetime(orders["InvoiceDate"])
    monthly = orders.groupby("OrderMonth", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique")
    )
    countries = lines.groupby("Country", dropna=False, as_index=False).agg(
        revenue=("LineRevenue", "sum"), units=("Quantity", "sum"), customers=("CustomerID", "nunique")
    ).sort_values("revenue", ascending=False)
    products = lines.groupby(["StockCode", "Description"], dropna=False, as_index=False).agg(
        revenue=("LineRevenue", "sum"), units=("Quantity", "sum"), invoices=("InvoiceNo", "nunique")
    ).sort_values("revenue", ascending=False)
    customer_orders = orders.dropna(subset=["CustomerID"]).groupby("CustomerID").agg(
        order_count=("InvoiceNo", "nunique"), revenue=("revenue", "sum")
    )
    frequency_distribution = customer_orders.groupby("order_count").size().rename("customers").reset_index()
    concentration = customer_orders.sort_values("revenue", ascending=False).reset_index()
    concentration["customer_share"] = (concentration.index + 1) / len(concentration)
    concentration["cumulative_revenue_share"] = concentration["revenue"].cumsum() / concentration["revenue"].sum()
    repeat_rate = float((customer_orders["order_count"] > 1).mean()) if len(customer_orders) else 0.0
    metrics = {
        "total_revenue_gbp": float(orders["revenue"].sum()),
        "total_orders": int(orders["InvoiceNo"].nunique()),
        "unique_customers": int(orders["CustomerID"].nunique()),
        "average_order_value_gbp": float(orders["revenue"].mean()),
        "repeat_purchase_rate": repeat_rate,
        "median_customer_orders": float(customer_orders["order_count"].median()),
        "customer_gini_revenue": _gini(customer_orders["revenue"]),
    }
    return {"metrics": metrics, "monthly": monthly, "countries": countries, "products": products, "concentration": concentration, "frequency_distribution": frequency_distribution}


def _gini(values: pd.Series) -> float:
    array = values.dropna().astype(float).sort_values().to_numpy()
    if len(array) == 0 or array.sum() == 0:
        return 0.0
    n = len(array)
    return float((2 * (array * range(1, n + 1)).sum()) / (n * array.sum()) - (n + 1) / n)
