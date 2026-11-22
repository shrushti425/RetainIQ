"""Monthly cohort retention and revenue analysis."""

from __future__ import annotations

import pandas as pd


def cohort_analysis(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = orders.dropna(subset=["CustomerID"]).copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["order_month"] = data["InvoiceDate"].dt.to_period("M")
    first = data.groupby("CustomerID")["order_month"].min().rename("cohort_month")
    data = data.join(first, on="CustomerID")
    data["cohort_index"] = (data["order_month"].dt.year - data["cohort_month"].dt.year) * 12 + (data["order_month"].dt.month - data["cohort_month"].dt.month)
    cohort_size = data.loc[data["cohort_index"].eq(0)].groupby("cohort_month")["CustomerID"].nunique()
    active = data.groupby(["cohort_month", "cohort_index"])["CustomerID"].nunique().unstack(fill_value=0)
    retention = active.div(cohort_size, axis=0)
    # Future periods beyond the source boundary are censored, not zero retention.
    last_observed_month = data["order_month"].max()
    for cohort_month in retention.index:
        maximum_age = (last_observed_month.year - cohort_month.year) * 12 + last_observed_month.month - cohort_month.month
        retention.loc[cohort_month, retention.columns > maximum_age] = float("nan")
    revenue = data.groupby(["cohort_month", "cohort_index"], as_index=False).agg(revenue=("revenue", "sum"), active_customers=("CustomerID", "nunique"))
    revenue["revenue_per_cohort_customer"] = revenue.apply(lambda row: row["revenue"] / cohort_size.loc[row["cohort_month"]], axis=1)
    left_censored_cohort = pd.Period("2010-12", freq="M")
    revenue["left_censored"] = revenue["cohort_month"].eq(left_censored_cohort)
    repeat = data.groupby("CustomerID").agg(orders=("InvoiceNo", "nunique"), revenue=("revenue", "sum"))
    lifecycle = pd.DataFrame([{
        "customers": len(repeat),
        "repeat_customers": int((repeat["orders"] > 1).sum()),
        "repeat_rate": float((repeat["orders"] > 1).mean()),
        "median_days_to_second_order": float(_median_days_to_second(data)),
    }])
    retention.index = retention.index.astype(str)
    revenue["cohort_month"] = revenue["cohort_month"].astype(str)
    retention.attrs["left_censored_cohorts"] = [str(left_censored_cohort)] if left_censored_cohort in data["cohort_month"].values else []
    return retention, revenue, lifecycle


def _median_days_to_second(data: pd.DataFrame) -> float:
    dates = data.sort_values("InvoiceDate").drop_duplicates(["CustomerID", "InvoiceNo"]).groupby("CustomerID")["InvoiceDate"].agg(list)
    gaps = [(values[1] - values[0]).total_seconds() / 86400 for values in dates if len(values) > 1]
    return float(pd.Series(gaps).median()) if gaps else float("nan")
