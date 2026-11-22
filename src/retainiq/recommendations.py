"""Create conservative, evidence-linked action notes from computed artifacts."""

from __future__ import annotations

import pandas as pd


def build_recommendations(profiles: pd.DataFrame, retention: pd.DataFrame, rules: pd.DataFrame, predictions: pd.DataFrame, snapshots: pd.DataFrame, selected_model: str) -> list[dict]:
    recs = []
    if not profiles.empty:
        highest = profiles.sort_values("mean_revenue", ascending=False).iloc[0]
        recs.append({
            "observation": f"{highest['segment']} has the highest average historical customer revenue among the derived clusters.",
            "support": f"Mean customer revenue £{highest['mean_revenue']:,.2f}; {int(highest['customers']):,} customers ({highest['customer_share']:.1%} of profiled customers).",
            "segment": highest["segment"],
            "action": "Review recent engagement and purchasing cadence for this group before selecting a retention treatment.",
            "relevance": "A high historical value base can justify prioritizing service review.",
            "limitation": "Historical cluster averages do not establish future value or prove that an intervention will improve retention.",
        })
        least_recent = profiles.sort_values("median_recency_days", ascending=False).iloc[0]
        if len(profiles) > 1:
            recs.append({
                "observation": f"{least_recent['segment']} has the longest median time since last purchase and a lower observed purchase cadence than the higher-activity cluster.",
                "support": f"Median recency {least_recent['median_recency_days']:.0f} days; mean {least_recent['mean_orders']:.2f} invoices and mean historical revenue £{least_recent['mean_revenue']:,.2f} per customer.",
                "segment": least_recent["segment"],
                "action": "Review the highest-value customers in this group for product-fit or service issues, then evaluate any reactivation idea with a controlled test.",
                "relevance": "Combines relative inactivity with a segment-level historical value baseline to focus manual review.",
                "limitation": "The segment is a retrospective K-Means profile, and its label is descriptive. The data contains no contact or intervention outcomes.",
            })
    if not predictions.empty and not snapshots.empty:
        best = predictions.loc[predictions["model"].eq(selected_model)].copy()
        features = snapshots.loc[snapshots["cutoff"].eq(best["cutoff"].iloc[0]), ["CustomerID", "revenue_180d"]]
        risk = best.merge(features, on="CustomerID", how="left")
        risk["exposure_proxy"] = (1 - risk["probability_repeat_60d"]) * risk["revenue_180d"].fillna(0)
        row = risk.sort_values("exposure_proxy", ascending=False).iloc[0]
        recs.append({
            "observation": "Some customers combine substantial recent revenue with lower predicted repeat probability.",
            "support": f"Largest customer-level proxy in the test snapshot is £{row['exposure_proxy']:,.2f} (customer {row['CustomerID']}); model {selected_model}, predicted repeat probability {row['probability_repeat_60d']:.1%}.",
            "segment": "Customer-level score; segment label is shown separately in the app",
            "action": "Review the highest exposure-proxy customers for service or product-fit issues and validate eligibility before any outreach.",
            "relevance": "Provides a ranked queue for human review.",
            "limitation": "Proxy = observed trailing 180-day revenue × predicted probability of no purchase in 60 days. It is not a forecast of lost revenue or an intervention effect.",
        })
    if not rules.empty:
        rule = rules.iloc[0]
        recs.append({
            "observation": f"The strongest retained association connects {rule['antecedents']} with {rule['consequents']}.",
            "support": f"Support {rule['support']:.2%}; confidence {rule['confidence']:.2%}; lift {rule['lift']:.2f}.",
            "segment": "Product relationship across all eligible invoice baskets",
            "action": "Consider this pair for a small merchandising or bundle test, with a control group and margin review.",
            "relevance": "Association strength can inform product discovery and placement hypotheses.",
            "limitation": "Co-purchase is observational and does not show that a bundle or recommendation will cause incremental sales.",
        })
    if not retention.empty:
        maturity = retention.groupby("cohort_month").first().sort_values("cohort_month")
        if len(maturity) >= 2:
            recs.append({
                "observation": "Monthly cohort retention differs across acquisition months; interpret only comparable cohort ages.",
                "support": "See cohort heatmap and revenue table in RetainIQ; each cell uses active customers divided by that cohort's month-zero size.",
                "segment": "Acquisition cohorts",
                "action": "Compare cohorts at the same elapsed month and investigate changes in acquisition mix or seasonal context.",
                "relevance": "Separates lifecycle timing from aggregate monthly revenue movements.",
                "limitation": "Cohort differences are descriptive and may reflect seasonality, acquisition channels, or customer mix.",
            })
    return recs
