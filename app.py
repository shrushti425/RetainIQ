"""RetainIQ: interactive customer intelligence and revenue analytics demo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from retainiq.market_basket import apriori_rules
from retainiq.retention import cohort_analysis

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/processed"
REPORTS = ROOT / "reports"
NAV = ["Executive Overview", "Customer Segmentation", "Predictive Intelligence", "Cohort Retention", "Product Intelligence", "Business Recommendations", "Data Explorer"]

st.set_page_config(page_title="RetainIQ | Customer Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; color: #e5edf8; }
[data-testid="stSidebar"] { background: #0f192a; border-right: 1px solid #22314a; }
[data-testid="stHeader"] { background: rgba(11,18,32,.86); }
h1,h2,h3 { color: #f5f8ff !important; letter-spacing: -.02em; }
p, label, [data-testid="stMarkdownContainer"] { color: #c3cee0; }
[data-testid="stMetric"] { background: #111d30; border: 1px solid #253650; padding: 16px 18px; border-radius: 12px; }
[data-testid="stMetricLabel"] { color: #8ea1bd; }
[data-testid="stMetricValue"] { color: #f3f7ff; }
.ri-eyebrow { color:#64d9c2; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; }
.ri-card { background:#111d30; border:1px solid #253650; border-radius:12px; padding:18px 20px; margin:8px 0 14px; }
.ri-muted { color:#94a7c2; font-size:.86rem; }
div[data-testid="stDataFrame"] { border: 1px solid #253650; border-radius: 10px; }
button[kind="primary"] { background:#17a98f; border-color:#17a98f; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading prepared RetainIQ data…")
def load_data() -> dict:
    data = {
        "lines": pd.read_csv(DATA / "sales_lines.csv", dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"}, parse_dates=["InvoiceDate"], low_memory=False),
        "orders": pd.read_csv(DATA / "orders.csv", dtype={"InvoiceNo": "string", "CustomerID": "string"}, parse_dates=["InvoiceDate"], low_memory=False),
        "customers": pd.read_csv(DATA / "customer_report.csv", dtype={"CustomerID": "string"}, parse_dates=["first_purchase", "last_purchase"], low_memory=False),
        "monthly": pd.read_csv(DATA / "monthly.csv"),
        "countries": pd.read_csv(DATA / "countries.csv"),
        "products": pd.read_csv(DATA / "products.csv", dtype={"StockCode": "string"}),
        "concentration": pd.read_csv(DATA / "concentration.csv", dtype={"CustomerID": "string"}),
        "frequency_distribution": pd.read_csv(DATA / "frequency_distribution.csv"),
        "segment_profiles": pd.read_csv(DATA / "segment_profiles.csv"),
        "secondary_k4_profiles": pd.read_csv(DATA / "secondary_k4_profiles.csv"),
        "cluster_stability": json.loads((DATA / "cluster_stability.json").read_text()),
        "rfm": pd.read_csv(DATA / "segmented_customers.csv", dtype={"CustomerID": "string"}),
        "retention": pd.read_csv(DATA / "retention_matrix.csv"),
        "cohort_revenue": pd.read_csv(DATA / "cohort_revenue.csv"),
        "lifecycle": pd.read_csv(DATA / "lifecycle_metrics.csv"),
        "rules": pd.read_csv(DATA / "association_rules.csv"),
        "itemsets": pd.read_csv(DATA / "frequent_itemsets.csv"),
        "metrics": pd.read_json(DATA / "prediction/model_metrics.json"),
        "curves": pd.read_csv(DATA / "prediction/model_curves.csv"),
        "reliability": pd.read_csv(DATA / "prediction/reliability_curve.csv"),
        "predictions": pd.read_csv(DATA / "prediction/predictions.csv", dtype={"CustomerID": "string"}),
        "snapshots": pd.read_csv(DATA / "prediction/prediction_snapshots.csv", dtype={"CustomerID": "string"}),
        "shap_importance": pd.read_csv(DATA / "prediction/shap_importance.csv"),
        "individual_shap": pd.read_csv(DATA / "prediction/individual_shap.csv", dtype={"CustomerID": "string"}),
        "quality": json.loads((DATA / "data_quality.json").read_text()),
        "prediction_report": json.loads((DATA / "prediction/prediction_report.json").read_text()),
        "run_summary": json.loads((REPORTS / "run_summary.json").read_text()),
        "recommendations": json.loads((REPORTS / "recommendations.json").read_text()),
    }
    data["best_model"] = data["prediction_report"]["selected_model_by_validation_pr_auc"]
    data["selected_predictions"] = data["predictions"].loc[data["predictions"]["model"].eq(data["best_model"])].copy()
    data["selected_predictions"] = data["selected_predictions"].merge(
        data["snapshots"].loc[data["snapshots"]["cutoff"].eq(data["prediction_report"]["test_cutoffs"][0]), ["CustomerID", "revenue_180d"]],
        on="CustomerID", how="left",
    )
    data["selected_predictions"]["exposure_proxy_gbp"] = (1 - data["selected_predictions"]["probability_repeat_60d"]) * data["selected_predictions"]["revenue_180d"].fillna(0)
    return data


def _filters(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    lines = data["lines"]
    customers = data["customers"].drop_duplicates("CustomerID")
    countries = sorted(lines["Country"].dropna().unique().tolist())
    segments = sorted(customers["segment"].dropna().unique().tolist())
    st.sidebar.markdown("<div class='ri-eyebrow'>Workspace</div>", unsafe_allow_html=True)
    st.sidebar.title("RetainIQ")
    st.sidebar.caption("Customer intelligence · Online Retail")
    page = st.sidebar.radio("Navigate", NAV, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown("<div class='ri-eyebrow'>Global filters</div>", unsafe_allow_html=True)
    dates = st.sidebar.date_input("Invoice date", value=(lines["InvoiceDate"].min().date(), lines["InvoiceDate"].max().date()), min_value=lines["InvoiceDate"].min().date(), max_value=lines["InvoiceDate"].max().date())
    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        start_date, end_date = dates
    else:
        start_date = end_date = dates
    country_choice = st.sidebar.multiselect("Country", countries, default=countries)
    segment_choice = st.sidebar.multiselect("Customer segment", segments, default=segments)
    customer_options = sorted(customers["CustomerID"].dropna().astype(str).unique().tolist())
    customer_choice = st.sidebar.multiselect("Customer ID (optional)", customer_options, default=[], max_selections=100)
    product_codes = sorted(lines["StockCode"].dropna().unique().tolist())
    product_names = lines.drop_duplicates("StockCode").set_index("StockCode")["Description"].fillna("(description unavailable)").to_dict()
    product_choice = st.sidebar.multiselect("Product codes", product_codes, default=[], format_func=lambda code: f"{code} · {product_names.get(code, '')}")
    low_rev = float(lines["LineRevenue"].min())
    high_rev = float(lines["LineRevenue"].max())
    revenue_range = st.sidebar.slider("Line revenue (GBP)", min_value=low_rev, max_value=high_rev, value=(low_rev, high_rev), step=max((high_rev - low_rev) / 1000, 0.01), format="£%.2f")
    selected_customer_mask = customers["segment"].isin(segment_choice)
    if country_choice and "country" in customers:
        selected_customer_mask &= customers["country"].isin(country_choice)
    selected_customers = set(customers.loc[selected_customer_mask, "CustomerID"].astype(str))
    mask = lines["InvoiceDate"].dt.date.between(start_date, end_date)
    if country_choice:
        mask &= lines["Country"].isin(country_choice)
    else:
        mask &= False
    mask &= lines["LineRevenue"].between(revenue_range[0], revenue_range[1])
    if product_choice:
        mask &= lines["StockCode"].isin(product_choice)
    if customer_choice:
        mask &= lines["CustomerID"].astype(str).isin(customer_choice)
    if set(segment_choice) != set(segments):
        mask &= lines["CustomerID"].isin(selected_customers)
    filtered_lines = lines.loc[mask].copy()
    invoice_ids = set(filtered_lines["InvoiceNo"].dropna().astype(str))
    filtered_orders = data["orders"].loc[data["orders"]["InvoiceNo"].astype(str).isin(invoice_ids)].copy()
    filtered_orders = filtered_orders.loc[filtered_orders["InvoiceDate"].dt.date.between(start_date, end_date)]
    if customer_choice:
        selected_customers &= set(customer_choice)
    context = {"page": page, "start": start_date, "end": end_date, "countries": country_choice, "segments": segment_choice, "customers": customer_choice, "products": product_choice, "revenue_range": revenue_range, "selected_customers": selected_customers}
    st.sidebar.caption(f"{len(filtered_lines):,} line items · {filtered_orders['InvoiceNo'].nunique():,} invoices in the filtered view")
    return filtered_lines, filtered_orders, context


def _header(title: str, subtitle: str) -> None:
    st.markdown("<div class='ri-eyebrow'>RetainIQ · Customer intelligence</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def _money(value: float) -> str:
    return f"£{value:,.0f}"


def _segment_map(data: dict) -> dict:
    return data["customers"].drop_duplicates("CustomerID").set_index("CustomerID")["segment"].to_dict()


def _filtered_risk_count(data: dict, context: dict) -> int:
    risk = data["selected_predictions"]
    risk = risk.loc[risk["probability_repeat_60d"] < 0.5]
    allowed = context["selected_customers"]
    return int(risk["CustomerID"].astype(str).isin(allowed).sum())


def executive(data: dict, lines: pd.DataFrame, orders: pd.DataFrame, context: dict) -> None:
    _header("Executive Overview", "A filtered view of observed sales, customer mix, and the temporal repeat-purchase scoring snapshot.")
    cust_counts = orders.dropna(subset=["CustomerID"]).groupby("CustomerID")["InvoiceNo"].nunique()
    repeat = float((cust_counts > 1).mean()) if len(cust_counts) else 0.0
    revenue = float(lines["LineRevenue"].sum())
    aov = float(orders["revenue"].mean()) if len(orders) else 0.0
    kpis = st.columns(6)
    for col, label, value in zip(kpis, ["Observed revenue", "Customers", "Orders", "Average order value", "Repeat purchase rate", "Lower repeat probability"], [_money(revenue), f"{orders['CustomerID'].nunique():,}", f"{orders['InvoiceNo'].nunique():,}", _money(aov), f"{repeat:.1%}", f"{_filtered_risk_count(data, context):,}"]):
        col.metric(label, value)
    st.caption("All commercial KPIs above respond to global filters. Repeat rate is the share of identified customers with >1 invoice in the filtered period. The risk count is scored at 10 Oct 2011 and uses a 50% repeat-probability review threshold.")
    left, right = st.columns([1.55, 1])
    if not lines.empty:
        trend = lines.assign(month=lines["InvoiceDate"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(revenue=("LineRevenue", "sum"))
        left.plotly_chart(px.line(trend, x="month", y="revenue", markers=True, title="Monthly observed revenue", labels={"month": "Invoice month", "revenue": "Revenue (£)"}, template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
        segmap = _segment_map(data)
        segment_revenue = orders.assign(segment=orders["CustomerID"].astype("string").map(segmap).fillna("Anonymous / unsegmented")).groupby("segment", as_index=False).agg(revenue=("revenue", "sum"), customers=("CustomerID", "nunique")).sort_values("revenue", ascending=False)
        right.plotly_chart(px.bar(segment_revenue, x="segment", y="revenue", color="segment", title="Revenue by historical RFM cluster profile", labels={"revenue": "Revenue (£)", "segment": "Segment"}, template="plotly_dark").update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    st.subheader("Signals in this filtered view")
    if not lines.empty:
        top_country = lines.groupby("Country")["LineRevenue"].sum().sort_values(ascending=False)
        top_product = lines.groupby("Description")["LineRevenue"].sum().sort_values(ascending=False)
        concentration = float(data["run_summary"]["executive_metrics"]["customer_gini_revenue"])
        st.markdown(f"- **Country concentration:** {top_country.index[0]} contributes {_money(float(top_country.iloc[0]))} of filtered revenue ({top_country.iloc[0] / revenue:.1%}).")
        st.markdown(f"- **Product leader:** {top_product.index[0]} is the leading filtered product by observed revenue ({_money(float(top_product.iloc[0]))}).")
        st.markdown(f"- **Customer revenue concentration:** full-period customer revenue Gini is {concentration:.3f}; this describes concentration and is not a risk or causal measure.")
        detail_left, detail_right = st.columns(2)
        country_data = lines.groupby("Country", as_index=False).agg(revenue=("LineRevenue", "sum")).sort_values("revenue", ascending=False).head(12)
        detail_left.plotly_chart(px.bar(country_data.sort_values("revenue"), x="revenue", y="Country", orientation="h", title="Country revenue in filtered view", template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
        if len(cust_counts):
            frequency_data = cust_counts.rename("orders").reset_index(name="order_count")
            distribution = frequency_data.groupby("order_count").size().rename("customers").reset_index()
            detail_right.plotly_chart(px.bar(distribution.head(25), x="order_count", y="customers", title="Customer purchase frequency distribution", template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")


def segmentation_page(data: dict, lines: pd.DataFrame, orders: pd.DataFrame, context: dict) -> None:
    _header("Customer Segmentation", "RFM clusters selected with silhouette evaluation; scatter metrics below recompute over the active date and business filters.")
    scoped = orders.dropna(subset=["CustomerID"]).copy()
    if scoped.empty:
        st.info("No identified customers in the selected filters.")
        return
    as_of = scoped["InvoiceDate"].max().normalize() + pd.Timedelta(1, unit="D")
    dynamic = scoped.groupby("CustomerID", as_index=False).agg(
        last_order=("InvoiceDate", "max"), frequency=("InvoiceNo", "nunique"), monetary=("revenue", "sum")
    )
    dynamic["recency"] = (as_of - dynamic["last_order"].dt.normalize()).dt.days
    dynamic["CustomerID"] = dynamic["CustomerID"].astype(str)
    dynamic["segment"] = dynamic["CustomerID"].map(_segment_map(data)).fillna("Unprofiled / new")
    dynamic = dynamic.merge(data["customers"][["CustomerID", "rfm_score", "cluster"]].drop_duplicates("CustomerID"), on="CustomerID", how="left")
    a, b = st.columns([1.3, 1])
    a.plotly_chart(px.scatter(dynamic, x="recency", y="monetary", size="frequency", color="segment", hover_data=["CustomerID", "rfm_score", "frequency"], title=f"RFM view · as of {as_of.date()}", labels={"recency": "Recency (days; lower is more recent)", "monetary": "Revenue (£)"}, template="plotly_dark", size_max=28).update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    profile = dynamic.groupby("segment", as_index=False).agg(customers=("CustomerID", "nunique"), median_recency=("recency", "median"), mean_orders=("frequency", "mean"), mean_revenue=("monetary", "mean"))
    b.plotly_chart(px.bar(profile.sort_values("customers", ascending=True), x="customers", y="segment", orientation="h", color="segment", title="Customer count in filtered period", template="plotly_dark").update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    st.dataframe(profile, width="stretch", hide_index=True)
    pca_view = data["rfm"].loc[data["rfm"]["CustomerID"].astype(str).isin(set(dynamic["CustomerID"]))]
    st.plotly_chart(px.scatter(pca_view, x="pca_1", y="pca_2", color="segment", hover_data=["CustomerID", "recency", "frequency", "monetary"], title="PCA projection of full-period standardized log-RFM clusters", template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    table = dynamic.merge(data["customers"].drop(columns=["recency", "frequency", "monetary", "rfm_score", "cluster", "segment", "pca_1", "pca_2"], errors="ignore"), on="CustomerID", how="left")
    st.subheader("Customer-level view")
    st.dataframe(table.sort_values("monetary", ascending=False), width="stretch", hide_index=True, height=380)
    st.download_button("Download filtered segmentation CSV", table.to_csv(index=False).encode(), "retainiq_customer_segmentation.csv", "text/csv")
    st.caption(f"Full-period cluster count selected by maximum silhouette among k=2…8: k={data['run_summary']['segmentation']['selected_k']}. Segment names are descriptive labels based on observed relative recency, frequency, and monetary profiles; these are not treatment groups.")
    with st.expander("Stability and secondary four-cluster view"):
        stability = data["cluster_stability"]["solutions"]
        stability_frame = pd.DataFrame([{"k": int(k), **values} for k, values in stability.items()])
        st.dataframe(stability_frame, width="stretch", hide_index=True)
        st.caption("Mean bootstrap Adjusted Rand Index compares each resampled partition with the full-sample partition; values closer to 1 indicate more consistent assignments.")
        st.subheader("Secondary k=4 profile table")
        st.dataframe(data["secondary_k4_profiles"], width="stretch", hide_index=True)
        st.write(data["run_summary"]["segmentation"]["secondary_tradeoff_note"])


def predictive_page(data: dict, context: dict) -> None:
    _header("Predictive Intelligence", "Temporal model comparison for any purchase in the next 60 days among customers active in the prior 180 days.")
    report = data["prediction_report"]
    st.info(f"Observation window: 180 days · Outcome: next 60 days · Validation cutoff: {', '.join(report['validation_cutoffs'])} · Test cutoff: {', '.join(report['test_cutoffs'])}. Test outcome window ends {report['test_window_end']}.")
    st.caption("Model family and operating threshold were selected using validation; test metrics are held out. Customer and country filters narrow the customer-level selector; the model evaluation itself remains the fixed temporal test. Customers may appear in earlier development snapshots, and customer ID is not a predictor.")
    display_metrics = data["metrics"].copy()
    view = display_metrics[["model", "validation_pr_auc", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "test_rows", "test_positive_rate"]].copy()
    st.dataframe(view.style.format({"validation_pr_auc": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "roc_auc": "{:.3f}", "pr_auc": "{:.3f}", "threshold": "{:.3f}", "test_positive_rate": "{:.1%}"}), width="stretch", hide_index=True)
    selected = data["best_model"]
    metrics = display_metrics.loc[display_metrics["model"].eq(selected)].iloc[0]
    left, right = st.columns(2)
    left.metric(f"Selected model · {selected}", f"PR-AUC {metrics['pr_auc']:.3f}", f"ROC-AUC {metrics['roc_auc']:.3f}")
    right.plotly_chart(px.bar(view, x="model", y="pr_auc", color="model", title="Held-out PR-AUC (primary under class imbalance)", template="plotly_dark").update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    calibration = report["calibration"]
    st.caption(f"Platt sigmoid calibration is fitted on validation predictions only. Validation Brier: {calibration['validation_brier_uncalibrated']:.3f} raw → {calibration['validation_brier_calibrated']:.3f} calibrated. Calibrated probabilities {'are' if calibration['use_calibrated_probabilities_for_exposure'] else 'are not'} used for the exposure proxy.")
    if "brier_score" in metrics.index and pd.notna(metrics["brier_score"]):
        st.metric("Selected model · test Brier score", f"{metrics['brier_score']:.3f}", f"Calibrated {metrics['calibrated_brier_score']:.3f}")
    reliability = data["reliability"]
    rel_chart = px.line(reliability, x="mean_predicted_probability", y="observed_fraction", color="calibration", markers=True, title="Reliability on held-out temporal test", labels={"mean_predicted_probability": "Mean predicted probability", "observed_fraction": "Observed purchase fraction"}, template="plotly_dark")
    rel_chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
    st.plotly_chart(rel_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    curves = data["curves"]
    roc = curves.loc[curves["curve"].eq("roc")]
    pr = curves.loc[curves["curve"].eq("precision_recall")]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(roc, x="x", y="y", color="model", title="ROC curves · temporal test", labels={"x": "False positive rate", "y": "True positive rate"}, template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    c2.plotly_chart(px.line(pr, x="x", y="y", color="model", title="Precision–recall curves · temporal test", labels={"x": "Recall", "y": "Precision"}, template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    cm = np.array(metrics["confusion_matrix"])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted no", "Predicted repeat"], y=["Actual no", "Actual repeat"], color_continuous_scale="Teal", title=f"Confusion matrix · threshold {metrics['threshold']:.2f}", template="plotly_dark"), width="stretch")
    st.subheader("Global SHAP feature effects")
    st.plotly_chart(px.bar(data["shap_importance"].sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", title=f"Mean absolute SHAP · {selected}", labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"}, template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    st.subheader("Individual prediction and explanation")
    shap_rows = data["individual_shap"]
    ids = sorted(set(shap_rows["CustomerID"].astype(str).unique()) & context["selected_customers"])
    if ids:
        chosen = st.selectbox("Customer scored in the held-out temporal test", ids)
        prediction = data["selected_predictions"].loc[data["selected_predictions"]["CustomerID"].astype(str).eq(chosen)].iloc[0]
        st.metric("Predicted probability of repeat purchase", f"{prediction['probability_repeat_60d']:.1%}", f"Historical value exposure proxy {_money(prediction['exposure_proxy_gbp'])}")
        one = shap_rows.loc[shap_rows["CustomerID"].astype(str).eq(chosen)].iloc[0]
        contributions = pd.DataFrame({"feature": report["feature_columns"], "SHAP value": [one[c] for c in report["feature_columns"]]}).sort_values("SHAP value")
        st.plotly_chart(px.bar(contributions, x="SHAP value", y="feature", orientation="h", color="SHAP value", color_continuous_scale="RdBu", title="Customer-level SHAP contributions"), width="stretch")
    st.markdown("**Limitations:** one end-of-history test cutoff; purchasing probability is not a treatment response; observed history covers only one year; the probability should be used for prioritization with review, not as a guarantee of churn.")


@st.cache_data(show_spinner="Calculating filtered Apriori rules…")
def filtered_basket(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apriori_rules(frame, top_products=35, min_support=0.01, min_confidence=0.15)


def cohort_page(orders: pd.DataFrame) -> None:
    _header("Cohort Retention", "Monthly acquisition cohorts, active-customer retention, and cohort revenue from the current filtered invoice set.")
    if orders.empty or orders["CustomerID"].notna().sum() == 0:
        st.info("No identified customer invoices are available for these filters.")
        return
    retention, revenue, lifecycle = cohort_analysis(orders)
    left_censored = set(revenue.loc[revenue["left_censored"], "cohort_month"].astype(str))
    st.warning(f"The first observed cohort ({', '.join(sorted(left_censored))}) is left-censored: customers may have purchased before the dataset begins. It remains visible with an explicit marker, but is excluded from the headline month-one retention statistic.", title="Cohort boundary caveat")
    eligible = retention.loc[~retention.index.astype(str).isin(left_censored)]
    month_one = eligible[1].dropna() if 1 in eligible.columns else pd.Series(dtype=float)
    a, b, c, d = st.columns(4)
    life = lifecycle.iloc[0]
    a.metric("Median month-one retention · eligible cohorts", f"{month_one.median():.1%}" if not month_one.empty else "—")
    b.metric("Eligible cohorts with observed month one", f"{len(month_one):,}")
    c.metric("Repeat customers · full observed history", f"{int(life['repeat_customers']):,}")
    d.metric("Median days to second order", f"{life['median_days_to_second_order']:.0f}" if pd.notna(life["median_days_to_second_order"]) else "—")
    matrix = retention.copy()
    matrix.index = [f"{month} * left-censored" if str(month) in left_censored else str(month) for month in matrix.index]
    st.plotly_chart(px.imshow(matrix, aspect="auto", color_continuous_scale="Teal", zmin=0, zmax=1, labels={"x": "Months since first purchase", "y": "Acquisition cohort", "color": "Retention"}, title="Monthly customer cohort retention", template="plotly_dark"), width="stretch")
    curves = matrix.reset_index(names="cohort_month").melt(id_vars="cohort_month", var_name="cohort_age", value_name="retention")
    st.plotly_chart(px.line(curves, x="cohort_age", y="retention", color="cohort_month", markers=True, title="Retention curves by acquisition month", labels={"cohort_age": "Months since acquisition", "retention": "Customer retention"}, template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch")
    st.subheader("Revenue by cohort")
    st.plotly_chart(px.density_heatmap(revenue, x="cohort_index", y="cohort_month", z="revenue", histfunc="sum", color_continuous_scale="Teal", title="Observed cohort revenue by elapsed month", labels={"cohort_index": "Months since first purchase", "cohort_month": "Acquisition cohort", "z": "Revenue (£)"}), width="stretch")
    st.dataframe(revenue, width="stretch", hide_index=True)
    st.caption("Cohort age zero is the first calendar month with a purchase. Retention is active identified customers in month N / customers in month zero. The Dec 2010 boundary cohort is marked left-censored and excluded from the headline month-one statistic; incomplete recent cohorts have fewer observable months.")


def product_page(data: dict, lines: pd.DataFrame, context: dict) -> None:
    _header("Product Intelligence", "Product revenue and co-purchase patterns using Apriori support, confidence, and lift.")
    if lines.empty:
        st.info("No product lines match the active filters.")
        return
    products = lines.groupby(["StockCode", "Description"], dropna=False, as_index=False).agg(revenue=("LineRevenue", "sum"), units=("Quantity", "sum"), invoices=("InvoiceNo", "nunique")).sort_values("revenue", ascending=False)
    a, b = st.columns([1.25, 1])
    a.plotly_chart(px.bar(products.head(20).sort_values("revenue"), x="revenue", y="Description", orientation="h", color="revenue", color_continuous_scale="Teal", title="Top products by observed revenue", labels={"revenue": "Revenue (£)"}), width="stretch")
    b.plotly_chart(px.treemap(products.head(40), path=["Description"], values="revenue", color="revenue", color_continuous_scale="Teal", title="Revenue contribution across leading products"), width="stretch")
    st.dataframe(products.head(100), width="stretch", hide_index=True)
    st.download_button("Download filtered product report", products.to_csv(index=False).encode(), "retainiq_filtered_products.csv", "text/csv")
    st.subheader("Apriori association rules")
    st.caption("Rules recalculate on the active line filters using up to 35 frequent product codes, minimum support 1%, confidence 15%, and itemsets up to length 3. The candidate cap controls computational cost; support is relative to invoices containing at least one candidate product.")
    _, rules = filtered_basket(lines)
    if not rules.empty:
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(100).style.format({"support": "{:.2%}", "confidence": "{:.2%}", "lift": "{:.2f}"}), width="stretch", hide_index=True)
        st.download_button("Download filtered association rules", rules.to_csv(index=False).encode(), "retainiq_filtered_apriori_rules.csv", "text/csv")
    else:
        st.info("No rules passed the minimum support, confidence, and lift filters for this selection.")


def recommendations_page(data: dict, context: dict) -> None:
    _header("Business Recommendations", "Decision support grounded in computed outputs; no estimated uplift or causal claims.")
    recommendations = data["recommendations"]
    allowed = set(context["segments"])
    for i, rec in enumerate(recommendations, 1):
        if rec["segment"] not in {"Acquisition cohorts", "Product relationship across all eligible invoice baskets", "Customer-level score; segment label is shown separately in the app"} and rec["segment"] not in allowed:
            continue
        st.markdown(f"<div class='ri-card'><div class='ri-eyebrow'>Recommendation {i} · {rec['segment']}</div><h3>{rec['observation']}</h3><p><b>Evidence:</b> {rec['support']}</p><p><b>Suggested review:</b> {rec['action']}</p><p><b>Business relevance:</b> {rec['relevance']}</p><p class='ri-muted'><b>Evidence limit:</b> {rec['limitation']}</p></div>", unsafe_allow_html=True)
    st.markdown("**Estimated exposure is a prioritization proxy:** trailing 180-day customer revenue multiplied by model-estimated probability of no purchase within 60 days. It is not a future revenue forecast, guaranteed loss, or causal effect.")
    risk = data["selected_predictions"].copy()
    risk = risk.loc[risk["CustomerID"].astype(str).isin(context["selected_customers"])].sort_values("exposure_proxy_gbp", ascending=False).head(30)
    if not risk.empty:
        risk = risk.merge(data["customers"][["CustomerID", "segment", "country"]].drop_duplicates("CustomerID"), on="CustomerID", how="left")
        st.subheader("Highest customer-level exposure proxies")
        st.dataframe(risk[["CustomerID", "segment", "country", "probability_repeat_60d", "revenue_180d", "exposure_proxy_gbp"]].style.format({"probability_repeat_60d": "{:.1%}", "revenue_180d": "£{:,.2f}", "exposure_proxy_gbp": "£{:,.2f}"}), width="stretch", hide_index=True)


def explorer(data: dict, lines: pd.DataFrame, orders: pd.DataFrame) -> None:
    _header("Data Explorer", "Inspect and download the processed sales lines, invoice-level orders, and cleaning audit summaries.")
    quality = data["quality"]
    a, b, c, d = st.columns(4)
    a.metric("Original source rows", f"{quality['source_rows']:,}")
    b.metric("Revenue sales lines", f"{quality['sales_line_rows']:,}")
    c.metric("Audited or segregated", f"{quality['excluded_or_segregated_rows']:,}")
    d.metric("Identified customers", f"{quality['customers']:,}")
    st.subheader("Reason-coded source audit")
    audit = pd.DataFrame([{"Decision code": key.replace("_", " ").title(), "Rows": value} for key, value in quality["audit_reason_counts"].items() if key != "valid_sale"])
    st.dataframe(audit, width="stretch", hide_index=True)
    st.json(quality["method"])
    st.subheader("Filtered sales lines")
    st.dataframe(lines.head(1000), width="stretch", hide_index=True, height=400)
    st.download_button("Download filtered sales lines", lines.to_csv(index=False).encode(), "retainiq_filtered_sales_lines.csv", "text/csv")
    st.subheader("Filtered invoice table")
    st.dataframe(orders.head(1000), width="stretch", hide_index=True)
    st.download_button("Download filtered invoices", orders.to_csv(index=False).encode(), "retainiq_filtered_orders.csv", "text/csv")
    for filename, label in [("cleaning_audit.csv", "Full cleaning audit"), ("returns.csv", "Segregated return lines"), ("cancellations.csv", "Cancellation invoice lines"), ("invalid_transactions.csv", "Invalid / zero-price lines")]:
        path = DATA / filename
        if path.exists():
            st.download_button(f"Download {label}", path.read_bytes(), filename, "text/csv", key=f"download-{filename}")


def main() -> None:
    try:
        data = load_data()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"RetainIQ analytical artifacts are unavailable: {exc}")
        st.code("PYTHONPATH=src python -m retainiq.run_pipeline --source data/raw/online_retail.zip --project-root .")
        st.stop()
    lines, orders, context = _filters(data)
    page = context["page"]
    if page == "Executive Overview":
        executive(data, lines, orders, context)
    elif page == "Customer Segmentation":
        segmentation_page(data, lines, orders, context)
    elif page == "Predictive Intelligence":
        predictive_page(data, context)
    elif page == "Cohort Retention":
        cohort_page(orders)
    elif page == "Product Intelligence":
        product_page(data, lines, context)
    elif page == "Business Recommendations":
        recommendations_page(data, context)
    else:
        explorer(data, lines, orders)
    st.divider()
    st.caption("RetainIQ prototype · UCI Online Retail · CC BY 4.0 · Historical descriptive and predictive analysis; not causal evidence.")


if __name__ == "__main__":
    main()
