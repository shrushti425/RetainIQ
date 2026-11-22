"""Run the reproducible RetainIQ data and analytical artifact pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from retainiq.analytics.eda import build_eda
from retainiq.data.profile import profile_dataset
from retainiq.market_basket import apriori_rules
from retainiq.prediction import train_and_evaluate
from retainiq.preprocessing import prepare_tables
from retainiq.recommendations import build_recommendations
from retainiq.retention import cohort_analysis
from retainiq.segmentation import segment_customers

LOG = logging.getLogger("retainiq.pipeline")


def run(source: Path, project_root: Path, skip_models: bool = False) -> dict:
    profile_dir = project_root / "reports/profiling"
    data_dir = project_root / "data/processed"
    report_dir = project_root / "reports"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_dataset(source)
    (profile_dir / "source_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    cleaning = prepare_tables(source, data_dir)
    lines = pd.read_csv(data_dir / "sales_lines.csv", dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"}, parse_dates=["InvoiceDate"], low_memory=False)
    orders = pd.read_csv(data_dir / "orders.csv", dtype={"InvoiceNo": "string", "CustomerID": "string"}, parse_dates=["InvoiceDate"], low_memory=False)
    customers = pd.read_csv(data_dir / "customers.csv", dtype={"CustomerID": "string"}, parse_dates=["first_purchase", "last_purchase"])

    eda = build_eda(lines, orders, customers)
    for name in ["monthly", "countries", "products", "concentration", "frequency_distribution"]:
        eda[name].to_csv(data_dir / f"{name}.csv", index=False)
    (report_dir / "executive_metrics.json").write_text(json.dumps(eda["metrics"], indent=2) + "\n")

    rfm, profiles, segmentation_meta = segment_customers(orders, data_dir)
    rfm.to_csv(data_dir / "segmented_customers.csv", index=False)
    customers.merge(rfm[["CustomerID", "recency", "rfm_score", "cluster", "segment", "pca_1", "pca_2"]], on="CustomerID", how="left").to_csv(data_dir / "customer_report.csv", index=False)

    retention, cohort_revenue, lifecycle = cohort_analysis(orders)
    retention.rename_axis("cohort_month").reset_index().to_csv(data_dir / "retention_matrix.csv", index=False)
    cohort_revenue.to_csv(data_dir / "cohort_revenue.csv", index=False)
    cohort_revenue[["cohort_month", "left_censored"]].drop_duplicates().to_csv(data_dir / "cohort_metadata.csv", index=False)
    lifecycle.to_csv(data_dir / "lifecycle_metrics.csv", index=False)

    frequent, rules = apriori_rules(lines)
    frequent.to_csv(data_dir / "frequent_itemsets.csv", index=False)
    rules.to_csv(data_dir / "association_rules.csv", index=False)
    (report_dir / "market_basket_metadata.json").write_text(json.dumps({"method": "Apriori", "candidate_products": 75, "minimum_invoice_support": 0.005, "minimum_confidence": 0.15, "max_itemset_size": 3, "basket_universe": "invoices containing at least one of the 75 most frequently invoiced product codes"}, indent=2) + "\n")

    model_result = {"report": {"skipped": True}, "metrics": []}
    predictions = pd.DataFrame()
    snapshots = pd.DataFrame()
    selected_model = ""
    if not skip_models:
        model_result = train_and_evaluate(orders, lines, data_dir / "prediction", seed=42)
        predictions = pd.read_csv(data_dir / "prediction/predictions.csv")
        snapshots = pd.read_csv(data_dir / "prediction/prediction_snapshots.csv")
        selected_model = model_result["report"]["selected_model_by_validation_pr_auc"]
    recommendations = build_recommendations(profiles, cohort_revenue.rename(columns={"cohort_month": "cohort_month"}), rules, predictions, snapshots, selected_model) if selected_model else []
    (report_dir / "recommendations.json").write_text(json.dumps(recommendations, indent=2) + "\n")
    summary = {
        "data_quality": cleaning,
        "executive_metrics": eda["metrics"],
        "segmentation": segmentation_meta,
        "lifecycle": lifecycle.iloc[0].to_dict(),
        "cohort_caveat": {"left_censored_cohorts": retention.attrs.get("left_censored_cohorts", []), "headline_retention_excludes_left_censored": True},
        "market_basket_rule_count": int(len(rules)),
        "prediction": model_result["report"],
        "model_metrics": model_result["metrics"],
    }
    (report_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _save_visualizations(report_dir / "visualizations", eda, rfm, retention, cohort_revenue, model_result["metrics"], data_dir)
    LOG.info("Pipeline complete; artifacts in %s", data_dir)
    return summary


def _save_visualizations(folder: Path, eda: dict, rfm: pd.DataFrame, retention: pd.DataFrame, cohort_revenue: pd.DataFrame, metrics: list, data_dir: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    charts = {
        "monthly_revenue.html": px.line(eda["monthly"], x="OrderMonth", y="revenue", markers=True, title="Monthly observed positive-sales revenue", template="plotly_dark"),
        "rfm_pca_clusters.html": px.scatter(rfm, x="pca_1", y="pca_2", color="segment", hover_data=["CustomerID", "recency", "frequency", "monetary"], title="PCA projection of RFM customer clusters", template="plotly_dark"),
        "top_products.html": px.bar(eda["products"].head(20).sort_values("revenue"), x="revenue", y="Description", orientation="h", title="Top products by positive-sales revenue", template="plotly_dark"),
        "customer_order_distribution.html": px.bar(eda["frequency_distribution"].head(25), x="order_count", y="customers", title="Identified customer purchase frequency distribution", template="plotly_dark"),
    }
    matrix = retention.copy()
    left_censored = set(cohort_revenue.loc[cohort_revenue["left_censored"], "cohort_month"].astype(str))
    matrix.index = [f"{month} *" if str(month) in left_censored else str(month) for month in matrix.index]
    charts["cohort_retention.html"] = px.imshow(matrix, aspect="auto", zmin=0, zmax=1, color_continuous_scale="Teal", title="Monthly customer cohort retention")
    if metrics:
        metrics_frame = pd.DataFrame(metrics)
        comparison = metrics_frame.melt(id_vars="model", value_vars=["roc_auc", "pr_auc"], var_name="metric", value_name="score")
        charts["model_comparison.html"] = px.bar(comparison, x="model", y="score", color="metric", barmode="group", title="Held-out temporal-test ranking performance", template="plotly_dark")
        shap_path = data_dir / "prediction/shap_importance.csv"
        if shap_path.exists():
            shap_frame = pd.read_csv(shap_path).sort_values("mean_abs_shap")
            charts["shap_global_importance.html"] = px.bar(shap_frame, x="mean_abs_shap", y="feature", orientation="h", title="Global SHAP importance for the selected model", template="plotly_dark")
    for filename, figure in charts.items():
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        figure.write_html(folder / filename, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/online_retail.zip"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-models", action="store_true", help="Run data and non-predictive analytics only")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(json.dumps(run(args.source, args.project_root, args.skip_models), indent=2, default=str))


if __name__ == "__main__":
    main()

