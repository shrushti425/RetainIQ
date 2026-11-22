# RetainIQ

RetainIQ is a customer intelligence and revenue analysis prototype built on the UCI Online Retail transaction history. It combines reproducible preprocessing, descriptive analytics, customer segmentation, temporal repeat-purchase modeling, cohort retention, product associations, and an interactive Streamlit application.

## What it answers

- What revenue, order, customer, country, and product patterns are visible in the source period?
- How do customers differ in recency, frequency, and monetary value?
- Which active customers are more or less likely to purchase in the next 60 days, based on historical behavior?
- How does repeat purchasing vary by acquisition cohort?
- Which products co-occur in invoice baskets?
- Which customers and segments may merit human review based on observed value and model scores?

The system reports observed revenue, predicted purchasing probability, and revenue exposure proxies separately. It makes no causal claims about retention actions.

## Dataset and attribution

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/352/online-retail), [DOI 10.24432/C5BW33](https://doi.org/10.24432/C5BW33). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source contains 541,909 invoice-line records from 1 December 2010 through 9 December 2011. Raw source files are stored under `data/raw/` locally and ignored by Git.

## Architecture

```text
app.py                                  Streamlit pages and global filters
src/retainiq/data/download.py           UCI archive download
src/retainiq/data/profile.py            source schema, checksum, and quality profile
src/retainiq/preprocessing.py           typed sales, orders, customers, row audit
src/retainiq/analytics/eda.py           commercial and concentration summaries
src/retainiq/segmentation.py             RFM, K-Means, evaluation, PCA
src/retainiq/prediction.py               temporal prediction, metrics, SHAP
src/retainiq/retention.py                cohort, repeat, and lifecycle analytics
src/retainiq/market_basket.py             Apriori associations
src/retainiq/recommendations.py           evidence-linked decision support
src/retainiq/run_pipeline.py              end-to-end artifact orchestration
data/processed/                          generated analytical tables and models
reports/                                 run metadata, recommendations, charts
tests/                                    deterministic unit checks
```

## Data preparation methodology

The source profile records schema, missingness, exact duplicates, cancellations, returns, suspicious numeric values, dates, and a workbook SHA-256. Preprocessing assigns each source row exactly one first-match reason so audit counts reconcile. It keeps the first exact copy, segregates invoice numbers beginning with `C`, segregates remaining negative-quantity returns without uncertain invoice matching, excludes invalid transaction fields, and removes zero-price lines from revenue outputs for review. Positive anonymous sales remain in aggregate line/order tables; customer analytics require an ID.

Invoice tables contain one row per invoice number. Some source invoice lines carry multiple timestamps, so the order timestamp is the earliest line timestamp. Revenue is positive quantity × positive unit price in nominal GBP; returns are not subtracted from gross positive-sales revenue. See [`docs/technical_report.md`](docs/technical_report.md) for complete analytical assumptions.

## Analytical methods

- **EDA:** positive-sales revenue, unique invoices/customers, AOV, repeat rate, monthly/country/product sales, order distribution, and customer revenue concentration.
- **RFM:** five-quantile RFM scores; K-Means on standardized log1p RFM values; k=2…8 compared using silhouette, inertia, and Davies–Bouldin. PCA is saved for visualization. Labels describe observed cluster profiles; they are not predetermined business personas.
- **Prediction:** among customers active in a prior 180-day observation period, predict any positive purchase in the next 60 days. Logistic Regression, Random Forest, and XGBoost use chronological train/validation/test cutoffs; class weighting handles imbalance. Validation chooses model family and F1 threshold; final test metrics remain held out. SHAP explains the selected model.
- **Cohorts:** first positive invoice month; retention is active customers at cohort age N divided by cohort month-zero customers. Future, unobserved months are treated as censored.
- **Market basket:** Apriori, min support 0.5%, min confidence 15%, max itemset size 3, among invoices containing one of the 75 most frequently invoiced products. Rules with lift ≤1 are excluded. Dashboard rules recalculate within active line filters and a smaller 35-product universe.
- **Revenue exposure:** trailing 180-day customer revenue × predicted probability of no purchase in 60 days. This is a review-prioritization proxy, not a forecast or a guaranteed loss.

## Install and run

Python 3.10+ is supported (the verified local environment is Python 3.13). From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m retainiq.data.download
PYTHONPATH=src python -m retainiq.run_pipeline --source data/raw/online_retail.zip --project-root .
PYTHONPATH=src streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The full pipeline profiles and cleans the data, builds analytics, trains the models, evaluates the temporal test, calculates SHAP explanations, and writes reports/interactive HTML figures. Model training is not repeated when navigating the dashboard; Streamlit reads saved artifacts. The local demo URL is `http://localhost:8501`.

## Dashboard pages

1. **Executive Overview:** filtered KPIs, monthly revenue, segment contribution, and descriptive signals.
2. **Customer Segmentation:** filtered RFM scatter, full-period PCA cluster view, segment profile table, customer CSV.
3. **Predictive Intelligence:** model metrics, confusion matrix, ROC/PR curves, global SHAP, selected individual explanations.
4. **Cohort Retention:** filtered retention heatmap, retention curves, repeat and revenue by cohort.
5. **Product Intelligence:** filtered top products and on-demand Apriori associations with CSV download.
6. **Business Recommendations:** evidence, proposed review, relevance, and limitations for each recommendation.
7. **Data Explorer:** processed records, audit summaries, and CSV downloads.

Date, country, customer-segment, product, and line-revenue filters are available in the sidebar. Model evaluation metrics describe a fixed historical temporal test; customer and country filters narrow customer-level review, but do not retrain or alter that evaluation.

## Generated outputs

- `reports/profiling/source_profile.json`: source audit profile and checksum.
- `data/processed/cleaning_audit.csv`, `cancellations.csv`, `returns.csv`, `invalid_transactions.csv`: row-level, reason-coded decisions.
- `data/processed/sales_lines.csv`, `orders.csv`, `customers.csv`, `customer_report.csv`: analysis tables.
- `data/processed/prediction/`: saved models, predictions, evaluation, curves, and SHAP explanations.
- `reports/run_summary.json`, `reports/recommendations.json`, `reports/visualizations/*.html`: reproducible result summaries and charts.

Generated data and model binaries are ignored by Git and can be recreated by the pipeline.

## Verified full-data findings

The latest pipeline run retained 524,878 positive-sales lines (including 132,186 anonymous lines) and created 19,960 unique invoice records for 4,338 identified customers. It recorded 17,031 mutually exclusive duplicate, cancellation, return, invalid-price, or zero-price rows. Gross positive-sales revenue is **£10.64m** (returns are not reconciled against original sales). The identified-customer repeat share is **65.58%** and average order value is **£533.17**.

| Temporal test model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.608 | 0.797 | 0.690 | 0.715 | 0.762 |
| Random Forest | 0.610 | 0.782 | 0.686 | 0.700 | 0.754 |
| XGBoost | 0.621 | 0.789 | 0.695 | 0.716 | 0.762 |

XGBoost was selected by validation PR-AUC (0.733) and achieved a test PR-AUC of 0.762 at a validation-selected threshold of 0.359. Its confusion matrix was `[[638, 771], [337, 1261]]` with actual classes as rows (no repeat, repeat). Logistic Regression's test PR-AUC was nearly identical, so the run does not show a robust winner. Test positive rate was 53.14% across 3,007 October customer snapshots.

Silhouette selected **k=2** (0.433). The higher-activity “Champions” cluster contains 1,668 customers (38.5%; mean historical revenue £4,535.74/customer; median recency 17 days). “Needs Attention” contains 2,670 customers (61.5%; mean historical revenue £494.98/customer; median recency 96 days). These labels summarize the fitted clusters; they do not imply a causal treatment effect. Global SHAP ranked observed active days and units highest for the selected model. The strongest Apriori rule had 0.64% support, 83.33% confidence, and 15.56 lift.

Saved interactive Plotly charts: [monthly revenue](reports/visualizations/monthly_revenue.html), [RFM PCA clusters](reports/visualizations/rfm_pca_clusters.html), [cohort retention](reports/visualizations/cohort_retention.html), [customer order distribution](reports/visualizations/customer_order_distribution.html), [model comparison](reports/visualizations/model_comparison.html), [global SHAP importance](reports/visualizations/shap_global_importance.html), and [top products](reports/visualizations/top_products.html).

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The test suite checks source profiling, schema validation, cleaning/audit decisions, cohort calculations and right-censoring, Apriori rules, RFM segmentation invariants, and temporal prediction-window boundaries.

## Limitations

This is one retailer's historical data for roughly one year, without marketing contacts, margins, inventory, channels, or treatment outcomes. Repeat purchase and revenue metrics are descriptive. The predictive test contains one late historical cutoff, so performance may vary by season or retailer. Product association is not evidence of incremental bundle lift. Segment names are post-hoc summaries of two selected clusters in the current run, not validated customer personas. Any operational decision should be monitored and experimentally evaluated.
