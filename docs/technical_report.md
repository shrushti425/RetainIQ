# RetainIQ technical report

## Scope and source

RetainIQ analyzes UCI Online Retail invoice lines from 1 December 2010 through 9 December 2011. The repository keeps the original UCI archive out of version control and records a checksum in `reports/profiling/source_profile.json`. Currency is nominal GBP; no inflation adjustment is applied.

## Preprocessing and audit policy

Rows are assigned one mutually exclusive first-match status so the audit counts reconcile to the source. Exact duplicates across the original eight fields are excluded after retaining their first occurrence. Invoice numbers beginning with `C` are segregated as cancellations. Remaining negative quantities are segregated as returns without attempting uncertain original-invoice matching. Rows with invalid date/quantity/price or missing invoice/product identifiers are excluded. Zero-price lines are excluded from revenue tables and retained in the audit for review. Positive sales with missing `CustomerID` remain in sales-line and invoice tables for aggregate/product analysis; customer-level features require an ID.

Invoice tables are keyed by invoice number. Forty-two invoices in the source's eligible sales had multiple line-level timestamps; invoice time is set to the earliest timestamp so revenue/order counts are not split. Customer frequency is a count of unique invoice numbers. Revenue is positive quantity × positive unit price.

## Descriptive analytics

The pipeline reports revenue, unique invoices/customers, average order value, repeat-customer share, monthly sales, country/product contributions, and customer revenue concentration. A repeat customer is an identified customer with more than one positive sale invoice during the observed data period. This is a retrospective descriptive rate, not a forecast.

## Segmentation

Recency, frequency, and monetary value are computed as of the day after the last source transaction. RFM scores use five quantile bands (recency reversed). K-Means uses standardized log1p RFM values with a fixed seed and k=2…8 comparison by silhouette, with Davies–Bouldin and inertia also retained. The selected k and full profiles are written to reports. Cluster names derive from relative cluster recency/frequency/revenue profiles; they are descriptive labels, not externally validated personas. PCA is fit on the transformed RFM matrix for visualization.

## Repeat-purchase prediction

The prediction unit is an identified customer snapshot active at least once in the prior 180 days. The label is any positive sales invoice during the following 60 days. Features are trailing order count, revenue, units, average order value, recency, active invoice dates, product diversity, and revenue in recent versus preceding 30-day periods. IDs and future information are not model features.

Train snapshot cutoff: 31 May 2011; validation cutoff: 31 July 2011; test cutoff: 10 October 2011. Validation labels are fully observable before the test cutoff; the test horizon ends on 9 December, the source boundary. Logistic Regression, balanced Random Forest, and weighted XGBoost are compared. Validation PR-AUC selects the reported model family; the threshold is selected on validation F1 and frozen for the final temporal test. Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and classification report are saved for every model. SHAP values explain the validation-selected model on a deterministic sample (up to 750 test snapshots).

The final test is one scoring occasion close to the dataset boundary. Multiple customer snapshots may occur across development splits, which simulates repeated scoring of known customers; the final test cutoff is chronologically later and no customer ID is provided as a feature. This short, single-company history cannot establish performance on a new retailer or season. Scores predict observed purchase, not response to marketing.

## Cohorts and product associations

Monthly cohorts are assigned by first positive invoice month. Retention for cohort age N is identified customers purchasing in month N divided by customers in month zero. Recent cohorts have incomplete follow-up. Revenue tables use the same cohort month and elapsed month.

Apriori is run over invoices containing at least one of the 75 most frequently invoiced products, with minimum support 0.5%, minimum confidence 15%, and maximum itemset size 3. Rules with lift ≤1 are excluded. The dashboard recalculates filtered product associations over up to 35 frequent products. Basket associations show co-occurrence only and do not prove incremental revenue from a bundle.

## Revenue exposure and recommendations

The customer exposure proxy equals trailing 180-day revenue multiplied by the model-estimated probability of no purchase in the next 60 days. It is a ranked review aid—not a forecast of revenue lost. Recommendations state supporting evidence and limits, and do not claim causal uplift.

## Reproducibility

All analysis uses fixed random seeds. `reports/run_summary.json` contains generated results; `reports/recommendations.json` contains evidence-linked recommendation text; `data/processed/` contains tables, audit files, predictions, curves, model binaries, and SHAP extracts. Generated data/model artifacts are not committed.

## Findings from the verified full-data run

The source profile contained 541,909 rows. Mutually exclusive cleaning statuses were 5,268 duplicate copies, 9,251 cancellation invoice lines, 1,336 remaining negative-quantity return lines, 2 invalid negative-price lines, and 1,174 zero-price lines. The resulting positive-sales line table contains 524,878 rows, including 132,186 anonymous lines; customer analysis includes 4,338 identified customers. The invoice table contains 19,960 distinct invoices. Gross positive-sales revenue is £10,642,110.80; this is before return reconciliation and excludes cancellation, zero-price, and invalid lines.

Across the full source period the identified-customer repeat share is 65.58%, average invoice value is £533.17, and median identified customer order count is 2. Revenue is concentrated in the UK (£9,001,744.09, approximately 84.6% of positive-sales revenue). “DOTCOM POSTAGE” has the highest line revenue (£206,248.77); it should be interpreted as a shipping-related line rather than a merchandise recommendation.

Silhouette selected two clusters (0.433 versus 0.337 for k=3). The derived “Champions” cluster contains 1,668 customers (38.5%), with median recency 17 days, mean frequency 8.44 invoices, and mean historical revenue £4,535.74. “Needs Attention” contains 2,670 (61.5%), with median recency 96 days, mean frequency 1.67, and mean revenue £494.98. The material gap is descriptive and the two-cluster solution is intentionally not split into six unsupported personas.

The repeat-purchase test contains 3,007 customer snapshots, with a 53.14% positive rate. Validation PR-AUC selected XGBoost (0.7326). Test results were:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.608 | 0.797 | 0.690 | 0.715 | 0.762 |
| Random Forest | 0.610 | 0.782 | 0.686 | 0.700 | 0.754 |
| XGBoost | 0.621 | 0.789 | 0.695 | 0.716 | 0.762 |

XGBoost test confusion matrix at validation-selected threshold 0.359 is `[[638, 771], [337, 1261]]` (actual rows: no repeat, repeat; predicted columns: no repeat, repeat). It has relatively high repeat recall but many false-positive repeat predictions. Logistic Regression's PR-AUC is almost identical to XGBoost's; the single-cutoff results do not establish a reliable winner. The most influential global SHAP feature by mean absolute attribution was `active_days_180d` (counted as distinct calendar days), followed by `units_180d`; SHAP is model attribution, not causal effect.

The highest retained Apriori rule has support 0.64%, confidence 83.33%, and lift 15.56. It associates a pair of products in the antecedent with WOODLAND CHARLOTTE BAG. The top customer revenue-exposure proxy is £33,819.33 for customer 15098 at a 15.3% predicted 60-day repeat probability. This proxy is historical revenue scaled by estimated no-repeat probability, not projected lost sales.
