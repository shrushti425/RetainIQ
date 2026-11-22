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

For k=2…6, 100 fixed-seed bootstrap resamples are clustered and compared to full-sample assignments on the sampled customers using Adjusted Rand Index. The selected k=2 has the highest silhouette (0.433) and mean bootstrap ARI 0.980; k=4 has silhouette 0.334 and mean ARI 0.884. A secondary k=4 profile table is saved for exploratory use. It splits the broad k=2 groups into more specific profiles, but adds complexity and has lower observed stability; k=2 remains primary.

## Repeat-purchase prediction

The prediction unit is an identified customer snapshot active at least once in the prior 180 days. The label is any positive sales invoice during the following 60 days. Features are trailing order count, revenue, units, average order value, recency, active invoice dates, product diversity, and revenue in recent versus preceding 30-day periods. IDs and future information are not model features.

Train snapshot cutoff: 31 May 2011; validation cutoff: 31 July 2011; test cutoff: 10 October 2011. Validation labels are fully observable before the test cutoff; the test horizon ends on 9 December, the source boundary. Logistic Regression, balanced Random Forest, and weighted XGBoost are compared with two non-probabilistic ranking baselines: descending negative recency and descending composite RFM score. Validation PR-AUC selects the learned model family; thresholds are selected on validation F1 and frozen for the final temporal test. Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and classification report are saved for every method. Baseline scores are ranks, not probabilities. SHAP values explain the validation-selected model on a deterministic sample (up to 750 test snapshots).

Each test ROC-AUC, average-precision PR-AUC, and F1 is reported with a 95% percentile interval from 1,000 stratified bootstrap resamples. Resampling is stratified by observed label to preserve both classes; F1 uses the validation-selected threshold. These intervals describe sampling variation within one test scoring occasion, not temporal or population uncertainty.

The selected model is calibrated with a Platt sigmoid map fitted only on validation predictions and labels. Brier score and quantile reliability points are calculated before and after calibration on validation and held-out test. Validation Brier changed from 0.2013 to 0.1990; held-out test Brier changed from 0.2134 to 0.2175. The decision to apply calibration is made using validation only, so calibrated probabilities are used in the exposure proxy; test labels do not select the scoring path. The validation improvement did not transfer to this held-out period. The calibrator itself is never fit on test labels, and the test comparison is diagnostic for this single period.

Three rolling-origin evaluations use 31 July, 31 August, and 30 September 2011. For each origin, a historical training snapshot is eligible only when its complete 60-day target window ends by the evaluation date. Positive rates were 44.35% (1,229 / 2,771 snapshots), 49.06% (1,360 / 2,772), and 52.55% (1,537 / 2,925), respectively. Mean ± sample standard deviation PR-AUC across these origins was 0.745 ± 0.018 for Logistic Regression, 0.742 ± 0.013 for Random Forest, 0.752 ± 0.014 for XGBoost, 0.595 ± 0.019 for recency, and 0.661 ± 0.019 for RFM score. The spread is descriptive across three origins, not a confidence interval; prevalence is listed because PR-AUC depends on the positive-class rate.

The final test is one scoring occasion close to the dataset boundary. Multiple customer snapshots may occur across development splits, which simulates repeated scoring of known customers; the final test cutoff is chronologically later and no customer ID is provided as a feature. This short, single-company history cannot establish performance on a new retailer or season. Scores predict observed purchase, not response to marketing.

## Cohorts and product associations

Monthly cohorts are assigned by first positive invoice month. Retention for cohort age N is identified customers purchasing in month N divided by customers in month zero. December 2010 is explicitly marked left-censored because the source starts during customers' potentially longer histories. It remains visible in the matrices and cohort metadata, but is excluded from the headline month-one retention statistic. The verified median month-one retention across the 11 other cohorts with observable month one is 19.0%. Recent cohorts have incomplete follow-up and are right-censored. Revenue tables use the same cohort month and elapsed month.

Apriori is run over invoices containing at least one of the 75 most frequently invoiced products, with minimum support 0.5%, minimum confidence 15%, and maximum itemset size 3. Rules with lift ≤1 are excluded. The dashboard recalculates filtered product associations over up to 35 frequent products. Basket associations show co-occurrence only and do not prove incremental revenue from a bundle.

## Revenue exposure and recommendations

The customer exposure proxy equals trailing 180-day revenue multiplied by the model-estimated probability of no purchase in the next 60 days. It is a ranked review aid—not a forecast of revenue lost. Recommendations state supporting evidence and limits, and do not claim causal uplift.

## Reproducibility

All analysis uses fixed random seeds. `reports/run_summary.json` contains generated results; `reports/recommendations.json` contains evidence-linked recommendation text; `data/processed/` contains tables, audit files, predictions, curves, model binaries, SHAP extracts, `cluster_stability.json`, `secondary_k4_profiles.csv`, `cohort_metadata.csv`, `reliability_curve.csv`, and `rolling_origin_metrics.json`. Generated data/model artifacts are not committed. `requirements.txt` uses exact direct-dependency pins; `scripts/check_requirements_pins.py` is run locally and in GitHub Actions before the unittest suite.

## Findings from the verified full-data run

The source profile contained 541,909 rows. Mutually exclusive cleaning statuses were 5,268 duplicate copies, 9,251 cancellation invoice lines, 1,336 remaining negative-quantity return lines, 2 invalid negative-price lines, and 1,174 zero-price lines. The resulting positive-sales line table contains 524,878 rows, including 132,186 anonymous lines; customer analysis includes 4,338 identified customers. The invoice table contains 19,960 distinct invoices. Gross positive-sales revenue is £10,642,110.80; this is before return reconciliation and excludes cancellation, zero-price, and invalid lines.

Across the full source period the identified-customer repeat share is 65.58%, average invoice value is £533.17, and median identified customer order count is 2. Revenue is concentrated in the UK (£9,001,744.09, approximately 84.6% of positive-sales revenue). “DOTCOM POSTAGE” has the highest line revenue (£206,248.77); it should be interpreted as a shipping-related line rather than a merchandise recommendation.

Silhouette selected two clusters (0.433 versus 0.337 for k=3). Their mean bootstrap ARI was 0.980. The derived “Champions” cluster contains 1,668 customers (38.5%), with median recency 17 days, mean frequency 8.44 invoices, and mean historical revenue £4,535.74. “Needs Attention” contains 2,670 (61.5%), with median recency 96 days, mean frequency 1.67, and mean revenue £494.98. The k=4 secondary profiles were: Needs Attention (1,177 customers; median recency 54 days; mean 4.12 orders; mean revenue £1,793.44), Potential Loyalists (889; 20 days; 1.96 orders; £493.28), Champions (731; 9 days; 13.53 orders; £7,923.97), and Hibernating Customers (1,541; 186 days; 1.33 orders; £353.92). k=4 silhouette was 0.334 and mean bootstrap ARI 0.884. Its extra granularity may support more specific hypotheses but is less stable and more operationally complex than k=2.

The repeat-purchase test contains 3,007 customer snapshots, with a 53.14% positive rate. Validation PR-AUC selected XGBoost (0.7326). Test results and stratified bootstrap 95% CIs were:

| Method | F1 (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---|---|---|
| Logistic Regression | 0.690 (0.677–0.703) | 0.715 (0.697–0.731) | 0.762 (0.746–0.778) |
| Random Forest | 0.686 (0.671–0.699) | 0.700 (0.681–0.719) | 0.754 (0.736–0.771) |
| XGBoost | 0.695 (0.681–0.708) | 0.716 (0.698–0.734) | 0.762 (0.744–0.778) |
| Recency baseline | 0.687 (0.674–0.699) | 0.613 (0.592–0.633) | 0.604 (0.584–0.625) |
| RFM-score baseline | 0.684 (0.669–0.696) | 0.640 (0.620–0.660) | 0.677 (0.657–0.696) |

Rolling-origin mean ± standard deviation PR-AUC across three earlier cutoffs: Logistic Regression 0.745 ± 0.018; Random Forest 0.742 ± 0.013; XGBoost 0.752 ± 0.014; recency 0.595 ± 0.019; RFM score 0.661 ± 0.019.

XGBoost test confusion matrix at validation-selected threshold 0.359 is `[[638, 771], [337, 1261]]` (actual rows: no repeat, repeat; predicted columns: no repeat, repeat). It has relatively high repeat recall but many false-positive repeat predictions. Logistic Regression's PR-AUC is almost identical to XGBoost's; the single-cutoff results do not establish a reliable winner. The most influential global SHAP feature by mean absolute attribution was `active_days_180d` (counted as distinct calendar days), followed by `units_180d`; SHAP is model attribution, not causal effect.

The highest retained Apriori rule has support 0.64%, confidence 83.33%, and lift 15.56. It associates a pair of products in the antecedent with WOODLAND CHARLOTTE BAG. The top customer revenue-exposure proxy is £33,819.33 for customer 15098 at a 15.3% predicted 60-day repeat probability. This proxy is historical revenue scaled by estimated no-repeat probability, not projected lost sales.
