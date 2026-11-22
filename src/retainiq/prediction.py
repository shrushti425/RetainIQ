"""Leakage-aware 60-day repeat-purchase prediction with chronological evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("retainiq.prediction")
FEATURES = ["orders_180d", "revenue_180d", "units_180d", "avg_order_value_180d", "recency_days", "active_days_180d", "product_diversity_180d", "revenue_recent_30d", "revenue_prior_30d"]
TRAIN_CUTOFFS = ["2011-05-31"]
VALIDATION_CUTOFFS = ["2011-07-31"]
TEST_CUTOFFS = ["2011-10-10"]


def make_snapshot(orders: pd.DataFrame, lines: pd.DataFrame, cutoff: str, obs_days: int = 180, horizon_days: int = 60) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(cutoff).normalize() + pd.Timedelta(1, unit="D") - pd.Timedelta(1, unit="us")
    start = cutoff_ts - pd.Timedelta(obs_days, unit="D")
    label_end = cutoff_ts + pd.Timedelta(horizon_days, unit="D")
    known_orders = orders.dropna(subset=["CustomerID"]).copy()
    known_orders["InvoiceDate"] = pd.to_datetime(known_orders["InvoiceDate"])
    known_orders["CustomerID"] = known_orders["CustomerID"].map(_customer_key)
    observed = known_orders.loc[known_orders["InvoiceDate"].between(start, cutoff_ts)]
    future = known_orders.loc[known_orders["InvoiceDate"].gt(cutoff_ts) & known_orders["InvoiceDate"].le(label_end)]
    if observed.empty:
        return pd.DataFrame(columns=["CustomerID", "cutoff", "label"] + FEATURES)
    agg = observed.groupby("CustomerID").agg(
        orders_180d=("InvoiceNo", "nunique"), revenue_180d=("revenue", "sum"),
        units_180d=("units", "sum"), last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
    )
    active_days = observed.assign(activity_day=observed["InvoiceDate"].dt.normalize()).groupby("CustomerID")["activity_day"].nunique()
    agg["active_days_180d"] = active_days
    agg["avg_order_value_180d"] = agg["revenue_180d"] / agg["orders_180d"].clip(lower=1)
    agg["recency_days"] = (cutoff_ts - agg["last_purchase"]).dt.total_seconds() / 86400
    agg["customer_age_180d"] = (cutoff_ts - agg["first_purchase"]).dt.total_seconds() / 86400
    line_dates = pd.to_datetime(lines["InvoiceDate"])
    line_slice = lines.loc[line_dates.between(start, cutoff_ts) & lines["CustomerID"].notna()].copy()
    line_slice["CustomerID"] = line_slice["CustomerID"].map(_customer_key)
    diversity = line_slice.groupby("CustomerID")["StockCode"].nunique()
    agg["product_diversity_180d"] = diversity
    recent_start = cutoff_ts - pd.Timedelta(30, unit="D")
    recent = observed.loc[observed["InvoiceDate"].gt(recent_start)].groupby("CustomerID")["revenue"].sum()
    prior_start = cutoff_ts - pd.Timedelta(60, unit="D")
    prior = observed.loc[observed["InvoiceDate"].between(prior_start, recent_start, inclusive="left")].groupby("CustomerID")["revenue"].sum()
    agg["revenue_recent_30d"] = recent
    agg["revenue_prior_30d"] = prior
    agg[["product_diversity_180d", "revenue_recent_30d", "revenue_prior_30d"]] = agg[["product_diversity_180d", "revenue_recent_30d", "revenue_prior_30d"]].fillna(0)
    bought = set(future["CustomerID"])
    agg = agg.reset_index()
    agg["CustomerID"] = agg["CustomerID"].map(_customer_key)
    agg["cutoff"] = cutoff_ts.date().isoformat()
    agg["label"] = agg["CustomerID"].isin(bought).astype(int)
    return agg[["CustomerID", "cutoff", "label"] + FEATURES]


def _customer_key(value) -> str:
    return str(int(float(value)))


def _select_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def _add_rfm_score(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Add a conventional 3-digit RFM score independently within each cutoff."""
    scored = snapshots.copy()
    scored["rfm_score"] = np.nan
    for _, idx in scored.groupby("cutoff").groups.items():
        group = scored.loc[idx]
        recency = pd.qcut(group["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        frequency = pd.qcut(group["orders_180d"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        monetary = pd.qcut(group["revenue_180d"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        scored.loc[idx, "rfm_score"] = (recency * 100 + frequency * 10 + monetary).to_numpy()
    return scored


def _metrics(y: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    pred = (probability >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y, probability)) if y.nunique() > 1 else None,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def _fit_validation_calibrator(validation_probability: np.ndarray, validation_labels: pd.Series, seed: int = 42) -> LogisticRegression:
    """Fit sigmoid calibration on validation predictions only."""
    clipped = np.clip(np.asarray(validation_probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=1e6, solver="lbfgs", random_state=seed).fit(logits, validation_labels)


def _apply_platt_calibrator(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _reliability_rows(labels: pd.Series, probability: np.ndarray, name: str, bins: int = 10) -> list[dict]:
    observed, predicted = calibration_curve(labels, probability, n_bins=bins, strategy="quantile")
    return [{"calibration": name, "mean_predicted_probability": float(x), "observed_fraction": float(y), "bin": i + 1} for i, (x, y) in enumerate(zip(predicted, observed))]


def bootstrap_metric_cis(labels, scores, threshold: float, n_resamples: int = 1000, seed: int = 42) -> dict:
    """Stratified percentile bootstrap CIs for ROC-AUC, average precision, and F1."""
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    if len(y) != len(score) or np.unique(y).size != 2:
        raise ValueError("Bootstrap metrics need equal-length scores and labels containing both classes.")
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == label) for label in (0, 1)]
    draws = {metric: np.empty(n_resamples, dtype=float) for metric in ("roc_auc", "pr_auc", "f1")}
    for i in range(n_resamples):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in by_class])
        sample_y, sample_score = y[sampled], score[sampled]
        draws["roc_auc"][i] = roc_auc_score(sample_y, sample_score)
        draws["pr_auc"][i] = average_precision_score(sample_y, sample_score)
        draws["f1"][i] = f1_score(sample_y, sample_score >= threshold, zero_division=0)
    point = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, score >= threshold, zero_division=0)),
    }
    result = {}
    for metric, values in draws.items():
        result[metric] = {"estimate": point[metric], "ci_95": [float(v) for v in np.quantile(values, [0.025, 0.975])], "resamples": n_resamples}
    return result


def _rolling_origin(orders: pd.DataFrame, lines: pd.DataFrame, factories: dict, seed: int) -> dict:
    """Evaluate 3 prior monthly cutoffs, training only on fully observed historical labels."""
    eval_cutoffs = ["2011-07-31", "2011-08-31", "2011-09-30"]
    candidate_train_cutoffs = ["2011-03-31", "2011-04-30", "2011-05-31", "2011-06-30", "2011-07-31"]
    scores: dict[str, list[float]] = {name: [] for name in [*factories, "Recency baseline", "RFM score baseline"]}
    by_origin = []
    for cutoff in eval_cutoffs:
        eligible = _eligible_training_cutoffs(cutoff, candidate_train_cutoffs, horizon_days=60)
        training = pd.concat([make_snapshot(orders, lines, c) for c in eligible], ignore_index=True)
        validation = make_snapshot(orders, lines, cutoff)
        if training.empty or training["label"].nunique() < 2 or validation.empty or validation["label"].nunique() < 2:
            raise ValueError(f"Rolling-origin split at {cutoff} lacks both classes or eligible historical data.")
        training = _add_rfm_score(training)
        validation = _add_rfm_score(validation)
        origin_result = {"cutoff": cutoff, "training_cutoffs": eligible, "models": {}}
        positive_weight = float((training["label"].eq(0).sum()) / max(training["label"].eq(1).sum(), 1))
        for model_i, (name, factory) in enumerate(factories.items()):
            model = factory() if name != "XGBoost" else _xgboost_pipeline(positive_weight, seed)
            model.fit(training[FEATURES], training["label"])
            probability = model.predict_proba(validation[FEATURES])[:, 1]
            value = float(average_precision_score(validation["label"], probability))
            scores[name].append(value)
            origin_result["models"][name] = value
        for name in ["Recency baseline", "RFM score baseline"]:
            value = float(average_precision_score(validation["label"], _baseline_score(validation, name)))
            scores[name].append(value)
            origin_result["models"][name] = value
        by_origin.append(origin_result)
    summary = {name: {"mean_pr_auc": float(np.mean(values)), "std_pr_auc": float(np.std(values, ddof=1)), "min_pr_auc": float(np.min(values)), "max_pr_auc": float(np.max(values)), "origins": values} for name, values in scores.items()}
    return {"evaluation_cutoffs": eval_cutoffs, "by_origin": by_origin, "summary": summary, "spread_definition": "sample standard deviation across 3 rolling-origin PR-AUC values"}


def _eligible_training_cutoffs(evaluation_cutoff: str, candidates: list[str], horizon_days: int = 60) -> list[str]:
    """Return train snapshot cutoffs whose complete outcomes precede evaluation."""
    evaluation_ts = pd.Timestamp(evaluation_cutoff)
    return [cutoff for cutoff in candidates if pd.Timestamp(cutoff) + pd.Timedelta(horizon_days, unit="D") <= evaluation_ts]


def _baseline_score(frame: pd.DataFrame, baseline: str) -> np.ndarray:
    """Higher scores indicate higher expected repeat-purchase rank."""
    if baseline == "Recency baseline":
        return -frame["recency_days"].to_numpy(dtype=float)
    if baseline == "RFM score baseline":
        return frame["rfm_score"].to_numpy(dtype=float)
    raise ValueError(f"Unknown ranking baseline: {baseline}")


def train_and_evaluate(orders: pd.DataFrame, lines: pd.DataFrame, output_dir: Path, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = pd.concat([
        make_snapshot(orders, lines, cutoff) for cutoff in TRAIN_CUTOFFS + VALIDATION_CUTOFFS + TEST_CUTOFFS
    ], ignore_index=True)
    snapshots = _add_rfm_score(snapshots)
    train = snapshots[snapshots["cutoff"].isin(TRAIN_CUTOFFS)]
    valid = snapshots[snapshots["cutoff"].isin(VALIDATION_CUTOFFS)]
    test = snapshots[snapshots["cutoff"].isin(TEST_CUTOFFS)]
    if train["label"].nunique() < 2 or valid["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise ValueError("Temporal split needs both classes in train, validation, and test; inspect generated label counts.")
    positive_weight = (train["label"].eq(0).sum() / max(train["label"].eq(1).sum(), 1))
    factories = {
        "Logistic Regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        "Random Forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=4, random_state=seed)),
        "XGBoost": lambda: _xgboost_pipeline(positive_weight, seed),
    }
    validation_scores = {}
    validation_probabilities = {}
    thresholds = {}
    for name, factory in factories.items():
        model = factory()
        model.fit(train[FEATURES], train["label"])
        probability = model.predict_proba(valid[FEATURES])[:, 1]
        validation_probabilities[name] = probability
        validation_scores[name] = float(average_precision_score(valid["label"], probability))
        thresholds[name] = _select_threshold(valid["label"], probability)
    baseline_specs = {
        "Recency baseline": ("recency_days", -1),
        "RFM score baseline": ("rfm_score", 1),
    }
    for name in baseline_specs:
        validation_rank = _baseline_score(valid, name)
        validation_scores[name] = float(average_precision_score(valid["label"], validation_rank))
        thresholds[name] = _select_threshold(valid["label"], validation_rank)
    selected = max(factories, key=validation_scores.get)
    calibrator = _fit_validation_calibrator(validation_probabilities[selected], valid["label"], seed)
    validation_brier_raw = float(brier_score_loss(valid["label"], validation_probabilities[selected]))
    validation_probability_calibrated = _apply_platt_calibrator(calibrator, validation_probabilities[selected])
    validation_brier_calibrated = float(brier_score_loss(valid["label"], validation_probability_calibrated))
    # Calibration use is selected using validation only; the held-out test is diagnostic.
    use_calibrated = validation_brier_calibrated < validation_brier_raw
    # Refit on train + validation only after choosing model family and threshold on validation.
    train_valid = pd.concat([train, valid], ignore_index=True)
    results = []
    test_curves = []
    reliability = []
    prediction_rows = []
    final_models = {}
    for name, factory in factories.items():
        model = factory()
        model.fit(train_valid[FEATURES], train_valid["label"])
        probability = model.predict_proba(test[FEATURES])[:, 1]
        metrics = _metrics(test["label"], probability, thresholds[name])
        metrics["bootstrap_ci_95"] = bootstrap_metric_cis(test["label"], probability, thresholds[name], n_resamples=1000, seed=seed + len(results))
        if name == selected:
            calibrated_probability = _apply_platt_calibrator(calibrator, probability)
            raw_test_brier = float(brier_score_loss(test["label"], probability))
            calibrated_test_brier = float(brier_score_loss(test["label"], calibrated_probability))
            effective_probability = calibrated_probability if use_calibrated else probability
            metrics["brier_score"] = raw_test_brier
            metrics["calibrated_brier_score"] = calibrated_test_brier
            metrics["effective_brier_score"] = float(brier_score_loss(test["label"], effective_probability))
            reliability.extend(_reliability_rows(test["label"], probability, "Uncalibrated"))
            reliability.extend(_reliability_rows(test["label"], calibrated_probability, "Platt calibrated"))
        else:
            effective_probability = probability
            metrics["brier_score"] = float(brier_score_loss(test["label"], probability))
        metrics.update({"model": name, "validation_pr_auc": validation_scores[name], "test_rows": len(test), "test_positive_rate": float(test["label"].mean())})
        results.append(metrics)
        fpr, tpr, _ = roc_curve(test["label"], probability)
        pr_precision, pr_recall, _ = precision_recall_curve(test["label"], probability)
        test_curves.extend([{"model": name, "curve": "roc", "x": float(x), "y": float(y)} for x, y in zip(fpr, tpr)])
        test_curves.extend([{"model": name, "curve": "precision_recall", "x": float(x), "y": float(y)} for x, y in zip(pr_recall, pr_precision)])
        preds = test[["CustomerID", "cutoff", "label"]].copy()
        preds["model"] = name
        preds["probability_repeat_60d_raw"] = probability
        preds["probability_repeat_60d"] = effective_probability
        preds["risk_band"] = pd.cut(probability, [-0.01, 0.3, 0.6, 1.01], labels=["Higher repeat likelihood", "Uncertain", "Lower repeat likelihood"])
        prediction_rows.append(preds)
        final_models[name] = model
        joblib.dump(model, output_dir / (name.lower().replace(" ", "_") + ".joblib"))
    for name in baseline_specs:
        score = _baseline_score(test, name)
        metrics = _metrics(test["label"], score, thresholds[name])
        metrics["bootstrap_ci_95"] = bootstrap_metric_cis(test["label"], score, thresholds[name], n_resamples=1000, seed=seed + len(results))
        metrics.update({"model": name, "model_type": "baseline", "validation_pr_auc": validation_scores[name], "test_rows": len(test), "test_positive_rate": float(test["label"].mean())})
        results.append(metrics)
        fpr, tpr, _ = roc_curve(test["label"], score)
        pr_precision, pr_recall, _ = precision_recall_curve(test["label"], score)
        test_curves.extend([{"model": name, "curve": "roc", "x": float(x), "y": float(y)} for x, y in zip(fpr, tpr)])
        test_curves.extend([{"model": name, "curve": "precision_recall", "x": float(x), "y": float(y)} for x, y in zip(pr_recall, pr_precision)])
        preds = test[["CustomerID", "cutoff", "label"]].copy()
        preds["model"] = name
        preds["probability_repeat_60d"] = np.nan
        preds["ranking_score"] = score
        preds["risk_band"] = pd.cut(pd.Series(score, index=test.index).rank(pct=True), [-0.01, 0.3, 0.6, 1.01], labels=["Lower rank", "Middle rank", "Higher rank"])
        prediction_rows.append(preds)
    pd.DataFrame(results).to_json(output_dir / "model_metrics.json", orient="records", indent=2)
    pd.DataFrame(reliability).to_csv(output_dir / "reliability_curve.csv", index=False)
    joblib.dump(calibrator, output_dir / "selected_model_platt_calibrator.joblib")
    pd.DataFrame(test_curves).to_csv(output_dir / "model_curves.csv", index=False)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    snapshots.to_csv(output_dir / "prediction_snapshots.csv", index=False)
    # Explain the validation-selected model on an interpretable, deterministic test subset.
    selected_model = final_models[selected]
    sample = test.sample(min(750, len(test)), random_state=seed)
    imputer = selected_model.named_steps["simpleimputer"]
    transformed = imputer.transform(sample[FEATURES])
    estimator = selected_model.steps[-1][1]
    import shap
    if selected == "Logistic Regression":
        background = train_valid[FEATURES].sample(min(500, len(train_valid)), random_state=seed)
        background_scaled = selected_model.named_steps["standardscaler"].transform(imputer.transform(background))
        explainer = shap.LinearExplainer(estimator, background_scaled)
        shap_values = explainer(selected_model.named_steps["standardscaler"].transform(transformed)).values
    else:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        if np.ndim(shap_values) == 3:
            shap_values = shap_values[:, :, -1]
    shap_values = np.asarray(shap_values)
    shap_frame = pd.DataFrame(shap_values, columns=FEATURES)
    shap_frame.insert(0, "CustomerID", sample["CustomerID"].values)
    shap_frame.insert(1, "cutoff", sample["cutoff"].values)
    shap_frame.to_csv(output_dir / "individual_shap.csv", index=False)
    importance = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "shap_importance.csv", index=False)
    report = {
        "problem": "Among customers with at least one order in the prior 180 days, predict any positive sale invoice in the next 60 days.",
        "observation_days": 180,
        "prediction_horizon_days": 60,
        "train_cutoffs": TRAIN_CUTOFFS,
        "validation_cutoffs": VALIDATION_CUTOFFS,
        "test_cutoffs": TEST_CUTOFFS,
        "test_window_end": "2011-12-09 (source end date)",
        "feature_columns": FEATURES,
        "selected_model_by_validation_pr_auc": selected,
        "selected_validation_pr_auc": validation_scores[selected],
        "calibration": {"method": "Platt sigmoid", "fit_split": "validation only", "decision_split": "validation only", "validation_brier_uncalibrated": validation_brier_raw, "validation_brier_calibrated": validation_brier_calibrated, "test_brier_uncalibrated": next(row["brier_score"] for row in results if row["model"] == selected), "test_brier_calibrated": next(row["calibrated_brier_score"] for row in results if row["model"] == selected), "use_calibrated_probabilities_for_exposure": bool(use_calibrated)},
        "rolling_origin": _rolling_origin(orders, lines, factories, seed),
        "class_imbalance_train": {"positive": int(train["label"].sum()), "negative": int((train["label"] == 0).sum()), "positive_weight_xgb": float(positive_weight)},
        "snapshot_rows": {"train": int(len(train)), "validation": int(len(valid)), "test": int(len(test))},
        "limitations": ["Repeated customer snapshots are allowed across dates and are evaluated as future scoring occasions; customer ID is not a model feature.", "The test set is one final cutoff near the end of this short historical dataset.", "This dataset has no outreach or treatment data; model scores estimate observed purchasing, not intervention response or causal retention impact."],
    }
    (output_dir / "rolling_origin_metrics.json").write_text(json.dumps(report["rolling_origin"], indent=2) + "\n")
    (output_dir / "prediction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"report": report, "metrics": results}


def _xgboost_pipeline(positive_weight: float, seed: int):
    from xgboost import XGBClassifier
    return make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.9, reg_lambda=2, scale_pos_weight=positive_weight, eval_metric="logloss", n_jobs=4, random_state=seed))
