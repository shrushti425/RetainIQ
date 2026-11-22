"""RFM scoring, K-Means selection, and data-driven segment descriptions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_customers(orders: pd.DataFrame, output_dir: Path | None = None, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    orders = orders.dropna(subset=["CustomerID"]).copy()
    orders["InvoiceDate"] = pd.to_datetime(orders["InvoiceDate"])
    as_of = orders["InvoiceDate"].max().normalize() + pd.Timedelta(1, unit="D")
    rfm = orders.groupby("CustomerID", as_index=False).agg(
        last_purchase=("InvoiceDate", "max"), frequency=("InvoiceNo", "nunique"), monetary=("revenue", "sum")
    )
    rfm["recency"] = (as_of - rfm["last_purchase"].dt.normalize()).dt.days
    rfm["recency_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["monetary_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm[["recency_score", "frequency_score", "monetary_score"]].astype(str).agg("".join, axis=1)

    feature = np.log1p(rfm[["recency", "frequency", "monetary"]].astype(float))
    scaler = StandardScaler()
    transformed = scaler.fit_transform(feature)
    evaluation = []
    possible_k = range(2, min(9, len(rfm)))
    for k in possible_k:
        candidate = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(transformed)
        evaluation.append({
            "k": k,
            "inertia": float(candidate.inertia_),
            "silhouette": float(silhouette_score(transformed, candidate.labels_, sample_size=min(5000, len(rfm)), random_state=seed)),
            "davies_bouldin": float(davies_bouldin_score(transformed, candidate.labels_)),
        })
    # Choose by silhouette among 2-8 clusters; inspect resulting profiles before applying business names.
    best_k = max(evaluation, key=lambda row: row["silhouette"])["k"]
    model = KMeans(n_clusters=best_k, random_state=seed, n_init=20).fit(transformed)
    rfm["cluster"] = model.labels_.astype(int)
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(transformed)
    rfm["pca_1"], rfm["pca_2"] = coordinates[:, 0], coordinates[:, 1]
    profiles = rfm.groupby("cluster").agg(
        customers=("CustomerID", "size"), median_recency_days=("recency", "median"),
        mean_orders=("frequency", "mean"), median_orders=("frequency", "median"),
        mean_revenue=("monetary", "mean"), median_revenue=("monetary", "median"),
    ).reset_index()
    labels = _describe_clusters(profiles)
    rfm["segment"] = rfm["cluster"].map(labels)
    profiles["segment"] = profiles["cluster"].map(labels)
    profiles["customer_share"] = profiles["customers"] / profiles["customers"].sum()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        rfm.to_csv(output_dir / "segmented_customers.csv", index=False)
        profiles.to_csv(output_dir / "segment_profiles.csv", index=False)
        (output_dir / "cluster_evaluation.json").write_text(json.dumps({"selected_k": int(best_k), "evaluations": evaluation, "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()}, indent=2))
    metadata = {"as_of_date": as_of.date().isoformat(), "selected_k": int(best_k), "evaluation": evaluation, "segment_labels": labels}
    return rfm, profiles, metadata


def _describe_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """Assign interpretable descriptions from observed within-cluster ranks."""
    ranked = profiles.copy()
    ranked["recency_rank"] = ranked["median_recency_days"].rank(pct=True, method="average")
    ranked["frequency_rank"] = ranked["mean_orders"].rank(pct=True, method="average")
    ranked["revenue_rank"] = ranked["mean_revenue"].rank(pct=True, method="average")
    result: dict[int, str] = {}
    for row in ranked.itertuples():
        high_value = row.frequency_rank >= 0.67 and row.revenue_rank >= 0.67
        recent = row.recency_rank <= 0.5
        if high_value and recent:
            label = "Champions" if row.frequency_rank >= 0.85 and row.revenue_rank >= 0.85 else "Loyal Customers"
        elif row.recency_rank >= 0.67 and row.frequency_rank <= 0.34:
            label = "Hibernating Customers" if row.recency_rank >= 0.85 else "At-Risk Customers"
        elif row.recency_rank <= 0.34 and row.frequency_rank <= 0.34:
            label = "New Customers"
        elif row.recency_rank <= 0.5 and row.frequency_rank >= 0.5:
            label = "Potential Loyalists"
        else:
            label = "Needs Attention"
        result[int(row.cluster)] = label
    return result
