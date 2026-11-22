"""RFM scoring, K-Means selection, and data-driven segment descriptions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_customers(orders: pd.DataFrame, output_dir: Path | None = None, seed: int = 42, stability_resamples: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
    stability = _bootstrap_kmeans_stability(transformed, seed=seed, n_resamples=stability_resamples)
    secondary_model = KMeans(n_clusters=4, random_state=seed, n_init=20).fit(transformed)
    rfm["cluster_k4"] = secondary_model.labels_.astype(int)
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
    secondary_profiles = _profile_table(rfm, "cluster_k4", "cluster_k4")
    secondary_labels = _describe_clusters(secondary_profiles.rename(columns={"cluster_k4": "cluster"}))
    rfm["segment_k4"] = rfm["cluster_k4"].map(secondary_labels)
    secondary_profiles["segment_k4"] = secondary_profiles["cluster_k4"].map(secondary_labels)
    secondary_profiles["customer_share"] = secondary_profiles["customers"] / secondary_profiles["customers"].sum()
    tradeoff_note = (
        f"The silhouette-selected k={best_k} solution is the primary segmentation because it had the strongest silhouette score among the tested values. "
        "The four-cluster view is an exploratory secondary lens that can expose subgroups hidden inside broader profiles, at the cost of more complex targeting and potentially less stable assignments. "
        "Use the bootstrap Adjusted Rand Index results and profile sizes to judge whether the extra detail is useful; cluster labels are descriptive, not validated personas."
    )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        rfm.to_csv(output_dir / "segmented_customers.csv", index=False)
        profiles.to_csv(output_dir / "segment_profiles.csv", index=False)
        secondary_profiles.to_csv(output_dir / "secondary_k4_profiles.csv", index=False)
        (output_dir / "cluster_stability.json").write_text(json.dumps(stability, indent=2) + "\n")
        (output_dir / "cluster_evaluation.json").write_text(json.dumps({"selected_k": int(best_k), "evaluations": evaluation, "stability": stability, "secondary_k": 4, "secondary_tradeoff_note": tradeoff_note, "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()}, indent=2))
    metadata = {"as_of_date": as_of.date().isoformat(), "selected_k": int(best_k), "evaluation": evaluation, "stability": stability, "secondary_k": 4, "secondary_tradeoff_note": tradeoff_note, "segment_labels": labels}
    return rfm, profiles, metadata


def _profile_table(rfm: pd.DataFrame, cluster_column: str, label_column: str) -> pd.DataFrame:
    """Build comparable behavioral profiles for any fitted cluster solution."""
    return rfm.groupby(cluster_column).agg(
        customers=("CustomerID", "size"), median_recency_days=("recency", "median"),
        mean_orders=("frequency", "mean"), median_orders=("frequency", "median"),
        mean_revenue=("monetary", "mean"), median_revenue=("monetary", "median"),
    ).reset_index().rename(columns={cluster_column: label_column})


def _bootstrap_kmeans_stability(transformed: np.ndarray, seed: int = 42, n_resamples: int = 100) -> dict:
    """Compare bootstrap-fitted partitions with full-sample partitions using ARI."""
    rng = np.random.default_rng(seed)
    results = {}
    for k in range(2, 7):
        reference = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(transformed).labels_
        scores = []
        for _ in range(n_resamples):
            indices = rng.integers(0, len(transformed), size=len(transformed))
            try:
                bootstrap = KMeans(n_clusters=k, random_state=int(rng.integers(0, 2**31 - 1)), n_init=10).fit(transformed[indices])
            except ValueError:
                continue
            projected_labels = bootstrap.predict(transformed[indices])
            scores.append(float(adjusted_rand_score(reference[indices], projected_labels)))
        results[str(k)] = {
            "mean_adjusted_rand_index": float(np.mean(scores)) if scores else float("nan"),
            "std_adjusted_rand_index": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "successful_resamples": len(scores),
            "requested_resamples": n_resamples,
        }
    return {"metric": "Adjusted Rand Index on bootstrap-sampled customers versus the full-sample partition", "seed": seed, "solutions": results}


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
