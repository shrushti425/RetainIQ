"""Apriori product association mining on invoice baskets."""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def apriori_rules(lines: pd.DataFrame, top_products: int = 75, min_support: float = 0.005, min_confidence: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = lines.dropna(subset=["InvoiceNo", "StockCode"]).copy()
    data["StockCode"] = data["StockCode"].astype(str)
    # Capping the candidate universe makes Apriori computationally practical and is explicit in report metadata.
    allowed = data.groupby("StockCode")["InvoiceNo"].nunique().nlargest(top_products).index
    data = data.loc[data["StockCode"].isin(allowed)]
    basket = pd.crosstab(data["InvoiceNo"], data["StockCode"]).gt(0)
    frequent = apriori(basket, min_support=min_support, use_colnames=True, max_len=3, low_memory=True)
    if frequent.empty:
        return frequent, pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules = rules.loc[(rules["lift"] > 1.0) & (rules["support"] >= min_support)].copy()
    desc = data.drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict()
    for column in ["antecedents", "consequents"]:
        rules[column + "_codes"] = rules[column].map(lambda values: " + ".join(sorted(values)))
        rules[column] = rules[column].map(lambda values: " + ".join(str(desc.get(code, code)) for code in sorted(values)))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    frequent["itemsets_codes"] = frequent["itemsets"].map(lambda values: " + ".join(sorted(values)))
    frequent["itemsets"] = frequent["itemsets"].map(lambda values: " + ".join(str(desc.get(code, code)) for code in sorted(values)))
    return frequent.sort_values("support", ascending=False), rules
