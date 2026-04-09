"""
RetailMiner — Model Training Module
=====================================
Trains four analytical models:
  A. Isolation Forest  — anomaly detection
  B. K-Means           — customer segmentation
  C. Apriori           — market basket analysis
  D. Temporal Mining   — peak hours, monthly trends, revenue spikes
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# A. ANOMALY DETECTION — Isolation Forest
# ═══════════════════════════════════════════════════════════════════════════

def train_anomaly_detector(
    invoice_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect anomalous invoices using Isolation Forest.

    Features: InvoiceTotal, ItemCount, Hour

    Returns
    -------
    pd.DataFrame with original invoice columns + AnomalyLabel + AnomalyScore + RiskLevel
    """
    features = ["InvoiceTotal", "ItemCount", "Hour"]
    X = invoice_df[features].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    invoice_df = invoice_df.copy()
    invoice_df["AnomalyLabel"] = model.predict(X_scaled)          # -1 = anomaly, 1 = normal
    invoice_df["AnomalyScore"]  = -model.score_samples(X_scaled)  # higher = more anomalous

    # Normalise score to [0, 1]
    s_min, s_max = invoice_df["AnomalyScore"].min(), invoice_df["AnomalyScore"].max()
    if s_max > s_min:
        invoice_df["AnomalyScore"] = (invoice_df["AnomalyScore"] - s_min) / (s_max - s_min)

    # Risk level buckets
    invoice_df["RiskLevel"] = pd.cut(
        invoice_df["AnomalyScore"],
        bins=[0, 0.4, 0.7, 1.01],
        labels=["Low", "Medium", "High"],
    )

    anomalies = invoice_df[invoice_df["AnomalyLabel"] == -1].copy()
    logger.info("Anomaly detection: %d anomalies found (%.1f%%)", len(anomalies), 100 * len(anomalies) / len(invoice_df))

    # Persist
    joblib.dump({"model": model, "scaler": scaler}, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    # Return BOTH the full scored invoice_df AND the filtered anomalies subset
    return invoice_df.reset_index(drop=True), anomalies.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# B. CUSTOMER SEGMENTATION — K-Means
# ═══════════════════════════════════════════════════════════════════════════

def _optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> int:
    """Elbow method to find optimal number of clusters."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Simple elbow: largest second-derivative
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    if len(diffs2) == 0:
        return 4
    elbow_idx = np.argmax(diffs2) + 2   # +2 because k_range starts at 2
    return list(k_range)[elbow_idx]


def train_segmentation(
    rfm_df: pd.DataFrame,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Segment customers using K-Means on RFM features.

    Returns
    -------
    pd.DataFrame with CustomerID, RFM features, Cluster label, ClusterName
    """
    features = ["Recency", "Frequency", "Monetary", "AvgSpend"]
    X = rfm_df[features].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_clusters is None:
        n_clusters = _optimal_k(X_scaled)
        logger.info("Optimal clusters determined by elbow: %d", n_clusters)
    else:
        logger.info("Using user-specified clusters: %d", n_clusters)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(X_scaled)

    result = rfm_df.copy()
    result["Cluster"] = labels

    # Name clusters by average Monetary value
    cluster_means = result.groupby("Cluster")["Monetary"].mean().sort_values()
    tier_names = ["Budget Shoppers", "Casual Buyers", "Regular Customers", "Premium Loyalists"]
    # Extend tier names if more clusters
    while len(tier_names) < n_clusters:
        tier_names.append(f"Segment {len(tier_names) + 1}")
    tier_map = {c: tier_names[i] for i, c in enumerate(cluster_means.index)}
    result["ClusterName"] = result["Cluster"].map(tier_map)

    logger.info("Segmentation: %d clusters, %d customers", n_clusters, len(result))

    joblib.dump({"model": model, "scaler": scaler, "n_clusters": n_clusters}, os.path.join(MODELS_DIR, "kmeans.pkl"))
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# C. MARKET BASKET ANALYSIS — Apriori
# ═══════════════════════════════════════════════════════════════════════════

def run_market_basket(
    df: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    min_lift: float = 1.0,
    max_items: int = 5,
) -> pd.DataFrame:
    """
    Perform market basket analysis using Apriori.

    Returns
    -------
    pd.DataFrame of association rules with antecedents, consequents,
    support, confidence, lift, conviction columns.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError as e:
        logger.error("mlxtend not installed: %s", e)
        return pd.DataFrame()

    # Build basket: unique items per invoice
    basket = df.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack(fill_value=0)
    basket = (basket > 0).astype(bool)

    # Drop products with description that look like codes
    basket = basket[[c for c in basket.columns if isinstance(c, str) and len(c) > 3]]

    logger.info("Basket matrix: %d invoices × %d products", *basket.shape)

    # Frequent itemsets
    try:
        frequent_items = apriori(basket, min_support=min_support, use_colnames=True, max_len=max_items)
    except Exception as e:
        logger.warning("Apriori failed — lowering min_support: %s", e)
        frequent_items = apriori(basket, min_support=0.005, use_colnames=True, max_len=3)

    if frequent_items.empty:
        logger.warning("No frequent itemsets found; try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_confidence]

    # Stringify frozensets for display
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # Add conviction label
    conditions = [rules["lift"] >= 3, rules["lift"] >= 2, rules["lift"] >= 1]
    labels = ["Strong 🔥", "Moderate ✅", "Weak ⚠️"]
    rules["RuleStrength"] = np.select(conditions, labels, default="Weak ⚠️")

    logger.info("Market basket: %d frequent itemsets → %d rules", len(frequent_items), len(rules))
    return rules


# ═══════════════════════════════════════════════════════════════════════════
# D. TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_temporal_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Extract temporal patterns:
      - Peak shopping hours
      - Monthly revenue trends
      - Revenue spikes (Z-score > 2)

    Returns
    -------
    dict with keys: peak_hours, monthly_trends, revenue_spikes
    """
    # Peak hours
    peak_hours = (
        df.groupby("Hour")["TotalAmount"]
        .agg(TotalRevenue="sum", TransactionCount="count")
        .reset_index()
        .sort_values("Hour")
    )
    peak_hours["IsPeak"] = peak_hours["TotalRevenue"] > peak_hours["TotalRevenue"].quantile(0.75)

    # Monthly trends
    df["MonthYear"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("MonthYear")["TotalAmount"]
        .agg(TotalRevenue="sum", OrderCount="count")
        .reset_index()
        .sort_values("MonthYear")
    )
    monthly["MoM_Change"] = monthly["TotalRevenue"].pct_change() * 100

    # Revenue spikes — daily Z-score
    df["Date"] = df["InvoiceDate"].dt.date
    daily = df.groupby("Date")["TotalAmount"].sum().reset_index()
    daily.columns = ["Date", "DailyRevenue"]
    daily["Date"] = pd.to_datetime(daily["Date"])

    z_scores = np.abs(stats.zscore(daily["DailyRevenue"].fillna(0)))
    daily["ZScore"] = z_scores
    spikes = daily[daily["ZScore"] > 2].copy()
    spikes["Deviation"] = spikes["DailyRevenue"] - daily["DailyRevenue"].mean()

    logger.info(
        "Temporal: %d peak hours | %d monthly points | %d spikes",
        len(peak_hours), len(monthly), len(spikes),
    )
    return {
        "peak_hours": peak_hours,
        "monthly_trends": monthly,
        "revenue_spikes": spikes,
    }
