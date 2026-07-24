"""
segmentation.py
----------------
Customer segmentation using K-Means Clustering.

Includes:
    - Elbow Method (inertia vs. k)
    - Silhouette Score
    - Final cluster assignment
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_elbow_curve(X_scaled: np.ndarray, k_range=range(2, 11)):
    """
    Fit K-Means for each k in k_range and return the inertia (within-cluster
    sum of squares) for each k, used to plot the Elbow Curve.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    return list(k_range), inertias


def compute_silhouette_scores(X_scaled: np.ndarray, k_range=range(2, 11)):
    """
    Compute the silhouette score for each k in k_range.
    """
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    return list(k_range), scores


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    """
    Train the final K-Means model with the chosen number of clusters.
    Returns the fitted model, cluster labels, and the silhouette score.
    """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    return kmeans, labels, sil_score


def get_best_k(k_range, silhouette_scores):
    """Return the k that yields the highest silhouette score."""
    best_index = int(np.argmax(silhouette_scores))
    return list(k_range)[best_index]


def assign_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels to the original dataframe."""
    df = df.copy()
    df["Cluster"] = labels
    return df


def cluster_summary(df_with_clusters: pd.DataFrame, features: list) -> pd.DataFrame:
    """Return mean feature values and customer counts per cluster."""
    summary = df_with_clusters.groupby("Cluster")[features].mean().round(2)
    summary["Customer_Count"] = df_with_clusters.groupby("Cluster").size()
    return summary.reset_index()
