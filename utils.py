"""
utils.py
--------
Shared plotting and persistence helper functions used by the
Streamlit dashboard and the training scripts.
"""

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

MODELS_DIR = "models"


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_pickle(obj, filename: str):
    """Save any Python object (model, scaler, encoder) to the models/ folder."""
    ensure_models_dir()
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename: str):
    """Load a pickled object from the models/ folder."""
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------- Plot helpers -----------------------------

def plot_elbow_curve(k_values, inertias):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, inertias, marker="o", color="#4C72B0", linewidth=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Method for Optimal k")
    fig.tight_layout()
    return fig


def plot_silhouette_scores(k_values, scores):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, scores, marker="o", color="#DD8452", linewidth=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by k")
    fig.tight_layout()
    return fig


def plot_cluster_scatter(df, x_col, y_col, cluster_col="Cluster"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette="viridis", ax=ax, alpha=0.75)
    ax.set_title(f"Customer Segments: {x_col} vs {y_col}")
    fig.tight_layout()
    return fig


def plot_cluster_counts(df, cluster_col="Cluster"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df[cluster_col].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count per Cluster")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.5, color="#4C72B0")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Prediction")
    ax.set_xlabel("Actual CLV")
    ax.set_ylabel("Predicted CLV")
    ax.set_title("Actual vs Predicted CLV")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_pred, residuals):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="#55A868")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted CLV")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    return fig
