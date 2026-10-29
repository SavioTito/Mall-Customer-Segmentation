"""Customer segmentation with K-Means clustering.

This module contains the data-mining logic used by main.py: loading the
dataset, finding a good number of clusters (elbow method and silhouette
score), fitting K-Means, plotting the results, and profiling each segment.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES = ["Annual_Income_k$", "Spending_Score"]
PALETTE = ["#3b6ea5", "#c44e52", "#55a868", "#8172b3", "#ccb974", "#64b5cd"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the customer dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def elbow_and_silhouette(X, k_range=range(1, 11)) -> Tuple[List[float], List[float]]:
    """Compute within-cluster sum of squares (WCSS) and silhouette scores.

    Returns two lists: wcss for every k, and silhouette for k >= 2.
    """
    wcss, sil = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_) if k >= 2 else float("nan"))
    return wcss, sil


def run_kmeans(X, k: int) -> KMeans:
    """Fit a K-Means model with k clusters."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X)
    return km


def plot_elbow(k_range, wcss, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(list(k_range), wcss, marker="o", color="#3b6ea5")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS (inertia)")
    ax.set_title("Elbow method for choosing k")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_clusters(df, labels, model, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(df.loc[m, FEATURES[0]], df.loc[m, FEATURES[1]],
                   s=42, color=PALETTE[c % len(PALETTE)],
                   label=f"Cluster {c + 1}", alpha=0.8,
                   edgecolors="white", linewidths=0.4)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
               s=220, marker="X", c="black", label="Centroids", zorder=5)
    ax.set_xlabel("Annual income (k$)")
    ax.set_ylabel("Spending score (1-100)")
    ax.set_title("Customer segments (K-Means)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def profile_clusters(df, labels) -> pd.DataFrame:
    """Return a summary table with the average profile of each segment."""
    tmp = df.copy()
    tmp["Cluster"] = labels
    profile = (tmp.groupby("Cluster")
                  .agg(Customers=("CustomerID", "count"),
                       Avg_Age=("Age", "mean"),
                       Avg_Income=("Annual_Income_k$", "mean"),
                       Avg_Spending=("Spending_Score", "mean"))
                  .round(1))
    profile.index = profile.index + 1
    return profile
