#!/usr/bin/env python3
"""Customer segmentation of mall customers using K-Means clustering.

Runs the full pipeline:
    1. Load the dataset.
    2. Use the elbow method and silhouette score to choose the number of clusters.
    3. Fit K-Means and assign every customer to a segment.
    4. Save the plots and print a profile of each segment.

Usage:
    python main.py
"""
from pathlib import Path

import segmentation as seg

DATA = Path("data/Mall_Customers.csv")
OUT = Path("outputs")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # 1. Load the data
    df = seg.load_data(DATA)
    X = df[seg.FEATURES]
    print(f"Loaded {len(df)} customers.")

    # 2. Choose k with the elbow method and silhouette score
    k_range = range(1, 11)
    wcss, sil = seg.elbow_and_silhouette(X, k_range)
    seg.plot_elbow(k_range, wcss, OUT / "elbow.png")
    best_k = max(range(2, 11), key=lambda k: sil[k - 1])
    print(f"Best k by silhouette score: {best_k} "
          f"(score = {sil[best_k - 1]:.3f})")

    # 3. Fit K-Means with the chosen k
    k = 5
    model = seg.run_kmeans(X, k)
    df["Cluster"] = model.labels_ + 1
    seg.plot_clusters(X, model.labels_, model, OUT / "clusters.png")

    # 4. Profile each segment
    profile = seg.profile_clusters(df, model.labels_)
    print("\nSegment profiles:")
    print(profile.to_string())

    df.to_csv(OUT / "customers_with_segments.csv", index=False)
    print("\nResults saved to the 'outputs' folder.")


if __name__ == "__main__":
    main()
