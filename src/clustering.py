"""
Smart Crop Recommendation System - Soil & Climate K-Means Clustering
====================================================================
This script performs unsupervised clustering on the 7 soil and climate features
using StandardScaler and KMeans (k=4).
It computes the silhouette score, performs PCA dimensionality reduction (2D)
for visualization, extracts cluster profiles in original units, and saves all artifacts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def load_data(data_path="data/Crop_recommendation.csv"):
    """
    Load dataset and extract the seven numerical soil & climate features.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    print(f"[1/5] Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    X = df[feature_cols]
    print(f"      Loaded {len(df)} records with features: {feature_cols}")
    return df, X, feature_cols


def perform_clustering(X, n_clusters=4, random_state=42, n_init=10):
    """
    Scale features using StandardScaler, fit KMeans (k=4), and compute silhouette score.
    """
    print(f"[2/5] Standardizing features and fitting KMeans (n_clusters={n_clusters}, n_init={n_init})...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate Silhouette Score
    score = silhouette_score(X_scaled, cluster_labels)
    print(f"      Calculated Silhouette Score: {score:.4f}")

    return scaler, kmeans, X_scaled, cluster_labels, float(score)


def perform_pca_projection(X_scaled, random_state=42):
    """
    Fit PCA to project the 7-dimensional standardized feature space into 2D.
    """
    print("[3/5] Performing 2D PCA projection for visualization...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_.tolist()
    print(f"      PCA Explained Variance: Component 1 = {explained_var[0]*100:.2f}%, Component 2 = {explained_var[1]*100:.2f}% (Total: {sum(explained_var)*100:.2f}%)")
    return pca, X_pca, explained_var


def compute_cluster_profiles(df, cluster_labels, feature_cols, X_pca):
    """
    Compute cluster sizes, mean profiles in original feature units, and dominant crops.
    """
    print("[4/5] Computing cluster profiles in original units...")
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels
    df_clustered["pca_1"] = X_pca[:, 0]
    df_clustered["pca_2"] = X_pca[:, 1]

    # Cluster sizes
    sizes = df_clustered["cluster"].value_counts().sort_index().to_dict()

    # Profiles in original units (mean values)
    profiles = {}
    top_crops = {}

    for c in sorted(df_clustered["cluster"].unique()):
        sub_df = df_clustered[df_clustered["cluster"] == c]
        mean_vals = sub_df[feature_cols].mean().round(2).to_dict()
        profiles[str(c)] = mean_vals

        # Top 5 most frequent crops in this cluster
        crop_counts = sub_df["label"].value_counts().head(5).to_dict()
        top_crops[str(c)] = [{"crop": k, "count": int(v)} for k, v in crop_counts.items()]

    return df_clustered, sizes, profiles, top_crops


def save_clustering_artifacts(scaler, kmeans, pca, df_clustered, silhouette, sizes, profiles, top_crops, explained_var, feature_cols, output_dir="models"):
    """
    Save scaler, kmeans, pca, clustered data, and metrics JSON to models/ directory.
    """
    print(f"[5/5] Saving clustering artifacts to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save Scaler
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"      Saved Scaler : {scaler_path}")

    # 2. Save KMeans
    kmeans_path = os.path.join(output_dir, "kmeans.joblib")
    joblib.dump(kmeans, kmeans_path)
    print(f"      Saved KMeans : {kmeans_path}")

    # 3. Save PCA
    pca_path = os.path.join(output_dir, "pca.joblib")
    joblib.dump(pca, pca_path)
    print(f"      Saved PCA    : {pca_path}")

    # 4. Save Clustered Dataset for fast Streamlit rendering
    data_save_path = os.path.join("data", "clustered_crops.csv")
    df_clustered.to_csv(data_save_path, index=False)
    print(f"      Saved Clustered CSV : {data_save_path}")

    # 5. Save clustering metrics JSON
    metrics_path = os.path.join(output_dir, "clustering_metrics.json")
    metrics_payload = {
        "algorithm": "K-Means",
        "n_clusters": 4,
        "random_state": 42,
        "n_init": 10,
        "features": feature_cols,
        "silhouette_score": silhouette,
        "pca_explained_variance_ratio": explained_var,
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "cluster_profiles_original_units": profiles,
        "top_crops_per_cluster": top_crops,
        "disclaimer": "These clusters represent statistical data-driven patterns across the 7 environmental and soil features, not agronomically verified suitability categories."
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"      Saved Metrics JSON  : {metrics_path}\n")


def main():
    # 1. Load data
    df, X, feature_cols = load_data("data/Crop_recommendation.csv")

    # 2. Cluster
    scaler, kmeans, X_scaled, cluster_labels, score = perform_clustering(
        X, n_clusters=4, random_state=42, n_init=10
    )

    # 3. PCA
    pca, X_pca, explained_var = perform_pca_projection(X_scaled, random_state=42)

    # 4. Cluster profiles
    df_clustered, sizes, profiles, top_crops = compute_cluster_profiles(
        df, cluster_labels, feature_cols, X_pca
    )

    # 5. Save artifacts
    save_clustering_artifacts(
        scaler, kmeans, pca, df_clustered, score, sizes, profiles, top_crops, explained_var, feature_cols
    )

    print("=" * 65)
    print("CLUSTERING SUMMARY & ACTUAL SILHOUETTE SCORE:")
    print(f"  • Number of Clusters (k): 4")
    print(f"  • Silhouette Score      : {score:.4f}")
    print(f"  • Cluster Sizes (0-3)   : {sizes}")
    print("=" * 65)
    print("K-Means clustering completed successfully!")


if __name__ == "__main__":
    main()
