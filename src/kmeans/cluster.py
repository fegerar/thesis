"""
K-means++ clustering for shapegraph tokenization.

Fits k-means++ on flat shapegraph feature vectors, evaluates clustering
quality, and persists results (centroids + assignments).
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = 512,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    normalize: bool = True,
) -> tuple[KMeans, StandardScaler | None]:
    """Fit k-means++ on feature matrix.

    Args:
        X: (N, D) feature matrix
        n_clusters: number of clusters (codebook size)
        n_init: number of k-means++ initializations
        max_iter: max iterations per run
        random_state: seed
        normalize: whether to standardize features before clustering

    Returns:
        kmeans: fitted KMeans model
        scaler: fitted StandardScaler (or None if normalize=False)
    """
    scaler = None
    X_fit = X
    if normalize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    kmeans.fit(X_fit)

    print(f"K-means converged in {kmeans.n_iter_} iterations")
    print(f"Inertia: {kmeans.inertia_:.2f}")

    return kmeans, scaler


def assign_clusters(X: np.ndarray, kmeans: KMeans,
                    scaler: StandardScaler | None = None) -> np.ndarray:
    """Assign cluster labels to feature vectors.

    Returns:
        labels: (N,) int array of cluster indices (tokens)
    """
    X_use = scaler.transform(X) if scaler is not None else X
    return kmeans.predict(X_use)


def evaluate_clustering(X: np.ndarray, labels: np.ndarray,
                        kmeans: KMeans,
                        scaler: StandardScaler | None = None,
                        sample_size: int = 50_000) -> dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        X: raw feature matrix
        labels: cluster assignments
        kmeans: fitted model
        scaler: optional scaler
        sample_size: subsample for expensive metrics (silhouette)

    Returns:
        Dictionary of metric name -> value
    """
    X_use = scaler.transform(X) if scaler is not None else X
    n = len(X_use)

    metrics = {
        "inertia": kmeans.inertia_,
        "n_samples": n,
        "n_clusters_used": len(np.unique(labels)),
        "n_clusters_total": kmeans.n_clusters,
        "utilization": len(np.unique(labels)) / kmeans.n_clusters,
    }

    # Subsample for expensive metrics
    if n > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, sample_size, replace=False)
        X_sub, labels_sub = X_use[idx], labels[idx]
    else:
        X_sub, labels_sub = X_use, labels

    # Only compute if there are at least 2 clusters used
    n_unique = len(np.unique(labels_sub))
    if n_unique >= 2:
        metrics["silhouette"] = silhouette_score(X_sub, labels_sub)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X_sub, labels_sub)
        metrics["davies_bouldin"] = davies_bouldin_score(X_sub, labels_sub)

    # Cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    metrics["cluster_size_mean"] = float(counts.mean())
    metrics["cluster_size_std"] = float(counts.std())
    metrics["cluster_size_min"] = int(counts.min())
    metrics["cluster_size_max"] = int(counts.max())

    return metrics


def save_results(
    output_dir: str | Path,
    kmeans: KMeans,
    scaler: StandardScaler | None,
    labels: np.ndarray,
    role_vocab: dict[str, int],
    metrics: dict,
    config: dict | None = None,
):
    """Save all clustering artifacts to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "kmeans": kmeans,
        "scaler": scaler,
        "labels": labels,
        "role_vocab": role_vocab,
        "metrics": metrics,
        "config": config,
    }

    path = output_dir / "kmeans_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {path}")

    # Also save a human-readable metrics summary
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in sorted(metrics.items()):
            f.write(f"{k}: {v}\n")
    print(f"Saved metrics to {metrics_path}")


def load_results(path: str | Path) -> dict:
    """Load saved clustering artifacts."""
    with open(path, "rb") as f:
        return pickle.load(f)
