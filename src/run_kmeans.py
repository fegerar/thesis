"""
Run k-means++ clustering on shapegraphs.

Usage:
    python src/run_kmeans.py --config config/kmeans_default.yaml
"""

import argparse
import time

import yaml

from kmeans import load_and_featurize, fit_kmeans, assign_clusters, evaluate_clustering, save_results


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="K-means++ clustering on shapegraphs")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load and featurize
    data_cfg = cfg["data"]
    print("Loading and featurizing shapegraphs...")
    t0 = time.time()
    X, role_vocab = load_and_featurize(
        data_path=data_cfg["path"],
        max_nodes=data_cfg.get("max_nodes", 22),
    )
    print(f"Featurization took {time.time() - t0:.1f}s")

    # Fit k-means++
    cluster_cfg = cfg["clustering"]
    print(f"\nFitting k-means++ with K={cluster_cfg['n_clusters']}...")
    t0 = time.time()
    kmeans, scaler = fit_kmeans(
        X,
        n_clusters=cluster_cfg["n_clusters"],
        n_init=cluster_cfg.get("n_init", 10),
        max_iter=cluster_cfg.get("max_iter", 300),
        random_state=data_cfg.get("seed", 42),
        normalize=cluster_cfg.get("normalize", True),
    )
    print(f"K-means fitting took {time.time() - t0:.1f}s")

    # Assign clusters
    labels = assign_clusters(X, kmeans, scaler)

    # Evaluate
    print("\nEvaluating clustering quality...")
    eval_cfg = cfg.get("evaluation", {})
    metrics = evaluate_clustering(
        X, labels, kmeans, scaler,
        sample_size=eval_cfg.get("silhouette_sample_size", 50_000),
    )

    print("\n--- Clustering Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # Save
    output_dir = cfg.get("output", {}).get("dir", "results/kmeans")
    save_results(output_dir, kmeans, scaler, labels, role_vocab, metrics, config=cfg)

    print(f"\nDone. {len(labels)} shapegraphs assigned to {metrics['n_clusters_used']} clusters.")


if __name__ == "__main__":
    main()
