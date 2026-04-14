"""Visualize frame clustering results.

Produces:
    * `mds.png` — 2-D classical-MDS scatter of all frames, coloured by cluster
      (judge of whether the clustering actually separates frames).
    * `cluster_{c}.png` — grid of random sample frames from cluster c, each
      rendered on the pitch with roles + joint zone grid (same visual style as
      `visualize_annotations.py`).

Usage:
    python -m src.visualize_clusters --match DFL-MAT-J03WMX \\
        --clusters clusters/clusters.json --output-dir cluster_viz
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.annotate.dfl import (
    attacking_sign, find_xmls, parse_match_info, pivot_to_frames,
)
from src.visualize_annotations import _plot_frame


def _mds_2d(dist):
    """Classical MDS to 2-D from an NxN distance matrix.

    Uses randomized SVD on the double-centered squared-distance matrix, so
    it scales to N ~ 1e4 without a full eigendecomposition.
    """
    d2 = (dist.double() ** 2)
    n = d2.shape[0]
    row_mean = d2.mean(dim=1, keepdim=True)
    col_mean = d2.mean(dim=0, keepdim=True)
    grand_mean = d2.mean()
    b = -0.5 * (d2 - row_mean - col_mean + grand_mean)
    b = 0.5 * (b + b.T)  # kill numerical drift from symmetry
    u, s, _ = torch.svd_lowrank(b, q=6, niter=4)
    coords = u[:, :2] * torch.sqrt(s[:2].clamp_min(0))
    return coords.float().numpy()


def _palette(k):
    cmap = plt.get_cmap("tab20" if k > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(k)]


def _plot_mds(coords, labels, k, output):
    colors = _palette(k)
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(k):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.5,
                   c=[colors[c]], label=f"cluster {c}  (n={int(mask.sum())})")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.9)
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    ax.set_title("Cluster separation — classical MDS on the distance matrix",
                 fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[cluster-viz] saved {output}")


def _plot_cluster_samples(c, frames_meta, pivoted, players, sign, match,
                          n_samples, output, rng):
    in_cluster = [fm for fm in frames_meta if fm["cluster"] == c]
    rng.shuffle(in_cluster)
    picked = []
    for fm in in_cluster:
        key = (fm["phase"], fm["frame_id"])
        if key in pivoted:
            picked.append(key)
        if len(picked) >= n_samples:
            break
    if not picked:
        print(f"[cluster-viz] cluster {c}: no renderable frames, skipping")
        return

    cols = min(3, len(picked))
    rows = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    fig.patch.set_facecolor("#1a1a2e")
    for i, key in enumerate(picked):
        _plot_frame(axes[i // cols, i % cols], match, key, pivoted, players, sign)
    for j in range(len(picked), rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(
        f"cluster {c}  —  {len(in_cluster)} frames total, "
        f"{len(picked)} sampled",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[cluster-viz] saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", required=True, help="match id, e.g. DFL-MAT-J03WMX")
    p.add_argument("--data-dir", default="data", help="raw DFL XML folder")
    p.add_argument("--clusters", required=True, help="clusters.json path")
    p.add_argument("--distance-matrix", default=None,
                   help="distance_matrix.pt path (defaults to clusters dir)")
    p.add_argument("--output-dir", default="cluster_viz")
    p.add_argument("--n-samples", type=int, default=6,
                   help="frames per cluster to render")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.clusters) as f:
        data = json.load(f)
    frames_meta = data["frames"]
    labels = np.array([fm["cluster"] for fm in frames_meta], dtype=np.int64)
    k = int(labels.max()) + 1
    print(f"[cluster-viz] {len(frames_meta)} frames, k={k}")

    dist_path = args.distance_matrix or os.path.join(
        os.path.dirname(args.clusters), "distance_matrix.pt")
    print(f"[cluster-viz] loading distance matrix from {dist_path}")
    dist = torch.load(dist_path, map_location="cpu")
    print("[cluster-viz] running MDS to 2D...")
    coords = _mds_2d(dist)
    _plot_mds(coords, labels, k, os.path.join(args.output_dir, "mds.png"))

    info_path, pos_path = find_xmls(args.data_dir, args.match)
    print(f"[cluster-viz] parsing {args.match} positions (this is the slow part)...")
    players, _teams = parse_match_info(info_path)
    pivoted = pivot_to_frames(pos_path, players)
    sign = attacking_sign(pivoted, players)

    rng = random.Random(args.seed)
    for c in range(k):
        out = os.path.join(args.output_dir, f"cluster_{c}.png")
        _plot_cluster_samples(c, frames_meta, pivoted, players, sign,
                              args.match, args.n_samples, out, rng)


if __name__ == "__main__":
    main()
