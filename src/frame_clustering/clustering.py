"""k-medoids (PAM-style Voronoi update) + elbow selection on a distance matrix.

Operates directly on an NxN symmetric distance matrix, so it's faithful to
the Sinkhorn-EMD geometry (unlike running k-means on distance-matrix rows
as if they were Euclidean feature vectors).
"""

import torch


def kmedoids(dist, k, max_iter=100, seed=42):
    """Return (labels, medoid_indices, inertia) for a k-medoids clustering.

    `dist` is a symmetric (N, N) non-negative matrix; `inertia` is the sum
    of distances from each point to its assigned medoid.
    """
    n = dist.shape[0]
    device = dist.device
    gen = torch.Generator(device=device).manual_seed(seed)

    medoids = torch.randperm(n, generator=gen, device=device)[:k].clone()
    # sentinel labels so the first iteration always runs a medoid update
    # (otherwise a "lucky" initial argmin can match zero-init and we'd break
    # before any reassignment, collapsing every point into cluster 0).
    labels = torch.full((n,), -1, dtype=torch.long, device=device)

    for _ in range(max_iter):
        new_labels = torch.argmin(dist[:, medoids], dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            idx = torch.where(labels == c)[0]
            if idx.numel() == 0:
                # re-seed empty cluster from the globally farthest point
                far = torch.argmax(dist[:, medoids].min(dim=1).values)
                medoids[c] = far
                continue
            sub = dist[idx][:, idx]
            medoids[c] = idx[sub.sum(dim=1).argmin()]

    inertia = float(dist[torch.arange(n, device=device), medoids[labels]].sum().item())
    return labels, medoids, inertia


def elbow_k(ks, inertias):
    """Pick k by maximum perpendicular distance to the (k_min, k_max) chord."""
    if len(ks) == 1:
        return ks[0]
    x = torch.tensor(ks, dtype=torch.float32)
    y = torch.tensor(inertias, dtype=torch.float32)
    x0, y0 = x[0], y[0]
    lx, ly = x[-1] - x0, y[-1] - y0
    line_norm = torch.sqrt(lx * lx + ly * ly) + 1e-9
    # 2-D cross product |lx*(yi-y0) - ly*(xi-x0)|
    d = torch.abs(lx * (y - y0) - ly * (x - x0)) / line_norm
    return ks[int(torch.argmax(d).item())]
