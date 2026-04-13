"""Batched Sinkhorn EMD over row-stochastic histograms (PyTorch)."""

import torch
from tqdm import tqdm


def sinkhorn_batch(a, b, cost, reg, n_iter, eps=1e-8):
    """Sinkhorn distance for a batch of histogram pairs sharing one cost.

    `a`, `b`: (B, n) / (B, m) row-stochastic histograms.
    `cost`:  (n, m) ground cost.  Returns (B,) regularized OT cost.
    """
    k_mat = torch.exp(-cost / reg)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        u = a / (v @ k_mat.T + eps)
        v = b / (u @ k_mat + eps)
    plan = u[:, :, None] * k_mat[None, :, :] * v[:, None, :]
    return (plan * cost[None, :, :]).sum(dim=(1, 2))


def pairwise_sinkhorn(x, cost, reg, n_iter, pair_batch, desc="sinkhorn"):
    """Full NxN symmetric pairwise Sinkhorn distance matrix."""
    n = x.shape[0]
    out = torch.zeros((n, n), dtype=torch.float32, device=x.device)
    triu = torch.triu_indices(n, n, offset=1, device=x.device)
    total = triu.shape[1]
    for start in tqdm(range(0, total, pair_batch), desc=desc,
                      unit="batch", dynamic_ncols=True):
        end = min(start + pair_batch, total)
        i, j = triu[0, start:end], triu[1, start:end]
        d = sinkhorn_batch(x[i], x[j], cost=cost, reg=reg, n_iter=n_iter)
        out[i, j] = d
        out[j, i] = d
    return out
