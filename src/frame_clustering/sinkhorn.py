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


def _iter_triu_batches(n, pair_batch):
    """Yield (i_cpu, j_cpu) int64 index batches covering the strict upper
    triangle of an (n, n) matrix, without materializing all pairs at once.

    We walk rows i; for each row the columns are i+1..n-1, and we slice that
    range into chunks of size up to `pair_batch`.
    """
    i = 0
    while i < n - 1:
        # how many full rows fit into a single batch starting at row i?
        cols_per_row = n - 1 - i
        rows_in_batch = max(1, pair_batch // max(cols_per_row, 1))
        end_row = min(n - 1, i + rows_in_batch)
        ii_list, jj_list = [], []
        taken = 0
        for r in range(i, end_row):
            width = n - 1 - r
            take = min(width, pair_batch - taken)
            if take <= 0:
                break
            jj_list.append(torch.arange(r + 1, r + 1 + take, dtype=torch.int64))
            ii_list.append(torch.full((take,), r, dtype=torch.int64))
            taken += take
            if taken >= pair_batch:
                end_row = r + 1
                break
        if taken == 0:
            # fall back: single partial row
            width = n - 1 - i
            take = min(width, pair_batch)
            jj_list = [torch.arange(i + 1, i + 1 + take, dtype=torch.int64)]
            ii_list = [torch.full((take,), i, dtype=torch.int64)]
            end_row = i + 1
        yield torch.cat(ii_list), torch.cat(jj_list)
        i = end_row


def _total_pairs(n):
    return n * (n - 1) // 2


def pairwise_sinkhorn(x, cost, reg, n_iter, pair_batch, desc="sinkhorn",
                      out_device=None):
    """Full NxN symmetric pairwise Sinkhorn distance matrix.

    `x` lives on a compute device (e.g. CUDA); the result is kept on
    `out_device` (defaults to CPU) so the NxN matrix does not compete with
    the compute memory.  Pair indices are generated lazily to avoid the
    6 GiB+ spike from `torch.triu_indices` at large N.
    """
    n = x.shape[0]
    if out_device is None:
        out_device = torch.device("cpu")
    out = torch.zeros((n, n), dtype=torch.float32, device=out_device)

    total = _total_pairs(n)
    pbar = tqdm(total=total, desc=desc, unit="pair", unit_scale=True,
                dynamic_ncols=True)
    for i_cpu, j_cpu in _iter_triu_batches(n, pair_batch):
        i_dev = i_cpu.to(x.device, non_blocking=True)
        j_dev = j_cpu.to(x.device, non_blocking=True)
        d = sinkhorn_batch(x[i_dev], x[j_dev], cost=cost, reg=reg, n_iter=n_iter)
        d_out = d.detach().to(out_device, non_blocking=True)
        out[i_cpu, j_cpu] = d_out
        out[j_cpu, i_cpu] = d_out
        pbar.update(i_cpu.numel())
    pbar.close()
    return out
