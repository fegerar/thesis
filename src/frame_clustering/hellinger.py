"""Smoothed Hellinger distance over 2-D grid histograms (PyTorch).

For each frame's 2-D histogram H:
    1. Gaussian-smooth H so one-cell player shifts are small perturbations.
    2. Normalize to a probability distribution p = H / sum(H).
    3. Take r = sqrt(p).
The Hellinger distance between two histograms is then
    d(p, q) = (1 / sqrt(2)) * || r_p - r_q ||_2
so the full NxN pairwise matrix is a single `cdist` on the stack of r vectors
per histogram block — no inner Sinkhorn loop, no regularization parameter.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _gaussian_kernel_2d(sigma, device):
    radius = max(1, int(round(3 * sigma)))
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g1 = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    return g1[:, None] * g1[None, :]


def smooth_histograms(h, sigma):
    """Smooth a stack (B, H, W) of 2-D histograms with a Gaussian kernel."""
    if sigma <= 0:
        return h
    kernel = _gaussian_kernel_2d(sigma, h.device)[None, None]
    pad = kernel.shape[-1] // 2
    x = h.unsqueeze(1)  # (B, 1, H, W)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel).squeeze(1)


def sqrt_prob(h, eps=1e-12):
    """Flatten and convert (B, H, W) counts -> (B, H*W) sqrt-probabilities.

    `h` may contain tiny negative values from FP round-off in the conv
    smoothing; we clamp to zero before the sqrt.
    """
    flat = h.reshape(h.shape[0], -1).clamp_min(0.0)
    sums = flat.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.sqrt(flat / sums)


def pairwise_hellinger(h, sigma, row_batch=2048, desc="hellinger",
                       out_device=None):
    """Full NxN Hellinger distance matrix for a stack of 2-D histograms.

    `h`: (N, H, W) non-negative counts on compute device.
    Output: (N, N) float32 on `out_device` (default CPU so the matrix does
    not compete with GPU memory at large N).

    Implementation: Hellinger = sqrt(1 - <sqrt(p), sqrt(q)>) (the squared
    distance is 1 minus the Bhattacharyya coefficient).  We compute it via
    a plain matmul + clamp + sqrt, which is numerically stable even when
    vectors are (near-)identical — unlike `torch.cdist(..., p=2)` which
    relies on `||a||^2 + ||b||^2 - 2<a,b>` and produces NaN for identical
    rows due to round-off making the argument of `sqrt` slightly negative.
    """
    if out_device is None:
        out_device = torch.device("cpu")
    r = sqrt_prob(smooth_histograms(h, sigma))
    n = r.shape[0]
    out = torch.zeros((n, n), dtype=torch.float32, device=out_device)
    for a in tqdm(range(0, n, row_batch), desc=desc, unit="row",
                  dynamic_ncols=True):
        b_end = min(a + row_batch, n)
        inner = r[a:b_end] @ r.T             # Bhattacharyya coefficient (B, n)
        h2 = (1.0 - inner).clamp_min(0.0)    # Hellinger^2, safe for sqrt
        out[a:b_end] = torch.sqrt(h2).to(out_device)
    return out
