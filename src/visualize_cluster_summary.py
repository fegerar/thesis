"""Per-cluster summary + cross-cluster overview for frame clustering runs.

Consumes the artefacts that `cluster_frames` writes to `--output-dir`:
    role_home.pt / role_guest.pt   (N, 5, 5)
    zone_home.pt / zone_guest.pt   (N, Z, Z)
    zone_ball.pt                   (N, Z, Z)
    clusters.json                  {frames: [{cluster, frame_id, phase}, ...]}

Produces:
    cluster_{c}_summary.png  — 1x5 strip of mean heatmaps for cluster c, plus
                               per-cluster tactical stats in the title.
    overview.png             — global view: timeline strip, cluster durations,
                               and the cluster-to-cluster transition matrix.

Usage:
    python -m src.visualize_cluster_summary --clusters-dir clusters --output-dir cluster_viz
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch


MAT_KEYS = ("role_home", "role_guest", "zone_home", "zone_guest", "zone_ball")


def _load_artefacts(clusters_dir):
    with open(os.path.join(clusters_dir, "clusters.json")) as f:
        meta = json.load(f)
    mats = {k: torch.load(os.path.join(clusters_dir, f"{k}.pt"),
                          map_location="cpu").float()
            for k in MAT_KEYS}
    labels = np.array([fm["cluster"] for fm in meta["frames"]], dtype=np.int64)
    return meta, mats, labels


def _palette(k):
    cmap = plt.get_cmap("tab20" if k > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(k)]


def _center_of_mass(grid):
    """(row_com, col_com) for a non-negative 2-D grid."""
    g = np.asarray(grid, dtype=float)
    s = g.sum()
    if s <= 0:
        return float("nan"), float("nan")
    rows = np.arange(g.shape[0])[:, None]
    cols = np.arange(g.shape[1])[None, :]
    return float((g * rows).sum() / s), float((g * cols).sum() / s)


def _spread(grid):
    """std along rows, std along cols (weighted by grid mass)."""
    g = np.asarray(grid, dtype=float)
    s = g.sum()
    if s <= 0:
        return float("nan"), float("nan")
    r_com, c_com = _center_of_mass(g)
    rows = np.arange(g.shape[0])[:, None]
    cols = np.arange(g.shape[1])[None, :]
    var_r = (g * (rows - r_com) ** 2).sum() / s
    var_c = (g * (cols - c_com) ** 2).sum() / s
    return float(np.sqrt(var_r)), float(np.sqrt(var_c))


def _draw_grid(ax, grid, title, origin, cmap):
    im = ax.imshow(grid, cmap=cmap, origin=origin, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


def _cluster_stats_line(mean_mats, zone_side):
    """Short textual stats summarizing the cluster's average configuration."""
    b_r, b_c = _center_of_mass(mean_mats["zone_ball"])
    h_r, h_c = _center_of_mass(mean_mats["zone_home"])
    g_r, g_c = _center_of_mass(mean_mats["zone_guest"])
    h_dr, h_dc = _spread(mean_mats["zone_home"])
    g_dr, g_dc = _spread(mean_mats["zone_guest"])
    sep = float(np.hypot(h_r - g_r, h_c - g_c))
    return (
        f"ball (c,r)=({b_c:.1f},{b_r:.1f})  "
        f"home=({h_c:.1f},{h_r:.1f}) σ=({h_dc:.1f},{h_dr:.1f})  "
        f"guest=({g_c:.1f},{g_r:.1f}) σ=({g_dc:.1f},{g_dr:.1f})  "
        f"team-sep={sep:.2f} (grid={zone_side})"
    )


def _render_cluster(c, mask, mats, output):
    n_c = int(mask.sum())
    means = {k: mats[k][mask].mean(dim=0).numpy() for k in MAT_KEYS}

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    _draw_grid(axes[0], means["zone_ball"],  "ball zone",  "upper", "magma")
    _draw_grid(axes[1], means["zone_home"],  "home zones", "upper", "Blues")
    _draw_grid(axes[2], means["zone_guest"], "guest zones","upper", "Reds")
    _draw_grid(axes[3], means["role_home"],  "home roles (depth↑, width→)",
               "lower", "Blues")
    _draw_grid(axes[4], means["role_guest"], "guest roles (depth↑, width→)",
               "lower", "Reds")
    zone_side = means["zone_ball"].shape[0]
    fig.suptitle(
        f"cluster {c}  —  {n_c} frames\n{_cluster_stats_line(means, zone_side)}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.88])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[summary] saved {output}")


def _contiguous_runs(frames_meta):
    """Yield (cluster_id, run_length_in_frames) for each contiguous same-
    cluster block, split across phase boundaries (firstHalf / secondHalf)."""
    prev_c = None
    prev_phase = None
    length = 0
    for fm in frames_meta:
        c, phase = fm["cluster"], fm["phase"]
        if c == prev_c and phase == prev_phase:
            length += 1
        else:
            if prev_c is not None:
                yield prev_c, length
            prev_c, prev_phase, length = c, phase, 1
    if prev_c is not None:
        yield prev_c, length


def _transition_matrix(frames_meta, k):
    counts = np.zeros((k, k), dtype=np.float64)
    prev_c, prev_phase = None, None
    for fm in frames_meta:
        c, phase = fm["cluster"], fm["phase"]
        if prev_c is not None and phase == prev_phase:
            counts[prev_c, c] += 1
        prev_c, prev_phase = c, phase
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, np.where(row_sums == 0, 1.0, row_sums),
                     out=np.zeros_like(counts), where=row_sums > 0)


def _render_overview(meta, labels, k, output):
    frames_meta = meta["frames"]
    colors = _palette(k)
    trans = _transition_matrix(frames_meta, k)

    # durations per cluster (runs, in stride-units — whatever the annotation stride was)
    durations = defaultdict(list)
    for c, L in _contiguous_runs(frames_meta):
        durations[c].append(L)

    fig = plt.figure(figsize=(17, 5.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 6], hspace=0.3, wspace=0.25)

    # --- timeline strip (top) ---------------------------------------------
    ax_tl = fig.add_subplot(gs[0, :])
    strip = np.zeros((1, len(labels), 3))
    for i, c in enumerate(labels):
        strip[0, i] = colors[int(c)][:3]
    halftime = sum(1 for fm in frames_meta if fm["phase"] == "firstHalf")
    ax_tl.imshow(strip, aspect="auto", interpolation="nearest")
    ax_tl.axvline(halftime, color="black", linewidth=1.0)
    ax_tl.set_xticks([0, halftime, len(labels) - 1])
    ax_tl.set_xticklabels(["KO", "HT", "FT"])
    ax_tl.set_yticks([])
    ax_tl.set_title("match timeline coloured by cluster",
                    fontsize=10, fontweight="bold", loc="left")

    # --- duration distribution (bottom-left) ------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    data = [durations.get(c, []) for c in range(k)]
    ax_d.boxplot(data, tick_labels=[f"c{c}" for c in range(k)], showfliers=False)
    for i, arr in enumerate(data):
        if arr:
            ax_d.scatter(np.full(len(arr), i + 1)
                         + (np.random.RandomState(0).rand(len(arr)) - 0.5) * 0.3,
                         arr, s=3, alpha=0.35, color=colors[i])
    ax_d.set_ylabel("run length (frames)")
    ax_d.set_title("cluster dwell-time", fontsize=10, fontweight="bold")
    ax_d.set_yscale("log")

    # --- transition matrix (bottom-center) --------------------------------
    ax_t = fig.add_subplot(gs[1, 1])
    im = ax_t.imshow(trans, cmap="viridis", vmin=0.0, vmax=1.0)
    for i in range(k):
        for j in range(k):
            ax_t.text(j, i, f"{trans[i, j]:.2f}",
                      ha="center", va="center",
                      color="white" if trans[i, j] < 0.55 else "black",
                      fontsize=8)
    ax_t.set_xticks(range(k)); ax_t.set_yticks(range(k))
    ax_t.set_xticklabels([f"c{c}" for c in range(k)])
    ax_t.set_yticklabels([f"c{c}" for c in range(k)])
    ax_t.set_xlabel("next cluster"); ax_t.set_ylabel("current cluster")
    ax_t.set_title("P(next | current)", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax_t, fraction=0.045, pad=0.04)

    # --- share per cluster (bottom-right) ---------------------------------
    ax_s = fig.add_subplot(gs[1, 2])
    shares = np.bincount(labels, minlength=k) / len(labels)
    ax_s.bar(range(k), shares, color=colors[:k])
    ax_s.set_xticks(range(k))
    ax_s.set_xticklabels([f"c{c}" for c in range(k)])
    ax_s.set_ylabel("frame share")
    ax_s.set_title("cluster frequency", fontsize=10, fontweight="bold")
    for i, v in enumerate(shares):
        ax_s.text(i, v, f"{100 * v:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("cluster overview", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[summary] saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters-dir", required=True,
                   help="folder written by cluster_frames (holds .pt + clusters.json)")
    p.add_argument("--output-dir", default="cluster_viz")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    meta, mats, labels = _load_artefacts(args.clusters_dir)
    k = int(labels.max()) + 1
    print(f"[summary] {len(labels)} frames, k={k}, "
          f"zone_side={mats['zone_ball'].shape[-1]}")

    labels_t = torch.from_numpy(labels)
    for c in range(k):
        mask = (labels_t == c)
        if not mask.any():
            continue
        out = os.path.join(args.output_dir, f"cluster_{c}_summary.png")
        _render_cluster(c, mask, mats, out)

    _render_overview(meta, labels, k, os.path.join(args.output_dir, "overview.png"))


if __name__ == "__main__":
    main()
