"""Per-cluster summary for team-level clustering runs.

Consumes the artefacts that `cluster_teams` writes to `--output-dir`:
    role.pt              (N, 5, 5)
    zone.pt              (N, Z, Z)
    team_clusters.json   {samples: [{team, frame_idx, cluster, phase}, ...]}

Produces:
    team_cluster_{c}_summary.png  — 1x2 strip of mean role + zone heatmaps.
    team_overview.png             — cluster frequency + home/guest split.

Usage:
    python -m src.visualize_team_cluster_summary \\
        --clusters-dir team_clusters --output-dir team_cluster_viz
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_artefacts(clusters_dir):
    with open(os.path.join(clusters_dir, "team_clusters.json")) as f:
        meta = json.load(f)
    role = torch.load(os.path.join(clusters_dir, "role.pt"),
                      map_location="cpu").float()
    zone = torch.load(os.path.join(clusters_dir, "zone.pt"),
                      map_location="cpu").float()
    labels = np.array([s["cluster"] for s in meta["samples"]], dtype=np.int64)
    teams = np.array([s["team"] for s in meta["samples"]])
    return meta, role, zone, labels, teams


def _palette(k):
    cmap = plt.get_cmap("tab20" if k > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(k)]


def _center_of_mass(grid):
    g = np.asarray(grid, dtype=float)
    s = g.sum()
    if s <= 0:
        return float("nan"), float("nan")
    rows = np.arange(g.shape[0])[:, None]
    cols = np.arange(g.shape[1])[None, :]
    return float((g * rows).sum() / s), float((g * cols).sum() / s)


def _spread(grid):
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
    ax.imshow(grid, cmap=cmap, origin=origin, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, fontweight="bold")


def _render_cluster(c, mask, role, zone, teams, output):
    n_c = int(mask.sum())
    n_home = int((teams[mask] == "home").sum())
    n_guest = n_c - n_home
    mean_role = role[mask].mean(dim=0).numpy()
    mean_zone = zone[mask].mean(dim=0).numpy()

    r_com_r, r_com_c = _center_of_mass(mean_role)
    r_sig_r, r_sig_c = _spread(mean_role)
    z_com_r, z_com_c = _center_of_mass(mean_zone)
    z_sig_r, z_sig_c = _spread(mean_zone)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    _draw_grid(axes[0], mean_role, "tactical roles (depth↑, width→)",
               "lower", "Greens")
    _draw_grid(axes[1], mean_zone, "zone occupancy (pitch frame)",
               "upper", "Purples")
    fig.suptitle(
        f"team cluster {c}  —  {n_c} team-frames  "
        f"(home={n_home}, guest={n_guest})\n"
        f"role com=({r_com_c:.1f},{r_com_r:.1f}) σ=({r_sig_c:.1f},{r_sig_r:.1f})  "
        f"zone com=({z_com_c:.1f},{z_com_r:.1f}) σ=({z_sig_c:.1f},{z_sig_r:.1f})  "
        f"(grid={mean_zone.shape[0]})",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.85])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[team-summary] saved {output}")


def _render_overview(labels, teams, k, output):
    colors = _palette(k)
    home_mask = teams == "home"
    guest_mask = teams == "guest"
    home_counts = np.bincount(labels[home_mask], minlength=k)
    guest_counts = np.bincount(labels[guest_mask], minlength=k)
    totals = home_counts + guest_counts
    shares = totals / max(1, totals.sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_s = axes[0]
    ax_s.bar(range(k), shares, color=colors[:k])
    ax_s.set_xticks(range(k))
    ax_s.set_xticklabels([f"c{c}" for c in range(k)])
    ax_s.set_ylabel("team-frame share")
    ax_s.set_title("cluster frequency", fontsize=10, fontweight="bold")
    for i, v in enumerate(shares):
        ax_s.text(i, v, f"{100 * v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax_b = axes[1]
    x = np.arange(k)
    width = 0.4
    ax_b.bar(x - width / 2, home_counts, width, label="home", color="#4C78A8")
    ax_b.bar(x + width / 2, guest_counts, width, label="guest", color="#E45756")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"c{c}" for c in range(k)])
    ax_b.set_ylabel("team-frame count")
    ax_b.set_title("home vs guest per cluster (sanity check: mostly balanced = "
                   "team-agnostic)", fontsize=10, fontweight="bold")
    ax_b.legend()

    fig.suptitle("team cluster overview", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.0, 1, 0.93])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[team-summary] saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters-dir", required=True)
    p.add_argument("--output-dir", default="team_cluster_viz")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    meta, role, zone, labels, teams = _load_artefacts(args.clusters_dir)
    k = int(labels.max()) + 1
    print(f"[team-summary] {len(labels)} team-frames, k={k}, "
          f"zone_side={zone.shape[-1]}")

    labels_t = torch.from_numpy(labels)
    for c in range(k):
        mask = (labels_t == c)
        if not mask.any():
            continue
        out = os.path.join(args.output_dir, f"team_cluster_{c}_summary.png")
        _render_cluster(c, mask, role, zone, teams, out)

    _render_overview(labels, teams, k,
                     os.path.join(args.output_dir, "team_overview.png"))


if __name__ == "__main__":
    main()
