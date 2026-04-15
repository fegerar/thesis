"""Visualize team-level frame-clustering results.

Counterpart to `visualize_clusters.py`, but for the team pipeline (where each
frame contributes two samples — one per team — and the cluster label is
team-specific). Produces:

    * `mds.png` — 2-D classical-MDS scatter of all team-samples, coloured by
      cluster (judge of whether clustering actually separates team-frames).
    * `team_cluster_{c}.png` — grid of random sampled team-frames for cluster c,
      rendered on the pitch with *only the assigned team* highlighted
      (the other team is faded grey, the ball is dimmed). An arrow marks the
      highlighted team's attacking direction.

Usage:
    python -m src.visualize_team_clusters --data-dir data --clusters team_clusters/team_clusters.json --output-dir team_cluster_viz

Samples are routed to their source match via the `match_id` each sample
carries (stamped from the annotated-JSON filename at clustering time). Each
match's raw XML is parsed lazily and cached, so only matches that actually
appear in the chosen cluster samples are loaded.
"""

import argparse
import json
import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.annotate.dfl import (
    attacking_sign, find_xmls, parse_match_info, pivot_to_frames, rotate_for_team,
)
from src.annotate.shape_graph import infer_joint_zone, infer_team_roles
from src.visualize_annotations import (
    _draw_grid, _draw_pitch, _frame_coords,
    BALL_COLOR, GUEST_COLOR, HOME_COLOR, PITCH_X, PITCH_Y,
)
from src.visualize_clusters import _mds_2d, _plot_mds


class MatchCache:
    """Lazy-loaded per-match raw XML data (pivoted frames, players, sign)."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._cache = {}

    def get(self, match_id):
        if match_id not in self._cache:
            info_path, pos_path = find_xmls(self.data_dir, match_id)
            print(f"[team-cluster-viz] parsing {match_id} positions "
                  f"(this is the slow part)...")
            players, _ = parse_match_info(info_path)
            pivoted = pivot_to_frames(pos_path, players)
            sign = attacking_sign(pivoted, players)
            self._cache[match_id] = (pivoted, players, sign)
        return self._cache[match_id]


def _plot_team_frame(ax, match_id, key, team, frames, players, sign,
                      n_levels=5):
    ids, coords, teams, is_gk, _fr = _frame_coords(frames, key, players)
    _draw_pitch(ax)
    include_mask = [not gk for gk in is_gk]
    _draw_grid(ax, coords, include_mask=include_mask, n_levels=n_levels)

    zones = infer_joint_zone(coords, include_mask=include_mask,
                             n_levels=n_levels)

    roles = {}
    idx = [i for i, t in enumerate(teams) if t == team]
    if len(idx) >= 5:
        rot = rotate_for_team(coords[idx], sign.get((key[0], team), 1))
        gk_local = next((j for j, i in enumerate(idx) if is_gk[i]), None)
        res = infer_team_roles(rot, gk_local)
        for j, i in enumerate(idx):
            roles[ids[i]] = res[j]

    team_color = HOME_COLOR if team == "home" else GUEST_COLOR
    for i, pid in enumerate(ids):
        x, y = coords[i]
        col, row = zones[i]
        if pid == "BALL":
            ax.scatter(x, y, c=BALL_COLOR, s=55, edgecolors="black",
                       linewidths=0.8, alpha=0.4, zorder=5)
            continue
        if teams[i] == team:
            edge = "yellow" if is_gk[i] else "white"
            ax.scatter(x, y, c=team_color, s=85, edgecolors=edge,
                       linewidths=1.5, zorder=6)
            label = f"{roles.get(pid, '?')}\n[{col},{row}]"
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=5.5,
                        color="white",
                        bbox=dict(facecolor="black", edgecolor="none",
                                  pad=1, alpha=0.55))
        else:
            ax.scatter(x, y, c="#888", s=35, edgecolors="none",
                       alpha=0.3, zorder=4)

    s = sign.get((key[0], team), 1)
    hx, hy = PITCH_X / 2, PITCH_Y / 2
    y_arrow = -hy + 3
    ax.annotate(
        "", xy=(s * (hx - 6), y_arrow), xytext=(-s * (hx - 6), y_arrow),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.8, alpha=0.85),
        zorder=7,
    )
    ax.text(0, y_arrow + 1.8, f"{team} attacking", ha="center",
            color="white", fontsize=7, alpha=0.9)

    ax.set_title(f"{match_id} — {key[0]} f{key[1]} — {team}",
                 fontsize=9, color="white", fontweight="bold")


def _plot_cluster_samples(c, samples, match_cache, n_samples, output, rng):
    in_cluster = [s for s in samples if s["cluster"] == c]
    rng.shuffle(in_cluster)
    picked = []
    for s in in_cluster:
        mid = s.get("match_id")
        if not mid:
            continue
        try:
            pivoted, players, sign = match_cache.get(mid)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[team-cluster-viz] skipping match {mid}: {exc}")
            continue
        key = (s["phase"], s["frame_id"])
        if key in pivoted:
            picked.append((mid, key, s["team"], pivoted, players, sign))
        if len(picked) >= n_samples:
            break
    if not picked:
        print(f"[team-cluster-viz] cluster {c}: no renderable samples; skipping")
        return

    cols = min(3, len(picked))
    rows = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    fig.patch.set_facecolor("#1a1a2e")
    for i, (mid, key, team, pivoted, players, sign) in enumerate(picked):
        _plot_team_frame(axes[i // cols, i % cols], mid, key, team,
                         pivoted, players, sign)
    for j in range(len(picked), rows * cols):
        axes[j // cols, j % cols].axis("off")

    handles = [
        mpatches.Patch(color=HOME_COLOR, label="home (when highlighted)"),
        mpatches.Patch(color=GUEST_COLOR, label="guest (when highlighted)"),
        mpatches.Patch(color="#888", label="other team (faded)"),
        mpatches.Patch(facecolor=BALL_COLOR, edgecolor="black", label="ball"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    fig.suptitle(
        f"team cluster {c}  —  {len(in_cluster)} team-samples total, "
        f"{len(picked)} shown",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[team-cluster-viz] saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="raw DFL XML folder")
    p.add_argument("--clusters", required=True,
                   help="team_clusters.json path")
    p.add_argument("--distance-matrix", default=None,
                   help="distance_matrix.pt path (defaults to clusters dir)")
    p.add_argument("--output-dir", default="team_cluster_viz")
    p.add_argument("--n-samples", type=int, default=6,
                   help="team-frames per cluster to render")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.clusters, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    labels = np.array([s["cluster"] for s in samples], dtype=np.int64)
    k = int(labels.max()) + 1
    match_ids = sorted({s.get("match_id") for s in samples if s.get("match_id")})
    print(f"[team-cluster-viz] {len(samples)} team-samples, k={k}, "
          f"matches={len(match_ids)}")
    if not match_ids:
        raise SystemExit(
            "samples have no match_id — re-run cluster_teams after updating "
            "the annotation/clustering pipeline (match_id is now stamped into "
            "each annotated frame)."
        )

    dist_path = args.distance_matrix or os.path.join(
        os.path.dirname(args.clusters), "distance_matrix.pt")
    print(f"[team-cluster-viz] loading distance matrix from {dist_path}")
    dist = torch.load(dist_path, map_location="cpu")
    print("[team-cluster-viz] running MDS to 2D...")
    coords = _mds_2d(dist)
    _plot_mds(coords, labels, k, os.path.join(args.output_dir, "mds.png"))

    match_cache = MatchCache(args.data_dir)
    rng = random.Random(args.seed)
    for c in range(k):
        out = os.path.join(args.output_dir, f"team_cluster_{c}.png")
        _plot_cluster_samples(c, samples, match_cache, args.n_samples,
                              out, rng)


if __name__ == "__main__":
    main()
