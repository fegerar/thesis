"""Visualize per-frame DFL shape-graph annotations.

For each sampled frame draws:
    * the pitch with both teams (home blue, guest red, ball yellow)
    * each player's inferred tactical role (LB / CDM / …)
    * the joint 5x5 zone grid derived from the joint shape graph, as dashed
      threshold lines on both axes (outer + inner splits)
    * the [col, row] zone label next to each player

Usage:
    python -m src.visualize_annotations --match DFL-MAT-J03WMX \\
        --frames 10000,20000,30000 --output annotations_viz.png
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.annotate.dfl import (
    attacking_sign, find_xmls, parse_match_info, pivot_to_frames, rotate_for_team,
)
from src.annotate.shape_graph import build_shape_graph, infer_joint_zone, infer_team_roles

PITCH_X, PITCH_Y = 105.0, 68.0
HOME_COLOR, GUEST_COLOR, BALL_COLOR = "#3498db", "#e74c3c", "#f1c40f"


def _draw_pitch(ax):
    """DFL raw coords are centered at (0,0); pitch is [-52.5, 52.5] x [-34, 34]."""
    hx, hy = PITCH_X / 2, PITCH_Y / 2
    ax.set_xlim(-hx - 2, hx + 2)
    ax.set_ylim(-hy - 2, hy + 2)
    ax.set_aspect("equal")
    ax.set_facecolor("#2e8b57")
    ax.add_patch(plt.Rectangle((-hx, -hy), PITCH_X, PITCH_Y,
                               fill=False, edgecolor="white", linewidth=1.5))
    ax.plot([0, 0], [-hy, hy], color="white", linewidth=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, edgecolor="white", linewidth=1))
    for sx in (-hx, hx - 16.5):
        ax.add_patch(plt.Rectangle((sx, -20.15), 16.5, 40.3,
                                   fill=False, edgecolor="white", linewidth=1))
    ax.set_xticks([]); ax.set_yticks([])


def _axis_thresholds(values, faces, face_proj, node_index, n_levels=5):
    """Return the list of (n_levels-1) grid lines that cut the shape graph,
    paired with a "depth" (pass index) so inner lines can be styled lighter.

    Mirrors `shape_graph._split_levels`: each pass peels off the min/max
    centroid projections of faces whose vertices lie entirely in the current
    inner set, then shrinks the inner set to values within those bounds.
    """
    if len(face_proj) == 0:
        return []
    n = len(values)
    local_to_vertex = np.asarray(node_index)
    current = np.ones(n, dtype=bool)
    center = (n_levels - 1) // 2
    thresholds = []
    for i in range(center):
        if current.sum() < 4:
            break
        inner_vertices = set(local_to_vertex[current].tolist())
        proj = [fp for f, fp in zip(faces, face_proj)
                if f and all(v in inner_vertices for v in f)]
        if not proj:
            break
        lo, hi = float(min(proj)), float(max(proj))
        thresholds.append((lo, i))
        thresholds.append((hi, i))
        current &= (values >= lo) & (values <= hi)
    return thresholds


def _draw_grid(ax, coords, include_mask=None, n_levels=5):
    """Overlay the shape-graph-derived 5x5 grid thresholds on the pitch.

    GKs are excluded from the shape graph when `include_mask` is provided —
    otherwise they would stretch the outer thresholds to the goal lines.
    """
    n = len(coords)
    if include_mask is None:
        mask = np.ones(n, bool)
    else:
        mask = np.asarray(include_mask, bool)
    sg_pts = coords[mask] if mask.any() else coords
    node_index = np.full(n, -1)
    node_index[mask] = np.arange(mask.sum())
    excluded = np.where(~mask)[0]
    if len(excluded):
        diff = coords[excluded, None, :] - sg_pts[None, :, :]
        node_index[excluded] = np.argmin((diff ** 2).sum(-1), axis=1)

    faces, cc = build_shape_graph(sg_pts)

    # the shape-graph polygon edges themselves — the "lines that cut the graph"
    drawn = set()
    for face in faces:
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            key = (a, b) if a < b else (b, a)
            if key in drawn:
                continue
            drawn.add(key)
            ax.plot([sg_pts[a, 0], sg_pts[b, 0]],
                    [sg_pts[a, 1], sg_pts[b, 1]],
                    color="#ffeb3b", linewidth=0.8, alpha=0.55, zorder=3)

    x_th = _axis_thresholds(coords[:, 0], faces,
                            cc[:, 0] if len(cc) else np.empty(0),
                            node_index, n_levels=n_levels)
    y_th = _axis_thresholds(coords[:, 1], faces,
                            cc[:, 1] if len(cc) else np.empty(0),
                            node_index, n_levels=n_levels)
    hx, hy = PITCH_X / 2, PITCH_Y / 2
    center = (n_levels - 1) // 2
    for th, is_vertical in ((x_th, True), (y_th, False)):
        for v, depth in th:
            # outer pass (depth 0) = bright white; inner passes fade to gray
            t = depth / max(1, center - 1) if center > 1 else 0.0
            gray = int(round(255 - 90 * t))
            color = f"#{gray:02x}{gray:02x}{gray:02x}"
            lw = 1.2 - 0.4 * t
            if is_vertical:
                ax.plot([v, v], [-hy, hy], color=color, linestyle="--",
                        linewidth=lw, alpha=0.7, zorder=2)
            else:
                ax.plot([-hx, hx], [v, v], color=color, linestyle="--",
                        linewidth=lw, alpha=0.7, zorder=2)


def _frame_coords(frames, key, players):
    fr = frames[key]
    ids, xy, teams, is_gk = [], [], [], []
    for pid, (x, y, role) in fr["points"].items():
        ids.append(pid); xy.append((x, y)); teams.append(role)
        is_gk.append(bool(pid != "BALL" and players.get(pid, {}).get("gk")))
    return ids, np.array(xy, float), teams, is_gk, fr


def _plot_frame(ax, match_id, key, frames, players, sign, n_levels=5):
    ids, coords, teams, is_gk, fr = _frame_coords(frames, key, players)
    _draw_pitch(ax)
    include_mask = [not gk for gk in is_gk]
    _draw_grid(ax, coords, include_mask=include_mask, n_levels=n_levels)

    zones = infer_joint_zone(coords, include_mask=include_mask, n_levels=n_levels)
    roles = {}
    for team in ("home", "guest"):
        idx = [i for i, t in enumerate(teams) if t == team]
        if len(idx) < 5:
            for i in idx:
                roles[ids[i]] = "UNK"
            continue
        rot = rotate_for_team(coords[idx], sign.get((key[0], team), 1))
        gk_local = next((j for j, i in enumerate(idx) if is_gk[i]), None)
        res = infer_team_roles(rot, gk_local)
        for j, i in enumerate(idx):
            roles[ids[i]] = res[j]

    for i, pid in enumerate(ids):
        x, y = coords[i]
        col, row = zones[i]
        if pid == "BALL":
            ax.scatter(x, y, c=BALL_COLOR, s=90, edgecolors="black",
                       linewidths=1.2, zorder=6, marker="o")
            label = f"BALL [{col},{row}]"
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6, color="black",
                        bbox=dict(facecolor=BALL_COLOR, edgecolor="none",
                                  pad=1, alpha=0.85))
            continue
        color = HOME_COLOR if teams[i] == "home" else GUEST_COLOR
        edge = "yellow" if is_gk[i] else "white"
        ax.scatter(x, y, c=color, s=75, edgecolors=edge, linewidths=1.4, zorder=5)
        label = f"{roles.get(pid, '?')}\n[{col},{row}]"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=5.5, color="white",
                    bbox=dict(facecolor="black", edgecolor="none",
                              pad=1, alpha=0.55))

    ax.set_title(f"{match_id} — {key[0]} frame {key[1]}",
                 fontsize=10, color="white", fontweight="bold")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--match", required=True, help="match id, e.g. DFL-MAT-J03WMX")
    p.add_argument("--frames", default=None,
                   help="comma-separated frame numbers (N); if omitted, picks 4 evenly spaced")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--output", default="annotations_viz.png")
    p.add_argument("--zone-levels", type=int, default=5,
                   help="odd integer >=3; granularity of the joint zone grid")
    args = p.parse_args()
    if args.zone_levels < 3 or args.zone_levels % 2 == 0:
        raise SystemExit("--zone-levels must be an odd integer >= 3")

    info_path, pos_path = find_xmls(args.data_dir, args.match)
    players, _ = parse_match_info(info_path)
    frames = pivot_to_frames(pos_path, players)
    sign = attacking_sign(frames, players)

    keys_fh = sorted(k for k in frames if k[0] == "firstHalf")
    if args.frames:
        wanted = {int(x) for x in args.frames.split(",")}
        keys = [k for k in keys_fh if k[1] in wanted]
    else:
        step = max(1, len(keys_fh) // (args.num_samples + 1))
        keys = keys_fh[step::step][:args.num_samples]
    if not keys:
        raise SystemExit("no matching frames found")

    n = len(keys)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    fig.patch.set_facecolor("#1a1a2e")

    for i, key in enumerate(keys):
        _plot_frame(axes[i // cols, i % cols], args.match, key, frames, players, sign,
                    n_levels=args.zone_levels)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    handles = [
        mpatches.Patch(color=HOME_COLOR, label="Home"),
        mpatches.Patch(color=GUEST_COLOR, label="Guest"),
        mpatches.Patch(facecolor=BALL_COLOR, edgecolor="black", label="Ball"),
        mpatches.Patch(facecolor="gray", edgecolor="yellow",
                       linewidth=2, label="GK (yellow ring)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    fig.suptitle("DFL shape-graph annotations: roles + joint 5x5 zone grid",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved -> {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
