"""Per-player x/y timelines over a full match (Brandes 2025 style).

For every player, two rows are shown on the timeline — the top row is the
player's y-level (depth 0..4, defender -> forward) and the bottom row is the
player's x-level (width 0..4, left flank -> right flank).

Two figures are produced (one per --output-prefix):
    * roles  — levels come from the PER-TEAM role inference (team-attacking-up
      coordinates). Depth 4 is always the attacking end for that team.
    * zones  — levels come from the JOINT 5x5 shape-graph zone, in raw pitch
      coords. col = pitch length axis, row = pitch width axis.

Halftime is marked by a vertical line. Substitutions appear as gaps where the
player is not on the pitch.

Usage:
    python -m src.visualize_timelines --match DFL-MAT-J03WMX \\
        --stride 5 --output-prefix timelines
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, to_rgb
from tqdm import tqdm

from src.annotate.dfl import (
    attacking_sign, find_xmls, parse_match_info, pivot_to_frames, rotate_for_team,
)
from src.annotate.shape_graph import infer_joint_zone, infer_team_positions


# Depth / width palettes share the same 5-step Brandes-style gradient so the
# two sub-rows of each player are visually comparable.
DEPTH_COLORS = [
    "#225CAE",  # 0 — defender / back
    "#799DCE",  # 1 — DefMid
    "#E1E1E1",  # 2 — Midfield
    "#D48681",  # 3 — AttMid
    "#B6362D",  # 4 — forward
]
WIDTH_COLORS = [
    "#8D6714",  # 0 — left flank
    "#D1C1A0",  # 1 — half-left
    "#E1E1E1",  # 2 — center
    "#C0C6A0",  # 3 — half-right
    "#627413",  # 4 — right flank
]
GK_COLOR = "#000000"
OFF_COLOR = "#000000"


def _collect(match_id, data_dir, stride, zone_levels):
    info_path, pos_path = find_xmls(data_dir, match_id)
    print(f"[{match_id}] parsing match info...")
    players, teams_info = parse_match_info(info_path)
    print(f"[{match_id}] parsing positions XML (this is the slow part)...")
    frames = pivot_to_frames(pos_path, players)
    print(f"[{match_id}] {len(frames)} frames loaded; inferring attacking direction...")
    sign = attacking_sign(frames, players)

    fh = sorted(k for k in frames if k[0] == "firstHalf")
    sh = sorted(k for k in frames if k[0] == "secondHalf")
    if stride > 1:
        fh = fh[::stride]; sh = sh[::stride]
    keys = fh + sh
    halftime = len(fh)
    T = len(keys)

    team_players = {"home": {}, "guest": {}}
    for fr in frames.values():
        for pid, (_, _, team) in fr["points"].items():
            if team not in team_players or pid in team_players[team]:
                continue
            meta = players.get(pid, {})
            team_players[team][pid] = {
                "shirt": meta.get("shirt", -1),
                "gk": bool(meta.get("gk")),
                "name": meta.get("name", pid),
            }

    # per-team: role-inference depth/width (in team-attacking-up coords)
    # per-team: joint-zone col/row (raw pitch coords)
    empty = lambda: np.full(T, -1, dtype=np.int8)
    role_depth = {t: {pid: empty() for pid in pl} for t, pl in team_players.items()}
    role_width = {t: {pid: empty() for pid in pl} for t, pl in team_players.items()}
    zone_col = {t: {pid: empty() for pid in pl} for t, pl in team_players.items()}
    zone_row = {t: {pid: empty() for pid in pl} for t, pl in team_players.items()}
    gk_mask = {t: {pid: np.zeros(T, dtype=bool) for pid in pl}
               for t, pl in team_players.items()}

    for fi, key in enumerate(tqdm(keys, desc=f"[{match_id}] shape graphs",
                                   unit="frame", dynamic_ncols=True)):
        fr = frames[key]
        ids, coords, teams, is_gk = [], [], [], []
        for pid, (x, y, team) in fr["points"].items():
            ids.append(pid); coords.append((x, y)); teams.append(team)
            is_gk.append(bool(pid != "BALL" and players.get(pid, {}).get("gk")))
        coords = np.array(coords, dtype=float)
        if len(coords) == 0:
            continue

        include_mask = [not gk for gk in is_gk]
        zones = infer_joint_zone(coords, include_mask=include_mask,
                                 n_levels=zone_levels)

        for team in ("home", "guest"):
            idx = [i for i, t in enumerate(teams) if t == team]
            if len(idx) < 5:
                continue
            s = sign.get((key[0], team), 1)
            rot = rotate_for_team(coords[idx], s)
            gk_local = next((j for j, i in enumerate(idx) if is_gk[i]), None)
            pos = infer_team_positions(rot, gk_local)
            for j, orig in enumerate(idx):
                pid = ids[orig]
                role_depth[team][pid][fi] = pos[j]["depth"]
                role_width[team][pid][fi] = pos[j]["width"]
                col, row = zones[orig]
                # orient col so it matches team-attacking-up: attacking end -> last level.
                zone_col[team][pid][fi] = col if s == 1 else (zone_levels - 1) - col
                zone_row[team][pid][fi] = row
                if is_gk[orig]:
                    gk_mask[team][pid][fi] = True

    return {
        "keys": keys, "halftime": halftime, "T": T, "zone_levels": zone_levels,
        "team_players": team_players, "teams_info": teams_info,
        "role_depth": role_depth, "role_width": role_width,
        "zone_col": zone_col, "zone_row": zone_row,
        "gk_mask": gk_mask,
    }


# --- matrix assembly --------------------------------------------------------


def _encode(values, gk, n_levels):
    """Encode per-frame 0..n_levels-1 level into colormap index.

    Reserved indices: n_levels = GK, n_levels+1 = off pitch.
    """
    enc = np.full_like(values, n_levels + 1, dtype=np.int8)
    ok = values >= 0
    enc[ok] = values[ok]
    enc[gk] = n_levels
    return enc


def _sort_players(team_players, role_depth, gk_mask):
    """GKs first, then outfield sorted by mean depth (back -> front)."""
    rows = []
    for pid, meta in team_players.items():
        d = role_depth[pid]
        present = d >= 0
        mean_d = float(d[present].mean()) if present.any() else -1.0
        is_gk = meta["gk"] or gk_mask[pid].any()
        rows.append((pid, meta["shirt"], is_gk, mean_d))
    rows.sort(key=lambda t: (0 if t[2] else 1, t[3]))
    return [r[0] for r in rows]


# --- plotting ---------------------------------------------------------------


def _sample_cmap(name, n):
    return [tuple(colormaps[name](t)[:3]) for t in np.linspace(0.0, 1.0, n)]


def _palettes(kind, n_levels):
    """Return (depth_colors, width_colors) for the given visualization kind."""
    if kind == "roles":
        return list(DEPTH_COLORS), list(WIDTH_COLORS)
    # zones: sample diverging colormaps so center stays neutral even for large n
    return _sample_cmap("RdBu_r", n_levels), _sample_cmap("BrBG_r", n_levels)


def _build_row_cmaps(depth_colors, width_colors):
    """Two colormaps (one per sub-row). 0..n-1 levels, n = GK, n+1 = off."""
    depth_cmap = ListedColormap(list(depth_colors) + [GK_COLOR, OFF_COLOR])
    width_cmap = ListedColormap(list(width_colors) + [GK_COLOR, OFF_COLOR])
    return depth_cmap, width_cmap


def _plot_team_timeline(ax, ordered_pids, y_mat, x_mat, team_players,
                        depth_cmap, width_cmap, halftime, T, title):
    """Draw an interleaved (y, x) pair of rows per player on `ax`."""
    n = len(ordered_pids)
    rgb = np.zeros((2 * n, T, 3), dtype=float)
    dcols = np.array([to_rgb(c) for c in depth_cmap.colors])
    wcols = np.array([to_rgb(c) for c in width_cmap.colors])
    for i in range(n):
        rgb[2 * i]     = dcols[y_mat[i]]
        rgb[2 * i + 1] = wcols[x_mat[i]]
    ax.imshow(rgb, aspect="auto", interpolation="nearest")

    yticks, ylabels = [], []
    for i, pid in enumerate(ordered_pids):
        meta = team_players[pid]
        name = meta.get("name", pid)
        shirt = meta.get("shirt", -1)
        yticks.append(2 * i); ylabels.append(f"{name} (#{shirt}) · y")
        yticks.append(2 * i + 1); ylabels.append(f"{name} (#{shirt}) · x")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7, color="black")
    for i in range(n):
        ax.axhline(2 * i - 0.5, color="black", linewidth=0.4, alpha=0.6)
    ax.axhline(2 * n - 0.5, color="black", linewidth=0.4, alpha=0.6)

    ax.set_xticks([0, halftime, T - 1])
    ax.set_xticklabels(["KO", "HT", "FT"], fontsize=9, color="black")
    ax.tick_params(axis="both", colors="black", length=0)
    for spine in ax.spines.values():
        spine.set_color("black"); spine.set_linewidth(0.5)
    ax.axvline(halftime, color="black", linewidth=1.0, alpha=0.9)
    ax.set_title(title, color="black", fontweight="bold", fontsize=11, loc="left")



def _legend(fig, kind, depth_colors, width_colors):
    if kind == "roles":
        y_labels = ["Defender", "DefMid", "Midfield", "AttMid", "Forward"]
        x_labels = ["L flank", "Half-L", "Center", "Half-R", "R flank"]
    else:
        y_labels = [f"row {i}" for i in range(len(depth_colors))]
        x_labels = [f"col {i}" for i in range(len(width_colors))]
    y_handles = [mpatches.Patch(color=c, label=f"y: {l}")
                 for c, l in zip(depth_colors, y_labels)]
    x_handles = [mpatches.Patch(color=c, label=f"x: {l}")
                 for c, l in zip(width_colors, x_labels)]
    gk = [mpatches.Patch(color=GK_COLOR, label="GK")]
    ncol = max(6, len(depth_colors) + 1)
    fig.legend(handles=y_handles + x_handles + gk, loc="lower center",
               ncol=ncol, fontsize=8, facecolor="#ffffff",
               edgecolor="black", labelcolor="black")


def _render(kind, data, source, output, match_id):
    team_players = data["team_players"]
    teams_info = data["teams_info"]
    halftime, T = data["halftime"], data["T"]
    y_series_fn, x_series_fn = source
    n_levels = 5 if kind == "roles" else data["zone_levels"]
    depth_colors, width_colors = _palettes(kind, n_levels)

    n_home = len(team_players["home"])
    n_guest = len(team_players["guest"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 0.35 * (n_home + n_guest) + 3),
                             gridspec_kw={"hspace": 0.35,
                                          "height_ratios": [n_home, n_guest]})
    fig.patch.set_facecolor("#ffffff")
    depth_cmap, width_cmap = _build_row_cmaps(depth_colors, width_colors)

    for ax, team in zip(axes, ["home", "guest"]):
        ordered = _sort_players(team_players[team], data["role_depth"][team],
                                data["gk_mask"][team])
        y_series = y_series_fn(team)
        x_series = x_series_fn(team)
        mat_y = np.stack([_encode(y_series[pid], data["gk_mask"][team][pid], n_levels)
                          for pid in ordered])
        mat_x = np.stack([_encode(x_series[pid], data["gk_mask"][team][pid], n_levels)
                          for pid in ordered])
        tname = teams_info.get(team, {}).get("name", team.upper())
        _plot_team_timeline(ax, ordered, mat_y, mat_x, team_players[team],
                            depth_cmap, width_cmap, halftime, T,
                            f"{tname} ({team})  —  {kind}")

    home_name = teams_info.get("home", {}).get("name", "Home")
    guest_name = teams_info.get("guest", {}).get("name", "Guest")
    fig.suptitle(f"{match_id}   {home_name}  vs  {guest_name}   —   per-player {kind} timeline",
                 color="black", fontsize=13, fontweight="bold")
    _legend(fig, kind, depth_colors, width_colors)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved -> {output}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--match", required=True)
    p.add_argument("--stride", type=int, default=5,
                   help="keep every Nth frame (25 fps raw)")
    p.add_argument("--output-prefix", default="timelines")
    p.add_argument("--zone-levels", type=int, default=5,
                   help="odd integer >=3; granularity of the joint zone grid")
    args = p.parse_args()
    if args.zone_levels < 3 or args.zone_levels % 2 == 0:
        raise SystemExit("--zone-levels must be an odd integer >= 3")

    data = _collect(args.match, args.data_dir, args.stride, args.zone_levels)

    _render(
        "roles", data,
        (lambda team: data["role_depth"][team],
         lambda team: data["role_width"][team]),
        f"{args.output_prefix}_roles.png",
        args.match,
    )
    _render(
        "zones", data,
        (lambda team: data["zone_row"][team],
         lambda team: data["zone_col"][team]),
        f"{args.output_prefix}_zones.png",
        args.match,
    )


if __name__ == "__main__":
    main()
