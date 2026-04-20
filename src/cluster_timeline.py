"""Print a per-frame cluster timeline to the console.

Watch alongside the real match video to label clusters by what you see.
By default shows only transitions (when a cluster assignment changes);
use --all-frames to print every frame.

Usage:
    python -m src.cluster_timeline \
        --clusters team_clusters/team_clusters.json \
        --match DFL-MAT-J03WMX

    # with possession split
    python -m src.cluster_timeline \
        --clusters team_clusters_poss \
        --match DFL-MAT-J03WMX

    # print every frame instead of transitions only
    python -m src.cluster_timeline \
        --clusters team_clusters/team_clusters.json \
        --match DFL-MAT-J03WMX --all-frames
"""

import argparse
import json
import os


def _load_samples(clusters_path):
    """Load samples from a single JSON or a split-possession directory."""
    samples = []
    if os.path.isdir(clusters_path):
        for tag in ("in_possession", "out_of_possession"):
            json_path = os.path.join(clusters_path, tag, "team_clusters.json")
            if not os.path.isfile(json_path):
                continue
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            for s in data["samples"]:
                s.setdefault("possession", tag)
                samples.append(s)
    else:
        with open(clusters_path, encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"]
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", required=True,
                   help="team_clusters.json or split-possession directory")
    p.add_argument("--match", default=None,
                   help="filter to a single match_id")
    p.add_argument("--all-frames", action="store_true",
                   help="print every frame (default: transitions only)")
    p.add_argument("--half", choices=("first", "second", "both"),
                   default="both")
    args = p.parse_args()

    samples = _load_samples(args.clusters)

    # build lookup: (match_id, phase, frame_id, team) -> (cluster, possession)
    lookup = {}
    for s in samples:
        mid = s.get("match_id")
        if args.match and mid != args.match:
            continue
        key = (mid, s["phase"], s["frame_id"], s["team"])
        poss = s.get("possession", "")
        lookup[key] = (s["cluster"], poss)

    if not lookup:
        print("no matching samples found")
        return

    # collect unique frames in order
    frame_keys = set()
    for mid, phase, fid, team in lookup:
        frame_keys.add((mid, phase, fid))
    frame_keys = sorted(frame_keys,
                        key=lambda k: (k[0] or "",
                                       0 if k[1] == "firstHalf" else 1,
                                       k[2]))
    if args.half != "both":
        target = "firstHalf" if args.half == "first" else "secondHalf"
        frame_keys = [(m, ph, f) for m, ph, f in frame_keys if ph == target]

    print(f"{'FRAME':>10}  {'PHASE':<12}  {'HOME':<20}  {'GUEST':<20}")
    print("-" * 68)

    prev_home, prev_guest = None, None
    for mid, phase, fid in frame_keys:
        h_entry = lookup.get((mid, phase, fid, "home"))
        g_entry = lookup.get((mid, phase, fid, "guest"))

        if h_entry:
            hc, hp = h_entry
            h_str = f"c{hc}"
            if hp and hp not in ("all", ""):
                h_str += f" ({hp[:2].upper()}P)"
        else:
            hc, h_str = None, "—"

        if g_entry:
            gc, gp = g_entry
            g_str = f"c{gc}"
            if gp and gp not in ("all", ""):
                g_str += f" ({gp[:2].upper()}P)"
        else:
            gc, g_str = None, "—"

        current = (hc, gc)
        if not args.all_frames and current == (prev_home, prev_guest):
            continue
        prev_home, prev_guest = current

        print(f"{fid:>10}  {phase:<12}  {h_str:<20}  {g_str:<20}")


if __name__ == "__main__":
    main()
