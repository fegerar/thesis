"""CLI: cluster team-frames (home + guest stacked) by smoothed-Hellinger
distance + k-medoids, using only role + zone histograms.

Example:
    python -m src.cluster_teams --input data_annotated --output-dir team_clusters --stride 5 --zone-levels 7
"""

import argparse

from .frame_clustering import run_team_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="annotated JSON path or directory containing annotated JSON files",
    )
    p.add_argument("--output-dir", default="team_clusters")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--zone-levels", type=int, default=None,
                   help="grid side length of the joint zone; inferred from the "
                        "data when omitted (must match the annotation run for "
                        "shape_graph mode; required for pitch mode)")
    p.add_argument("--zone-mode", choices=("shape_graph", "pitch"),
                   default="shape_graph",
                   help="shape_graph: use the per-frame shape-graph zone (default). "
                        "pitch: bin players on a fixed NxN grid over the pitch "
                        "(uses pitch_xy; requires re-annotated frames).")
    p.add_argument("--smooth-sigma-role", type=float, default=0.3,
                   help="Gaussian sigma (in grid cells) applied to the role "
                        "histogram before the Hellinger distance; 0 disables it")
    p.add_argument("--smooth-sigma-zone", type=float, default=0.7,
                   help="Gaussian sigma (in grid cells) applied to the zone "
                        "histogram before the Hellinger distance; 0 disables it")
    p.add_argument("--row-batch", type=int, default=2048)
    p.add_argument("--weight-role", type=float, default=0.5)
    p.add_argument("--weight-zone", type=float, default=0.5)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--kmedoids-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-possession", action="store_true",
                   help="split team-frames into in-possession vs "
                        "out-of-possession and cluster each group separately. "
                        "Requires re-annotated data with the 'possession' field "
                        "(from DFL BallPossession).")
    args = p.parse_args()
    run_team_pipeline(args)


if __name__ == "__main__":
    main()
