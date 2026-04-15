"""CLI: cluster annotated DFL frames by smoothed-Hellinger distance + k-medoids.

Example:
    python -m src.cluster_frames --input data_annotated \\
        --output-dir clusters --stride 5 --zone-levels 7
"""

import argparse

from .frame_clustering import run_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="annotated JSON path or directory containing annotated JSON files",
    )
    p.add_argument("--output-dir", default="clusters")
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
    p.add_argument("--smooth-sigma", type=float, default=0.7,
                   help="Gaussian sigma (in grid cells) applied to each "
                        "histogram before the Hellinger distance; 0 disables it")
    p.add_argument("--row-batch", type=int, default=2048,
                   help="#rows per Hellinger cdist chunk")
    p.add_argument("--weight-role", type=float, default=0.4)
    p.add_argument("--weight-zone", type=float, default=0.4)
    p.add_argument("--weight-ball", type=float, default=0.2)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--kmedoids-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
