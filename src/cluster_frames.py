"""CLI: cluster annotated DFL frames by Sinkhorn EMD + k-medoids.

Example:
    python -m src.cluster_frames --input data_annotated/DFL-MAT-J03WMX.json \\
        --output-dir clusters --stride 5 --zone-levels 7
"""

import argparse

from frame_clustering import run_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="annotated JSON path")
    p.add_argument("--output-dir", default="clusters")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--zone-levels", type=int, default=None,
                   help="grid side length of the joint zone; inferred from the "
                        "data when omitted (must match the annotation run)")
    p.add_argument("--reg", type=float, default=0.08, help="Sinkhorn regularization")
    p.add_argument("--sinkhorn-iters", type=int, default=80)
    p.add_argument("--pair-batch", type=int, default=4096,
                   help="#pairs per Sinkhorn batch")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--kmedoids-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
